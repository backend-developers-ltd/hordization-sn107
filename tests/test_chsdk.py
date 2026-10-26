import asyncio
import contextvars
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from compute_horde_sdk import v2 as chsdk

from templates import _chsdk, _common, bcftools, deepvariant, freebayes, gatk
from utils import scoring


class Engine:
    def __init__(self, blocked=(), chunks=None, returncode=0):
        stages = ("image", "create", "attach", "start", "logs", "remove")
        self.gates = {stage: threading.Event() for stage in stages}
        self.entered = {stage: threading.Event() for stage in stages}
        for stage in stages:
            if stage not in blocked:
                self.gates[stage].set()
        self.calls = []
        self.chunks = chunks if chunks is not None else [(1, b"stdout\r\n"), (2, b"stderr\r\n")]
        self.returncode = returncode
        self.api = self
        self.remove_error = None
        self.create_error = None

    def stage(self, stage):
        self.calls.append(stage)
        self.entered[stage].set()
        assert self.gates[stage].wait(10), f"Test did not release Docker stage {stage}"

    def inspect_image(self, image):
        self.stage("image")
        return {"Os": "linux", "Architecture": "amd64"}

    def create_host_config(self, **kwargs):
        return kwargs

    def create_container(self, **kwargs):
        self.create_kwargs = kwargs
        self.stage("create")
        if self.create_error:
            raise self.create_error
        return {"Id": kwargs["name"]}

    def start(self, identity):
        self.stage("start")

    def stream(self):
        for channel, data in self.chunks:
            yield (data, None) if channel == 1 else (None, data)
        self.stage("logs")

    def attach(self, identity, **kwargs):
        assert kwargs == dict(stdout=True, stderr=True, stream=True, demux=True)
        self.stage("attach")
        engine = self

        class Stream:
            def __init__(self):
                self.chunks = engine.stream()

            def __next__(self):
                return next(self.chunks)

            def close(self):
                engine.gates["logs"].set()

        return Stream()

    def wait(self, identity):
        return {"StatusCode": self.returncode}

    def inspect_container(self, identity):
        return {"Id": identity, "Config": {"Labels": self.create_kwargs["labels"]}}

    def remove_container(self, identity, *, force, v):
        assert force and v
        self.stage("remove")
        if self.remove_error:
            raise self.remove_error
        self.gates["logs"].set()

    def close(self):
        pass


def command(name="minos-test", *args):
    return ["docker", "run", "--rm", "--name", name, "image", *args]


def capture(monkeypatch, module, after=None):
    calls = []
    def run(cmd, **kwargs):
        spec = _chsdk.parse_command(cmd)
        _chsdk._Output(kwargs)
        # Compare against the untouched suffix of the original argument list.
        image_index = cmd.index(spec.docker_image)
        assert spec.args == tuple(cmd[image_index + 1:])
        calls.append((spec, kwargs))
        if after:
            after(spec, kwargs)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(module, "run_container", run)
    return calls


@pytest.fixture
def inputs(tmp_path):
    bam, ref, out = (tmp_path / name for name in ("sample reads.bam", "ref.fa", "out.vcf.gz"))
    for path in (bam, ref, ref.with_suffix(".fa.fai"), bam.with_suffix(".bam.bai")):
        path.touch()
    return bam, ref, out


@pytest.fixture
def runtime(monkeypatch, caplog):
    # Bittensor can raise existing loggers' levels during test collection.
    caplog.set_level("WARNING", logger=_chsdk.__name__)
    token = _common._job_round.set(None)
    _common._open_round = None
    _common._live_containers.clear()
    runner = _chsdk._Runner()
    monkeypatch.setattr(_chsdk, "_runner", runner)
    engines = []
    queued = []

    def factory():
        engine = queued.pop(0) if queued else Engine()
        engines.append(engine)
        return engine

    monkeypatch.setattr("compute_horde_sdk._internal.local.client.from_env", factory)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("CHSDK path called native subprocess"))
    with ThreadPoolExecutor(max_workers=12) as pool:
        def launch(cmd=None, **kwargs):
            return pool.submit(contextvars.copy_context().run, _common.run_container,
                               cmd or command(), capture_output=True, **kwargs)
        yield runner, queued, engines, launch, pool
        for engine in engines:
            engine.remove_error = None
            for gate in engine.gates.values():
                gate.set()
        _common.reap_live_containers(2)
    runner.call(runner.loop.shutdown_asyncgens())
    runner.call(runner.loop.shutdown_default_executor())
    runner.loop.call_soon_threadsafe(runner.loop.stop)
    runner.thread.join(2)
    runner.loop.close()
    _common._job_round.reset(token)
    _common._open_round = None
    _common._live_containers.clear()


# Docker command parsing


@pytest.mark.parametrize("option", [["--network=none"], ["--entrypoint", "sh"],
                                    ["--privileged"], ["-e", "INHERITED"],
                                    ["-v", "/tmp:/data:z"], ["-v", "volume:/data"]])
def test_parser_rejects_unsupported_options(option):
    with pytest.raises(ValueError):
        _chsdk.parse_command(["docker", "run", "--rm", "--name", "parser-test", *option, "image"])


def test_parser_explicit_rw_duplicate_sources_and_argv(tmp_path):
    cmd = ["docker", "run", "--rm", "--name=parser-test", "--cpus", "1.5", "--memory=2g",
           "-u", "123:456", "--platform=linux/amd64", "-e", "VALUE=a b=c",
           "-v", f"{tmp_path}:/input:ro", "-v", f"{tmp_path}:/output",
           "-v", f"{tmp_path}:/other:rw", "image", "sh", "-lc", "a | b; echo 'x y'", ""]
    spec = _chsdk.parse_command(cmd)
    assert (spec.args, spec.env, spec.cpus, spec.memory, spec.user, spec.platform) == (
        ("sh", "-lc", "a | b; echo 'x y'", ""), {"VALUE": "a b=c"}, 1.5, "2g", "123:456", "linux/amd64")
    assert [(v.source, v.destination, v.read_only) for v in spec.volumes] == [
        (tmp_path, "/input", True), (tmp_path, "/output", False), (tmp_path, "/other", False)]


# Template and scoring commands


@pytest.mark.parametrize("module,destinations", [
    (gatk, ["/data/bams", "/data/reference", "/data/output"]),
    (deepvariant, ["/input", "/reference", "/output"]),
])
def test_gatk_and_deepvariant(monkeypatch, inputs, module, destinations):
    calls = capture(monkeypatch, module)
    module.variant_call(*inputs, "chr20:1-1000", {"threads": 2, "memory_gb": 6, "timeout": 99})
    assert len(calls) == 1
    spec, kwargs = calls[0]
    assert [(v.source, v.destination, v.read_only) for v in spec.volumes] == [
        (inputs[0].parent, dest, ro) for dest, ro in zip(destinations, [True, True, False])]
    assert (spec.cpus, spec.memory, kwargs["timeout"]) == (2, "6g", 99)
    if module is deepvariant:
        assert any(a.startswith("--intermediate_results_dir=/output/") for a in spec.args)


@pytest.mark.parametrize("arch,platform", [("arm64", "linux/amd64"), ("x86_64", None)])
def test_bcftools_index_and_pipeline(monkeypatch, inputs, arch, platform):
    inputs[0].with_suffix(".bam.bai").unlink()
    monkeypatch.setattr(bcftools.platform, "machine", lambda: arch)
    calls = capture(monkeypatch, bcftools)
    bcftools.variant_call(*inputs, "chr20:1-1000", {"threads": 2})
    assert len(calls) == 2
    index, pipeline = (call[0] for call in calls)
    assert [(s.cpus, s.memory, s.platform) for s in (index, pipeline)] == [(2, None, platform)] * 2
    assert [(v.destination, v.read_only) for v in index.volumes] == [("/data", False)]
    assert [(v.destination, v.read_only) for v in pipeline.volumes] == [
        ("/data", False), ("/ref", False), ("/output", False)]
    assert pipeline.user == f"{os.getuid()}:{os.getgid()}"
    assert pipeline.args[:2] == ("sh", "-lc")
    assert len(pipeline.args) == 3 and " | bcftools call " in pipeline.args[2]
    assert "'sample reads.bam'" in pipeline.args[2]
    assert calls[0][1]["timeout"] == 300


@pytest.mark.parametrize("arch,platform", [("arm64", "linux/amd64"), ("x86_64", None)])
def test_freebayes_file_then_compression(monkeypatch, inputs, arch, platform):
    monkeypatch.setattr(freebayes.platform, "machine", lambda: arch)
    def output(spec, kwargs):
        if "stdout" in kwargs:
            kwargs["stdout"].write("##VCF\n" * 100)
    calls = capture(monkeypatch, freebayes, output)
    freebayes.variant_call(*inputs, "chr20:1-1000", {"threads": 2, "memory_gb": 6})
    assert len(calls) == 2
    call, compress = (c[0] for c in calls)
    assert [(s.cpus, s.memory, s.platform) for s in (call, compress)] == [(2, "6g", platform)] * 2
    assert [(v.destination, v.read_only) for v in call.volumes] == [("/data", True), ("/ref", True)]
    assert [(v.destination, v.read_only) for v in compress.volumes] == [("/data", False)]
    assert calls[0][1]["stderr"] == subprocess.PIPE
    assert calls[0][1]["stdout"].closed  # caller, not CHSDK, owns the file lifetime
    assert compress.args[:2] == ("sh", "-c") and len(compress.args) == 3
    assert calls[1][1]["timeout"] == 120


@pytest.mark.parametrize("same_directory", [True, False])
def test_truth_slice_and_index(monkeypatch, tmp_path, same_directory):
    source = tmp_path / "truth.vcf.gz"
    source.touch()
    source.with_suffix(".gz.csi").touch()
    target = (tmp_path if same_directory else tmp_path / "other") / "slice.vcf.gz"
    calls = capture(monkeypatch, scoring, lambda *args: target.touch())
    assert scoring.slice_truth_vcf(source, target, "chr20:1-1000")
    assert len(calls) == 2
    assert [c[1]["timeout"] for c in calls] == [300, 60]
    destinations = ["/data"] if same_directory else ["/source", "/target"]
    assert [v.destination for v in calls[0][0].volumes] == destinations
    for spec, _ in calls:
        assert all(not v.read_only for v in spec.volumes)
        assert (spec.cpus, spec.memory) == (None, None)
    expected_target = f"{destinations[-1]}/slice.vcf.gz"
    assert calls[0][0].args[-2:] == ("-o", expected_target)


@pytest.mark.parametrize("bed_location", [None, "truth", "separate"])
def test_happy_entrypoint_env_sdf_and_optional_bed(monkeypatch, tmp_path, bed_location):
    truth_dir, work, ref_dir, bed_dir = (tmp_path / p for p in ("truth", "work", "ref", "bed"))
    for directory in (truth_dir, work, ref_dir, bed_dir):
        directory.mkdir()
    truth, query, ref = truth_dir / "truth.vcf", work / "query.vcf", ref_dir / "ref.fa"
    for path in (truth, query, ref):
        path.write_text("# header\n")
    sdf = ref_dir / "sdf"
    sdf.mkdir()
    bed = None if bed_location is None else (truth_dir if bed_location == "truth" else bed_dir) / "confident.bed"
    if bed:
        bed.write_text("chr20\t0\t1000\n")
    monkeypatch.setattr(scoring, "generate_synthetic_regions_bed", lambda *args: False)
    calls = capture(monkeypatch, scoring)
    scoring.HappyScorer().score_vcf(str(truth), str(query), reference_fasta=str(ref),
                                  reference_sdf=str(sdf), confident_bed=str(bed) if bed else None)
    assert len(calls) == 1
    spec, kwargs = calls[0]
    assert spec.args[:2] == ("/data/truth/truth.vcf", "/data/output/query.vcf")
    assert spec.env == {"HGREF": "/data/reference/ref.fa", "RTG_MEM": "8g"}
    assert (spec.cpus, spec.memory, kwargs["timeout"]) == (None, None, 600)
    assert spec.args[spec.args.index("--threads") + 1] == "4"
    assert [(v.destination, v.read_only) for v in spec.volumes] == [
        ("/data/truth", False), ("/data/reference", False), ("/data/output", False),
        ("/data/sdf", True), *([("/data/bed", False)] if bed_location == "separate" else [])]


def test_happy_optional_query_mount_shape(tmp_path):
    # Current score_vcf sets output_dir=query.parent, so this branch is dormant.
    # Cover its exact mount/argv form without changing the scoring path.
    cmd = ["docker", "run", "--rm", "--name", "optional-query",
           "-v", f"{tmp_path}:/data/output", "-v", f"{tmp_path}:/data/query",
           "happy", "/data/truth/truth.vcf", "/data/query/query.vcf", "--threads", "4"]
    spec = _chsdk.parse_command(cmd)
    assert [(v.destination, v.read_only) for v in spec.volumes] == [("/data/output", False), ("/data/query", False)]
    assert spec.cpus is None


# Execution, output and launch failures


@pytest.mark.parametrize("text", [True, False])
@pytest.mark.parametrize("returncode", [0, 7, 125, 137])
def test_results(runtime, text, returncode):
    runner, queued, engines, _, _ = runtime
    queued.append(Engine(returncode=returncode))
    _common.begin_round()
    cmd = command("minos-result", "a b", "", "$(unexpanded)")
    result = _common.run_container(cmd, capture_output=True, text=text)
    assert vars(result) == dict(args=cmd, returncode=returncode,
                               stdout="stdout\n" if text else b"stdout\r\n",
                               stderr="stderr\n" if text else b"stderr\r\n")
    assert engines[0].create_kwargs["command"] == ["a b", "", "$(unexpanded)"]
    assert (runner.jobs, _common._live_containers) == ({}, set())
    assert _common.reap_live_containers() == 0


def test_check_and_encoding(runtime):
    _, queued, _, _, _ = runtime
    queued.append(Engine(chunks=[(1, b"\xff\r"), (2, b"error")], returncode=7))
    with pytest.raises(subprocess.CalledProcessError) as error:
        _common.run_container(command(), capture_output=True, encoding="latin-1", check=True)
    assert (error.value.returncode, error.value.stdout, error.value.stderr) == (7, "ÿ\n", "error")


def test_stdout_file_exact_bytes_and_stderr(runtime, tmp_path):
    runner, queued, _, _, _ = runtime
    data = b"##VCF\r\n" + b"x\t1\n" * 800000 + b"\xff"
    queued.append(Engine(chunks=[(1, data[:11]), (2, b"diagnostic"), (1, data[11:])]))
    path = tmp_path / "result.vcf"
    with path.open("w") as sink:
        result = _common.run_container(command(), stdout=sink, stderr=subprocess.PIPE, text=True)
        assert not sink.closed
        assert path.read_bytes() == data
    assert (result.stdout, result.stderr, runner.jobs) == (None, "diagnostic", {})


@pytest.mark.parametrize("kwargs", [{"stdin": subprocess.PIPE}, {"shell": True}, {"cwd": "/tmp"},
                                    {"stderr": subprocess.STDOUT}, {"stdout": subprocess.DEVNULL},
                                    {"capture_output": True, "stdout": subprocess.PIPE}])
def test_unsupported_kwargs_fail_before_submission(runtime, kwargs):
    _, _, engines, _, _ = runtime
    with pytest.raises((TypeError, ValueError)):
        _common.run_container(command(), **kwargs)
    assert engines == []


def test_launch_failure_releases_registry(runtime):
    runner, queued, _, _, _ = runtime
    engine = Engine()
    engine.create_error = OSError("daemon failed")
    queued.append(engine)
    with pytest.raises(chsdk.LocalJobLaunchError, match="daemon failed"):
        _common.run_container(command(), capture_output=True)
    assert (runner.jobs, _common._live_containers, engine.calls[-1]) == ({}, set(), "remove")


def test_duplicate_name_failure_does_not_stop_original(runtime):
    _, queued, _, launch, _ = runtime
    engine = Engine(blocked=("logs",))
    queued.append(engine)
    original = launch()
    assert engine.entered["logs"].wait(2)
    with pytest.raises(ValueError, match="already registered"):
        launch().result(2)
    assert "remove" not in engine.calls
    _common.reap_container("minos-test")
    with pytest.raises(chsdk.LocalJobCancelledError):
        original.result(2)


# Timeouts and container cleanup


def test_timeout_covers_creation_and_keeps_partial_bytes(runtime):
    runner, queued, _, _, _ = runtime
    engine = Engine(blocked=("logs",))
    queued.append(engine)
    with pytest.raises(subprocess.TimeoutExpired) as error:
        _common.run_container(command(), capture_output=True, text=True, timeout=0.05)
    assert (error.value.cmd, error.value.timeout, error.value.stdout, error.value.stderr) == (
        command(), 0.05, b"stdout\r\n", b"stderr\r\n")
    assert engine.calls[-1] == "remove"
    assert runner.jobs == {}
    queued.append(Engine(blocked=("image",)))
    with pytest.raises(subprocess.TimeoutExpired):
        _common.run_container(command(), capture_output=True, timeout=0.02)
    assert runner.jobs == {}


def test_timeout_with_unconfirmed_creation_cleanup_keeps_handle(runtime, caplog):
    runner, queued, _, launch, _ = runtime
    runner.client.cleanup_timeout = 0.02
    engine = Engine(blocked=("create",))
    queued.append(engine)
    future = launch(timeout=0.02)
    assert engine.entered["create"].wait(2)
    with pytest.raises(subprocess.TimeoutExpired):
        future.result(2)
    assert set(runner.jobs) == {"minos-test"}
    assert not _common._live_containers
    assert "retaining handle" in caplog.text
    runner.loop.call_soon_threadsafe(engine.gates["create"].set)
    assert _common.reap_live_containers(1) == 1
    assert "start" not in engine.calls
    assert not runner.jobs


def test_single_cleanup_and_repeated_cleanup(runtime):
    runner, queued, _, launch, _ = runtime
    engine = Engine(blocked=("logs",))
    queued.append(engine)
    future = launch()
    assert engine.entered["logs"].wait(2)
    _common.reap_container("minos-test")
    _common.reap_container("minos-test")
    _common.reap_container("")
    with pytest.raises(chsdk.LocalJobCancelledError):
        future.result(2)
    assert (engine.calls.count("remove"), runner.jobs, _common._live_containers) == (1, {}, set())


def test_single_cleanup_uses_30_second_budget_and_retains_failures(runtime, monkeypatch, caplog):
    runner, queued, _, launch, _ = runtime
    queued.append(Engine(blocked=("logs",)))
    future = launch()
    # Wait for a registered running job, then replace only the public stop method.
    deadline = time.monotonic() + 2
    while not runner.jobs and time.monotonic() < deadline:
        time.sleep(0.001)
    job = runner.jobs["minos-test"]
    original = job.stop
    budgets = []
    async def fail(timeout):
        budgets.append(timeout)
        raise chsdk.LocalCleanupError("retry required")
    monkeypatch.setattr(job, "stop", fail)
    _common.reap_container("minos-test")
    assert 29 < budgets[0] <= 30
    assert "retry required" in caplog.text
    monkeypatch.setattr(job, "stop", original)
    with pytest.raises(chsdk.LocalJobCancelledError):
        future.result(2)
    assert not runner.jobs


def test_cleanup_error_keeps_handle_for_later_reap(runtime, caplog):
    runner, queued, _, _, _ = runtime
    engine = Engine()
    engine.remove_error = OSError("daemon unavailable")
    queued.append(engine)
    with pytest.raises(chsdk.LocalCleanupError):
        _common.run_container(command(), capture_output=True)
    assert set(runner.jobs) == {"minos-test"}
    assert not _common._live_containers
    assert _common.reap_live_containers(1) == 0
    engine.remove_error = None
    assert _common.reap_live_containers(1) == 1
    assert not runner.jobs


# Round lifecycle and cleanup budgets


def test_reaping_closes_registration_before_cleanup_finishes(runtime):
    """A late job cannot escape the reap while existing jobs are being removed."""
    runner, queued, engines, launch, pool = runtime
    engine = Engine(blocked=("logs", "remove"))
    queued.append(engine)
    _common.begin_round()
    running = launch(command("running-job"))
    assert engine.entered["logs"].wait(2)

    reaping = pool.submit(_common.reap_live_containers, 5)
    try:
        assert engine.entered["remove"].wait(2)
        assert not reaping.done()
        assert _common._open_round is None
        with pytest.raises(_common.RegistryClosed):
            launch(command("late-job")).result(2)
        # Rejection happens before allocating or starting another SDK job.
        assert engines == [engine]
        assert set(runner.jobs) == {"running-job"}
    finally:
        runner.loop.call_soon_threadsafe(engine.gates["remove"].set)

    assert reaping.result(2) == 1
    with pytest.raises(chsdk.LocalJobCancelledError):
        running.result(2)
    assert engine.calls.count("remove") == 1
    assert not runner.jobs and not _common._live_containers


def test_round_closed_before_submission_and_old_context(runtime):
    _, _, engines, launch, _ = runtime
    _common.begin_round()
    old = contextvars.copy_context()
    assert _common.reap_live_containers() == 0
    with pytest.raises(_common.RegistryClosed):
        launch().result(2)
    _common.begin_round()
    with pytest.raises(_common.RegistryClosed):
        old.run(_common.run_container, command(), capture_output=True)
    assert engines == []
    assert launch().result(2).returncode == 0


def test_to_thread_carries_round_generation(runtime):
    async def run():
        _common.begin_round()
        _common.reap_live_containers()
        with pytest.raises(_common.RegistryClosed):
            await asyncio.to_thread(_common.run_container, command(), capture_output=True)
    asyncio.run(run())


def test_round_closes_during_submission_before_chsdk_execution(runtime, monkeypatch):
    runner, _, engines, launch, _ = runtime
    prepare = runner.client.prepare_job
    _common.begin_round()
    def prepare_then_close(*args, **kwargs):
        job = prepare(*args, **kwargs)
        # Closure is queued before job.start schedules the CHSDK's execution.
        runner.loop.create_task(runner.reap_all(time.monotonic() + 1))
        return job
    monkeypatch.setattr(runner.client, "prepare_job", prepare_then_close)
    with pytest.raises(chsdk.LocalJobCancelledError):
        launch().result(2)
    assert (engines, runner.jobs, _common._live_containers) == ([], {}, set())


@pytest.mark.parametrize("stage,started", [("image", False), ("create", False),
                                           ("attach", False), ("start", True), ("logs", True)])
def test_round_closure_during_chsdk_lifecycle(runtime, stage, started):
    runner, queued, _, launch, pool = runtime
    engine = Engine(blocked=(stage, "logs"))
    queued.append(engine)
    _common.begin_round()
    future = launch()
    assert engine.entered[stage].wait(2)
    # A zero budget still closes registration and requests stopping immediately.
    assert pool.submit(_common.reap_live_containers, 0).result(2) == 0
    assert _common._open_round is None
    runner.loop.call_soon_threadsafe(engine.gates[stage].set)
    with pytest.raises(chsdk.LocalJobCancelledError):
        future.result(2)
    assert ("start" in engine.calls) is started
    assert not runner.jobs and not _common._live_containers
    with pytest.raises(_common.RegistryClosed):
        launch(command("late-job")).result(2)
    assert _common.reap_live_containers() == 0


def test_round_budget_is_shared_with_partial_success_and_retry(runtime, caplog):
    runner, queued, _, launch, _ = runtime
    engines = [Engine(blocked=("logs", "remove")) for _ in range(5)]
    fast = Engine(blocked=("logs",))
    queued.extend([fast, *engines])
    _common.begin_round()
    futures = []
    for index, engine in enumerate([fast, *engines]):
        futures.append(launch(command(f"budget-{index}")))
        assert engine.entered["logs"].wait(2)
    start = time.monotonic()
    assert _common.reap_live_containers(0.08) == 1
    assert time.monotonic() - start < 0.3  # serial per-job budgets would exceed 0.4s
    assert len(runner.jobs) == 5
    assert not _common._live_containers
    assert "Could not reap CHSDK" in caplog.text
    for engine in engines:
        runner.loop.call_soon_threadsafe(engine.gates["remove"].set)
    _common.reap_live_containers(1)
    for future in futures:
        with pytest.raises(chsdk.LocalJobCancelledError):
            future.result(2)
    assert not runner.jobs
