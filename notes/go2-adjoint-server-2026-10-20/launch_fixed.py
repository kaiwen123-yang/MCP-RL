"""Execute the committed protocol once; instrumentation never changes learning."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback

RUN = Path(__file__).resolve().parent
SOURCE = RUN / "source"
OUTPUT = SOURCE / "work/results"
EVENTS = RUN / "execution-events.jsonl"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.sched_setaffinity(0, {6, 7})
os.nice(10)
sys.path.insert(0, str(SOURCE / "src"))
from go2_learning import runner
from go2_learning.learning import PPOUpdater

config_path = SOURCE / "configs/go2_adjoint_learning.json"
expected_sha = "724e5385cb92c2bdfbc0a9b783ed5340f0634b66e65254223907135ab375ce20"
actual_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
if actual_sha != expected_sha or OUTPUT.exists() or EVENTS.exists():
    raise RuntimeError("Fixed source/config or fresh-output identity failed")
started = time.perf_counter()
current_path = None


def emit(kind, **fields):
    record = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "elapsed_seconds": time.perf_counter() - started, "event": kind, **fields}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def relative(path):
    return str(Path(path).relative_to(OUTPUT)).replace("\\", "/")


original_collect = runner.collect_phase


def collect_logged(system, config, assets, path, seed, mode):
    global current_path
    current_path = relative(path)
    emit("phase_start", operation="collect", path=current_path, mode=mode)
    at = time.perf_counter()
    result = original_collect(system, config, assets, path, seed, mode)
    completed = [e for e in result["episodes"] if e["complete"]]
    emit("phase_end", operation="collect", path=current_path, mode=mode,
         wall_seconds=time.perf_counter() - at,
         success=sum(e["success"] for e in completed),
         complete_episodes=len(completed),
         reasons=[e["reason"] for e in result["episodes"]],
         stats=result["stats"])
    return result


original_update = PPOUpdater.update


def update_logged(self, batch, direction, arm):
    emit("phase_start", operation="ppo", path=current_path, arm=arm)
    at = time.perf_counter()
    result = original_update(self, batch, direction, arm)
    emit("phase_end", operation="ppo", path=current_path, arm=arm,
         wall_seconds=time.perf_counter() - at, stats=result)
    return result


original_fit = runner.fit_phase


def fit_logged(*args, **kwargs):
    emit("phase_start", operation="fit_facts", path=current_path)
    at = time.perf_counter()
    result = original_fit(*args, **kwargs)
    emit("phase_end", operation="fit_facts", path=current_path,
         wall_seconds=time.perf_counter() - at, stats=result)
    return result


original_directions = runner.directions


def directions_logged(*args, **kwargs):
    emit("phase_start", operation="adjoint", path=current_path)
    at = time.perf_counter()
    result = original_directions(*args, **kwargs)
    emit("phase_end", operation="adjoint", path=current_path,
         wall_seconds=time.perf_counter() - at, stats=result[1])
    return result


runner.collect_phase = collect_logged
runner.fit_phase = fit_logged
runner.directions = directions_logged
PPOUpdater.update = update_logged
import importlib.metadata as metadata
runtime = {name: metadata.version(name) for name in ("torch", "numpy", "scipy", "mujoco", "PyYAML")}
runner.write_json(RUN / "runtime.json", {
    "algorithm_commit": "ddd85783a75895f95bdb7f170edf92a08ca8cf6c",
    "configuration_sha256": actual_sha,
    "authorization": "2026-10-20 user authorized one fixed three-arm comparison",
    "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "pid": os.getpid(), "python": sys.version, "runtime": runtime,
    "affinity": sorted(os.sched_getaffinity(0)), "nice": os.getpriority(os.PRIO_PROCESS, 0),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "automatic_retry": False, "automatic_monitor": False,
})
emit("experiment_start", runtime=runtime, configuration_sha256=actual_sha)
status = "FAILED"
try:
    runner.train(runner.read_config(config_path), RUN / "assets", OUTPUT)
    status = "COMPLETED"
except BaseException as exc:
    failure = {"status": "FAILED", "exception_type": type(exc).__name__,
               "message": str(exc), "automatic_retry": False}
    runner.write_json(RUN / "execution_failure.json", failure)
    if OUTPUT.exists():
        runner.write_json(OUTPUT / "execution_failure.json", failure)
    emit("experiment_failure", **failure)
    traceback.print_exc()
    raise
finally:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {"status": status, "wall_seconds": time.perf_counter() - started,
              "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime,
              "peak_rss_kib": usage.ru_maxrss, "pid": os.getpid()}
    runner.write_json(RUN / "exit.json", result)
    emit("experiment_exit", **result)
