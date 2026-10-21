"""Finite reference evaluations followed by the single authorized clean suite."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time
import traceback

root = Path(__file__).resolve().parent
source = root / "source"
python = root / "env/bin/python"
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
references = []
started = time.perf_counter()
try:
    for case in ("nominal", "narrow", "wide"):
        command = ["taskset", "-c", "6,7", "nice", "-n", "10", str(python),
                   "scripts/run_go2_fair.py", "--suite", "primary", "--mode", "follower",
                   "--speed", "fast", "--case", case, "--assets", str(root / "assets"),
                   "--output", "work/follower-reference/evidence",
                   "--raw-output", "work/follower-reference/raw"]
        begin = time.perf_counter()
        with (root / f"reference-{case}.log").open("xb") as log:
            result = subprocess.run(command, cwd=source, env=environment,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        references.append({"case": case, "returncode": result.returncode,
                           "wall_seconds": time.perf_counter() - begin})
        (root / "reference-execution.json").write_text(json.dumps({
            "status": "RUNNING" if case != "wide" else "COMPLETED",
            "fixed_cases_only": True, "records": references,
            "wall_seconds": time.perf_counter() - started}, indent=2))
        if result.returncode:
            raise RuntimeError(f"Fixed follower reference process failed: {case}")
    with (root / "training-console.log").open("xb") as log:
        result = subprocess.run([str(python), "-u", str(root / "launch_fixed.py")],
                                cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    (root / "job-exit.json").write_text(json.dumps({
        "returncode": result.returncode, "wall_seconds": time.perf_counter() - started,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "automatic_retry": False}, indent=2))
    sys.exit(result.returncode)
except BaseException as exc:
    if not isinstance(exc, SystemExit):
        (root / "job-failure.json").write_text(json.dumps({
            "exception_type": type(exc).__name__, "message": str(exc),
            "wall_seconds": time.perf_counter() - started, "automatic_retry": False}, indent=2))
        traceback.print_exc()
    raise
