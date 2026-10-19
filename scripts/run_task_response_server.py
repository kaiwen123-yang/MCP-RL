"""One authorized fixed-protocol run with process-local resource limits.

No scheduler, retry loop, background monitor, or global environment change.
Learning configuration is read verbatim; resource limits apply to all arms.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task_response_pilot.json")
    parser.add_argument("--output", default="work/server-pilot-20261019-r1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpus", default="")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = (root / args.output).resolve()
    if not output.is_relative_to(root / "work"):
        raise ValueError("Run output must stay in the ignored project work directory")
    metadata = output.parent / (output.name + "-execution")
    metadata.mkdir(parents=True, exist_ok=False)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.nice(10)
    if args.cpus:
        os.sched_setaffinity(0, {int(x) for x in args.cpus.split(",")})
    sys.path.insert(0, str(root / "src"))
    import torch
    from task_response.runner import read_config, train
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        torch.cuda.reset_peak_memory_stats(0)
    config = read_config(args.config)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "scipy", "gymnasium", "tianshou")}
    manifest = {"status": "RUNNING", "pid": os.getpid(), "python": platform.python_version(),
                "versions": versions, "device": args.device,
                "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                "cpu_affinity": sorted(os.sched_getaffinity(0)), "nice": os.getpriority(os.PRIO_PROCESS, 0),
                "torch_threads": torch.get_num_threads(), "gpu_memory_fraction": args.gpu_memory_fraction,
                "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                "learning_config_modified": False, "automatic_monitoring": False}
    record = metadata / "execution.json"
    record.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    started, cpu_started = time.perf_counter(), time.process_time()
    exit_code = 0
    try:
        result = train(config, args.device, output)
        manifest["status"] = result["status"]
    except BaseException as error:
        manifest["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        (metadata / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        exit_code = 1
        traceback.print_exc()
    finally:
        manifest.update({"wall_seconds": time.perf_counter()-started,
                         "process_cpu_seconds": time.process_time()-cpu_started,
                         "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if args.device == "cuda" else 0,
                         "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0) if args.device == "cuda" else 0,
                         "exit_code": exit_code})
        record.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
