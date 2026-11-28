"""Run one W61 physical campaign under the shared D57 slot and resource budget."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

POOL = Path(os.environ.get("W61_POOL", "/root/autodl-tmp/MCP-RL-w61-joint/work/w61/pool"))
CG = Path("/sys/fs/cgroup")
GIB = 2**30


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write(path, value):
    temp = path.with_suffix(".pending")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def fields(name):
    return dict(line.split() for line in (CG / name).read_text().splitlines())


def resources():
    start, cpu = time.monotonic(), int(fields("cpu.stat")["usage_usec"])
    time.sleep(.6)
    cores = (int(fields("cpu.stat")["usage_usec"]) - cpu) / (time.monotonic()-start) / 1e6
    quota, period = (CG / "cpu.max").read_text().split()
    stat = fields("memory.stat")
    available = int((CG/"memory.max").read_text()) - int((CG/"memory.current").read_text()) + int(stat["inactive_file"])
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True).splitlines()[0].split(",")
    io = float((CG/"io.pressure").read_text().splitlines()[0].split()[1].split("=")[1])
    return dict(total_cpu_cores=cores, quota_cores=int(quota)/int(period), available_ram_gib=available/GIB,
                free_gpu_mib=int(gpu[0]), gpu_util_percent=int(gpu[1]), io_pressure_pct=io,
                scratch_free_gib=shutil.disk_usage(POOL).free/GIB, scratch_free_inodes=os.statvfs(POOL).f_favail)


def claim(args):
    with (POOL/"admission.lock").open("a") as gate, ExitStack() as held:
        fcntl.flock(gate, fcntl.LOCK_EX)
        config = json.loads((POOL/"capacity.json").read_text())
        locks, reserved = [], [0., 0., 0.]
        for n in range(config["physical_slots"]):
            lock = (POOL/f"slot-{n}.lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.seek(0)
                entry = json.loads(lock.read())
                if time.time() - entry["start"] < config["startup_s"]:
                    for i, name in enumerate(("cores", "ram", "gpu")):
                        reserved[i] += entry[name]
                lock.close()
            else:
                held.enter_context(lock)
                locks.append((n,lock))
        snap = resources()
        # A fast campaign returning after a closed batch must not repeatedly take
        # the last free slot ahead of an older eligible single or paired job.
        waiting=[]
        for row in (POOL/"jobs").glob("*.json"):
            job=json.loads(row.read_text())
            if job["state"]!="waiting" or job["slots_requested"]>len(locks):continue
            if not Path(f"/proc/{job['owner_pid']}").exists():continue
            demands=job.get("demand",dict(cores=job['slots_requested'],ram_gib=2*job['slots_requested'],gpu_mib=0,scratch_gib=1))
            if (snap["quota_cores"]-snap["total_cpu_cores"]>=config["spare_cpu"]+demands["cores"]+reserved[0] and
                snap["available_ram_gib"]>=config["spare_ram_gib"]+demands["ram_gib"]+reserved[1] and
                snap["free_gpu_mib"]>=demands["gpu_mib"]+(config["spare_gpu_mib"] if demands["gpu_mib"] else 0)+reserved[2] and
                snap["scratch_free_gib"]>=config["spare_scratch_gib"]+demands["scratch_gib"]):
                waiting.append((job["registered"],job["job"]))
        if (waiting and min(waiting)[1]!=args.job) or (len(locks) < args.slots or
            snap["quota_cores"] - snap["total_cpu_cores"] < config["spare_cpu"] + args.cores + reserved[0] or
            snap["available_ram_gib"] < config["spare_ram_gib"] + args.ram_gib + reserved[1] or
            (args.gpu_mib and snap["free_gpu_mib"] < config["spare_gpu_mib"] + args.gpu_mib + reserved[2]) or
            snap["io_pressure_pct"] > config["max_io_pressure_pct"] or
            snap["scratch_free_gib"] < config["spare_scratch_gib"] + args.scratch_gib or
            snap["scratch_free_inodes"] < config["spare_inodes"]):
            return None, snap
        chosen = locks[:args.slots]
        for n, lock in chosen:
            lock.seek(0)
            lock.truncate()
            json.dump(dict(job=args.job, start=time.time(), cores=args.cores/args.slots,
                           ram=args.ram_gib/args.slots, gpu=args.gpu_mib/args.slots), lock)
            lock.flush()
        # Preserve exactly the selected locks until the entire child group returns.
        for _, lock in locks[args.slots:]:
            lock.close()
        return (held.pop_all(), chosen, config["cpu_ids"]), snap


def main(args):
    jobs = POOL/"jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    record = jobs/(args.job+".json")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    with (jobs/(args.job+".lock")).open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if record.exists():
            prev = json.loads(record.read_text())
            if prev["command"] != command or prev["runtime"] != str(args.runtime.resolve()):
                raise ValueError("Job identity already belongs to a different command or runtime")
            if prev["state"] == "ended":
                return prev["exit_code"]
            if prev["state"] == "running":
                raise RuntimeError("Previous lease owner exited during this run; inspect its original before resuming")
        data = dict(job=args.job, runtime=str(args.runtime.resolve()), command=command, slots_requested=args.slots,
                    registered=stamp(), owner_pid=os.getpid(), state="waiting", deadline=None,
                    wall_deadline_enabled=False, kill_on_conversation_end=False)
        data["demand"]=dict(cores=args.cores,ram_gib=args.ram_gib,gpu_mib=args.gpu_mib,scratch_gib=args.scratch_gib)
        write(record,data)
        while True:
            leased, snap = claim(args)
            data.update(resource_sample=snap, sampled=stamp())
            write(record,data)
            if leased:
                break
            time.sleep(12)
        holding, chosen, cpu_ids = leased
        with holding:
            ids = [n for n,_ in chosen]
            groups = [cpu_ids[3*n:3*n+3] for n in ids]
            cpus = ",".join(str(c) for group in groups for c in group)
            fd = tuple(handle.fileno() for _,handle in chosen)
            env = dict(os.environ, W61_SLOTS=",".join(map(str,ids)), W61_PAIR_CPUS=json.dumps(groups),
                       S1_W58_PAIR_CPUS=json.dumps(groups), S1_W59_PAIR_CPUS=json.dumps(groups),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
            child = subprocess.Popen(["taskset","-c",cpus,*command], cwd=args.runtime, env=env, pass_fds=fd)
            data.update(state="running", pid=child.pid, slots=ids, cpu_groups=groups, started=stamp())
            write(record,data)
            rc = child.wait()
            write(record,dict(data,state="ended",exit_code=rc,ended=stamp()))
            return rc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",required=True,type=Path)
    parser.add_argument("--job",required=True)
    parser.add_argument("--slots",required=True,type=int,choices=(1,2))
    parser.add_argument("--cores",type=float,default=1.)
    parser.add_argument("--ram-gib",type=float,default=2.)
    parser.add_argument("--gpu-mib",type=int,default=0)
    parser.add_argument("--scratch-gib",type=float,default=1.)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    sys.exit(main(parser.parse_args()))
