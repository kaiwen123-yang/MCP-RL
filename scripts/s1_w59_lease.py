"""Hold a single D57 W59 lease across one job, including both collectors and their tails."""
import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from publish_s1_go2_w54 import write

ZONE = ZoneInfo('Asia/Shanghai')
ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT/'work/w59/pool'
BG = dict(deadline=None, wall_deadline_enabled=False, kill_on_conversation_end=False)


def stamp():
    return dt.datetime.now(ZONE).isoformat(timespec='seconds')


def admit(slots):
    handles = ExitStack()
    chosen = []
    with (POOL/'admission.lock').open('a') as gate:
        fcntl.flock(gate, fcntl.LOCK_EX)
        config = json.loads((POOL/'capacity.json').read_text())
        health_path = ROOT/'work/w59/last-resource.json'
        claims = None
        if health_path.is_file():
            sample = json.loads(health_path.read_text())
            quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
            mem = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
            available = int(Path('/sys/fs/cgroup/memory.max').read_text()) - int(Path('/sys/fs/cgroup/memory.current').read_text()) + int(mem['inactive_file'])
            observed_rss = max([2 * 2**30]+[v['rss_bytes'] for v in sample.get('usage',{}).values()])
            # Existing owners keep their slots. New admission uses actual cgroup room,
            # the largest observed collector RSS, and the project's three-core reserve.
            # Two coordinators may obtain slots before the next resource sample sees
            # either worker. Reserve those in-flight starts within this sample.
            claims_path = POOL/'starts-at-resource-sample.json'
            claims = json.loads(claims_path.read_text()) if claims_path.is_file() else {}
            pending = claims.get('slots', 0) if claims.get('sample') == sample['unix_s'] else 0
            demand = pending + slots
            if available < demand * observed_rss + 3 * 2**30:
                return None
            total_usage = sample.get('total_cpu_cores')
            if total_usage is not None and total_usage + demand > int(quota)/int(period) - 3:
                return None
            claims = {'sample':sample['unix_s'], 'slots':demand}
        count = int(config['physical_slots'])
        if not 1 <= count <= 6:
            raise ValueError(count)
        for i in range(count):
            f = (POOL/f'slot-{i}.lock').open('a')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            handles.enter_context(f)
            chosen.append((i, f.fileno()))
            if len(chosen) == slots:
                ids = config['cpu_ids']
                groups = [ids[3*i:3*i+3] for i,_ in chosen]
                if any(len(g) != 3 for g in groups):
                    handles.close()
                    raise ValueError('Each physical slot needs its registered CPU partition')
                if claims is not None:
                    write(claims_path, claims)
                return handles, chosen, groups
        handles.close()
    return None


def main(a):
    POOL.mkdir(parents=True, exist_ok=True)
    record = POOL/'jobs'/(a.job+'.json')
    record.parent.mkdir(exist_ok=True)
    with (record.with_suffix('.lock')).open('a') as job_lock:
        # Another coordinator can reach the same already admitted episode. Join its
        # terminal receipt before returning; the physical slot remains held only once.
        fcntl.flock(job_lock, fcntl.LOCK_EX)
        cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
        if record.is_file():
            previous = json.loads(record.read_text())
            if previous['state'] == 'ended':
                return previous['exit_code']
            if previous['state'] == 'running':
                return 0
        data = dict(job=a.job, requested_slots=a.slots, command=cmd, runtime=str(a.runtime),
                    owner_pid=os.getpid(), registered_at=stamp(), state='queued', **BG)
        write(record, data)
        while (leased := admit(a.slots)) is None:
            time.sleep(5)
        handles, slots, groups = leased
        with handles:
            fds = tuple(fd for _,fd in slots)
            ids = ','.join(str(i) for i,_ in slots)
            env = dict(os.environ, S1_W59_SLOT=ids, S1_W58_SLOT=ids,
                       S1_W58_LEASE_FDS=','.join(map(str,fds)), S1_W58_PAIR_CPUS=json.dumps(groups),
                       S1_W59_PAIR_CPUS=json.dumps(groups), OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       W59_TMUX=os.environ.get('W59_TMUX','w59-'+a.job))
            recorder_manifest = POOL/'recording.json'
            if recorder_manifest.is_file():
                recording = json.loads(recorder_manifest.read_text())
                if a.job in recording['natural_episodes']:
                    import hashlib
                    directory = Path(recording['directory'])
                    source = directory/'sitecustomize.py'
                    if hashlib.sha256(source.read_bytes()).hexdigest() != recording['sha256']:
                        raise ValueError('The predeclared passive recorder differs')
                    env['W59_RECORD_FROZEN_LEGAL_REQUESTS'] = '1'
                    env['PYTHONPATH'] = str(directory)+':'+env['PYTHONPATH']
                    data['passive_recorder'] = recording['sha256']
            cpus = ','.join(str(c) for g in groups for c in g)
            data.update(state='running', slots=[i for i,_ in slots], cpu_groups=groups, started_at=stamp())
            write(record, data)
            child = subprocess.Popen(['taskset','-c',cpus,*cmd], cwd=a.runtime, env=env, pass_fds=fds)
            data.update(pid=child.pid, start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(') ',1)[1].split()[19]))
            write(record,data)
            rc = child.wait()
            write(record, dict(data, state='ended', exit_code=rc, ended_at=stamp()))
            return rc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--job', required=True)
    p.add_argument('--slots', required=True, type=int, choices=(1,2))
    p.add_argument('command', nargs=argparse.REMAINDER)
    sys.exit(main(p.parse_args()))
