"""Observe actual W59 clock, CPU, memory, I/O and GPU use at natural running nodes."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

from s1_w59_lease import BG, POOL, ROOT
from publish_s1_go2_w54 import write

WORK = ROOT/'work/w59'
REL = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
STATE, RAW = ROOT/'results'/REL/'w59', ROOT/'outputs'/REL
CG = Path('/sys/fs/cgroup')


def data(p):
    return json.loads(p.read_text())


def ps(pid):
    stat = (Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
    return int(stat[19]),(int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK'),int(stat[21])*os.sysconf('SC_PAGE_SIZE')


def sample():
    own = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                cwd=os.readlink(p/'cwd')
                if cwd.startswith(str(ROOT)):
                    own[p.name]=ps(p.name)[:2]
            except (OSError,IndexError):
                continue
    slots, clocks, io = {}, {}, {}
    for p in sorted((POOL/'jobs').glob('*.json')):
        task=data(p)
        if task['state']!='running' or 'pid' not in task:
            continue
        try:
            if ps(task['pid'])[0]!=task['start_ticks']:
                continue
        except FileNotFoundError:
            continue
        slots[p.stem]=task['slots']
        live=RAW/p.stem/'live.json'
        if live.exists():
            row=data(live)
            clocks[p.stem]=[row['physical_time_s'],row['unix_s'],row['phase']]
            usage=RAW/'jobs'/p.stem/'resources.jsonl'
            if usage.is_file():
                with usage.open('rb') as f:
                    f.seek(max(0,usage.stat().st_size-8192))
                    closed=f.read().split(b'\n')[:-1]
                    if closed:
                        entry=json.loads(closed[-1]);io[p.stem]={k:entry[k] for k in ('rss_bytes','read_bytes','write_bytes','user_cpu_s','system_cpu_s')}
    total=dict(line.split() for line in (CG/'cpu.stat').read_text().splitlines())
    mem=dict(line.split() for line in (CG/'memory.stat').read_text().splitlines())
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    unix=time.time()
    row=dict(shanghai=datetime.fromtimestamp(unix,ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        unix_s=unix,slots=slots,clock=clocks,usage=io,cgroup_cpu_seconds=int(total['usage_usec'])*1e-6,
        usable_memory_bytes=int((CG/'memory.max').read_text())-int((CG/'memory.current').read_text())+int(mem['inactive_file']),
        io_pressure=(CG/'io.pressure').read_text().splitlines()[0],scratch_free_bytes=shutil.disk_usage(ROOT).free,
        gpu_util_used_total_MiB=[int(x.strip()) for x in gpu.strip().split(',')],**BG)
    return row,own


def monitor():
    last=None
    while True:
        row,own=sample()
        if last:
            prev,earlier=last
            elapsed=row['unix_s']-prev['unix_s']
            cpu=sum(v[1]-earlier[k][1] for k,v in own.items() if k in earlier and v[0]==earlier[k][0])/elapsed
            total=(row['cgroup_cpu_seconds']-prev['cgroup_cpu_seconds'])/elapsed
            row.update(W59_cpu_cores=cpu,total_cpu_cores=total,other_cpu_cores=max(0.,total-cpu))
            row['sim_s_per_wall_s']={k:(v[0]-prev['clock'][k][0])/(v[1]-prev['clock'][k][1])
                        for k,v in row['clock'].items() if k in prev['clock'] and v[1]>prev['clock'][k][1]}
        with (WORK/'resource-samples.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        write(WORK/'last-resource.json',row)
        if (STATE/'outcome.json').is_file() and not row['slots']:
            return
        last=row,own
        time.sleep(20)


if __name__=='__main__':
    monitor()
