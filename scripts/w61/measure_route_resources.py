"""Measure, on the execution host, the actual process tree owned by each physical lease."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


def process_table():
    parent=defaultdict(list); rows={}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            text=(path/'stat').read_text()
            stat=text[text.rfind(')')+2:].split()
            p=int(path.name);ppid=int(stat[1]);parent[ppid].append(p)
            io=dict(line.split(': ',1) for line in (path/'io').read_text().splitlines())
            rows[p]=(int(stat[11])+int(stat[12]),int(stat[21]),int(io['read_bytes']),int(io['write_bytes']),
                     int(stat[19]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
    return parent,rows


def measure(args):
    hz=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE')
    args.out.mkdir(exist_ok=True,parents=True)
    clock=ZoneInfo('Asia/Shanghai');streams={};summary={};counts=defaultdict(int);previous={}
    for job in args.jobs:
        streams[job]=(args.out/(job+'.jsonl')).open('a',buffering=1)
        summary[job]=dict(samples=0,peak_aggregate_rss_bytes=0,peak_process_gpu_MiB=0,
            observed_cpu_s=0.,observed_read_bytes=0,observed_write_bytes=0,
            interval_s=args.interval,coverage='sampled after observer attachment; process lifetime counters are differenced by PID and start time')
    while True:
        assignments={j:json.loads((args.pool/(j+'.json')).read_text()) for j in args.jobs}
        active={j:a for j,a in assignments.items() if a['state']=='running'}
        if active:
            tree,rows=process_table()
            gpu={int(p.strip()):float(m.strip()) for p,m in (line.split(',') for line in subprocess.check_output(
                ['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines() if line)}
            for job,a in active.items():
                pending=[a['pid']];ids=set()
                while pending:
                    k=pending.pop()
                    if k in ids:continue
                    ids.add(k);pending.extend(tree.get(k,[]))
                processes={k:rows[k] for k in ids if k in rows}
                rss=sum(r[1] for r in processes.values())*page
                mem_gpu=sum(gpu.get(k,0.) for k in processes)
                total=[0,0,0]
                for k,r in processes.items():
                    key=(job,k,r[4]);now=(r[0],r[2],r[3]);old=previous.get(key)
                    if old:
                        total=[v+x-y for v,x,y in zip(total,now,old)]
                    previous[key]=now
                obj=summary[job];obj['samples']+=1
                obj['peak_aggregate_rss_bytes']=max(obj['peak_aggregate_rss_bytes'],rss)
                obj['peak_process_gpu_MiB']=max(obj['peak_process_gpu_MiB'],mem_gpu)
                obj['observed_cpu_s']+=total[0]/hz
                obj['observed_read_bytes']+=total[1];obj['observed_write_bytes']+=total[2]
                streams[job].write(json.dumps(dict(t=datetime.now(clock).isoformat(),slots=a['slots'],pids=sorted(processes),
                    process_rss_sum_bytes=rss,allocated_gpu_MiB=mem_gpu,cpu_observed_delta_s=total[0]/hz,
                    read_delta_bytes=total[1],write_delta_bytes=total[2]))+'\n')
        for j,a in assignments.items():
            if a['state']=='ended':
                summary[j]['lease']=dict(start=a.get('started'),end=a['ended'],exit_code=a['exit_code'])
        (args.out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
        if all(a['state']=='ended' for a in assignments.values()):
            break
        time.sleep(args.interval)
    for f in streams.values():
        f.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--pool',required=True,type=Path)
    p.add_argument('--interval',type=float,default=10.);p.add_argument('jobs',nargs='+');measure(p.parse_args())
