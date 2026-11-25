"""Read real solver timings and resource samples for the W58 physical-slot decisions."""
from datetime import datetime
from pathlib import Path
import json
import numpy as np

from inspect_s1_w58_predecessor import BG, REL, SH, read
from publish_s1_go2_w54 import write

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'results'/REL/'w58'
RAW=ROOT/'outputs'/REL
WORK=ROOT/'work/w58'


def timed_values(path,key,start,end):
    if not path.is_file():return []
    result=[]
    with path.open() as source:
        for line in source:
            if not line.endswith('\n'):continue
            row=json.loads(line)
            t=row['time_s']
            if t>end:break
            if t>start and row.get(key) is not None:result.append(row[key])
    return result


def percentile(seconds):
    return dict(n=len(seconds),p95_ms=float(np.percentile(seconds,95)*1000) if len(seconds) else None)


def stream_timing(name,flow,start_unix,end_unix):
    raw=RAW/name; data=raw/'w55-f';lo,hi=flow['first_s'],flow['last_s']
    result=dict(candidate_event=percentile(timed_values(data/'w58-handover.jsonl','planning_wall_s',lo,hi)),
                mpc_solve=percentile(timed_values(data/'mpc-solves.jsonl','solve_time_s',lo,hi)),
                feedback=None,wbc_solve_p95_ms=None)
    resource=RAW/'jobs'/name/'resources.jsonl'
    if resource.is_file():
        with resource.open() as source:
            rows=[r for line in source if line.endswith(chr(10)) for r in [json.loads(line)]
                  if start_unix<=r['unix_s']<=end_unix]
        if len(rows)>1:
            a,b=rows[0],rows[-1];elapsed=b['unix_s']-a['unix_s']
            cpu=sum(b[k]-a[k] for k in ('user_cpu_s','system_cpu_s'))/elapsed
            result['process_cpu_core_s_per_wall_s']=cpu
            result['process_cpu_s_per_physics_s']=cpu/flow['physics_s_per_wall_s']
    episode=STATE/'episodes'/(name+'.json')
    if episode.is_file() and (raw/'trajectory.npz').is_file():
        with np.load(raw/'trajectory.npz') as z:
            t=z['time'];feedback=z['feedback_s']
            if len(t)==len(feedback):
                values=feedback[(t>lo)&(t<=hi)]
                result['feedback']=percentile(values[np.isfinite(values)])
        totals=read(episode).get('w54_f_recording',{}).get('timing',{}).get('wbc_solve')
        if totals:result['whole_episode_wbc_total']=totals
    return result


def summarize():
    cases={}
    for medium,filename in [('with_personal','evaluation-capacity.json'),('scratch','evaluation-scratch-capacity.json')]:
        path=STATE/'dispatch'/filename
        if path.is_file():cases[medium]=read(path)
    events=sorted(((event['at'],medium,event) for medium,case in cases.items() for event in case['events']),
                  key=lambda row:row[0])
    with (WORK/'throughput.jsonl').open() as f:
        trace=[json.loads(line) for line in f if line.endswith('\n')]
    decisions=[]
    for _,medium,event in events:
        sample=event['measurement'];start=datetime.fromisoformat(sample['start']).timestamp()
        rows=[r for r in trace if start<=r['unix_s']<=sample['end_unix']+.001]
        if not rows:raise ValueError('Missing measured resource window')
        seconds=sum(r['dt_s'] for r in rows)
        means={k:sum(r['dt_s']*r[k] for r in rows)/seconds
               for k in ('w58_core_s_per_s','auxiliary_core_s_per_s','other_core_s_per_s','io_pressure_pct','host_iowait_fraction')}
        decisions.append(dict(at=event['at'],storage_stratum=medium,action=event['action'],width=sample['width'],start=sample['start'],
            wall_s=sample['wall_s'],raw_physics_s_per_wall_s=sample['physics_s_per_wall_s'],
            reference_work_per_wall_s=sample['reference_work_per_wall_s'],comparison=event['comparison'],
            resources=dict(weighted_means=means,peak_project_rss_GiB=max(r['project_rss_bytes'] for r in rows)/2**30,
                           peak_project_GPU_MiB=max(r['project_gpu_MiB'] for r in rows),
                           minimum_available_ram_GiB=min(r['ram_available_bytes'] for r in rows)/2**30,
                           max_device_outstanding_io={k:max(r['host_disk_outstanding_io'].get(k,0) for r in rows)
                                                      for k in rows[0]['host_disk_outstanding_io']}),
            solvers={name:stream_timing(name,flow,start,sample['end_unix']) for name,flow in sample['streams'].items()}))
    current=read(STATE/'pool.json').get('independent_evaluation',{})
    data=dict(observed_at=datetime.now(SH).isoformat(),current=current,
              storage_strata={m:dict(admission_limit=v['admission_limit'],retained=v['retained'],blocked=v['blocked']) for m,v in cases.items()},
              decisions=decisions,
              timing_scope='Candidate event and MPC solve have timestamped samples. Whole feedback is available after a trajectory closes. The frozen WBC timer stores aggregate calls/wall/CPU, so isolated per-call WBC P95 is unavailable.',
              **BG)
    write(STATE/'dispatch/evaluation-throughput-readback.json',data)
    print(data['observed_at'],[(r['width'],r['action'],round(r['raw_physics_s_per_wall_s'],5)) for r in decisions])

if __name__=='__main__':summarize()
