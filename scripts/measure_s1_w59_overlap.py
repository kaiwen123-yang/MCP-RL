"""Measure the existing four mechanisms and two earlier recovery starts on the same D57 clock."""
from bisect import bisect_right
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

from publish_s1_go2_w54 import write
from s1_w59_lease import ROOT

WORK=ROOT/'work/w59'
STATE=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59'
ZONE=ZoneInfo('Asia/Shanghai')
SECONDS=240.


def sh(t):
    return datetime.fromtimestamp(t,ZONE).isoformat(timespec='seconds')


def points(rows,name):
    observations={}
    for sample in rows:
        if name in sample['clock']:
            sim,wall,phase=sample['clock'][name]
            observations[wall]=(sim,phase)
    return sorted((t,*value) for t,value in observations.items())


def clock(data,t):
    i=bisect_right([x[0] for x in data],t)
    if i==0 or i==len(data):
        raise ValueError('A live clock has not bracketed the common wall interval')
    left,right=data[i-1],data[i]
    return left[1]+(t-left[0])*(right[1]-left[1])/(right[0]-left[0])


def window(rows,names,start):
    end=start+SECONDS
    chosen=[r for r in rows if start<=r['unix_s']<=end]
    if not chosen or any(set(r['slots'])!=names for r in chosen):
        raise ValueError('The selected window contains a slot ownership change')
    streams={}
    for name in sorted(names):
        samples=points(rows,name)
        a,b=clock(samples,start),clock(samples,end)
        streams[name]=dict(start_sim_s=a,end_sim_s=b,sim_s_per_wall_s=(b-a)/SECONDS,
            phases=sorted({phase for t,_,phase in samples if start<=t<=end}))
    return dict(start_shanghai=sh(start),end_shanghai=sh(end),wall_s=SECONDS,streams=streams,
        aggregate_sim_s_per_wall_s=sum(v['sim_s_per_wall_s'] for v in streams.values()),
        cpu_W59=mean(r['W59_cpu_cores'] for r in chosen),cpu_other=mean(r['other_cpu_cores'] for r in chosen),
        gpu_board_median_pct=median(r['gpu_util_used_total_MiB'][0] for r in chosen),
        min_usable_ram_GiB=min(r['usable_memory_bytes'] for r in chosen)/2**30,
        min_free_scratch_GiB=min(r['scratch_free_bytes'] for r in chosen)/2**30)


def main():
    rows=[json.loads(line) for line in (WORK/'resource-samples.jsonl').open()]
    short={f'w59-{arm}-m1-{case}-development-short-a01' for arm in ('b3','fixed-action') for case in 'ab'}
    recovered={f'w59-{arm}-m1-a-development-recovered-a01' for arm in ('b3','fixed-action')}
    first=min(datetime.fromisoformat(json.loads((WORK/'pool/jobs'/(n+'.json')).read_text())['started_at']).timestamp() for n in recovered)
    four_end=max(r['unix_s'] for r in rows if r['unix_s']<first and set(r['slots'])==short)
    six=short|recovered
    six_start=next(r['unix_s'] for r in rows if r['unix_s']>=first and set(r['slots'])==six
                   and all(n in r['clock'] and r['clock'][n][0]>=10 for n in six))
    b=window(rows,short,four_end-SECONDS);a=window(rows,six,six_start)
    inc=sum(a['streams'][n]['sim_s_per_wall_s'] for n in short)
    result=dict(scope='Two 240 s natural scratch windows. Interpolated at common absolute wall boundaries from actual plant clock publications. The four incumbent PIDs/attempts did not change. New tasks are the already registered recovered-A cases. Different contact phases and external workload remain visible.',
        before_four=b,after_six=a,raw_total_change_pct=100*(a['aggregate_sim_s_per_wall_s']/b['aggregate_sim_s_per_wall_s']-1),
        incumbent_four_change_pct=100*(inc/b['aggregate_sim_s_per_wall_s']-1),
        physical_steps_added_for_measurement=0,official_denominator_changed=False)
    # Confirm on the following non-overlapping interval once each clock covers it.
    if all(points(rows,n)[-1][0] > six_start+2*SECONDS for n in six):
        second=window(rows,six,six_start+SECONDS)
        result['following_six']=second
        result['following_total_change_pct']=100*(second['aggregate_sim_s_per_wall_s']/b['aggregate_sim_s_per_wall_s']-1)
        result['following_incumbent_change_pct']=100*(sum(second['streams'][n]['sim_s_per_wall_s'] for n in short)/b['aggregate_sim_s_per_wall_s']-1)
    write(STATE/'dispatch/mechanism-overlap.json',result)
    print(json.dumps({k:v for k,v in result.items() if k.endswith('pct')}),
          b['aggregate_sim_s_per_wall_s'],a['aggregate_sim_s_per_wall_s'],b['start_shanghai'],a['end_shanghai'])


if __name__=='__main__':
    main()
