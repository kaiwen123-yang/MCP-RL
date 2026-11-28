"""Independent per-packet CPU oracle and paired real-collector end-to-end grading."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from paired import SPEC,RAW,DEST,STUDY
FROZEN=Path('/root/autodl-tmp/MCP-RL-w59')
sys.path.insert(0,str(FROZEN/'scripts'))
from s1_w60_cpu_observer import ARRAY,BYTE,INDEX


def parity(folder):
    lib=ct.CDLL(str(FROZEN/'work/w57-m/material.so')).w57_packet
    lib.argtypes=[ARRAY,ct.c_int,ct.c_int,ARRAY,ARRAY,ARRAY,INDEX,ct.c_double,ARRAY,BYTE,BYTE,ARRAY,ARRAY,BYTE,ct.c_int,ARRAY,ARRAY]
    with np.load(folder/'benchmark-timing.npz') as f:info=f['packets']
    fused={int(x[0]):bool(x[3]) for x in info}
    records=[]
    with np.load(folder/'parity-samples.npz') as z:
        model=z['model']
        for c in sorted(int(k.split('_',1)[0][1:]) for k in z.files if k.endswith('_n')):
            get=lambda name:z[f'p{c}_'+name]
            n,ticks=int(get('n')),int(get('ticks'));st=get('start_state');s=get('start_settled');l=get('start_loaded')
            f=np.zeros((ticks,4));v=f.copy();tr=np.zeros((ticks,n),'u1');diag=np.zeros((4,8))
            bad=lib(model,n,ticks,get('dt'),get('normal'),get('area'),get('owners'),float(get('gamma')),st,s,l,f,v,tr,int(get('correct')),get('measured'),diag)
            end=ticks if bad<0 else bad//n
            records.append(dict(seq=c,columns=n,fused=fused[c],same_failure=bad==int(get('bad')),
                same_active_masks=bool(np.array_equal(s,get('end_settled')) and np.array_equal(l,get('end_loaded')) and np.array_equal(tr[:end],get('trace')[:end])),
                max_state_abs=float(np.max(abs(st-get('end_state')))),max_force_abs_N=float(np.max(abs(f-get('forces')))),
                max_diagnostics_abs=float(np.max(abs(diag-get('diagnostics'))))))
    return records


def compare(w,baseline,trt,gate):
    pairs={};soft={}
    for case in ('A','B'):
        left=Path(baseline['raw_directory'])/case;right=Path(trt['raw_directory'])/case
        def load_logs(p):
            with np.load(p/'trajectory.npz') as a:x=a['control']
            events=[json.loads(i) for i in (p/'events.jsonl').read_text().splitlines()]
            return x,events
        u,ue=load_logs(left);v,ve=load_logs(right)
        n=min(len(u),len(v))
        l=baseline['cases'][case];r=trt['cases'][case]
        req=lambda row:(row['mode'],row['rng_key'],row['rng_state_sha256'],row['end_kind'])
        exact=len(ue)==len(ve) and all(req(i)==req(j) for i,j in zip(ue,ve))
        dt=max((abs(i['end_s']-j['end_s']) for i,j in zip(ue,ve)),default=0.)
        actions=max((float(np.max(abs(np.asarray(i['issued_action'])-j['issued_action']))) for i,j in zip(ue,ve)),default=0.)
        ds=float(np.max(abs(u[:n,2:21]-v[:n,2:21]))) if n else None
        # q-pos19, q-vel18, control12, torque12, foot XYZ12, measured4x3.
        grf=float(np.max(abs(u[:n,75:87]-v[:n,75:87]))) if n else None
        pairs[case]=dict(matched_frames=n,CPU_frames=len(u),GPU_frames=len(v),qpos_max_abs=ds,
                measured_GRF_max_abs_N=grf,issued_hybrid_action_max_abs=actions,event_time_max_abs_s=dt,
                exact_modes_rng_terminal_and_event_count=exact,coincident_terminal=l['reason']==r['reason'] and l['physical_steps']==r['physical_steps'],
                full_feedback_chain_p95_ratio=r['control_full_ms_p50_p95_p99'][1][1]/l['control_full_ms_p50_p95_p99'][1][1])
        for interval in l['soft_windows']:
            if interval not in r['soft_windows']:continue
            soft.setdefault(interval,{})[case]=dict(CPU=l['soft_windows'][interval],GPU=r['soft_windows'][interval],
                  per_stream_speedup=l['soft_windows'][interval]['wall_s']/r['soft_windows'][interval]['wall_s'])
    group={}
    for interval,parts in soft.items():
        if set(parts)!={'A','B'}:continue
        group[interval]={}
        for arm in ('CPU','GPU'):
            entries=[h[arm] for h in parts.values()]
            wall=max(t['end_monotonic_s'] for t in entries)-min(t['start_monotonic_s'] for t in entries)
            lo,hi=map(float,interval.split('-'))
            group[interval][arm]=dict(group_wall_s=wall,physics_steps_s=2*(hi-lo)/.0002/wall,
                complete_events_s=sum(t['events'] for t in entries)/wall,nonempty_packets=sum(t['nonempty_packets'] for t in entries),
                genuinely_fused_packets=sum(t['fused_packets'] for t in entries),
                shared_wall_overlap_s=max(0.,min(t['end_monotonic_s'] for t in entries)-max(t['start_monotonic_s'] for t in entries)))
        group[interval]['group_speedup']=group[interval]['CPU']['group_wall_s']/group[interval]['GPU']['group_wall_s']
    arrays={case:parity(Path(trt['raw_directory'])/case) for case in ('A','B')}
    fused=[p for case in arrays.values() for p in case if p['fused']]
    ratio=baseline['group_wall_s']/trt['group_wall_s']
    raw_pass=bool(fused) and all(d['same_failure'] and d['same_active_masks'] and max(d['max_force_abs_N'],d['max_state_abs'])<=gate['maximum_state_stress_or_force_same_input_abs'] for d in fused)
    trajectory=all(v['coincident_terminal'] and v['exact_modes_rng_terminal_and_event_count'] and v['qpos_max_abs'] is not None and v['qpos_max_abs']<=gate['maximum_physical_qpos_abs_m_or_rad'] for v in pairs.values())
    tail=all(v['full_feedback_chain_p95_ratio']<=gate['maximum_worse_full_p95_ratio'] for v in pairs.values())
    all_packets=sum(v['nonempty_packets'] for v in trt['cases'].values())
    all_fused=sum(v['fused_packets'] for v in trt['cases'].values())
    return dict(stage=baseline['phase'],window=w,arms={'cpu':baseline,'gpu_batch':trt},
        group_wall_speedup_including_broker_and_process_start=ratio,
        complete_event_throughput_ratio=trt['complete_events_s']/baseline['complete_events_s'],
        soft_windows=group,case_soft_windows=soft,batch_width_histogram=trt.get('batch_width_histogram',{}),
        fused_fraction=all_fused/all_packets if all_packets else None,fused_probe_count=len(fused),same_input_parity=arrays,
        task_and_rng_parity=pairs,checks=dict(same_input=raw_pass,physical_trajectory=trajectory,tails=tail,
            net_speedup=ratio>=gate['minimum_each_group_speedup'],multiple_batch=trt.get('mixed_worker_kernel_launches',0)>1),
        note='Diagnostic-prefix data do not enter the formal policy budget or authorize a production backend switch.')


def main():
    from paired import write
    cfg=json.loads(SPEC.read_text())
    previous=None
    while True:
        results={};latest={}
        for w in cfg['windows']:
            rows={}
            for arm in ('cpu','gpu_batch'):
                f=DEST/f'{w}-{arm}.json'
                if f.is_file():rows[arm]=json.loads(f.read_text())
            if rows:latest[w]={k:dict(status=v['status'],group_wall_s=v['group_wall_s'],complete_events_s=v['complete_events_s']) for k,v in rows.items()}
            file=DEST/f'{w}-comparison.json'
            if file.is_file():results[w]=json.loads(file.read_text());continue
            if len(rows)==2 and all(v['status']=='complete' for v in rows.values()):
                results[w]=compare(w,rows['cpu'],rows['gpu_batch'],cfg['admission']);write(file,results[w])
                print(w,results[w]['group_wall_speedup_including_broker_and_process_start'],results[w]['fused_fraction'],results[w]['checks'],flush=True)
        key=str(latest)
        if key!=previous:
            full=[v for v in results.values() if v['stage']=='full_training_collector_work']
            proposed=len(full)==2 and all(all(v['checks'].values()) for v in full)
            index=dict(stage='full_complete' if len(full)==2 else 'complete_prefix_full_pending' if 'prefix12' in results else 'collecting_prefix',
                   production_eligible=proposed,production_backend='cpu',
                   adoption='Only after a positive complete result may policy/gait/integrator register a new immutable runtime at a natural boundary.',
                   windows=latest,comparisons={w:dict(path=f'{w}-comparison.json',speedup=v['group_wall_speedup_including_broker_and_process_start'],fused_fraction=v['fused_fraction'],checks=v['checks']) for w,v in results.items()},
                   config=str(SPEC),soft_windows_s=cfg['soft_windows_s'],admission=cfg['admission'],new_formal_used=0)
            write(DEST/'index.json',index)
            from paired import SITE
            files=[str(p.relative_to(SITE)) for p in DEST.glob('*.json')]
            subprocess.run(['git','add','--sparse','--',*files],cwd=SITE,check=True)
            if subprocess.run(['git','diff','--cached','--quiet'],cwd=SITE).returncode==1:
                subprocess.run(['git','commit','-qm','Report actual paired E CPU and fused-collector GPU stage evidence'],cwd=SITE,check=True)
            previous=key
        if len(results)==3:return
        time.sleep(8)

if __name__=='__main__':main()
