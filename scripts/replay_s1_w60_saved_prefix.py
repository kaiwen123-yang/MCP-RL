"""Replay only recorded r58 E300 foot-boundary prefixes on the deployed r60 CPU observer."""
from bisect import bisect_left
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from s1_w56_window import WindowObserver
from s1_w60_cpu_observer import CPUWindowObserver, ROOT, LIBRARY
from tm02.s1_material import LearnedColumns
from tm02.s1_material_prediction import normal_reference

RUN=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
RAW=ROOT.parent/'MCP-RL-w56-material'/'outputs'/RUN
WEIGHTS=ROOT/'work/assets/B_random1000--ideal-potentials.npz'
PARAM=json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
OUT=ROOT/'results'/RUN/'w56/c/r60-saved-E300-prefix.json'
HISTORICAL=ROOT.parent/'MCP-RL-w55-integration/work/r58-e300-stall/full-window.json'
WORK=ROOT/'work/w60-replay'


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def publish(obs, rows):
    for r in rows:obs.completed(r[0],r[1],r[6:18].reshape(4,3),r[18:30].reshape(4,3))
    obs.observe(rows[-1,1],None,None)


def episode(case, previous, gpu):
    raw,=RAW.glob(f'job-*-f-{case}-r58-eval*-E300-attempt01')
    end=round(previous['end_packet_s'])
    obs=CPUWindowObserver(PARAM,WEIGHTS)
    t0=time.perf_counter();c0=time.process_time()
    last_normal=[]
    with (raw/'w55-f/normal-execution.jsonl').open() as src:
        for line in src:
            d=json.loads(line);t=d['consume_s']
            if t>end+1e-8:break
            if last_normal and t!=last_normal[-1]['consume_s']:last_normal=[]
            last_normal.append(d)
    inverse_comparisons=[]
    next_progress=0
    for i in range(end):
        with np.load(raw/f'material-{i:04d}.npz') as f:
            rows=f['foot'][:,:30]
        for j in range(0,len(rows),10):
            packet=rows[j:j+10]
            final=(i==end-1 and j==len(rows)-10)
            if final:
                # Only this comparison receives the same starting state. The
                # CPU replay itself began at virgin t=0 and never reads plant e/q.
                other=WindowObserver(PARAM,None,law=gpu)
                g=obs.ground.clone();g.prediction_steps=None;g.law=gpu
                other.ground=g;other.time=obs.time
                sample=time.perf_counter()
            publish(obs,packet)
            if final:cpu_packet=time.perf_counter()-sample
            if last_normal and abs(obs.time-last_normal[0]['consume_s'])<1e-8:
                legs=packet[-1,6:18].reshape(4,3)
                for d in last_normal:
                    started=time.perf_counter()
                    step,info=normal_reference(obs.ground,legs,d['leg'],d['target_N'],d['measured_N'],.002)
                    inverse_comparisons.append(dict(leg=d['leg'],consume_s=obs.time,wall_s=time.perf_counter()-started,
                        observed_area_error_m2=info['owned_area_m2']-d['owned_area_m2'],
                        target_residual_N=info['target_residual_N'],measured_residual_N=info['measured_residual_N'],
                        target_displacement_m=info['target_delta_m'],measured_displacement_m=info['measured_delta_m'],
                        archived_normal_increment_error_m=step-(d['measured_delta_m']-d['target_delta_m'])))
        if time.monotonic()>=next_progress:
            write(WORK/'progress.json',dict(case=case,physical_s=obs.time,archive_end=end,
                 world_cells=len(obs.ground.ids),relaxing=len(obs.relaxing),elapsed_s=time.perf_counter()-t0))
            next_progress=time.monotonic()+5
    elapsed=time.perf_counter()-t0;cs=time.process_time()-c0
    started=time.perf_counter();publish(other,packet);jax_packet=time.perf_counter()-started
    g=obs.ground;ref=other.ground
    if list(g.ids)!=list(ref.ids) or len(g.ids)!=previous['ground_cells']:
        raise ValueError('Wrong world map or wrong archival endpoint')
    world_delta={n:float(np.max(np.abs(getattr(g,n)-getattr(ref,n)))) for n in ('e','q','stress')}
    if max(world_delta['e'],world_delta['q'])>1e-8 or world_delta['stress']>1e-6 or np.any(g.loaded!=ref.loaded):
        raise ValueError(f'Archived final-window state mismatch {world_delta}')
    t=obs.time
    plans=[json.loads(s) for s in (raw/'w55-f/mpc-solves.jsonl').open()]
    idx=bisect_left([v['time_s'] for v in plans],t)-1
    with np.load(raw/'w55-f'/plans[idx]['file']) as z:
        query=z['foot_center_plan'][int(abs(z['times']-(t+.2)).argmin()),:,:2]
    transition=next(d for d in map(json.loads,(RAW/'evaluation-policy'/raw.name/'transitions.jsonl').open()) if abs(d['time_s']-t)<1e-8)
    expected=np.array(transition['next_pann_recovery_m'])
    started=time.perf_counter();actual=obs.recovery(query)[0].ravel();query_wall=time.perf_counter()-started
    if np.max(abs(actual-expected))>1e-10:raise ValueError('Actor recovery differs from archived behavior')
    if len(inverse_comparisons)!=len(last_normal) or max((abs(v['archived_normal_increment_error_m']) for v in inverse_comparisons),default=0)>1e-10:
        raise ValueError('Reference did not match actual archived soft-stance call')
    result=dict(attempt=raw.name,end_observed_time_s=t,total_measured_substeps=obs.updates,world_cells=len(g.ids),
        original_query_plan=plans[idx]['file'],whole_real_observation_prefix_wall_s=elapsed,
        prefix_cpu_core_s=cs,CPU_final_packet_s=cpu_packet,JAX_final_packet_s=jax_packet,
        comparison_initialization='Same replayed CPU prefix state for the final JAX packet; archived actor comparison uses original recorded behavior.',
        final_window_max_state_errors=world_delta,loaded_mask_equal=bool(np.array_equal(g.loaded,ref.loaded)),
        max_cpu_column_residual_MPa=float(g.max_residual),max_principal_strain=float(g.max_strain),
        archived_query_recovery_m=expected.tolist(),r60_query_recovery_m=actual.tolist(),
        max_error_vs_archived_recovery_m=float(np.max(abs(actual-expected))),query_wall_s=query_wall,
        final_query_zero_or_nonzero=bool(np.any(actual!=0)),latest_archived_normal_reference=inverse_comparisons,
        whole_prefix_max_simultaneously_integrated_columns=obs.max_packet_columns,
        whole_prefix_column_substeps=obs.integration_column_steps,final_relaxing_columns=len(obs.relaxing),
        prior_independent_r58_recorder_replay=previous)
    if case=='a':
        goal=previous['free_tail_hypothesis']['end_s']
        pos=packet[-1,6:18].reshape(4,3);velocity=packet[-1,18:30].reshape(4,3)
        if any(g.interior(p[:2]) for p in pos):raise ValueError('Archived final A feet are not all outside the strip')
        s=obs.time;started=time.perf_counter()
        while s<goal-1e-8:
            for _ in range(10):
                nxt=s+.0002;obs.completed(s,nxt,pos,velocity);s=nxt
            obs.observe(s,None,None)
        result['free_tail_hypothesis']=dict(start_s=t,end_s=obs.time,wall_s=time.perf_counter()-started,
            assumption='Held final measured positions outside soft terrain; free recovery only, no robot integration, no observed trajectory after 119.000.')
    else:
        result['coverage']='Last archived packet ended at 78.000; no observation for the missing interval to 78.14.'
    return result


def main():
    hist=json.loads(HISTORICAL.read_text())
    res=dict(source='Original completed 5000Hz measurement windows; initial material state is virgin',
         cpu_library_sha256=hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),pid=os.getpid(),
         background=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False),
         start_unix_s=time.time(),cases={})
    gpu=LearnedColumns(WEIGHTS)
    for case in ('b','a'):
        prior=next(v for v in hist['cases'] if v['case']==case)
        res['cases'][case]=episode(case,prior,gpu)
        res['updated_unix_s']=time.time();write(OUT,res)
        print(case,{k:res['cases'][case][k] for k in ('end_observed_time_s','world_cells','whole_real_observation_prefix_wall_s','max_error_vs_archived_recovery_m')},flush=True)


if __name__=='__main__':main()
