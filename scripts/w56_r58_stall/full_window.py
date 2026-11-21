"""Check the last fully recorded 500 Hz material packet, preserving world-cell order.

On D57, compile columns.cc with -O3 -shared -fPIC -std=c++17 and -I pointing
to the frozen r58 scripts directory. Write prefix.so to HERE. Use the r58
src and scripts directories on PYTHONPATH and the original FP64 JAX settings.
The CPU reconstruction uses observed sole boundaries and the original weights.
It has no access to the live process's unflushed world state.
"""
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import json
import time
from pathlib import Path
import numpy as np
from s1_w56_window import WindowObserver
from s1_w55_c_replay import pack
from tm02.s1_material import LearnedColumns
from tm02.s1_contact import Ground

ROOT=Path('/root/autodl-tmp/MCP-RL-w56-material')
HERE=Path('/root/autodl-tmp/MCP-RL-w55-integration/work/r58-e300-stall')
RAW=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
WEIGHTS=ROOT/'work/assets/B_random1000--ideal-potentials.npz'
PARAM=json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']

def histories(raw):
    half=np.sqrt(.001)/2
    chunks=sorted(raw.glob('material-[0-9]*.npz'))
    times=[]; segments=defaultdict(list); first={}; offset=0
    for file in chunks:
        with np.load(file) as f: rows=f['foot']
        if times and times[-1][-1]!=rows[0,0]:
            raise ValueError('Recorded physical clock discontinuity')
        times.append(rows[:,1].copy())
        bottom=rows[:,6:18].reshape(-1,4,3)
        for leg,feet in enumerate(bottom.transpose(1,0,2)):
            xy=feet[:,:2]; normal=feet[:,2]/.05
            interior=(xy[:,0]>=.8+half)&(xy[:,0]<=2.2-half)&(abs(xy[:,1])<=.6-half)
            lower=np.floor((xy-half)/.02).astype(int)
            for dx in range(3):
                for dy in range(3):
                    ij=lower+[dx,dy]
                    area=np.prod(np.maximum(np.minimum(xy+half,(ij+1)*.02)-np.maximum(xy-half,ij*.02),0.),axis=1)
                    mask=interior&(area>0)
                    # Within the material strip, the integer code uniquely identifies a world cell.
                    codes=np.where(mask,ij[:,0]*1000+ij[:,1]+200,-1)
                    ix=np.r_[0,np.flatnonzero(np.diff(codes))+1,len(rows)]
                    for s,e in zip(ix[:-1],ix[1:]):
                        if codes[s]<0: continue
                        key=tuple(int(v) for v in ij[s])
                        when=(offset+int(s),leg,*key)
                        if key not in first or when<first[key]:first[key]=when
                        segments[key].append((offset+int(s),normal[s:e].copy()))
        offset+=len(rows)
    return np.concatenate(times),segments,sorted(first,key=first.get),rows[-10:].copy()

def compare(case,law):
    raw=next(RAW.glob(f'job-*-f-{case}-r58-eval-*-E300-attempt01'))
    start=time.time()
    end,segments,keys,packet=histories(raw)
    ticks=len(end)-10
    model=pack(np.load(WEIGHTS))
    lib=ct.CDLL(str(HERE/'prefix.so'))
    vec=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
    lib.prefix_state.argtypes=[vec,ct.c_int,vec,vec,vec]
    def replay_cell(key):
        # For multiply covered cells, the lower sole supplies a single boundary.
        normal=np.full(len(end),np.inf)
        for lo,span in segments[key]:
            hi=lo+len(span)
            np.minimum(normal[lo:hi],span,out=normal[lo:hi])
        state=np.zeros(4)
        bad=lib.prefix_state(model,ticks,end,normal,state)
        if bad!=-1: raise ValueError((key,bad))
        return state
    print('Reconstructing full observer',case,len(keys),'cells',ticks,'steps',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool: state=np.stack(list(pool.map(replay_cell,keys)))
    obs=WindowObserver(PARAM,None,law=law)
    g=obs.ground
    g.ids={key:i for i,key in enumerate(keys)}
    g.xy=[tuple((np.array(k)+.5)*g.cell) for k in keys]
    g.e,g.q=state[:,:2].copy(),state[:,2:].copy()
    g.stress=np.zeros_like(g.e);g.loaded=np.zeros(len(keys),bool)
    g.last_owner=np.full(len(keys),-1);g.last_release=np.full(len(keys),np.nan)
    g.ever_loaded=np.zeros(len(keys),bool);g.last_normal=np.zeros(len(keys))
    g.last_area=np.zeros(len(keys));g.last_load_event_id=np.full(len(keys),-1)
    obs.time=float(packet[0,0]);pre=time.perf_counter()
    for row in packet:
        obs.completed(*row[:2],row[6:18].reshape(4,3),row[18:30].reshape(4,3))
    print('Calling original WindowObserver.observe',case,packet[-1,1],flush=True)
    obs.observe(float(packet[-1,1]),None,None)
    completed=time.perf_counter()-pre
    t=obs.time
    logs=[json.loads(s) for s in (raw/'w55-f/mpc-solves.jsonl').open()]
    idx=bisect_left([r['time_s'] for r in logs],t)-1
    with np.load(raw/'w55-f'/logs[idx]['file']) as z:
        q=z['foot_center_plan'][int(abs(z['times']-(t+.2)).argmin()),:,:2]
    actual=next(d for d in map(json.loads,(raw.parent/'evaluation-policy'/raw.name/'transitions.jsonl').open()) if abs(d['time_s']-t)<1e-8)
    forecast=obs.recovery(q)[0].ravel()
    expected=np.asarray(actual['next_pann_recovery_m'])
    ret=dict(case=case,attempt=raw.name,ground_cells=len(keys),original_substeps_in_prefix=ticks,
      begin_packet_s=float(packet[0,0]),end_packet_s=t,observer_returned_s=obs.time,
      original_observe_wall_s=completed,recovery_max_abs_difference_from_recorded_m=float(abs(expected-forecast).max()),
      recorded_recovery_max_abs_m=float(abs(expected).max()),replay_wall_s=time.time()-start)
    if case=='a':
        # A's missing 0.7 s tail is not an archived measurement. Check the
        # separately labelled free-boundary hypothesis, including the next replan.
        lo=t; pos=packet[-1,6:18].reshape(4,3); vel=packet[-1,18:30].reshape(4,3)
        if any(g.interior(p[:2]) for p in pos): raise ValueError('Final A packet has a soft foot')
        last=logs[-1]['time_s']; future=t
        while future<last+.020-1e-8:
            for _ in range(10):
                next_time=future+.0002;obs.completed(future,next_time,pos,vel);future=next_time
            obs.observe(future,None,None)
        ret['free_tail_hypothesis']=dict(start_s=lo,end_s=obs.time,all_world_cells=len(keys),
           assumption='No new material footprint after archived packet. Applies the original material window only; no robot integration or live-process memory read.')
    return ret

if __name__=='__main__':
    report=dict(start_unix_s=time.time(),cases=[])
    law=LearnedColumns(WEIGHTS)
    for case in ('b','a'):
        r=compare(case,law);report['cases'].append(r);report['updated_unix_s']=time.time()
        (HERE/'full-window.json').write_text(json.dumps(report,indent=2)+'\n')
        print(r,flush=True)
