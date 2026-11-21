"""Replay the final completely archived 10-boundary r58 window, for selected natural cells."""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import time
import numpy as np
from tm02.s1_material import LearnedColumns
from s1_w55_c_replay import pack
from s1_w55_c_data import geometry

ROOT=Path('/root/autodl-tmp/MCP-RL-w56-material')
HERE=Path('/root/autodl-tmp/MCP-RL-w55-integration/work/r58-e300-stall')
DATA=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
WEIGHTS=ROOT/'work/assets/B_random1000--ideal-potentials.npz'
LIB=Path('/root/autodl-tmp/MCP-RL-w56-integration/work/w56-query-columns.so')

def replay(case, law):
    folder=next(DATA.glob(f'job-*-f-{case}-r58-eval-*-E300-attempt01'))
    parts=sorted(folder.glob('material-[0-9]*.npz'))
    with np.load(parts[-1]) as saved:
        endpoint=saved['foot'][-1]
    # Select the last actually loaded cell in each foot's event history before
    # the archival boundary. The observed kinematics, not hidden plant q, are inputs.
    keys={}
    for line in (folder/'material-events.jsonl').open():
        e=json.loads(line)
        if e['time_s']<=endpoint[1] and e['type'] in ('load','revisit'):
            keys[e['leg']]=tuple(e['world_key'])
    selected=np.array(sorted(set(keys.values())),dtype=int)
    normal=[];clock=[];last_done=None
    for p in parts:
        with np.load(p) as f:
            foot=f['foot']
        if last_done is not None and foot[0,0]!=last_done:
            raise ValueError('Recorded material blocks are discontinuous')
        last_done=foot[-1,1]
        n,_,_=geometry(foot[:,6:18].reshape(-1,4,3),selected)
        normal.append(n);clock.append(foot[:,1])
    n=np.concatenate(normal)
    t=np.r_[0.,np.concatenate(clock)]
    tix=np.array([len(n)-10,len(n)],dtype='i')
    coeff=pack(np.load(WEIGHTS))
    lib=ct.CDLL(str(LIB))
    d=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
    lib.query_states.argtypes=[d,ct.c_int,d,d,np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),ct.c_int,d]
    batch=[]
    for k in range(len(selected)):
        out=np.empty((2,7))
        code=lib.query_states(coeff,len(t),t,np.r_[np.inf,n[:,k]],tix,2,out)
        if code!=-1:
            raise ValueError((case,k,code))
        batch.append(out[:,:4])
    st=np.array(batch)
    e,q=st[:,0,:2],st[:,0,2:]
    worst=0.; mixed=0; start=time.perf_counter()
    for k in range(len(n)-10,len(n)):
        dt=t[k+1]-t[k]
        free_e,free_q,_,res=law.step(e,q,np.zeros(len(e)),dt,np.zeros(len(e),bool))
        worst=max(worst,float(np.max(res)))
        use=np.flatnonzero(n[k]<=free_e[:,0])
        if len(use):
            ce,cq,stress,res=law.step(e[use],q[use],n[k,use],dt,np.ones(len(use),bool))
            mixed+=1;worst=max(worst,float(np.max(res)))
            active=stress[:,0]<0
            free_e[use[active]]=ce[active];free_q[use[active]]=cq[active]
        e,q=free_e,free_q
    return dict(case=case,source_attempt=folder.name,last_material_block=parts[-1].name,
            select='Most recent recorded loading event for each foot at the end of the last archived material block',
            world_keys=selected.tolist(),window_start_s=t[-11],window_end_s=t[-1],free_calls=10,mixed_calls=mixed,
            observer_state_before='CPU FP64 reconstruction from original boundaries since t=0, same frozen weights; not original process memory',
            max_abs_cpu_gpu_final_state_diff=float(np.max(np.abs(np.c_[e,q]-st[:,1,:]))),
            max_device_residual_MPa=worst,elapsed_s=time.perf_counter()-start)

if __name__=='__main__':
    result=dict(start_unix_s=time.time(),weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),cases=[])
    print('Initializing independent material context',flush=True)
    law=LearnedColumns(WEIGHTS)
    for case in ('b','a'):
        print('Reconstructing',case,flush=True)
        result['cases'].append(replay(case,law))
        print(result['cases'][-1],flush=True)
        result['updated_unix_s']=time.time()
        (HERE/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
