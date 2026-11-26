"""FP64 causal/true-release decompositions and error bands at predeclared recovery ages."""
import ctypes as ct
import gzip
import json
import time
from pathlib import Path

import numpy as np
from s1_w59_model_bank import OUT,WORK,ROOT,write
from s1_w59_model_fields import BG
from s1_w59_model_recovery import episodes,read,stats
from score_s1_w57_material import truth_at
from s1_w60_cpu_observer import CPUColumns
from s1_w55_c_data import WEIGHTS,DT

HORIZONS=(.02,.10,.35,.64,1.3,2.)


def free(law,states,ticks):
    ptr=ct.c_void_p
    fn=ct.CDLL(str(ROOT/'work/w57-m/forecast_cpu.so')).w57_forecast_cpu
    fn.argtypes=[ptr,ptr,ptr,ptr,ct.c_int,ct.c_double,ct.c_int,ptr];fn.restype=None
    x=np.ascontiguousarray(states,float);n=np.ascontiguousarray(ticks,np.int32);boundary=np.zeros(1);out=np.empty((len(n),6))
    fn(law.model.ctypes.data,x.ctypes.data,boundary.ctypes.data,n.ctypes.data,len(n),DT,1,out.ctypes.data)
    if out[:,4].max()>1e-7:raise RuntimeError('free recurrence exceeded the original solver residual')
    return out[:,:4]


def main():
    started=time.time();law=CPUColumns(WEIGHTS)
    fitted=json.loads((OUT/'recovery.json').read_text())['parameters']
    a,rate=fitted['release_prior_blend'],fitted['positive_recovery_rate_s_inv']
    result={};flat=[]
    for ep,role,raw,f0,f1,_ in episodes():
        m1=[r for r in read(f1) if 'release_state' in r and r['event']['cross_leg']];m0={r['event_id']:r for r in read(f0)}
        blocks=sorted(raw.glob('material-[0-9]*.npz'))
        truth0=truth_at(blocks,[dict(event_id=r['event_id'],target=r['release_publication_tick'],keys=[r['key']]) for r in m1])
        scopes={}
        for h in HORIZONS:
            length=round(h/DT);rows=[r for r in m1 if r['release_publication_tick']+length<=r['event_tick']]
            ix=len(rows)
            scopes[str(h)]=dict(all_cross_foot_events=len(m1),still_unloaded=ix,reloaded_before_horizon=len(m1)-ix)
            if not ix:continue
            xt=truth_at(blocks,[dict(event_id=r['event_id'],target=r['release_publication_tick']+length,keys=[r['key']]) for r in rows])
            true0=np.array([truth0[r['event_id']][0,:4] for r in rows])
            true=np.array([xt[r['event_id']][0,:4] for r in rows])
            init1=np.array([r['release_state'][:4] for r in rows]);init0=np.array([m0[r['event_id']]['release_state'][:4] for r in rows]);init3=init1+a*(init0-init1)
            states={'M1':free(law,init1,np.full(ix,length)),'M1_true_initial':free(law,true0,np.full(ix,length)),
                    'M3':init3*np.exp(-rate*h),'M3_true_initial':true0*np.exp(-rate*h)}
            for name,estimate in states.items():
                err=-.05*(estimate[:,0]-true[:,0]);qe=estimate[:,2]-true[:,2]
                scopes[str(h)][name]=dict(depth_m=stats(err),q11_offline_only=stats(qe))
                for q,e in zip(rows,err):
                    flat.append(dict(episode=ep,role=role,horizon_s=h,event_id=q['event_id'],world_key=q['key'],model=name,depth_error_m=float(e)))
        result[ep]=dict(role=role,horizons=scopes)
        print(ep,flush=True)
    bands={}
    for h in HORIZONS:
        bands[str(h)]={}
        for mode in ('M1','M3'):
            fit=[abs(r['depth_error_m']) for r in flat if r['role']=='fit' and r['model']==mode and r['horizon_s']==h]
            evalrows=[r for r in flat if r['role']=='development_validation' and r['model']==mode and r['horizon_s']==h]
            band=float(np.quantile(fit,.95)) if fit else None
            bands[str(h)][mode]=dict(calibration_fit_events=len(fit),development_validation_events=len(evalrows),
                p95_half_width_m=band,coverage=(float(np.mean([abs(r['depth_error_m'])<=band for r in evalrows])) if evalrows and band is not None else None))
    with gzip.open(WORK/'free-horizons.json.gz','wt') as f:json.dump(flat,f,allow_nan=False)
    write(OUT/'free-horizons.json',dict(by_episode=result,uncertainty=bands,
        note='Censored by real reload; full cross-foot denominator retained at each age. True initial state/q11 used only for post-fit decomposition. Fit-episode residual bands are descriptive.',wall_s=time.time()-started,**BG))

if __name__=='__main__':main()
