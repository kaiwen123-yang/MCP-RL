"""One bounded least-squares calibration of the frozen positive dissipation coefficients."""
from collections import Counter
import hashlib
import json
import os
import threading
import subprocess
import time

import numpy as np
from scipy.optimize import least_squares
from s1_w55_c_data import RAW,COMPACT,WEIGHTS,DT,H,reference_run,write
from s1_w55_c_replay import candidate,weights,run,cpu_usage


def runs(mask):
    change=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)))


def contact_families(normal,owners):
    """Labels only depend on the input foot, time and support geometry."""
    phys=np.zeros_like(owners,dtype=np.int32)
    for leg in range(4):
        mask=(owners==leg)&(normal<0)
        for event,(start,stop) in enumerate(runs(np.any(mask,axis=1)),1):
            view=phys[start:stop]
            view[mask[start:stop]]=(leg+1)*10000+event
    return [tuple(int(k) for k in np.unique(phys[:,j]) if k) for j in range(normal.shape[1])]


def synthetic():
    T=650
    amplitudes=[.003,.008,.014,.018,.012,.016]
    widths=[24,30,16,18,35,12]
    gaps=[None,None,None,70,130,230]
    n=np.full((T,len(widths)),np.inf)
    conditions=[]
    for j,(amplitude,width,gap) in enumerate(zip(amplitudes,widths,gaps)):
        starts=[10] if gap is None else [10,10+width+gap]
        for p,s in enumerate(starts):
            a=amplitude*(1.-.2*p)
            n[s:s+width,j]=-a*np.sin(np.pi*(np.arange(width)+.5)/width)
        conditions.append(dict(synthetic=True,axial_peak_strain=amplitude,steps_per_pulse=width,release_gap_steps=gap))
    return n,reference_run(n,DT),conditions


def sequence_residual(pred,truth,groups):
    """Equal group weight; balance integrated strain after release and loaded stress.
    The released state is freely integrated, including reloaded intervals."""
    nn=Counter(groups);vectors=[]
    for j,g in enumerate(groups):
        active=truth[:,j,4]<-1e-7
        after=np.maximum.accumulate(active)
        free=~active&after
        scale=np.sqrt(len(nn)*nn[g])
        if free.any():
            denom=max(1.,float(np.max(np.abs(truth[free,j,0]))*H*1e6))
            vectors.append((pred[free,j,0]-truth[free,j,0])*H*1e6/(scale*denom*np.sqrt(free.sum())))
        if active.any():
            denom=max(.25,float(np.max(np.abs(truth[active,j,4]))))
            vectors.append((pred[active,j,4]-truth[active,j,4])/(scale*denom*np.sqrt(active.sum())))
    return np.concatenate(vectors)


def memory_sampler(stop,peaks):
    own=str(os.getpid())
    while not stop.is_set():
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],
                              capture_output=True,text=True)
        for line in result.stdout.splitlines():
            fields=[x.strip() for x in line.split(',')]
            if len(fields)==2 and fields[0]==own and fields[1].isdigit():
                peaks.append(int(fields[1]))
        stop.wait(.5)


def main():
    t0=time.time();c0=cpu_usage();base=weights(WEIGHTS);report=COMPACT/'fit.json'
    confirm=json.loads((COMPACT/'frozen-N44.json').read_text())
    qualifying=[r for r in confirm['fine'] if r['unloaded_reference_peak_um']>=1. and
                r['max_continuous_loading_rate_per_s']>=1. and r['recovery_error_over_peak']>=.15]
    if not qualifying:raise ValueError('Independent, physically resolved reference recovery threshold was not reached.')
    with np.load(RAW/'N44-histories.npz') as z:
        source_normal=z['normal'];source_keys=z['keys'];source_owner=z['owner'];source_truth=z['reference_fine']
    path_ids=[hashlib.sha256(np.ascontiguousarray(n).tobytes()).hexdigest() for n in source_normal.T]
    _,idx=np.unique(path_ids,return_index=True);idx=np.sort(idx)
    groups=contact_families(source_normal,source_owner)
    fam=[groups[j] for j in idx]
    real_n=source_normal[:,idx];real_t=source_truth[:,idx]
    sn,st,specs=synthetic()
    stop=threading.Event();mem=[]
    monitor=threading.Thread(target=memory_sampler,args=(stop,mem),daemon=True);monitor.start()
    rows=[]
    def residual(theta):
        w=candidate(base,theta)
        a,ta=run(w,real_n,DT,'gpu');b,tb=run(w,sn,DT,'gpu')
        r=np.r_[sequence_residual(a,real_t,fam),sequence_residual(b,st,list(range(sn.shape[1])))]/np.sqrt(2.)
        rows.append(dict(theta=theta.tolist(),cost=float(.5*r@r),GPU=[ta,tb]))
        write(report,dict(state='fitting',start_unix=t0,cpu_time_s=cpu_usage()-c0,objective_evaluations=len(rows),
             measured_process_memory_MiB_peak=max(mem,default=None),last_evaluation=rows[-1],deadline=None,wall_deadline_enabled=False))
        print('eval',len(rows),'theta',theta,'cost',rows[-1]['cost'], 'gpu_wall',ta['wall_s']+tb['wall_s'],flush=True)
        return r
    # An explicit absolute step is more reliable here than a tiny relative
    # perturbation at the exact frozen initialization. No held-out scores enter.
    def jac(theta):
        cached=cache['value']
        return np.stack([(residual(theta+e*1e-3)-cached)/1e-3 for e in np.eye(2)],axis=1)
    cache={}
    def obj(theta):
        cache['value']=residual(theta)
        return cache['value']
    result=least_squares(obj,np.zeros(2),jac=jac,bounds=(np.log([.5,.25]),np.log([2.,4.])),
                         xtol=1e-4,ftol=1e-4,gtol=1e-5,max_nfev=16)
    stop.set();monitor.join()
    path=RAW/'W55-short-load-ideal-potentials.npz'
    np.savez_compressed(path,**candidate(base,result.x))
    write(report,dict(state='candidate_fitted',start_unix=t0,end_unix=time.time(),wall_s=time.time()-t0,cpu_time_s=cpu_usage()-c0,
        measured_process_memory_MiB_peak=max(mem,default=None),memory_sampling_interval_s=.5,
        lesson='Objective forward passes run on the GPU; SciPy trust-region update and residual aggregation run on CPU.',
        source_weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),candidate=str(path),candidate_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        initial_cost=rows[0]['cost'],final_cost=float(result.cost),termination=result.message,optimizer_nfev=int(result.nfev),
        optimizer_njev=int(result.njev),objective_including_difference_calls=len(rows),theta=result.x.tolist(),positive_multipliers=np.exp(result.x).tolist(),
        training_episode=confirm['episode'],qualifying_cells=qualifying,real_paths=len(idx),real_event_families=len(set(fam)),
        train_full_world_keys=[source_keys[j].tolist() for j in idx],per_history_family=[list(g) for g in fam],
        synthetic_conditions=specs,synthetic_steps=len(sn),iterations=rows,gunits='FP64 material',deadline=None,wall_deadline_enabled=False,
        new_PPO_samples=0,new_PPO_updates=0,donor=0))
    print('FROZEN CANDIDATE',path,result.x,result.cost,flush=True)


if __name__=='__main__':main()
