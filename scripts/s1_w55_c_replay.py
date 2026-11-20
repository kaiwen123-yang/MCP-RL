"""Deploy the axisymmetric convex potentials on complete measured boundary histories."""
import ctypes as ct
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from s1_w55_c_data import CASES, COMPACT, RAW, ROOT, WEIGHTS, REVIEW, DT, H, FORCE_SCALE, write

D = ct.POINTER(ct.c_double)


def softplus(x):
    return np.logaddexp(0.,x)


def positive_scale(x, log_factor):
    positive=softplus(x)*np.exp(log_factor)
    return positive+np.log(-np.expm1(-positive))


def candidate(base, theta):
    """Change only existing positive dissipation coefficients. A rate-only
    constant does not affect stress; y-only output terms cancel at zero rate."""
    w={k:v.copy() for k,v in base.items()}
    a,b=theta
    for key in ('phi/u2','phi/x2'):
        w[key]=positive_scale(w[key],a)
    w['phi/u1'][2]=positive_scale(w['phi/u1'][2],b)
    w['phi/x2'][2]=positive_scale(w['phi/x2'][2],b)
    return w


def pack(w):
    norm=lambda k:w['norm/'+k]
    parts=[]
    for family in ('eq','ov'):
        wi,wo,skip=[softplus(w[family+'/'+k]) for k in ('wi','wo','skip')]
        b=w[family+'/b'];z=(-norm('inv_m')/norm('inv_s'))@wi+b
        slope0=(wi[0]@(wo.ravel()/(1+np.exp(-z)))+skip.ravel()[0])/norm('inv_s')[0]
        parts.extend((wi,b,wo,skip,slope0))
    for k in ('v','bv','g1','bg1','u1','y1','b1','gu','bgu','gx','bgx','u2','x2'):
        value=w['phi/'+k];parts.append(softplus(value) if k in ('u1','u2','x2') else value)
    for k in ('inv_m','inv_s','rate_inv_m','rate_inv_s','state_inv_m','state_inv_s','energy_s','dissipation_s'):
        parts.append(norm(k))
    return np.ascontiguousarray(np.concatenate([np.asarray(a,dtype='d').ravel() for a in parts]))


def cpu_usage():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime+r.ru_stime


def packed_run(p, normal, dt, device):
    lib=ct.CDLL(str(ROOT/f'work/w55-c/column_{device}.so'))
    if len(p)!=lib.model_doubles():
        raise ValueError('Packed model field count differs from compiled layout')
    n=np.ascontiguousarray(normal,dtype='d');T,K=n.shape
    out=np.empty((T,K,6));perf=np.full(3,np.nan)
    fn=getattr(lib,device+'_run')
    fn.argtypes=[D,D,ct.c_int,ct.c_int,ct.c_double,D]+([D] if device=='gpu' else [])
    fn.restype=ct.c_int if device=='gpu' else None
    ptr=lambda x:x.ctypes.data_as(D)
    begin=time.time();c0=cpu_usage()
    ret=fn(ptr(p),ptr(n),K,T,dt,ptr(out),*([ptr(perf)] if device=='gpu' else []))
    end=time.time()
    if ret or not np.isfinite(out).all() or out[:,:,5].max()>1e-7:
        raise RuntimeError(f'{device} status={ret} finite={np.isfinite(out).all()} maximum residual={out[:,:,5].max()}')
    timing=dict(device=device,start_unix=begin,end_unix=end,wall_s=end-begin,cpu_s=cpu_usage()-c0,paths=K,steps_per_path=T,
                max_residual_MPa=float(out[:,:,5].max()),cpu_affinity=sorted(os.sched_getaffinity(0)))
    if device=='gpu':
        timing.update(gpu_setup_H2D_s=float(perf[0]),gpu_kernel_s=float(perf[1]),gpu_D2H_s=float(perf[2]),
                      explicit_gpu_bytes=int(8*(len(p)+T*K*7+4*K)))
    return out,timing


def run(w,normal,dt,device):
    return packed_run(pack(w),normal,dt,device)


def physical_metrics(a,b,area,keys):
    # The fine reference does not set a tensile force. Use compressive stress
    # and its actual material area rather than tiny free-state solve residuals.
    rows=[]
    for i,key in enumerate(keys):
        sa=a[:,i,4];sb=b[:,i,4];aa=area[:,i]
        active_a=sa < -1e-7;active_b=sb < -1e-7
        loaded=active_a|active_b
        unloaded=~loaded
        recovery_ref=-b[unloaded,i,0]*H*1e6
        recovery_delta=(a[unloaded,i,0]-b[unloaded,i,0])*H*1e6
        rr=float(max(0.,recovery_ref.max(initial=0)))
        er=float(np.max(np.abs(recovery_delta),initial=0.))
        force_err=float(np.max(np.abs(np.where(active_a,-sa,0.)-np.where(active_b,-sb,0.))*aa*FORCE_SCALE))
        peak=float(np.max(np.abs(sb),initial=0.))
        row=dict(key=list(map(int,key)),stress_abs_max_MPa=float(np.max(np.abs(sa-sb))),normal_force_abs_max_N=force_err,
                 stress_loaded_rmse_MPa=float(np.sqrt(np.mean((sa[loaded]-sb[loaded])**2))) if loaded.any() else None,
                 q_rmse=float(np.sqrt(np.mean((a[:,i,2:4]-b[:,i,2:4])**2))),
                 unloaded_recovery_error_um=er,unloaded_reference_peak_um=rr,recovery_error_over_peak=(er/rr if rr>1e-12 else None),
                 reference_peak_stress_MPa=peak,contact_disagreement_samples=int(np.count_nonzero(active_a!=active_b)))
        rows.append(row)
    return rows


def interclock(a,b,area_a,area_b):
    sa=a[:,:,4];sb=b[:,:,4]
    fa=np.where(sa<-1e-7,-sa*area_a*FORCE_SCALE,0.);fb=np.where(sb<-1e-7,-sb*area_b*FORCE_SCALE,0.)
    return dict(max_force_N=float(np.max(np.abs(fa-fb))),rms_force_N=float(np.sqrt(np.mean((fa-fb)**2))),
                max_abs_stress_MPa=float(np.max(np.abs(sa-sb))),max_free_compression_delta_um=float(
                    np.max(np.where((sa>=-1e-7)&(sb>=-1e-7),np.abs(a[:,:,0]-b[:,:,0])*H*1e6,0))),
                contact_disagreement_samples=int(np.count_nonzero((sa<-1e-7)!=(sb<-1e-7))))


def evaluate_case(name,w,tag,device):
    with np.load(RAW/f'{name}-histories.npz') as z:
        n=z['normal'];aa=z['area'];keys=z['keys'];idx=z['coarse'];ref=z['reference_fine'];obs=z['reference_observed']
        held=z['reference_endpoint_fine']
        # Physical load/release episode families are never split across cohorts.
        original=json.loads((COMPACT/f'{name}-inventory.json').read_text())
        identifiers=[original['full_boundary_sha256'][','.join(map(str,k))] for k in keys]
        _,unique,inv=np.unique(identifiers,return_index=True,return_inverse=True)
        pf,tf=run(w,n[:,unique],DT,device)
        pc,tc=run(w,n[idx][:,unique],.002,device)
        np.savez_compressed(RAW/f'{tag}-{name}-model.npz',unique=unique,inverse=inv,fine=pf,observer=pc)
        fine=physical_metrics(pf[:,inv],ref,aa,keys)
        coarse=physical_metrics(pc[:,inv],obs,aa[idx],keys)
        for j,r in enumerate(fine):
            loaded=ref[:,j,4]<-1e-7
            prev=loaded[1:]&loaded[:-1]
            r['max_continuous_loading_rate_per_s']=float(np.max(np.abs(np.diff(ref[:,j,0]))[prev],initial=0)/DT)
        common=idx>0;ii=idx[common]
        ctl=aa[ii];phy=aa[ii-1]
        metrics=dict(episode=name,model=tag,weight_variables_sha256=hashlib.sha256(pack(w).tobytes()).hexdigest(),
                     fine=fine,observer=coarse,execution=[tf,tc],source_inventory=f'{name}-inventory.json',
                     fine_vs_endpoint_sampling=interclock(held[common],ref[ii-1],ctl,phy),
                     same_endpoint_2ms_vs_02ms_integration=interclock(obs[common],held[common],ctl,ctl),
                     combined_sampling_and_dt=interclock(obs[common],ref[ii-1],ctl,phy),
                     sampler_semantics='fixed measured endpoint repeated 10 times at .2ms; preceding true boundaries from that window; same output timestamp',
                     regular_coarse_excludes_finish_only=True)
        write(COMPACT/f'{tag}-{name}.json',metrics)
        high=[r for r in fine if r['max_continuous_loading_rate_per_s']>=1 and r['unloaded_reference_peak_um']>=1.]
        print(name,tag,[(r['key'],round(100*r['recovery_error_over_peak'],2)) for r in sorted(high,key=lambda r:-r['recovery_error_over_peak'])[:3]],
              tf,flush=True)
        return metrics


def weights(path):
    with np.load(path) as f:return {k:np.asarray(f[k],dtype='d') for k in f.files}


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',type=Path,default=WEIGHTS);ap.add_argument('--tag',default='frozen')
    ap.add_argument('--device',choices=['gpu','cpu'],required=True);ap.add_argument('cases',nargs='+',choices=CASES)
    args=ap.parse_args();w=weights(args.weights)
    for name in args.cases:evaluate_case(name,w,args.tag,args.device)
