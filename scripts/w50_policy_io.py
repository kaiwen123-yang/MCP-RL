"""The complete W50 policy binary and fixed-size feedback provenance."""
from pathlib import Path
import numpy as np
from w49_common import fields
PROVENANCE_FIELDS=['schema','mode','reference_epoch','reference_call','reference_generation_time','reference_start','reference_end',
 'K_epoch','K_call','K_generation_time','H_epoch','H_call','H_generation_time','alpha','baseline_install_id',
 'baseline_reference_generation_time','baseline_K_epoch','baseline_K_call','baseline_K_generation_time',
 'baseline_H_epoch','baseline_H_call','baseline_H_generation_time','baseline_mu','baseline_H_mode',
 'new_backward_mu','new_backward_rate','new_backward_attempt','derivative_epoch','derivative_call','derivative_time',
 'baseline_scale_absorbed','raw_center_evaluations']
def read_policy(path):
 raw=np.fromfile(path,dtype='<f8');h=raw[:16].copy();i=16
 T,ns,nu,nx,nr=map(int,h[3:8]);p={}
 for name,shape in fields(T,ns,nu,nx,nr):
  n=int(np.prod(shape));p[name]=raw[i:i+n].reshape(shape).copy();i+=n
 ext=raw[i:i+6].copy();i+=6;n=int(ext[2])
 if ext[0]!=49:raise ValueError('Complete delta extension missing')
 p.update(delta_header=ext,delta_times=raw[i:i+n].copy());i+=n
 p['delta_values']=raw[i:i+n*nu].reshape(n,nu).copy();i+=n*nu
 p['semantics']=np.array([ext[1],h[8],h[9],h[10],h[11],h[12],h[13]])
 if h[0]==50:p['provenance']=raw[i:i+32].copy();i+=32
 if i!=len(raw):raise ValueError('Unexpected complete-policy payload')
 return h,p

def provenance_summary(policy,query_time=None):
 v=policy.get('provenance')
 result=dict(zip(PROVENANCE_FIELDS,(float(x) for x in v))) if v is not None else {'schema':49,'mode':0}
 result['mode_name']={0:'historical_sources_unavailable',1:'native_bootstrap',2:'homotopy'}.get(int(result['mode']),'unknown')
 result['unknown_source_sentinel']=-1
 times=policy['reference_times'];result['reference_start']=float(times[0]);result['reference_end']=float(times[-1])
 if query_time is not None:result.update(reference_covers_query=bool(times[0]<=query_time<=times[-1]),reference_expired_by_s=max(0.,float(query_time-times[-1])))
 result['metric_single_mu']=None if policy['semantics'][1]<0 else float(policy['semantics'][1])
 result['derivative_fields_role']='captured_backward_not_mixed_derivatives'
 return result
