#!/usr/bin/env python3
"""One pure-mathematical evaluation of all 301 saved W51 IK contexts."""
import ctypes,json,subprocess,time
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 started=time.time();out=ROOT/'work/w52-geometry';out.mkdir(exist_ok=True)
 source=out/'geometry.cc'
 source.write_text(r"""
#include "experiments/terrain_memory/w52_geometry.h"
extern "C" void geometry(const double* in,double* out) {
 std::array<double,3> p,v,e;for(int j=0;j<3;++j){p[j]=in[3+j];v[j]=in[6+j];e[j]=in[9+j];}
 w52::Surface surf;auto s=w52::Make(in[0],in[1],in[2]!=0,p,v,e,surf);
 out[0]=s.valid;out[1]=s.min_clearance;out[2]=s.max_clearance;out[3]=s.apex;
 out[4]=s.reason=="available"?0:s.reason=="late_swing_still_grounded"?1:
  s.reason=="continuous_surface_clearance"?2:s.reason=="original_apex_amplitude"?3:4;
 out[5]=out[6]=out[7]=out[8]=0;
 if(s.pieces.empty())return;
 const auto& a=s.pieces.front();const auto& b=s.pieces.back();
 for(int j=0;j<3;++j) {
  out[5]=std::max(out[5],std::abs(double(w52::Eval(a.xyz[j],0))-p[j]));
  out[6]=std::max(out[6],std::abs(double(w52::Eval(w52::Derivative(a.xyz[j]),0))/(a.end-a.begin)-v[j]));
  out[7]=std::max(out[7],std::abs(double(w52::Eval(b.xyz[j],1))-e[j]));
 }
 if(s.pieces.size()==2)for(int order=0;order<3;++order) {
  auto x=a.xyz[2],y=b.xyz[2];for(int j=0;j<order;++j){x=w52::Derivative(x);y=w52::Derivative(y);}
  double error=double(w52::Eval(x,1))/std::pow(a.end-a.begin,order)-double(w52::Eval(y,0))/std::pow(b.end-b.begin,order);
  out[8]=std::max(out[8],std::abs(error));
 }
}
""")
 subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(ROOT),str(source),'-o',str(out/'geometry.so')],check=True)
 fn=ctypes.CDLL(str(out/'geometry.so')).geometry
 pointer=ctypes.POINTER(ctypes.c_double);fn.argtypes=[pointer,pointer]
 attempt='job-20261115T113036Z-w51-d1-command-repair-eval-PANN_PPO_S5-s1-D8-attempt01'
 original=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/attempt
 seeds=[json.loads(x) for x in (original/'w51-seeds.jsonl').read_text().splitlines() if '"motion"' in x]
 tr=np.load(original/'trajectory.npz');rows=[];reasons=Counter();swings=[];errors=np.zeros(4)
 for seed in seeds:
  m=seed['motion'];t=m['time_s'];idx=int(np.argmin(abs(tr['time']-t)))
  assert abs(tr['time'][idx]-t)<1e-9
  # Read every original IK payload as provenance, but never load a physics engine.
  ik=np.load(original/seed['objects']['IK']['file'])['kinematics_payload'];assert int(ik[1])==seed['call_id']
  legs=[]
  for i,j in enumerate([0,2,1,3]):
   foot=m['foot_context'][i];p=tr['feet'][idx,j];v=tr['feet_velocity'][idx,j]
   e=foot['touchdown'] if foot['swing'] else foot['anchor']
   inp=np.array([t,foot['end'],foot['swing'],*p,*v,*e],dtype=np.float64);res=np.empty(9)
   fn(inp.ctypes.data_as(pointer),res.ctypes.data_as(pointer))
   reason=['available','late_swing_still_grounded','continuous_surface_clearance','original_apex_amplitude','late_swing_not_descending'][int(res[4])]
   reasons[reason]+=1;errors=np.maximum(errors,res[5:9])
   row=dict(foot=i,swing=foot['swing'],valid=bool(res[0]),reason=reason,min_clearance_m=res[1],
    max_clearance_m=res[2],apex_s=res[3],initial_velocity=v.tolist())
   legs.append(row)
   if foot['swing']:swings.append(row)
  rows.append(dict(call_id=seed['call_id'],time_s=t,current_segments_all_valid=all(x['valid'] for x in legs),feet=legs))
 result=dict(status='completed',source_attempt=attempt,IK_originals=len(seeds),current_swing_snapshots=len(swings),
  current_segments_all_valid_calls=sum(x['current_segments_all_valid'] for x in rows),
  current_swing_valid=sum(x['valid'] for x in swings),reason_counts=dict(reasons),
  endpoint_max_error_m=errors[0],initial_velocity_max_error_m_s=errors[1],
  touchdown_max_error_m=errors[2],apex_C2_max_error=errors[3],
  accepted_swing_min_clearance_m=min(x['min_clearance_m'] for x in swings if x['valid']),
  scope='All 301 saved contexts, current segments only, same compiled W52 geometry header. Original targets are all on pre-material flat ground. This is not a repaired-plant success rate or future-swing dynamic result.',
  physics_steps=0,native_rollouts=0,FD=0,PPO=0,wall_s=time.time()-started,calls=rows)
 dest=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w52/geometry-existing-301.json'
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='calls'}),flush=True)
if __name__=='__main__':main()
