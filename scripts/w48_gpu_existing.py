#!/usr/bin/env python3
"""W48 C: existing repeated GPU identities, first divergence; no integration."""
import json,hashlib,os,time,subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np,mujoco
from w48_build import ROOT,sha
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/task-c'
LARGE=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/task-c'
GPU=Path('/root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w46')
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def first(a):
 ids=np.flatnonzero(np.any(a!=0,axis=1));return int(ids[0]) if len(ids) else None
def main():
 begin=time.perf_counter();SMALL.mkdir(parents=True,exist_ok=True);LARGE.mkdir(parents=True,exist_ok=True)
 prior=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/task-c'
 original=json.loads((prior/'run.json').read_text())
 expected={x['path']:x['sha256'] for x in original['sources']}
 sources=[]
 for p in [GPU/'paired-states-actions.npz',GPU/'action-algebra.npz',GPU/'scenarios.json',GPU/'result.json']:
  digest=sha(p);assert digest==expected[str(p)]
  sources.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest))
 info=json.loads((prior/'gpu-corrected.json').read_text())
 tags=json.loads((GPU/'scenarios.json').read_text());z=np.load(GPU/'paired-states-actions.npz')
 groups=defaultdict(list)
 for row in info['scenarios']:groups[row['scenario_execution_sha256']].append(row['scenario_index'])
 rows=[];arrays={}
 for group,(ident,ids) in enumerate((x for x in groups.items() if len(x[1])>1)):
  assert len(ids)==2
  i,j=ids;m=mujoco.MjModel.from_xml_path(str(GPU/(tags[i]['label']+'-model.xml')))
  row=dict(identity=ident,indices=ids,tags=[tags[i],tags[j]],model_sha256=sha(GPU/(tags[i]['label']+'-model.xml')),
   origin_s=info['scenarios'][i]['origin_s'],world_index_distinct=True)
  for backend in ['native','gpu']:
   x,y=z[backend+'_states'][:,i],z[backend+'_states'][:,j];u,v=z[backend+'_actions'][:,i],z[backend+'_actions'][:,j]
   d=x-y;du=u-v;dq=[]
   for a,b in zip(y,x):
    p=np.zeros(m.nv)
    if not np.array_equal(a[:m.nq],b[:m.nq]):mujoco.mj_differentiatePos(m,p,1.,a[:m.nq],b[:m.nq])
    dq.append(np.r_[p,b[m.nq:]-a[m.nq:]])
   dq=np.asarray(dq);k=first(d);ku=first(du)
   key=f'group{group}_{backend}';arrays[key+'_state_difference']=d;arrays[key+'_tangent_velocity_difference']=dq;arrays[key+'_held_difference']=du
   row[backend]=dict(initial_state_bitwise_equal=bool(np.array_equal(x[0],y[0])),
    first_state_node=k,first_state_after_origin_s=.002*k if k is not None else None,
    first_held_node=ku,first_held_decision_after_origin_s=.002*ku if ku is not None else None,
    max_base_position_m=float(abs(dq[:,:3]).max()),max_orientation_rad=float(abs(dq[:,3:6]).max()),
    max_joint_position_rad=float(abs(dq[:,6:18]).max()),max_linear_velocity_m_s=float(abs(dq[:,18:21]).max()),
    max_angular_velocity_rad_s=float(abs(dq[:,21:24]).max()),max_joint_velocity_rad_s=float(abs(dq[:,24:]).max()),
    max_held_difference_rad=float(abs(du).max()))
   if k is not None:row[backend]['first_state_difference_components']=np.flatnonzero(d[k]!=0).tolist()
  rows.append(row)
 assert len(rows)==6
 np.savez_compressed(LARGE/'duplicate-timedifferences.npz',**arrays)
 result=dict(status='completed_existing_arrays',rows=rows,duplicate_groups=6,new_physics_steps=0,new_GPU_allocations=0,
  new_native_solves=0,new_PPO_samples=0,new_PPO_updates=0,policy_generation_build='historical W45 S5-r34',
  action_evaluation_build='historical W46 GPU FP64 formula plus host H-QP and FP32 engine; not r36',
  optional_action_batch='not_run; existing duplicate identities analyzed without adding GPU work',
  available=np.array(z.files).tolist(),source_files=sources,
  missing_fields=['per-world initial qacc_warmstart','per-substep contact/normal-force/KKT/solver traces','complete hidden engine allocator/initialization state'],
  conclusion='Repeated declared identity is exact in saved initial x; native and GPU differences are reported per time. Hidden initialization versus numerical/world-index effects cannot be uniquely separated from these arrays.',
  artifact=dict(path=str(LARGE/'duplicate-timedifferences.npz'),bytes=(LARGE/'duplicate-timedifferences.npz').stat().st_size,sha256=sha(LARGE/'duplicate-timedifferences.npz')),
  affinity=sorted(os.sched_getaffinity(0)),wall_s=time.perf_counter()-begin,deadline_unix_s=None,wall_deadline_enabled=False)
 write(SMALL/'duplicates.json',result)
 print(json.dumps(dict(groups=6,rows=[dict(indices=x['indices'],gpu_first_state=x['gpu']['first_state_after_origin_s'],gpu_max_action=x['gpu']['max_held_difference_rad'],native_max_action=x['native']['max_held_difference_rad']) for x in rows],wall_s=result['wall_s'])))
if __name__=='__main__':main()
