#!/usr/bin/env python3
"""One FP64 CUDA reduction of the existing 114 W50 residual trajectories; no physics."""
import json,os,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from compare_s1_go2_w44_saved import parse_candidates
from publish_s1_go2_w51 import ROOT,SMALL,LARGE,write,compact,deliver
def main():
 started=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 src=LARGE/'w50';out=LARGE/'w51-C';out.mkdir(exist_ok=False)
 groups=[];rows=[];residuals=[];states=[];definitions=None;fixtures=[]
 for family,labels in [('task-b',['A237','B236','A328','B625']),('deployment',['A237','B236'])]:
  for label in labels:
   fixture=json.loads((src/'inputs'/label/'input.json').read_text());definitions=fixture['cost_definitions']
   records=parse_candidates(src/family/label/'snapshot.bin.r33-candidates')
   first=len(rows)
   for r in records:
    rows.append(dict(group=family+'/'+label,physics_evidence_usable=family=='task-b',**r['metadata']))
    residuals.append(r['arrays']['residual']);states.append(r['arrays']['states'])
    fixtures.append(fixture)
   groups.append(dict(id=family+'/'+label,start=first,end=len(rows),physics_evidence_usable=family=='task-b'))
 R=np.stack(residuals);X=np.stack(states)
 device='cuda';torch.cuda.reset_peak_memory_stats()
 tensor=torch.tensor(R,device=device,dtype=torch.float64)
 torch.cuda.synchronize();start_reduce=time.perf_counter()
 gpu_nodes=[];cpu_nodes=[]
 for d in definitions:
  q=tensor[:,:,d['start']:d['end']];qn=R[:,:,d['start']:d['end']];p=d['norm_parameter']
  if d['norm']==0:v=.5*(q*q).sum(-1);vn=.5*(qn*qn).sum(-1)
  elif d['norm']==2:v=torch.sqrt((q*q).sum(-1)+p*p)-p;vn=np.sqrt((qn*qn).sum(-1)+p*p)-p
  elif d['norm']==6:v=(torch.sqrt(q*q+p*p)-p).sum(-1);vn=(np.sqrt(qn*qn+p*p)-p).sum(-1)
  else:raise ValueError('Unknown original norm')
  gpu_nodes.append(v*d['weight']);cpu_nodes.append(vn*d['weight'])
 nodes=torch.stack(gpu_nodes,-1);terms=nodes.mean(1);cost=terms.sum(-1)
 torch.cuda.synchronize();reduce_s=time.perf_counter()-start_reduce
 gpu=cost.cpu().numpy();gterms=terms.cpu().numpy();gnodes=nodes.cpu().numpy()
 cnodes=np.stack(cpu_nodes,-1);cterms=cnodes.mean(1);native=np.array([r['native_J'] for r in rows])
 # Original roundoff envelope and original risk metadata; never mix problems.
 n=R.shape[1]*(8*R.shape[2]+4*len(definitions)+2)+1;u=np.finfo(np.float64).eps/2;gamma=n*u/(1-n*u)
 offset=sum(2*abs(d['weight']*d['norm_parameter'])*(d['end']-d['start'] if d['norm']==6 else 1)
  for d in definitions if d['norm'] in (2,6))
 budget=gamma*(np.abs(cnodes).mean(1).sum(-1)+offset)
 def better(i,j,J):
  a,b=rows[i],rows[j]
  if a['failure'] or not np.isfinite([J[i],a['trunk_violation']]).all():return False
  if b['failure'] or not np.isfinite([J[j],b['trunk_violation']]).all():return True
  return a['trunk_violation']<b['trunk_violation'] or (a['trunk_violation']==b['trunk_violation'] and J[j]-J[i]>budget[i]+budget[j])
 for group in groups:
  idx=range(group['start'],group['end'])
  mismatches=[(i,j) for i in idx for j in idx if better(i,j,gpu)!=better(i,j,native)]
  group.update(candidate_count=len(idx),native_risk_and_roundoff_pairwise_mismatches=mismatches,
    comparisons=len(idx)**2,native_J_range=[float(native[list(idx)].min()),float(native[list(idx)].max())])
 # Pure state algebra whose native branches are established by the saved Trot context.
 xt=torch.tensor(X,device=device,dtype=torch.float64);quat=xt[:,:,3:7];quat=quat/torch.linalg.vector_norm(quat,dim=-1,keepdim=True)
 w,x,y,z=quat.unbind(-1)
 upright=torch.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)-1],-1)
 heading=torch.tensor([f['settings']['task_parameters']['Heading'] for f in fixtures],device=device,dtype=torch.float64)[:,None]
 hx=1-2*(y*y+z*z);hy=2*(x*y+w*z);hn=torch.sqrt(hx*hx+hy*hy)
 orientation=torch.stack([hx/hn-torch.cos(heading),hy/hn-torch.sin(heading)],-1)
 homes=[];heights=[]
 for f in fixtures:
  model=ET.parse(f['model_path'])
  homes.append([float(v) for v in model.find(".//key[@name='home']").attrib['qpos'].split()][7:19])
  heights.append(float(model.find(".//numeric[@name='r10_material_support_height']").attrib['data'].split()[0]))
 home=torch.tensor(homes,device=device,dtype=torch.float64)[:,None,:]
 posture=(xt[:,:,7:19]-home)*torch.tensor([2,1,1]*4,device=device,dtype=torch.float64)
 height=(xt[:,:,2]-torch.tensor(heights,device=device,dtype=torch.float64)[:,None])[:,:,None]/.04
 rebuilt={'Upright':upright,'Orientation':orientation,'Posture':posture,'MaterialSupportHeight':height}
 # These terms are gated to exact zero by the recorded moving (non-Stand) mode.
 for name in ('StandLinearVelocity','StandAngularVelocity','StandAbsoluteHeight','StandFootPosition','StandFootVelocity','StandSupportForce'):
  d=next(d for d in definitions if d['name']==name)
  rebuilt[name]=torch.zeros_like(tensor[:,:,d['start']:d['end']])
 errors={}
 for name,value in rebuilt.items():
  d=next(d for d in definitions if d['name']==name)
  errors[name]=float((value-tensor[:,:,d['start']:d['end']]).abs().max().item())
 dependency={
  'Upright':'base quaternion, moving-material-upright mode',
  'Height':'torso inertial COM position, all foot geom centers, original height target',
  'Position':'head site FK, R26 immutable packet, mocap and time',
  'Gait':'foot geom FK, geom radii, ground/material reference, private absolute phase and C2 parameters',
  'Balance':'subtree COM/velocity, support hull and foot positions',
  'Effort':'actuator_force and original actuator force range; node action alone is insufficient',
  'Posture':'qpos joints, home keyframe, stand/support gating',
  'Orientation':'base quaternion, original heading parameter',
  'Angmom':'torso angular-momentum sensor; full model inertias and velocities for reconstruction',
  'StandLinearVelocity':'qvel and mode; inactive in these Trot records',
  'StandAngularVelocity':'qvel and mode; inactive in these Trot records',
  'StandAbsoluteHeight':'base height and support target; inactive in these Trot records',
  'StandFootPosition':'foot FK and support userdata; inactive in these Trot records',
  'StandFootVelocity':'foot Jacobian/qvel and support mode; inactive in these Trot records',
  'StandSupportForce':'contact frame, geom ids and solver contact force; inactive here, active-mode solver data absent',
  'LandingVelocity':'foot FK/Jacobian, phase derivative, material patch rest and gating',
  'LandingIndentation':'foot FK, radius, material patch rest, stiffness and load scale',
  'MaterialSupportHeight':'base z, original support target, moving/support-region gate',
  'MaterialDomainMargin':'foot FK, material strip and exact native domain function/context',
  'TrunkClearance':'collision geom transforms/shapes and original clearance function',
  'ImpactMargin':'foot FK/Jacobian plus original approach/contact proxy state and bounds',
  'InputReference':'held control, current epoch R35 u/R/time grid; terminal node is zero by original definition'}
 exact_fields={'Upright': ['qpos[3:7]', 'mode', 'gait', 'r11_material_upright flags'], 'Height': ['xipos[trunk]', 'geom_xpos[FL,RL,FR,RR]', 'userdata[1]', 'mode'], 'Position': ['site_xpos[head]', 'userdata[84:96]', 'mocap_pos[goal]', 'time', 'r26_reference'], 'Gait': ['geom_xpos[FL,RL,FR,RR]', 'geom_size[foot]', 'phase_start/phase_start_time/phase_velocity', 'amplitude/duty/gait', 'flat_ground_height'], 'Balance': ['sensordata[torso_subtreecom]', 'sensordata[torso_subtreelinvel]', 'geom_xpos[feet]', 'mode/gait'], 'Effort': ['actuator_force[12] at each residual node', 'actuator_forcerange'], 'Posture': ['qpos[7:19]', 'key_qpos[home]', 'userdata[22:34]', 'mode/gait'], 'Orientation': ['qpos[3:7]', 'Heading parameter', 'mode'], 'Angmom': ['sensordata[torso_angmom] or full model inertia/FK/qvel'], 'StandLinearVelocity': ['qvel[0:3]', 'stand_equilibrium gate'], 'StandAngularVelocity': ['qvel[3:6]', 'stand_equilibrium gate'], 'StandAbsoluteHeight': ['xpos[trunk,2]', 'userdata[2]', 's1_stand_absolute_height', 'stand/support gates'], 'StandFootPosition': ['geom_xpos[feet]', 'userdata[10:22]', 'support gate'], 'StandFootVelocity': ['geom spatial velocity (mj_objectVelocity)', 'support gate'], 'StandSupportForce': ['ncon', 'contact.geom1/geom2', 'contact.frame', 'contact.efc_address', 'efc_force', 'userdata[58:62]', 'track_load gate'], 'LandingVelocity': ['geom_xpos[feet]', 'geom spatial velocity', 'userdata[62:78]', 'C2 FootStep derivative', 'r10/r11 settings'], 'LandingIndentation': ['geom_xpos[feet]', 'geom_size[feet]', 'userdata[62:78]', 'body_subtreemass[trunk]', 'stiffness/gravity', 'material gate'], 'MaterialSupportHeight': ['xpos[trunk,2] (exact derived native position)', 'r10_material_support_height', 'moving/material support gate'], 'MaterialDomainMargin': ['geom_xpos[feet]', 'geom_size[feet]', 'userdata[62:78]', 'r11_domain_margin and strip bounds'], 'TrunkClearance': ['xpos/xmat', 'collision geom types/sizes/transforms', 'r12_trunk_clearance'], 'ImpactMargin': ['geom_xpos[feet]', 'geom spatial velocity', 'userdata[62:78]', 'r26_impact_margin'], 'InputReference': ['current-epoch r35 origin/dt/u/R', 'actual held control at residual node', 'terminal node identity']}
 coverage=[]
 for d in definitions:
  name=d['name'];coverage.append(dict(**d,cost_reduction='FP64_GPU_from_saved_native_residual',
   residual_reconstruction='implemented_state_algebra_or_recorded_inactive_branch' if name in rebuilt else
    'requires_contact_solver_data_for_active_branch' if name=='StandSupportForce' else 'not_implemented_from_saved_state',
   dependency=dependency[name],exact_reconstruction_fields=exact_fields[name],max_residual_error=errors.get(name),
   active_mode_scope='All six recorded groups are original Quadruped/Trot; inactive Stand zeros are analytical gates, not filling missing measurements.'))
 np.savez_compressed(out/'fp64-costs.npz',node_terms=gnodes,term_mean=gterms,total_J=gpu,native_J=native,
  original_roundoff_budget=budget)
 result=dict(status='completed',started_unix_s=started,ended_unix_s=time.time(),deadline=None,wall_deadline_enabled=False,
  candidates=len(rows),saved_problem_candidates=76,wrong_bridge_algebra_only_candidates=38,
  residual_shape=list(R.shape),cost_terms=len(definitions),dtype='float64',device=torch.cuda.get_device_name(),
  physics_steps=0,native_rollouts=0,PPO_samples=0,PPO_updates=0,batches=1,
  peak_allocated_bytes=torch.cuda.max_memory_allocated(),reduction_wall_s=reduce_s,
  timings_scope='one observed run while another project uses the GPU; not a throughput benchmark',
  max_abs_gpu_vs_native_J_error=float(np.max(abs(gpu-native))),
  max_abs_gpu_vs_cpu_term_error=float(np.max(abs(gterms-cterms))),
  max_abs_gpu_vs_cpu_node_term_error=float(np.max(abs(gnodes-cnodes))),
  groups=groups,coverage=coverage,candidates_metadata=rows,
  residual_reconstruction_max_errors=errors,formal_installation_allowed=False,
  complete_native_cost_port=False,scope='Saved residual-to-cost reduction; partial pure state algebra, no new physics or complete contact/solver residual migration.')
 write(SMALL/'w51/task-c.json',result);write(out/'result.json',result)
 print(json.dumps({k:result[k] for k in ['status','candidates','physics_steps','max_abs_gpu_vs_native_J_error','residual_reconstruction_max_errors']}),flush=True)
 compact([SMALL/'w51/task-c.json'],'Record W51 C FP64 GPU cost reduction and 22-term coverage')
 publication=deliver('C-costs',[out]);result['delivery']=publication;write(SMALL/'w51/task-c.json',result)
 compact([SMALL/'w51/task-c.json'],'Record W51 C verified direct delivery')
if __name__=='__main__':main()
