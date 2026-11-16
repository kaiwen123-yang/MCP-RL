#!/usr/bin/env python3
"""Correct one saved-state residual using the actual Agent XML transport precision."""
import json,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np,torch
from compare_s1_go2_w44_saved import parse_candidates
from publish_s1_go2_w52 import ROOT,SMALL,LARGE,write
def main():
 started=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 out=LARGE/'w52-C';out.mkdir(exist_ok=False)
 src=LARGE/'w50';groups=[];z=[];native=[];heights=[];rows=[]
 for family,labels in [('task-b',['A237','B236','A328','B625']),('deployment',['A237','B236'])]:
  for label in labels:
   fixture=json.loads((src/'inputs'/label/'input.json').read_text());p=Path(fixture['model_path'])
   model=mujoco.MjModel.from_xml_path(str(p));serialized=out/(label+'-agent-transport.xml')
   # Exactly the Agent.init(send_as="xml") path; no Agent process or physics.
   mujoco.mj_saveLastXML(str(serialized),model)
   original=ET.parse(p);transport=ET.parse(serialized)
   original_height=float(original.find(".//numeric[@name='r10_material_support_height']").attrib['data'])
   height=float(transport.find(".//numeric[@name='r10_material_support_height']").attrib['data'])
   torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'trunk')
   joint=int(model.body_jntadr[torso])
   assert model.body_parentid[torso]==0 and model.jnt_type[joint]==mujoco.mjtJoint.mjJNT_FREE and model.jnt_qposadr[joint]==0
   moving=transport.find(".//numeric[@name='r11_moving_body_support']")
   assert moving is not None and float(moving.attrib['data'])==1 and fixture['settings']['declared_parameters']['Gait']=='Trot'
   # Under this exact active gate, native xpos[trunk,2] equals qpos[2].
   definition=next(x for x in fixture['cost_definitions'] if x['name']=='MaterialSupportHeight')
   records=parse_candidates(src/family/label/'snapshot.bin.r33-candidates');start=len(rows)
   for r in records:
    z.append(r['arrays']['states'][:,2]);native.append(r['arrays']['residual'][:,definition['start']])
    heights.append(height);rows.append(r['metadata'])
   groups.append(dict(family=family,label=label,start=start,count=len(records),source_height=original_height,
    transported_height=height,expected_old_residual_offset=(height-original_height)/.04,
    source_model=str(p),serialized_context=str(serialized.relative_to(ROOT)),
    physical_evidence_usable=family=='task-b',native_gate='r11_moving_body_support=1, original Quadruped Trot',
    native_geometry='trunk is free root body, translation qpos[0:3] is xpos[trunk]'))
 torch.cuda.reset_peak_memory_stats()
 Z=torch.tensor(np.array(z),device='cuda',dtype=torch.float64)
 H=torch.tensor(heights,device='cuda',dtype=torch.float64)[:,None]
 R=torch.tensor(np.array(native),device='cuda',dtype=torch.float64)
 rebuilt=(Z-H)/.04
 error=(rebuilt-R).abs();delta_term=(.5*rebuilt.square()-.5*R.square()).mean(dim=1)
 corrected_delta=delta_term.cpu().numpy()
 old=json.loads((SMALL/'w51/task-c.json').read_text())
 coverage=old['coverage']
 for term in coverage:
  if term['name']=='MaterialSupportHeight':
   term.update(residual_reconstruction='exact_from_saved_state_and_actual_serialized_model_in_recorded_active_Trot_gate',
    max_residual_error=float(error.max()),active_mode_scope='Original moving body support gate only; no new active Stand/material-only branch evidence')
 for group in groups:
  a=group['start'];b=a+group['count'];group['max_residual_error']=float(error[a:b].max())
  group['max_abs_J_change_vs_native']=float(np.max(abs(corrected_delta[a:b])))
 np.savez_compressed(out/'height-reconstruction.npz',native=np.array(native),rebuilt=rebuilt.cpu().numpy(),
   targeted_J_delta=corrected_delta)
 result=dict(status='completed',started_unix_s=started,ended_unix_s=time.time(),
  root_cause='W51 reconstruction read high-precision source XML target .29020336039643685; actual Agent XML transport saves .290203. Constant target offset, not inferred physical xpos error.',
  code_path='mujoco_mpc/agent.py Agent.init: mj_saveLastXML -> native init XML',
  old_max_residual_error=old['residual_reconstruction_max_errors']['MaterialSupportHeight'],
  corrected_max_residual_error=float(error.max()),max_targeted_J_change_vs_native=float(np.max(abs(corrected_delta))),
  original_rankings_unchanged=bool(np.all(corrected_delta==0)),candidates=114,saved_problem_candidates=76,wrong_bridge_algebra_only_candidates=38,
  active_state_terms=4,inactive_Stand_terms=6,remaining_unreconstructed_terms=12,coverage=coverage,groups=groups,
  device=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
  physics_steps=0,native_rollouts=0,FD=0,PPO_samples=0,PPO_updates=0,
  previous_full_reduction_repeated=False,formal_installation_allowed=False,complete_native_cost_port=False,
  deadline=None,wall_deadline_enabled=False,wall_s=time.time()-started)
 write(out/'result.json',result);write(SMALL/'w52/task-c.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ('groups','coverage')}),flush=True)
if __name__=='__main__':main()
