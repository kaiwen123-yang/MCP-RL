"""Replay saved measured states through the planner; never integrate the plant."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import mujoco
from mujoco_mpc.agent import Agent
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--label',required=True)
 p.add_argument('--without-moving-support',action='store_true');a=p.parse_args()
 large=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
 source=large/'train-PANN_PPO_S5-s0-ep00118'
 out=large/'job-20261101T133954Z'/'regression'/a.label;out.mkdir(parents=True,exist_ok=True)
 resolved=json.loads((ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/train-PANN_PPO_S5-s0-ep00118-resolved.json').read_text())
 c=resolved['r12_configuration'];data=np.load(source/'trajectory.npz')
 model=mujoco.MjModel.from_xml_path(str(source/'model/G1-project-go2.xml'))
 frozen=ROOT/'work/r11-frozen'/a.build
 env=dict(os.environ,LD_LIBRARY_PATH=str(frozen/'lib'),MJPC_TASKS_DIR=str(frozen/'runtime-tasks'),
          S1_REGRESSION_DIAGNOSTIC=str(out/'planner.jsonl'))
 observations=[];started=time.time()
 with (out/'server.log').open('w') as log,Agent(task_id='Quadruped Flat',server_binary_path=str(frozen/'agent_server'),
      extra_flags=['--mjpc_workers=20'],subprocess_kwargs=dict(stdout=log,stderr=log,env=env),run_init=False) as agent:
  agent.init(task_id='Quadruped Flat',model=model,send_as='xml');agent.set_mode('Quadruped')
  agent.set_task_parameter('Gait switch','Manual');previous=None
  for i,t in enumerate(data['time']):
   if t>2.021:break
   phase=int(data['phase'][i]);standing=phase==0
   table=c['task']['stand_settings' if standing else 'move_and_turn_settings']
   def state():
    agent.set_state(time=float(t),qpos=data['qpos'][i],qvel=data['qvel'][i],
      mocap_pos=data['mocap_pos'][i],mocap_quat=data['mocap_quat'][i],
      userdata=data['support_reference_userdata'][i])
   if phase!=previous:agent.set_task_parameter('Gait',table['Gait'])
   agent.set_task_parameter('Heading',float(data['heading_target'][i]))
   state()
   if phase!=previous:
    agent.set_task_parameters({k:v for k,v in table.items() if k not in ('Gait','Balance','Upright','Height')})
    agent.set_cost_weights({k:table[k] for k in ('Balance','Upright','Height')}|{'Orientation':c['task']['orientation_cost_weight']})
    # Reuse the actual complete readback of this phase, not guessed weights.
    setting=next(x for x in resolved['applied_settings'] if abs(x['time_s']-t)<1e-6)
    agent.set_cost_weights(setting['cost_weights'])
    if a.without_moving_support:agent.set_cost_weights({'MovingSupport':0.})
    state();previous=phase
   if i==0:
    for _ in range(c['startup']['planner_steps']):agent.planner_step()
   elif data['planner_s'][i]>0:agent.planner_step()
   if t>=1.999:
    actual=agent.get_action(time=float(t),nominal_action=False,averaging_duration=0)
    nominal=agent.get_action(time=float(t),nominal_action=True,averaging_duration=0)
    observations.append(dict(t_s=float(t),actual=actual.tolist(),nominal=nominal.tolist(),
       correction_max_rad=float(np.max(abs(actual-nominal))),
       old_recorded_correction_max_rad=float(np.max(abs(data['q_target'][i]-data['nominal_q_target'][i]))),
       action_difference_from_recorded=float(np.max(abs(actual-data['q_target'][i]))),
       cost_at_measured_state=agent.get_cost_term_values()))
    print(json.dumps(observations[-1]),flush=True)
 result=dict(build=a.build,label=a.label,without_moving_support=a.without_moving_support,
    source_episode=source.name,plant_integration_steps=0,wall_s=time.time()-started,observations=observations,
    scope='Recomputed planner from saved measured states/history;original gains were not saved and are not claimed recovered.')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
