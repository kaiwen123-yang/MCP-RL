#!/usr/bin/env python3
"""Native r34 full-policy confirmations in the GPU branch's two-CPU process."""
import argparse,json,os,shutil,time
import numpy as np,mujoco
from mujoco_mpc.agent import Agent
from compare_s1_go2_w45_saved import ROOT,OUT,write,initialize_agent,parse_candidates
def main():
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--directory',required=True,type=__import__('pathlib').Path);a=p.parse_args()
 fixture=json.loads((OUT/a.label/'input.json').read_text());batch=json.loads((a.directory/'candidates.json').read_text())
 order=json.loads((a.directory/'gpu-ranking.json').read_text())['confirmation_order']
 result=dict(status='running',rows=[],new_native_rollouts=0,cache_hits=0,host_cpus=2)
 native=a.directory/'native';native.mkdir(exist_ok=False)
 model=mujoco.MjModel.from_xml_path(str(ROOT/fixture['model_path']));frozen=ROOT/'work/r11-frozen/S5-r34'
 state=dict(np.load(OUT/a.label/'input-state.npz'));state['time']=float(state['time'])
 env=dict(os.environ,MJPC_TASKS_DIR=str(frozen/'runtime-tasks'),LD_LIBRARY_PATH=str(frozen/'lib'),
  MJPC_R33_INPUT_POLICY=str(native/'input.bin'),MJPC_R34_CONTEXT=str(OUT/a.label/'context.bin'),
  MJPC_R34_SCORE_ONLY='1',MJPC_R34_ACCEPTANCE='strict',MJPC_R33_MECHANISM='N',
  MJPC_R29_SNAPSHOT=str(native/'r29-current-plan.bin'))
 start=time.perf_counter();cache={}
 with (native/'agent.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,server_binary_path=frozen/'agent_server',
  extra_flags=['--mjpc_workers=2'],subprocess_kwargs=dict(stdout=log,stderr=log,env=env),run_init=False) as agent:
  initialize_agent(agent,fixture,state);result['native_setup_s']=time.perf_counter()-start
  for number,i in enumerate(order):
   entry=batch[i];t=time.perf_counter();key=entry['sha256']
   if key in cache:
    row=dict(cache[key],slot=i,cache_hit=True,wall_s=0.);result['cache_hits']+=1
   else:
    shutil.copyfile(a.directory/entry['file'],native/'input.bin')
    agent.set_state(**state);agent.planner_step()
    c=parse_candidates(native/'r29-current-plan.bin.r33-candidates')[0]
    row=dict(c['metadata'],slot=i,cache_hit=False,wall_s=time.perf_counter()-t)
    row['full_policy_sha256']=key
    np.savez_compressed(native/f'slot{i:02d}.npz',**c['arrays'])
    cache[key]=row;result['new_native_rollouts']+=1
   result['rows'].append(row)
   if number+1==len(set(order[:4])|{0,1}):result['top4_and_baseline_confirmation_s']=time.perf_counter()-start
   result['elapsed_s']=time.perf_counter()-start;write(a.directory/'native-results.json',result)
 result.update(status='completed',total_native_s=time.perf_counter()-start)
 write(a.directory/'native-results.json',result)
if __name__=='__main__':main()
