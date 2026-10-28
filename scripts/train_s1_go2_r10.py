#!/usr/bin/env python3
"""One independent standard-PPO stream using the same complete S4 runner."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_s1_go2_r10 import arguments,configuration
from run_s1_go2_wb_baseline import run,dump
from s1_go2_r10_policy import PolicyStream,PolicyMaterial
from mujoco_mpc.agent import Agent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',required=True,choices=['PPO_RAW_S4','PANN_PPO_S4'])
    p.add_argument('--seed',required=True,type=int,choices=[0,1,2])
    a=p.parse_args()
    small=ROOT/'results/terrain_memory/simulation/tm-s1-go2-soft-landing-r10'
    frozen=json.loads((small/'frozen-controller.json').read_text())
    c,r9=configuration(frozen['formulation'],frozen['build']);contract=c['r10_contract']
    output=ROOT/contract['outputs']['large_root']/'training'/f'{a.group}-seed{a.seed}'
    stream=PolicyStream(ROOT,a.group,a.seed,contract,output)
    os.environ['MJPC_TASKS_DIR']=str(ROOT/'work/r10-frozen'/frozen['build']/'runtime-tasks')
    binary=ROOT/'work/r10-frozen'/frozen['build']/'agent_server'
    with (output/'persistent-mpc.log').open('w') as log, Agent(task_id='Quadruped Flat',
            server_binary_path=str(binary),extra_flags=['--mjpc_workers=2'],
            subprocess_kwargs=dict(stdout=log,stderr=log),run_init=False) as agent:
        dump(output/'stream-identity.json',dict(group=a.group,seed=a.seed,frozen_controller=frozen,
            port=agent.port,device='PPO CPU, PANN GPU',normalization=stream.normalization,
            environment=dict(affinity=sorted(os.sched_getaffinity(0)),python=sys.version),
            ppo=stream.backend.provenance()))
        while stream.samples<stream.target:
            deadline=json.loads((small/'training-deadline.json').read_text())['deadline_unix_s']
            if time.time()>=deadline:break
            # Curriculum changes only here, at a natural episode/session boundary.
            stage=0 if stream.samples<8192 else 1 if stream.samples<16384 else 2
            session=60 if stage==0 else 300 if stage==1 else 600
            condition=['nominal','yaw_plus5','yaw_minus5'][stream.episode%3]
            stream.episode+=1
            name=f'train-{a.group}-s{a.seed}-ep{stream.episode:05d}'
            args=arguments(name,frozen['formulation'],condition,'training','single' if stage==0 else 'long',build=frozen['build'])
            c,r9=configuration(frozen['formulation'],frozen['build'])
            c.update(r10_training=True,r10_policy=True)
            if stage:
                c['long_mission_protocol'].update(duration_s=session,full_round_trips=10**12,
                    route_completion_deadline_s=session)
            joint=PolicyMaterial(ROOT,r9,stream,stage,session,agent=agent,
                full_recording=(stream.episode==1))
            joint.training_deadline_path=small/'training-deadline.json'
            result=run(args,c,joint)
            dump(output/f'episode-{stream.episode:05d}.json',dict(
                result_path=str(small/(name+'.json')),stage=stage,transitions=stream.samples,
                reason=result['reason'],simulation_s=result['simulation_s'],wall_s=result['wall_s'],
                completed_round_trips=result['completed_round_trips'],policy=result['policy']))
            if result['reason'] in ('campaign_budget_truncated','training_budget_truncated','execution_error','training_checkpoint_complete'):
                break
    stream.log.close();stream.learning.close()
    dump(output/'stopped.json',dict(actual_transitions=stream.samples,
        last_completed_checkpoint=stream.samples//256*256,unupdated_fragment=len(stream.buffer),
        actual_stage_samples=stream.stage_samples,physics_s=stream.physics_s,
        standard_target=32768,reason='target_reached' if stream.samples>=32768 else 'deadline_or_technical_stop'))


if __name__=='__main__':main()
