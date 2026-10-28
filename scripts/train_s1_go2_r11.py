#!/usr/bin/env python3
"""An independent R11 PPO stream: evidence-based continuous-session curriculum."""
import argparse,json,os,sys,time
from pathlib import Path
from run_s1_go2_r11 import configuration,arguments
from s1_go2_r11_protocol import ROOT,CONTRACT,CONDITIONS
from s1_go2_r11_runtime import run,dump,index_file
from s1_go2_r11_policy import PolicyStream,PolicyMaterial
from mujoco_mpc.agent import Agent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',required=True,choices=['PPO_RAW_S5','PANN_PPO_S5'])
    p.add_argument('--seed',required=True,type=int,choices=[0,1,2])
    p.add_argument('--resume');a=p.parse_args()
    small=ROOT/CONTRACT['outputs']['compact_root'];index=json.loads((small/'index.json').read_text())
    frozen=index['frozen_controller']
    output=ROOT/CONTRACT['outputs']['large_root']/'training'/f'{a.group}-seed{a.seed}'
    stream=PolicyStream(ROOT,a.group,a.seed,CONTRACT,output,checkpoint=a.resume)
    binary=ROOT/'work/r11-frozen'/frozen['build']/'agent_server'
    os.environ['MJPC_TASKS_DIR']=str(binary.parent/'runtime-tasks')
    with (output/'persistent-mpc.log').open('a') as log, Agent(task_id='Quadruped Flat',
            server_binary_path=str(binary),extra_flags=['--mjpc_workers=2'],
            subprocess_kwargs=dict(stdout=log,stderr=log),run_init=False) as agent:
        dump(output/'stream-identity.json',dict(group=a.group,seed=a.seed,frozen_controller=frozen,
            normalization=stream.normalization,ppo=stream.backend.provenance(),
            protocol_revision=CONTRACT['protocol_revision'],initial_condition_rotation=CONDITIONS,
            resume_checkpoint=a.resume,continuous_environment_restored=False))
        while stream.samples<stream.target:
            deadline=json.loads((small/'training-deadline.json').read_text())['deadline_unix_s']
            if time.time()>=deadline:break
            # Promotion happens here only, after a complete preceding natural session.
            stage=stream.curriculum_stage;session=[60,300,600][stage]
            condition=CONDITIONS[stream.episode%3];stream.episode+=1
            name=f'train-{a.group}-s{a.seed}-ep{stream.episode:05d}'
            c,r9=configuration(frozen['build'],'E60',frozen['cap'],training_duration=session)
            c.update(r11_training=True,r11_policy=True,case_key=None,case_identity=None)
            args=arguments(name,frozen['build'],condition,'training','continuing_acquisition')
            joint=PolicyMaterial(ROOT,r9,stream,stage,session,agent=agent,full_recording=(stream.episode==1))
            joint.training_deadline_path=small/'training-deadline.json'
            before=stream.samples;result=run(args,c,joint)
            whole=result['simulation_s']>=session-1e-7 and result['stage_status']=='acquisition_boundary'
            circuits=result['completed_round_trips']
            promoted=whole and stage<2 and circuits>=[1,5][stage]
            record=dict(episode=stream.episode,attempt=name,condition=condition,stage=stage,
                session_limit_s=session,transitions=stream.samples-before,reason=result['reason'],
                stage_status=result['stage_status'],simulation_s=result['simulation_s'],wall_s=result['wall_s'],
                completed_round_trips=circuits,complete_session=whole,promoted=bool(promoted),
                no_promotion_reason=None if promoted else 'final_curriculum_stage' if stage==2 else
                    'incomplete_session' if not whole else 'insufficient_complete_circuits')
            if promoted:
                stream.curriculum_stage+=1
                stream.promotions.append(dict(from_stage=stage,to_stage=stream.curriculum_stage,evidence=record,
                    effective_at='next_natural_independent_session'))
            stream.session_records.append(record);dump(output/f'episode-{stream.episode:05d}.json',record)
            if result['stage_status'] in ('technical_interruption','budget_truncated'):break
    stream.log.close();stream.learning.close()
    dump(output/'stopped.json',dict(actual_transitions=stream.samples,
        last_completed_checkpoint=stream.samples//256*256,unupdated_fragment=len(stream.buffer),
        actual_stage_samples=stream.stage_samples,physics_s=stream.physics_s,session_records=stream.session_records,
        promotions=stream.promotions,curriculum_stage=stream.curriculum_stage,
        standard_target=32768,reason='target_reached' if stream.samples>=32768 else 'deadline_or_technical_stop'))
if __name__=='__main__':main()
