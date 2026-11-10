#!/usr/bin/env python3
"""Re-solve retained W42 inputs with r32; no new plant trajectory or training."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
import mujoco
from mujoco_mpc.agent import Agent
from s1_go2_r29_records import R29Records
ROOT=Path(__file__).resolve().parents[1]
REL=Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
OUT=ROOT/REL/'w43-optimizer-inputs'/'saved-input-v2'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w43-optimizer-inputs'
def main():
    OUT.mkdir(parents=True,exist_ok=False);SMALL.mkdir(parents=True,exist_ok=True)
    report=dict(controller='S5-r32',based_on='S5-r31',started_unix_s=time.time(),
      scope='W42 saved real t0 startup20, last Stand2.0, first outbound2.002, and4s inputs. Own r32 startup warm context retained through sparse input sequence; intervening online optimizations and historical K/private gait state not restored. Not equivalent replay of old full controller history.',
      new_plant_episodes=0,streams={})
    for suffix,seed,stamp in [('a',1,'20261108T154719Z'),('b',2,'20261108T154727Z')]:
        name=f'job-{stamp}-w42-{suffix}-eval-PANN_PPO_S5-s{seed}-E30'
        source=ROOT/REL/name;dest=OUT/suffix;dest.mkdir()
        resolved=json.loads((ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/f'{name}-resolved.json').read_text())
        saved=np.load(source/'trajectory.npz')
        # W42 has no separate smoothing XML; runtime loads this retained task XML.
        model=mujoco.MjModel.from_xml_path(str(source/'model/G1-project-go2.xml'))
        env=dict(os.environ,MJPC_TASKS_DIR=str(ROOT/'work/r11-frozen/S5-r32/runtime-tasks'),
            LD_LIBRARY_PATH=str(ROOT/'work/r11-frozen/S5-r32/lib'),
            MJPC_R29_SNAPSHOT=str(dest/'r29-current-plan.bin'),
            MJPC_R29_CALL_RECORD=str(dest/'planner-calls.jsonl'),
            MJPC_R27_PLAN_RECORD=str(dest/'planner-records.jsonl'),
            MJPC_R27_ACTION_RECORD=str(dest/'feedback-actions.jsonl'))
        record=dict(source=str(source.relative_to(ROOT)),inputs=[],status='running')
        report['streams'][suffix]=record
        records=R29Records(dest,model);records.controller='S5-r32'
        try:
            with (dest/'agent-server.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
                 server_binary_path=ROOT/'work/r11-frozen/S5-r32/agent_server',
                 extra_flags=['--mjpc_workers=10'],subprocess_kwargs=dict(stdout=log,stderr=log,env=env),
                 run_init=False) as agent:
                agent.init(task_id='Quadruped Flat',model=model,send_as='xml')
                agent.set_mode('Quadruped');agent.set_task_parameters({'Gait switch':'Manual'})
                for target,count in [(0.,20),(2.,1),(2.002,2),(4.,2)]:
                    idx=int(np.abs(saved['time']-target).argmin());t=float(saved['time'][idx])
                    settings=[x for x in resolved['applied_settings'] if x['time_s']<=t+1e-10][-1]
                    params=dict(time=t,qpos=saved['qpos'][idx],qvel=saved['qvel'][idx],act=[],
                        mocap_pos=saved['mocap_pos'][idx],mocap_quat=saved['mocap_quat'][idx],
                        userdata=saved['support_reference_userdata'][idx])
                    np.savez_compressed(dest/f'input-{target:.3f}.npz',**params,
                        source_index=idx,held_target=saved['q_target'][idx])
                    agent.set_task_parameter('Gait',settings['task_parameters']['Gait'])
                    agent.set_state(**params)
                    agent.set_task_parameters(settings['task_parameters'])
                    agent.set_cost_weights(settings['cost_weights'])
                    agent.set_state(**params)
                    item=dict(time=t,source_index=idx,repeat=count,settings=settings);record['inputs'].append(item)
                    for i in range(count):
                        agent.planner_step()
                        records.plan(t);records.event(t,'saved_input',source_index=idx,repeat_index=i)
                        action=agent.get_action(time=t)
                        item.setdefault('actions',[]).append(np.asarray(action).tolist())
                        print(json.dumps(dict(stream=suffix,input_time=t,repeat=i+1)),flush=True)
            record['status']='completed_saved_input_sequence'
        except Exception as error:
            record.update(status='technical_failure',error=repr(error),traceback=traceback.format_exc())
        finally:
            records.finish(float(saved['time'][idx]),record['status'])
            calls=[json.loads(x) for x in (dest/'planner-calls.jsonl').read_text().splitlines()]
            record['calls']=calls
            initial=[x for x in calls if x['origin']==0]
            stopped=[x for x in initial if x['epoch_stopped']]
            record['observations']=dict(
                startup_epochs=sorted({x['problem_epoch'] for x in initial}),
                first_stopped_call=stopped[0]['call_id'] if stopped else None,
                attempts_after_first_stop=[x['backward_attempts'] for x in initial
                    if stopped and x['call_id']>stopped[0]['call_id']],
                new_epoch_first_used=[dict(time=x['origin'],epoch=x['problem_epoch'],
                    reason=x['epoch_reset_reason'],attempt=x['search_attempts'][0] if x['search_attempts'] else None)
                    for x in calls if x['epoch_reset_reason']!='same_problem'],
                accepted_calls=sum(x['deployment_status']=='accepted_native_descent' for x in calls),
                reused_calls=sum(x['deployment_status']=='reused_current_native_scored_incumbent' for x in calls),
                runtime_wall_s=sum(x['wall_s'] for x in calls))
            (SMALL/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    report['finished_unix_s']=time.time()
    (SMALL/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:dict(status=v['status'],observations=v['observations']) for k,v in report['streams'].items()},indent=2))
if __name__=='__main__':main()
