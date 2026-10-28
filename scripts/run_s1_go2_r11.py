#!/usr/bin/env python3
"""Independently predeclare and execute one R11 stage with exact-case qualification."""
import argparse,copy,json,os,time
from pathlib import Path
from types import SimpleNamespace
from s1_go2_r11_protocol import ROOT,CONFIG,CONTRACT,RUN,stage,case_identity,prerequisite
from s1_go2_r11_runtime import run,render,index_file,dump
from s1_go2_joint_material import JointMaterial
def configuration(build,level,cap=.215,terrain='tm02_anchor',training_duration=None):
    starting=json.loads((ROOT/CONTRACT['configuration_reuse']['starting_resolved']).read_text())['r10_configuration']
    c=copy.deepcopy(starting)
    c.pop('r10_contract',None)
    frozen=ROOT/'work/r11-frozen'/build
    identity=json.loads((frozen/'identity.json').read_text())
    c.update(run_id=RUN,stage='S1R11',controller_build=build,implementation_status='S5_development',
        r11_contract=CONTRACT,actual_terrain=terrain,fixed_cap_m_s=cap,r11_stage=stage(level,training_duration))
    c['outputs']['root']=CONTRACT['outputs']['compact_root']
    c['campaign_budget']=CONTRACT['budget']
    c['task']['command_speed_m_s']=cap
    c['task']['maximum_episode_s']=c['r11_stage']['duration_s']
    if c['r11_stage']['task_kind'] in ('continuous_finite','continuing_acquisition'):
        c['long_mission_protocol'].update(duration_s=c['r11_stage']['duration_s'],
            full_round_trips=c['r11_stage']['target_circuits'] or 10**12,
            route_completion_deadline_s=c['r11_stage']['route_deadline_s'] or c['r11_stage']['duration_s'])
    if terrain=='rigid':c['local_contact_model'].update(enabled=False,causal_force_anchor=False)
    for key,value in identity.get('configuration_overrides',{}).items():c[key]=value
    return c,json.loads((ROOT/'experiments/terrain_memory/s1-go2-joint.json').read_text())
def arguments(attempt,build,condition,purpose,kind,workers=2):
    frozen=ROOT/'work/r11-frozen'/build;assets=ROOT/'work/assets'
    return SimpleNamespace(config=CONFIG,stage='G1',purpose=purpose,condition=condition,arm='S',
        mission='long' if kind in ('continuous_finite','continuing_acquisition') else 'single',
        cohort=CONTRACT['protocol_revision'],attempt=attempt,workers=workers,author_source=str(frozen),
        agent_server=str(frozen/'agent_server'),original_robot_dir=str(assets/'go2'),
        original_scene=str(assets/'scene_flat.xml'),original_resolved=str(assets/'rigid-T1R-resolved.json'))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',required=True);p.add_argument('--build',required=True)
    p.add_argument('--level',choices=['E30','E60','E300','E600'],required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cap',type=float,default=.215);p.add_argument('--condition',choices=['nominal','yaw_plus5','yaw_minus5'],default='nominal')
    p.add_argument('--purpose',choices=['development','formal','rigid_regression'],default='development')
    p.add_argument('--terrain',choices=['tm02_anchor','rigid'],default='tm02_anchor')
    p.add_argument('--group',choices=['FIXED_S5','PPO_RAW_S5','PANN_PPO_S5'],default='FIXED_S5')
    p.add_argument('--seed',type=int,choices=[0,1,2]);p.add_argument('--checkpoint');p.add_argument('--render',action='store_true')
    a=p.parse_args();c,r9=configuration(a.build,a.level,a.cap,a.terrain)
    args=arguments(a.attempt,a.build,a.condition,a.purpose,c['r11_stage']['task_kind'],workers=a.workers)
    os.environ['MJPC_TASKS_DIR']=str(ROOT/'work/r11-frozen'/a.build/'runtime-tasks')
    if a.render:render(args,c);return
    normalization=None
    if a.group=='FIXED_S5':
        joint=JointMaterial(ROOT,r9,a.cap,terrain=a.terrain);joint.group=a.group
    else:
        from s1_go2_r11_policy import PolicyStream,PolicyMaterial
        if a.checkpoint is None or a.seed is None:raise ValueError('Learned evaluation requires the frozen primary checkpoint and its seed')
        stream=PolicyStream(ROOT,a.group,a.seed,CONTRACT,ROOT/CONTRACT['outputs']['large_root']/'evaluation-policy'/a.attempt,
            training=False,checkpoint=a.checkpoint)
        normalization=stream.normalization;c['r11_policy']=True
        joint=PolicyMaterial(ROOT,r9,stream,0,c['r11_stage']['duration_s'],full_recording=True)
    identity,key=case_identity(c,a.group,a.seed,a.condition,a.checkpoint,normalization)
    identity['terrain']=a.terrain
    from s1_go2_r11_protocol import stable_digest
    key=stable_digest(identity)
    c.update(case_identity=identity,case_key=key)
    small=ROOT/c['outputs']['root']
    with index_file(small,c) as idx:
        if idx.get('campaign_deadline_unix_s',float('inf'))<=time.time():raise ValueError('R11 campaign wall clock expired')
        if a.purpose=='development' and time.time()>=idx.get('campaign_started_unix_s',float('inf'))+24*3600:raise ValueError('R11 development wall clock expired')
        if idx.get('frozen_controller') and a.purpose=='development':raise ValueError('S5 already frozen after same-version E60')
        elapsed=time.time()-idx.get('campaign_started_unix_s',time.time())
        reserve=idx.get('evaluation_reserve',{}).get('wall_s',1.2*990*59.8586081547833)
        if a.purpose=='development' and elapsed>=min(24*3600,48*3600-reserve):
            raise ValueError('Protected nominal ladder reserve ends current development window')
        c['predecessor_attempt']=prerequisite(c,idx,a.purpose,a.terrain)
        idx.setdefault('evaluation_reserve',dict(wall_s=reserve,
            protected_chains=['FIXED nominal E300/E600','RAW seed0 nominal E30/E60/E300/E600','PANN seed0 nominal E30/E60/E300/E600'],
            basis='Initial conditional critical-path estimate from R10 dev11, assuming at most three protected evaluations overlap. Re-estimate from actual R11 missions and sampling. Downstream costs retained until predecessor failure.'))
        idx.update(protocol_revision=CONTRACT['protocol_revision'],base_commit='337b0d17dc4a96f70acce69774e5b610cd685fda',
            training_transitions=idx.get('training_transitions',0),training_runs=idx.get('training_runs',0))
    try:
        result=run(args,c,joint)
    except Exception as error:
        with index_file(small,c) as idx:
            row=next((x for x in idx['attempts'] if x['attempt']==a.attempt),None)
            if row is not None:
                row.update(status='initialization_failed' if not row['physical_started'] else 'interrupted',
                    stage_status='technical_interruption',error=repr(error),success=False)
        raise
    if a.level=='E60' and a.condition=='nominal' and a.group=='FIXED_S5' and a.terrain=='tm02_anchor' and result['success']:
        with index_file(small,c) as idx:
            if not idx.get('frozen_controller'):
                idx['frozen_controller']=dict(build=a.build,cap=a.cap,case_key=key,
                    E30=c['predecessor_attempt'],E60=a.attempt,case_identity=identity,
                    frozen_unix_s=time.time(),workers=a.workers,controller_identity=json.loads((ROOT/'work/r11-frozen'/a.build/'identity.json').read_text()))
    if a.group!='FIXED_S5':stream.log.close();stream.learning.close()
if __name__=='__main__':main()
