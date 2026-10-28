#!/usr/bin/env python3
"""R10 complete missions on the declared S4 development formulation."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from run_s1_go2_wb_baseline import run, render, dump
from s1_go2_joint_material import JointMaterial

ROOT = Path(__file__).resolve().parents[1]


def configuration(form):
    r10=json.loads((ROOT/'experiments/terrain_memory/s1-go2-soft-landing-r10.json').read_text())
    r9=json.loads((ROOT/'experiments/terrain_memory/s1-go2-joint.json').read_text())
    c=json.loads((ROOT/r10['configuration_reuse']['baseline']).read_text())
    c.update(run_id=r10['run_id'],stage='S1R10',r10_contract=r10,
        implementation_status='S4_development_'+form,controller_formulation=form)
    c['outputs']['root']=r10['outputs']['compact_root']
    c['campaign_budget']=dict(max_new_physical_starts=61,max_project_development_starts=12,
        wall_clock_hard_cap_h=48,reserve_evaluation_h_at_least=6)
    if form == 'GN':
        c['normal_landing']=dict(height_profile='symmetric quintic, original swing phase',
            approach_band_m=.005,gate_width_m=.002,velocity_scale_m_s=.05,
            velocity_target='-.05*tanh(max(bottom_z,0)/.001)',velocity_weight=.02,
            indentation_target_m=.00019889981664,indentation_scale_m=.001,
            indentation_weight=.1,positive_part_smoothing_m=.000001,
            force='Original native MPC prediction; real TM02 unchanged')
    return c,r9


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',required=True)
    p.add_argument('--form',choices=['G','GN'],required=True)
    p.add_argument('--cap',type=float,default=.25)
    p.add_argument('--condition',default='nominal')
    p.add_argument('--purpose',default='development')
    p.add_argument('--render',action='store_true')
    a=p.parse_args();c,r9=configuration(a.form)
    frozen=ROOT/'work/r10-frozen'/a.form
    assets=ROOT/'work/assets'
    args=SimpleNamespace(config='experiments/terrain_memory/s1-go2-soft-landing-r10.json',
        stage='G1',purpose=a.purpose,condition=a.condition,arm='S',mission='single',
        cohort='parallel_long_training_v2',attempt=a.attempt,workers=2,
        author_source=str(frozen),agent_server=str(frozen/'agent_server'),
        original_robot_dir=str(assets/'go2'),original_scene=str(assets/'scene_flat.xml'),
        original_resolved=str(assets/'rigid-T1R-resolved.json'))
    os.environ['MJPC_TASKS_DIR']=str(frozen/'runtime-tasks')
    if a.render:
        render(args,c)
        return
    joint=JointMaterial(ROOT,r9,a.cap)
    joint.group='S4_DEVELOPMENT_'+a.form
    result=run(args,c,joint)
    result['controller_formulation']=a.form
    result['controller_identity']=json.loads((frozen/'identity.json').read_text())
    dump(ROOT/c['outputs']['root']/(a.attempt+'.json'),result)


if __name__=='__main__':
    main()
