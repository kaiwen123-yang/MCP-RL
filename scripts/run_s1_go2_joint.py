#!/usr/bin/env python3
"""R9 complete material missions using the unchanged R8 S3 MPC binary."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_s1_go2_wb_baseline import run,render,dump,sha
from s1_go2_joint_material import JointMaterial

def configuration():
    contract=json.loads((ROOT/'experiments/terrain_memory/s1-go2-joint.json').read_text())
    c=json.loads((ROOT/contract['frozen_baseline']['configuration']).read_text())
    c['run_id']=contract['run_id'];c['outputs']['root']=contract['outputs']['root']
    c['campaign_budget']=dict(max_new_physical_starts=32,max_project_development_starts=8,
        wall_clock_hard_cap_h=48,reserve_evaluation_h_at_least=6)
    c['r9_contract']=contract
    return c,contract

def arguments(name,cap,condition='nominal',purpose='development',mission='single'):
    old=ROOT.parent/'terrain-memory-s1-execution'
    return SimpleNamespace(config='experiments/terrain_memory/s1-go2-joint.json',stage='G1',
        purpose=purpose,condition=condition,arm='S',mission=mission,cohort='r9-material-v1',
        attempt=name,workers=2,author_source=str(ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2'),
        agent_server=str(ROOT/'work/r8-s3-agent_server'),
        original_robot_dir=str(old/'work/envs/pympc/lib/python3.11/site-packages/gym_quadruped/robot_model/go2'),
        original_scene=str(old/'work/envs/pympc/lib/python3.11/site-packages/gym_quadruped/utils/mujoco/assets/scene_flat.xml'),
        original_resolved=str(old/'results/terrain_memory/simulation/tm-s1-contact-execution-20261026/development-v1-seed4/rigid-T1R-resolved.json'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',required=True);p.add_argument('--cap',type=float,default=.25)
    p.add_argument('--profile',default='constant',choices=['constant','switch_2s','switch_05s','sinusoid_4s'])
    p.add_argument('--condition',default='nominal');p.add_argument('--purpose',default='development')
    p.add_argument('--mission',default='single');p.add_argument('--render',action='store_true')
    a=p.parse_args();c,contract=configuration()
    args=arguments(a.attempt,a.cap,a.condition,a.purpose,a.mission)
    os.environ['MJPC_TASKS_DIR']=str(ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2/build/mjpc/tasks')
    if a.render:
        render(args,c)
        return
    joint=JointMaterial(ROOT,contract,a.cap,a.profile)
    run(args,c,joint)

if __name__=='__main__':main()

