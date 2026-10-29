#!/usr/bin/env python3
"""Run the single R12 W1 material E30 using the existing r21 executor."""
import argparse
import copy
import json
import os
import time
from pathlib import Path

from run_s1_go2_r11 import configuration, arguments
from s1_go2_r11_protocol import ROOT, case_identity, stable_digest
from s1_go2_r11_runtime import run, index_file
from s1_go2_joint_material import JointMaterial

CONFIG = 'experiments/terrain_memory/s1-go2-local-readiness-r12.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--window-start-unix', type=float, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--assets', type=Path, default=ROOT/'work/assets')
    a = p.parse_args()
    contract = json.loads((ROOT/CONFIG).read_text())
    hard_deadline = a.window_start_unix + contract['execution']['maximum_window_wall_s']
    deadline = min(a.deadline_unix, hard_deadline)
    if time.time() >= deadline:
        raise ValueError('The declared R12 window has ended')
    c, r9 = configuration('S5-r21', 'E30', .215, 'tm02_anchor')
    c.update(run_id=contract['run_id'], stage='S1R12', r12_contract=contract,
        campaign_budget=contract['execution'], predecessor_attempt=None,
        window=dict(task='W1', started_unix_s=a.window_start_unix,
            run_deadline_unix_s=deadline, hard_deadline_unix_s=hard_deadline))
    c['outputs']['root'] = contract['outputs']['compact_root']
    args = arguments(a.attempt, 'S5-r21', 'nominal', 'development',
        c['r11_stage']['task_kind'], workers=a.workers)
    args.config = CONFIG
    args.cohort = contract['protocol_revision']
    args.original_robot_dir = str(a.assets/'go2')
    args.original_scene = str(a.assets/'scene_flat.xml')
    args.original_resolved = str(a.assets/'rigid-T1R-resolved.json')
    # Python MuJoCo is already loaded from kaiwen-learn; the child Agent uses its frozen library.
    os.environ['LD_LIBRARY_PATH'] = str(ROOT/'work/r11-frozen/S5-r21/lib')
    os.environ['MJPC_TASKS_DIR'] = str(ROOT/'work/r11-frozen/S5-r21/runtime-tasks')
    joint = JointMaterial(ROOT, r9, .215, terrain='tm02_anchor')
    joint.group = 'FIXED_S5'
    # Window bookkeeping must not change the controller's qualification case.
    case_config = copy.deepcopy(c)
    case_config.pop('window')
    case_config.pop('r12_contract')
    identity, _ = case_identity(case_config, joint.group, None, 'nominal')
    identity.update(protocol_revision=contract['formal_evaluation_protocol'], terrain='tm02_anchor')
    c.update(case_identity=identity, case_key=stable_digest(identity), evaluation_workers=a.workers)
    small = ROOT/c['outputs']['root']
    try:
        result = run(args, c, joint)
    except Exception as error:
        with index_file(small, c) as idx:
            row = next((x for x in idx['attempts'] if x['attempt']==a.attempt), None)
            if row is not None:
                row.update(status='initialization_failed' if not row['physical_started'] else 'interrupted',
                    stage_status='technical_interruption', error=repr(error), success=False)
        raise
    with index_file(small, c) as idx:
        idx.update(protocol_revision=contract['protocol_revision'], current_task='W1',
            status='W1_partial' if result['stage_status'] in ('technical_interruption', 'budget_truncated') else 'W1_complete',
            window=c['window'])
        idx.setdefault('training_transitions', 0)
        idx.setdefault('training_runs', 0)
    return result


if __name__ == '__main__':
    main()
