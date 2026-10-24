#!/usr/bin/env python3
"""Execute complete, state-terminated Go2 missions for the fixed S1 study."""
import argparse
import os
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['MUJOCO_GL'] = 'egl'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/terrain_memory/s1-go2-contact.json')
    parser.add_argument('--terrain', choices=['rigid', 'tm02_anchor', 'kv_soft', 'kv_moderate'])
    parser.add_argument('--controller', choices=['C0', 'C1', 'C2', 'C3', 'C4'])
    parser.add_argument('--matrix', action='store_true', help='All 80 primary episodes, then the five frozen-B episodes.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--cohort', default='development-v1')
    parser.add_argument('--source-output-root', default='/home/fanyuan/projects/kaiwen-learning/terrain-memory-study/outputs/terrain_memory/reproduction/material/tm02-memory-study-20261023')
    args = parser.parse_args()
    if args.matrix:
        cases = [(terrain, method) for terrain in ('rigid','tm02_anchor','kv_soft','kv_moderate')
                 for method in ('C0','C1','C2','C3')] + [('tm02_anchor','C4')]
        for terrain, method in cases:
            subprocess.run([sys.executable, str(Path(__file__)), '--config', args.config,
                '--terrain',terrain,'--controller',method,'--seeds','0','1','2','3','4',
                '--cohort',args.cohort,'--source-output-root',args.source_output_root],check=True)
        return
    if args.terrain is None or args.controller is None:
        parser.error('Specify --matrix, or both --terrain and --controller.')
    from tm02.s1_robot import run
    run(ROOT, args)


if __name__ == '__main__':
    main()
