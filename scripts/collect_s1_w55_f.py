"""Collect completed F stages from their recorded originals on the execution host."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from analyze_s1_w55_f import ROOT, SMALL, analyze


def collect_stage(case_dir, level):
    status = case_dir / (level + '.json')
    while not status.exists():
        time.sleep(10)
    while True:
        state = json.loads(status.read_text())
        if state.get('video'):
            break
        if state.get('status') in ('declared', 'running'):
            time.sleep(10)
            continue
        if state.get('reason') == 'process_exited_without_completed_result':
            raise RuntimeError('No physical result or video for ' + str(status))
        pane = subprocess.run(['tmux','display-message','-p','-t',state['tmux'],'#{pane_dead}'],
                              text=True, capture_output=True)
        if pane.returncode or pane.stdout.strip() == '1':
            state = json.loads(status.read_text())
            if not state.get('video'):
                raise RuntimeError('Original renderer ended without full video for ' + str(status))
            break
        time.sleep(10)
    name = state['attempt']
    summary = analyze(name)
    paths = [case_dir/'job.json',case_dir/'execution-contract.json',status,
             SMALL/(name+'.json'),SMALL/(name+'-resolved.json'),SMALL/(name+'-analysis.json'),
             SMALL/'index.json']
    rel = [str(p.relative_to(ROOT)) for p in paths]
    # Existing F index lock also serializes commits from two naturally finishing cases.
    with (SMALL/'index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(['git','add','--',*rel], cwd=ROOT, check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',*rel], cwd=ROOT).returncode == 1:
            subprocess.run(['git','commit','--only','-m',f'Record W55 {case_dir.name} {level} original outcome','--',*rel],
                           cwd=ROOT, check=True)
    print(json.dumps(dict(attempt=name,level=level,status=summary['result'],
                          physical_s=summary['simulation_s'],video=state['video']['path'],
                          deadline=None,wall_deadline_enabled=False)),flush=True)
    return summary['passed']


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--case-dir',type=Path,required=True)
    p.add_argument('--levels',choices=('E30','E60','E300','E600'),nargs='+',required=True)
    args=p.parse_args()
    for step in args.levels:
        if not collect_stage(args.case_dir.resolve(),step):
            break
