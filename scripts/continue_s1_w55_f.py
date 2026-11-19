"""Sequential original stages for one scientifically selected, frozen W55 F kernel."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_s1_w55_f import SMALL, launch

LEVELS = ('E30','E60','E300','E600')


def remaining(case, build, selection):
    evidence = json.loads(selection.read_text())
    if evidence['build'] != build or evidence['normal_sustained_motion'] is not True:
        raise ValueError('F selection does not identify a reviewed normally moving kernel')
    original = SMALL / f'F-A-{build.removeprefix("S5-")}' / 'E30.json'
    reviewed = json.loads(original.read_text())
    if reviewed['status'] != 'passed' or not reviewed.get('video') or reviewed['attempt'] != evidence['attempt']:
        raise ValueError('Selection must match passed A E30 with its full recorded video')
    directory = SMALL / f'F-{case}-{build.removeprefix("S5-")}'
    first = 1 if case == 'A' else 0
    for level in LEVELS[first:]:
        status = directory / (level+'.json')
        if not status.exists():
            launch(build,case,level)
        row = json.loads(status.read_text())
        while row['status'] in ('declared','running'):
            if subprocess.run(['tmux','has-session','-t','='+row['tmux']],capture_output=True).returncode:
                raise RuntimeError(f'Exact stage tmux session missing: {row["tmux"]}')
            dead = subprocess.check_output(['tmux','display-message','-p','-t',row['tmux'],'#{pane_dead}'],text=True).strip()
            if dead == '1':
                row = json.loads(status.read_text())
                if row['status'] in ('declared','running'):
                    raise RuntimeError(f'Stage process ended before final state: {row["tmux"]}')
                break
            time.sleep(5)
            row = json.loads(status.read_text())
        print(json.dumps(dict(case=case,level=level,status=row['status'],result=row.get('reason'))),flush=True)
        if row['status'] != 'passed':
            break
    return 0


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--case',required=True,choices=('A','B'))
    p.add_argument('--build',required=True)
    p.add_argument('--selection',required=True,type=Path)
    options=p.parse_args()
    sys.exit(remaining(options.case,options.build,options.selection))
