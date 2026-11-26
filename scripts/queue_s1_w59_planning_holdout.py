"""Release the independently frozen planning controls after learning-budget selection."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time

from run_s1_w59_research import ROOT, SMALL
from publish_s1_go2_w54 import write

WORK=ROOT/'work/w59'


def read(p):
    return json.loads(p.read_text())


def main():
    freeze=SMALL/'learning/pre-heldout-frozen.json'
    decision=SMALL/'learning/dispatch.json'
    action=SMALL/'action/stage.json'
    while not freeze.exists():
        # Planning remains evaluable if no learned head earns independent evaluation.
        if (decision.is_file() and read(decision).get('state') in ('blocked','terminal')
            and action.is_file() and read(action).get('state') in ('sealed_scored','blocked_independent_sources')):
            freeze=decision
            break
        time.sleep(30)
    named=[('B3','M1'),('Fixed-action','M1')]
    recipe=SMALL/'action/p3-development-recipe.json'
    if recipe.is_file() and read(recipe).get('ranking_development_eligible'):
        named += [('P3-full','M1'),('P3-private','M1')]
    def execute(item):
        arm,material=item
        dev=SMALL/'mechanism'/f'{arm.lower()}-{material.lower()}-development.json'
        while not dev.is_file():
            prior=[read(f) for f in (SMALL/'attempts').glob('*.json')]
            failures=[x for x in prior if x.get('arm')==arm and x.get('task_mode')=='original'
                      and x.get('w59',{}).get('material')==material and not x.get('w59',{}).get('learner_config')
                      and x.get('ended_unix_s') and x['status'] not in ('passed','technical_interruption')]
            if failures:
                return dict(arm=arm,status='original_qualification_failed',denominator=4)
            time.sleep(30)
        build=read(dev)['build']
        which='runtime-research.txt' if arm.startswith('P3') else 'runtime-geometry.txt'
        rt=Path((WORK/which).read_text().strip())
        cmd=[sys.executable,str(ROOT/'scripts/queue_s1_w59_mechanisms.py'),'--runtime',str(rt),
             '--arm',arm,'--material',material,'--split','heldout','--build',build,'--freeze',str(freeze)]
        if arm.startswith('P3'):
            cmd+=['--wm-recipe',str(recipe)]
            for model in read(recipe)['model_paths']:
                cmd+=['--wm-checkpoint',model]
        subprocess.run(cmd,check=True,cwd=ROOT)
        return dict(arm=arm,status='see_own_heldout_receipt',denominator=4)
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(execute,named))
    write(SMALL/'mechanism/planning-holdout.json',dict(arms=results,contract_planning_denominator=16,
          material='M1',extra_M3_generalization_claim=False,decision=str(freeze)))


if __name__=='__main__':
    main()
