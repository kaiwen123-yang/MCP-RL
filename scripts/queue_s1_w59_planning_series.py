"""Continue the P3 controls after the primary post-calibration original identity has been admitted."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from publish_s1_go2_w54 import write
from run_s1_w59_research import ROOT, SMALL
from s1_w59_lease import BG

WORK=ROOT/'work/w59'


def read(p):
    return json.loads(p.read_text())


def drive(rt,arm,material,model_paths,recipe):
    for level in ('E30','E60'):
        jobs=[]
        for c in 'AB':
            existing=[read(p) for p in (SMALL/'attempts').glob('*.json')]
            prior=[v for v in existing if v.get('w59',{}).get('arm')==arm and v.get('w59',{}).get('material')==material and
                   v.get('task_mode')=='original' and v.get('level')==level and v.get('case')==c and
                   not v.get('w59',{}).get('learner_config') and v.get('w59',{}).get('model_paths')==model_paths]
            parent=next((x for x in prior if x.get('status')=='passed'),None)
            if parent:
                jobs.append((parent['attempt'],None))
                continue
            name=prior[-1]['attempt'] if prior else f'w59-{arm.lower()}-{material.lower()}-{c.lower()}-original-{level.lower()}-a01'
            cmd=[sys.executable,str(ROOT/'scripts/s1_w59_lease.py'),'--runtime',str(rt),'--job',name,'--slots','1','--',
                 sys.executable,'-u','scripts/run_s1_w59_research.py','--case',c,'--arm',arm,'--material',material,
                 '--task','original','--level',level,'--attempt',name,'--wm-recipe',str(recipe)]
            for m in model_paths: cmd+=['--wm-checkpoint',m]
            jobs.append((name,None if prior else subprocess.Popen(cmd,cwd=rt)))
        for name,child in jobs:
            if child: child.wait()
            a=SMALL/'attempts'/(name+'.json')
            while a.is_file() and not read(a).get('ended_unix_s'):
                time.sleep(15)
            if not a.is_file() or read(a)['status']!='passed':
                return
            build=read(a)['controller']
    cmd=[sys.executable,str(ROOT/'scripts/queue_s1_w59_mechanisms.py'),'--runtime',str(rt),'--arm',arm,
         '--material',material,'--split','development','--build',build,'--workers','2','--wm-recipe',str(recipe)]
    for m in model_paths: cmd+=['--wm-checkpoint',m]
    subprocess.run(cmd,cwd=ROOT,check=True)


def main():
    from concurrent.futures import ThreadPoolExecutor
    with (WORK/'planning-series.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        recipe=SMALL/'action/p3-development-recipe.json'
        primary=None
        while primary is None:
            for f in (SMALL/'attempts').glob('*.json'):
                a=read(f);w=a.get('w59',{})
                if w.get('arm')=='P3-full' and w.get('material')=='M1' and not w.get('learner_config') and a.get('task_mode')=='original' and a.get('level')=='E30':
                    primary=a;break
            if primary is None or not recipe.is_file():
                final=SMALL/'action/stage.json'
                if final.exists() and read(final).get('state') in ('technical_block','independent_sources_unavailable','sealed_scored') and not primary:
                    write(SMALL/'mechanism/planning-series.json',dict(status='blocked_by_action_model',source=str(final),**BG))
                    return
                time.sleep(20)
                primary=None
        rt=Path((WORK/'runtime-research.txt').read_text().strip())
        models=primary['w59']['model_paths']
        def after_owned_qualification():
            while True:
                done=[read(p) for p in (SMALL/'attempts').glob('*.json')]
                stage=[r for r in done if r.get('controller')==primary['controller'] and r.get('task_mode')=='original']
                if any(r.get('ended_unix_s') and r['status']!='passed' for r in stage):
                    return
                if {r['case'] for r in stage if r['level']=='E60' and r['status']=='passed'}=={'A','B'}:
                    break
                time.sleep(20)
            cmd=[sys.executable,str(ROOT/'scripts/queue_s1_w59_mechanisms.py'),'--runtime',str(rt),'--arm','P3-full',
                '--material','M1','--split','development','--build',primary['controller'],'--wm-recipe',str(recipe)]
            for m in models:cmd+=['--wm-checkpoint',m]
            subprocess.run(cmd,cwd=ROOT,check=True)
        with ThreadPoolExecutor(3) as execs:
            futures=[execs.submit(after_owned_qualification),execs.submit(drive,rt,'P3-private','M1',models,recipe)]
            mat=SMALL/'model/recovery.json'
            if mat.exists() and read(mat).get('offline_candidate_accepted'):
                futures.append(execs.submit(drive,rt,'P3-full','M3',models,recipe))
            for f in futures:f.result()
        write(SMALL/'mechanism/planning-series.json',dict(status='eligible_development_finished',
          primary_material='M1',paired_material='M3',selection_inputs='paired original qualification, phase, fixed footprint and applied-action evidence',**BG))


if __name__=='__main__':
    main()
