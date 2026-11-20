"""Finalize this exact N/A r56 attempt on its own D57 Git branch."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from publish_s1_go2_w54 import ROOT,LARGE,write
from run_s1_w56_n import SMALL,RESULTS

CASE=SMALL/'N-A-r56'
STAGE=CASE/'E30.json'
RECEIPT=CASE/'E30-collected.json'
MARKER=ROOT/'work/w55-n-r56/E30-compact-commit.json'

def read_stage(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def main():
    while not (stage:=read_stage(STAGE)) or not stage.get('attempt'):
        time.sleep(3)
    attempt=stage['attempt'];job=stage['job'];session=stage['tmux']
    if not attempt.startswith(job+'-eval-PANN_PPO_S5-s1-E30-') or session!='s1-w55-n-a-r56-e30':
        raise ValueError('The requested N/A r56 E30 attempt identity differs')
    while True:
        probe=subprocess.run(['tmux','display-message','-p','-t','='+session,'#{pane_dead}'],capture_output=True,text=True)
        stage=read_stage(STAGE)
        if stage and (probe.stdout.strip()=='1' or not Path('/proc',str(stage.get('runner_pid','pending'))).exists()):
            if stage.get('runner_pid'):break
        time.sleep(3)
    # Physical release is observable before video/asset processing. The
    # coordinator can take the slot as soon as this real child exit is known.
    if not (ROOT/'work/w55-n-r56/slot-released.json').exists() and stage.get('ended_unix_s') and stage.get('exit_code') is not None:
        write(ROOT/'work/w55-n-r56/slot-released.json',dict(status='naturally_ended',target='N-A-r56/E30',
            released_unix_s=time.time(),job=job,attempt=attempt,child_pid=stage.get('pid'),
            child_exit_code=stage['exit_code'],physical_started=bool(stage.get('physical_samples')),
            stage_file=str(STAGE),reason=stage.get('reason'),deadline=None,wall_deadline_enabled=False))
    out=LARGE/attempt;jobdir=LARGE/'jobs'/job
    result_path=RESULTS/(attempt+'.json');resolved=RESULTS/(attempt+'-resolved.json')
    result=read_stage(result_path)
    log=jobdir/(attempt+'.log');lines=log.read_text(errors='replace').splitlines()[-22:] if log.exists() else []
    ledger=read_stage(RESULTS/'index.json') or {}
    index_row=next((r for r in ledger.get('attempts',[]) if r.get('attempt')==attempt),{})
    video=out/'video-asset.json';analysis=CASE/'E30-actual-analysis.json'
    if result is not None and result.get('physical_samples',0)>0 and (out/'trajectory.npz').exists():
        if not video.exists():
            import render_s1_go2_r12 as renderer
            renderer.SMALL=RESULTS
            renderer.render(attempt)
        if not analysis.exists():
            from analyze_s1_w55_n import measure
            measure(CASE,'E30',RESULTS)
    manifest=read_stage(video);mp4=None
    if manifest:
        mp4=next(out.glob('full-mission*.mp4'))
        with mp4.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if manifest.get('sha256') and digest!=manifest['sha256']:raise ValueError('Rendered video digest differs')
        manifest.update(original_path=str(mp4),actual_size=mp4.stat().st_size,actual_sha256=digest)
    originals=[p for p in (out,jobdir,LARGE/'evaluation-policy'/attempt) if p.exists()]
    receipt=dict(case='N-A-r56',level='E30',attempt=attempt,runner_tmux=session,
        final_stage=stage.get('status'),reason=result.get('reason') if result else stage.get('reason'),
        exception=result.get('error') if result else index_row.get('error'),physical_steps=result.get('physical_samples') if result else None,
        physical_started=index_row.get('physical_started'),stage_process_exit_code=stage.get('exit_code'),
        branch='codex/w55-n-r56',observed_unix_s=time.time(),deadline=None,wall_deadline_enabled=False,
        compact_stage=str(STAGE.relative_to(ROOT)),original_result=str(result_path.relative_to(ROOT)) if result else None,
        analysis=str(analysis.relative_to(ROOT)) if analysis.exists() else None,
        full_video=manifest,raw_roots=[str(p) for p in originals],execution_log=str(log),
        terminal_log_tail=lines if result is None or stage.get('status')=='technical_interruption' else None,
        new_PPO_samples=0,new_PPO_updates=0,donor=0)
    if result is None and index_row.get('physical_started') is False:receipt['physical_steps']=0
    write(RECEIPT,receipt)
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='codex/w55-n-r56':
        raise ValueError('Collector is not on the N lane')
    compact=[CASE,SMALL/'queue-r56.json',RESULTS/'index.json',result_path,resolved]
    compact=[str(p.relative_to(ROOT)) for p in compact if p.exists()]
    subprocess.run(['git','add','--',*compact],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet','--',*compact],cwd=ROOT).returncode:
        subprocess.run(['git','commit','--only','-m','Record W55 N/A r56 original E30 outcome and full evidence pointers','--',*compact],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(MARKER,dict(commit=head,attempt=attempt,branch='codex/w55-n-r56',
        receipt=str(RECEIPT.relative_to(ROOT)),raw_roots=[str(p) for p in originals],
        finished_unix_s=time.time(),deadline=None,wall_deadline_enabled=False))
    print(head,receipt['final_stage'],attempt,flush=True)
if __name__=='__main__':main()
