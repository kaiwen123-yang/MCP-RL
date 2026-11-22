"""Finish W57 frozen evaluations and earned original qualifications, using the same two physical leases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from publish_s1_go2_w54 import write
from summarize_s1_w57 import ROOT, R, RAW, read

CAMPAIGN=ROOT/'work/w57-r/pilot02'
SNAPSHOTS={'B':('MCP-RL-w57-m1-b','5209716d8185b37c7437f9c43a6702b2961ed6c0'),
           'P':('MCP-RL-w57-p1-final','c4b80ec573e8fa01e6544d8843ef384aa78dcb98'),
           'R':('MCP-RL-w57-pilot02','99b2209967195f1061ccf59de8faef45d4ac69bf')}
BACKGROUND=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
WORK=ROOT/'work/w57'


def source_for(arm):
    key='P' if arm=='PR' else 'R' if arm=='blind' else arm
    folder,rev=SNAPSHOTS[key]
    src=ROOT.parent/folder
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()
    if actual!=rev:raise ValueError(f'{arm} snapshot differs from the declared source')
    return src


def run_episode(arm,case,mode,level='E60'):
    snapshot=source_for(arm)
    slug=f'w57-m1-{arm.lower()}-{case.lower()}-{mode}'
    if mode=='original':slug+='-'+level.lower()
    attempt=slug+'-a01'
    state=R/'attempts'/(attempt+'.json')
    if state.exists():
        while 'ended_unix_s' not in (row:=read(state)):
            pid=row.get('runner_pid')
            if not pid or not (Path('/proc')/str(pid)).exists():
                raise RuntimeError(f'{attempt} has an unfinished receipt without its recorded runner {pid}')
            time.sleep(10)
        return row
    if arm in ('R','PR','blind'):
        head='blind' if arm=='blind' else 'full'
        timing=['--timing-config',str(CAMPAIGN/head/'frozen-evaluation.json')]
    else:timing=[]
    cmd=[sys.executable,'-u','scripts/s1_w56_slot.py','--priority','F','--label',attempt,'--',
         sys.executable,'-u','scripts/run_s1_w57.py','episode','--case',case,'--arm',arm,'--task',mode,
         '--material','M1','--level',level,'--attempt',attempt,*timing]
    env=dict(os.environ,PYTHONPATH=str(snapshot/'src')+':'+str(snapshot/'work/upstreams/mujoco_mpc/python'),
             NUMEXPR_NUM_THREADS='1',W57_TMUX=os.environ.get('W57_TMUX','w57-frozen-evaluations'))
    with (WORK/(attempt+'.log')).open('a') as log:
        done=subprocess.run(cmd,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not state.exists():raise RuntimeError(f'{attempt} exited {done.returncode} before creating its attempt receipt')
    row=read(state)
    if done.returncode or row['status']=='technical_interruption':
        print(json.dumps(dict(attempt=attempt,stage=row['status'],returncode=done.returncode,
                              reason=row.get('reason'))),flush=True)
    return row


def campaign():
    target=R/'dispatch/frozen-evaluations.json'
    declaration=dict(command=sys.argv,stage='waiting_for_natural_final_R_node',common_observer='M1',
        source_snapshots=SNAPSHOTS,full_and_blind_campaign=str(CAMPAIGN),mechanism_cycle_target=12,
        attempted_conditions=['short','recovered'],primary_arms=['B','P','R','PR'],cases=['A','B'],
        original='Every genuinely new physical controller gets its own A/B E30 followed by E60 only after own E30 passes. Training and mechanism trials have no original qualification.',
        final_candidate='Frozen PR, independently by case, after own E60 passes, both mechanism durations complete and at least one observed target cross-foot cycle, an actually selected material candidate and a witnessed timing change (handover or transfer ramp). These are execution information and task eligibility, not evidence of superior average performance. Continue that case E300 and then E600 only after its own E300 passes.',
        **BACKGROUND)
    write(target,declaration)
    while not (CAMPAIGN/'final.json').exists():time.sleep(20)
    final=read(CAMPAIGN/'final.json')
    arms=['B','P','R','PR']+(['blind'] if final['blind'] else [])
    declaration.update(stage='frozen_evaluations_running',full_distinct=final['full'],
                       blind_distinct=final['blind'],frozen_checkpoints={k:v for k,v in final.items() if 'checkpoint' in k})
    write(target,declaration)
    unavailable=set()
    # Complete original E30 before E60 for each case, using the genuine common evaluator.
    # Run short trials early while both lead tests and their raw streams are available.
    def plan(level):
        return [(a,c) for a in arms for c in ('A','B') if (a,c) not in unavailable]
    with ThreadPoolExecutor(max_workers=2) as slots:
        tasks=plan('E30')
        rows=list(slots.map(lambda ac:run_episode(*ac,'original','E30'),tasks))
        original30=dict(zip(tasks,rows))
        unavailable.update(ac for ac,row in original30.items() if row['status']=='technical_interruption')
        for mode in ('short','recovered'):
            work=[(a,c) for a in arms for c in ('A','B') if (a,c) not in unavailable]
            for ac,row in zip(work,slots.map(lambda ac:run_episode(*ac,mode),work)):
                if row['status']=='technical_interruption':
                    unavailable.add(ac)
        eligible=[ac for ac,row in original30.items() if row['status']=='passed' and ac not in unavailable]
        res=dict(zip(eligible,slots.map(lambda ac:run_episode(*ac,'original','E60'),eligible)))
        from analyze_s1_w57_episodes import compare
        compare([read(x) for x in sorted((R/'attempts').glob('*.json'))],'M1')
        selections={}
        for case in ('A','B'):
            ac=('PR',case)
            metrics=[R/'mechanics'/f'w57-m1-pr-{case.lower()}-{m}-a01.json' for m in ('short','recovered')]
            entries=[read(p) for p in metrics if p.exists()]
            conditions=dict(
                own_original_E60=res.get(ac,{}).get('status')=='passed',
                both_observation_horizons=len(entries)==2 and all(
                    r['observed_physical_s']+1.e-6>=s for r,s in zip(entries,(100.,196.))),
                actual_cross_foot_cycle=any(r['mechanism']['confirmed_cycle_count']>0 for r in entries),
                actual_material_candidate=any(r['planner_winners'].get('material',0)>0 and r['mapped_input_rows']>0 for r in entries),
                actual_timing_authority=any((r['timing'] or {}).get('liftoff_changed_on_same_gate_sequence',0)>0 or
                    (r['timing'] or {}).get('only_transfer_ramp_changed',0)>0 for r in entries))
            selections[case]=dict(eligible=all(conditions.values()),conditions=conditions)
        write(R/'dispatch/final-qualification.json',dict(candidate='PR',case=selections,**BACKGROUND))
        long_cases=[c for c in ('A','B') if selections[c]['eligible']]
        res300=dict(zip(long_cases,slots.map(lambda c:run_episode('PR',c,'original','E300'),long_cases)))
        res600=list(slots.map(lambda c:run_episode('PR',c,'original','E600'),
                              [c for c in long_cases if res300[c]['status']=='passed']))
    declaration.update(stage='queue_ended',technical_pairs=sorted(unavailable),
                       original_E300_attempts={c:v['attempt'] for c,v in res300.items()},
                       original_E600_attempts=[r['attempt'] for r in res600],finished_unix_s=time.time())
    write(target,declaration)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with (WORK/'frozen-evaluations.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign()
