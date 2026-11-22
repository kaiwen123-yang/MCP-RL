"""Postprocess naturally finished P/PR simulations as the authorized W57 queue advances."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from analyze_s1_w57_planner_outcomes import ROOT,REL,analyze
from compare_s1_w57_original_planning import run as paired
from replay_s1_w57_confirmed_history_actions import evidence,first_actual

R=ROOT/'results'/REL/'w57'
RAW=ROOT/'outputs'/REL
OUT=R/'p/physical'
BACKGROUND=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)


def read(file):
    return json.loads(file.read_text())


def write(file,value):
    tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(file)


def collect():
    receipts=[read(f) for f in sorted((R/'attempts').glob('*.json'))]
    finished=[r for r in receipts if r.get('ended_unix_s') and (R/'episodes'/(r['attempt']+'.json')).is_file()]
    controls={(r['material'],r['case'],r['level']):r for r in finished if r['arm']=='B' and r['task_mode']=='original'}
    interested=[r for r in finished if r['arm'] in ('P','PR')]
    for r in interested:
        a=r['attempt'];events=OUT/(a+'.json')
        if not events.exists():
            print(json.dumps(analyze(a)),flush=True)
        if r['task_mode']=='original' and r['arm']=='P':
            b=controls.get((r['material'],r['case'],r['level']))
            if b and not (OUT/(a+'-paired-B.json')).exists():
                paired(b['attempt'],a)
        if (r['arm']=='P' and r['case']=='A' and r['material']=='M1' and
                r['settings']['planner_history']=='full' and
                (r['task_mode']=='short' or (r['task_mode']=='original' and r['level']=='E30'))):
            selected=read(events)
            if selected['eligible_winners'].get('material',0) and not (OUT/(a+'-first-selected.json')).exists():
                first_actual(a)
            if not (OUT/(a+'-private-action.json')).exists():
                print(json.dumps(evidence(a)),flush=True)
    rough={}
    for mode in ('original','short'):
        chosen=[r for r in interested if r['arm']=='P' and r['case']=='A' and r['material']=='M1'
                and r['settings']['planner_history']=='full' and r['task_mode']==mode
                and (mode=='short' or r['level']=='E30')]
        if not chosen:continue
        a=chosen[0]['attempt'];f=OUT/(a+'-private-action.json')
        if f.is_file():
            result=read(f)
            actual=read(R/'episodes'/(a+'.json'))
            boundary=(bool(actual['success']) if mode=='original' else
                      actual['simulation_s']+1.e-7>=actual['evaluation_stage']['duration_s'])
            rough[mode]=dict(attempt=a,physical_status=chosen[0]['status'],
                original_pass_or_complete_mechanism_observation=boundary,
                mechanism_confirmed_cycles=(actual.get('mechanism') or {}).get('confirmed_cycle_count'),
                exact_revisit_criterion_met_for_every_matched_short_target=(True if mode=='short' else None),
                distinct_matched_targets=result['distinct_target_events'],
                mapped_changes_above_reconstruction=result['above_reconstruction'])
    if set(rough)=={'short','original'}:
        gate=dict(selected_case='A',task='short',paired_episodes=rough,
                  eligible_for_separate_physics=all(r['original_pass_or_complete_mechanism_observation'] and r['distinct_matched_targets']>0
                       and r['mapped_changes_above_reconstruction']>0 for r in rough.values()),
                  private_physics_separate_from_16=True,**BACKGROUND)
        saved=OUT/'private-short-A-eligibility.json'
        if not saved.exists() or read(saved)!=gate:write(saved,gate)
    return len(interested)


def main(follow):
    OUT.mkdir(parents=True,exist_ok=True)
    if follow:
        print(json.dumps(dict(pid=os.getpid(),command=sys.argv,role='postprocess_completed_plans',**BACKGROUND)),flush=True)
    while True:
        n=collect()
        phase=R/'dispatch/frozen-evaluations.json'
        if not follow or (phase.is_file() and read(phase).get('stage')=='queue_ended'):
            print(json.dumps(dict(processed=n)),flush=True);return
        time.sleep(45)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--follow',action='store_true');main(p.parse_args().follow)
