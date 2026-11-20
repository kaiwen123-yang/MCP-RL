"""Derive W56 qualifications from original stage results and exact D57 process identities."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REL = Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
LARGE = Path('outputs')/Path(*REL.parts[1:])
WINDOW = ROOT/REL/'w56'
STAGES = ('E30','E60','E300','E600')
HOST = ROOT.parent
BASE = HOST/'MCP-RL-w55-f'
F = HOST/'MCP-RL-w56-f'
MATERIAL = HOST/'MCP-RL-w56-material'
N = HOST/'MCP-RL-w56-native'
CASES = (('F-A-r55',BASE,STAGES),('F-B-r55',BASE,STAGES),
         ('F-A-r57',F,STAGES[:2]),('F-B-r57',F,STAGES[:2]),
         ('F-A-r58',MATERIAL,STAGES),('F-B-r58',MATERIAL,STAGES),('N-A-r59',N,STAGES[:1]))
BACKGROUND = dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.w56-tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def short(source, names):
    return {key:source[key] for key in names if key in source}


def verified_pid(pid,attempt):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    return b'--attempt' in cmd and cmd[cmd.index(b'--attempt')+1] == attempt.encode()


def stage_info(label,tree,stage):
    lane,_,build=label.split('-')
    window='w55' if build=='r55' else 'w56'
    folder=tree/REL/window/lane.lower()
    # Contracts may use independent compact roots for different builds.
    cases=list(folder.glob('**/'+label))
    if len(cases)>1:
        raise ValueError(f'Ambiguous original case directory for {label}: {cases}')
    case=cases[0] if cases else folder/label
    row=dict(label=label,level=stage,window=window,worktree=str(tree),status='unrun')
    contract=case/'execution-contract.json'
    if (case/'queue.json').is_file():
        queue=read(case/'queue.json')
        row.update(queue_record=str((case/'queue.json').relative_to(tree)),queue_status=queue.get('status'),queue_tmux=queue.get('tmux'))
    path=case/(stage+'.json')
    if not path.is_file():
        return row
    job=read(path)
    row.update(short(job,['attempt','job','tmux','pid','runner_pid','reason','exit_code','started_unix_s','ended_unix_s']))
    row['stage_record']=str(path.relative_to(tree))
    attempt=job.get('attempt')
    if not attempt:
        row['status']=job.get('status','unrun')
        return row
    compact=tree/read(contract)['outputs']['compact_root']
    result=compact/(attempt+'.json')
    if result.is_file():
        data=read(result)
        if data.get('attempt')!=attempt or data.get('evaluation_stage',{}).get('level')!=stage:
            raise ValueError('Case and original evaluator result disagree: '+str(result))
        row.update(short(data,['success','reason','error','physical_samples','simulation_s','physical_started_unix_s',
             'physical_ended_unix_s','case_key','first_completion_s','max_x_m','min_body_height_m']))
        row.update(status='passed' if data.get('success') and data['stage_status']=='passed' else 'failed',
                   evaluator_status=data['stage_status'],source=str(result.relative_to(tree)),
                   identity=short(data['case_identity'],['seed','controller_build','checkpoint_sha256','normalizer_sha256']),
                   legal_route=short(data['legal_route'],['completed_roundtrips','legal_outbound_traversals','legal_return_traversals','side_exits']))
    else:
        alive=bool(job.get('pid') and verified_pid(job['pid'],attempt))
        row['pid_verified']=alive
        live=tree/LARGE/attempt/'live.json'
        if live.is_file():
            h=read(live)
            if h.get('pid')==job.get('pid'):
                row['heartbeat']=short(h,['unix_s','physical_time_s','phase','position_m','pid'])
        row['status']='running' if alive else ('declared' if job.get('status')=='declared' else
                      'interrupted' if job.get('reason')=='process_exited_without_completed_result' else 'awaiting_evaluator_record')
        for field in ('physical_started_unix_s','physical_ended_unix_s'):
            if field in job: row[field]=job[field]
        if alive and 'physical_started_unix_s' not in row and (compact/'index.json').exists():
            entry=next((x for x in read(compact/'index.json').get('attempts',[]) if x.get('attempt')==attempt),{})
            row.update(short(entry,['physical_started_unix_s']))
    for receipt_tree in (ROOT,tree,F,MATERIAL,N):
        delivery=receipt_tree/REL/window/'delivery'/(label+'-'+stage).lower()/'delivery.json'
        if delivery.is_file():
            receipt=read(delivery)
            assets=receipt.get('assets',[])
            row['publication']=dict(status=receipt.get('status'),receipt=str(delivery.relative_to(receipt_tree)),
                asset_count=len(assets),verified_asset_count=sum(bool(a.get('remote_digest')) for a in assets),
                asset_bytes=sum(a['size'] for a in assets))
            break
    return row


def derive():
    rows=[stage_info(label,tree,stage) for label,tree,levels in CASES for stage in levels]
    windows={k:[r for r in rows if r['window']==k] for k in ('w55','w56')}
    now=time.time()
    overlap=[]
    for i,a in enumerate(rows):
        for b in rows[:i]:
            starts=[r.get('physical_started_unix_s') for r in (a,b)]
            ends=[r.get('physical_ended_unix_s') or (r.get('heartbeat',{}).get('unix_s') if r['status']=='running' else None) for r in (a,b)]
            if None not in starts+ends and min(ends)>max(starts):
                overlap.append(dict(attempts=[a['attempt'],b['attempt']],begin_unix_s=max(starts),end_unix_s=min(ends),wall_s=min(ends)-max(starts)))
    phase=dict(snapshot_unix_s=now,source='D57 original per-case evaluator and same-attempt child PID; running interval ends at observed physical heartbeat.',
               cases=rows,overlapping_physics=overlap,**BACKGROUND)
    summary={}
    for window,data in windows.items():
        counts={s:sum(r['status']==s for r in data) for s in sorted({r['status'] for r in data})}
        summary[window]=dict(status_counts=counts,ended=sum(counts.get(k,0) for k in ('passed','failed','interrupted')),
            published_stages=sum(r.get('publication',{}).get('status') in ('published','verified','uploaded') for r in data),
            passed_original_levels={l:[r['level'] for r in data if r['label']==l and r['status']=='passed']
                                    for l in sorted({r['label'] for r in data})})
    results=list((WINDOW/'exposure').glob('*.json'))
    sources={d.name:str(d.relative_to(ROOT)) for d in results}
    newer=windows['w56']
    r57=(ROOT/'experiments/terrain_memory/s1-w56-f-r57.json').is_file()
    started=any(r.get('physical_samples',0)>0 or r.get('heartbeat',{}).get('physical_time_s',0)>0 for r in newer)
    index=dict(window='W56',implementation_base='3578b494f05b6a10c64a75cfceaaf5d3d4bc558c',
        status='executing',snapshot_unix_s=now,phase=str(REL/'w56/phase.json'),frozen_r55_scope='Two W44 nominal cases. E600 stages keep their W55 identities.',
        workspaces=dict(integration=str(HOST/'MCP-RL-w55-integration'),F=str(F),C=str(MATERIAL),N=str(N)),
        original_qualification=summary,baseline_array_checks={p.stem:str(p.relative_to(ROOT)) for p in sorted((WINDOW/'baseline').glob('*.json'))},actor_query_reconstruction=sources,robot_control_implemented=r57,robot_physics_started=started,
        new_PPO_samples=0,new_PPO_updates=0,donor=0,**BACKGROUND)
    save(WINDOW/'index.json',index)
    save(WINDOW/'phase.json',phase)
    current=read(ROOT/REL/'current.json')
    current.update(date=datetime.fromtimestamp(now,timezone.utc).astimezone().isoformat(timespec='seconds'),
        status='W56_execution',latest_executed_window='W56',latest_results=str(REL/'w56/index.json'),
        latest_execution_state=str(REL/'w56/phase.json'),result_source_commit_scope='W55 reviewed baseline. Current W56 per-case results and live process snapshot are in latest_execution_state.')
    current['W56'].update(status='executing',index=index['phase'].replace('phase.json','index.json'),phase=index['phase'],
        robot_control_implemented=r57,robot_physics_started=started,original_qualification=summary['w56'])
    current['continuation']['lane_workspaces'].update(W56_F=str(F),W56_C=str(MATERIAL),W56_N=str(N))
    current['continuation']['rule']='Resume exact jobs reported in W56 phase and original case records. W55 r55 E600 keeps its original case identity.'
    current['W55']['r55_latest_qualification']=summary['w55']
    current['W55']['active_stages_at_snapshot']=[dict(attempt=r['attempt'],pid=r['pid'],physical_s=r.get('heartbeat',{}).get('physical_time_s')) for r in windows['w55'] if r['status']=='running']
    current['W55']['active_stage_source']=index['phase']
    current['W55']['snapshot_unix_s']=now
    current['W55']['F']='Per-case nominal qualifications and observed E600 state are derived from the original evaluator under r55_latest_qualification.'
    current['W55']['status']='r55_original_E600_in_progress' if current['W55']['active_stages_at_snapshot'] else 'see_r55_original_stage_records'
    save(ROOT/REL/'current.json',current)
    config=ROOT/'experiments/terrain_memory/s1-go2-local-readiness-r12.json'
    d=read(config)
    plan=d['next_authorized_plan']
    if plan['entrypoint']=='docs/S1_W56_STABILITY_PLAN.md':
        plan.update(status=index['status'],robot_control_implemented=r57,robot_physics_started=started,last_actual_index=str(REL/'w56/index.json'),
                    current_execution_identity=str(REL/'w56/phase.json'),qualification_scope='W55 historical aggregate via current.W55; all W56 stages independently qualified.')
        w55=read(ROOT/REL/'w55/index.json')
        plan.update({level+'_passes':count for level,count in w55['original_stage_passes'].items()})
        plan['highest_passed_original_stage']=w55['highest_passed_original_stage']
        save(config,d)
    return index


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(derive()['original_qualification']))
