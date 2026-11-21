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
R60 = HOST/'MCP-RL-w56-r60'
CASES = (('F-A-r55',BASE,STAGES),('F-B-r55',BASE,STAGES),
         ('F-A-r57',F,STAGES[:2]),('F-B-r57',F,STAGES[:2]),
         ('F-A-r58',MATERIAL,STAGES),('F-B-r58',MATERIAL,STAGES),('N-A-r59',N,STAGES[:1]),
         ('F-A-r60',R60,STAGES),('F-B-r60',R60,STAGES))
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
        lane_queue=folder/('queue-'+build+'.json')
        if stage=='E30' and lane_queue.exists():
            ticket=read(lane_queue)
            row.update(queue_record=str(lane_queue.relative_to(tree)),queue_status=ticket['status'],
                       queue_tmux=ticket.get('tmux'))
            if ticket['status'].startswith('waiting'):
                row['status']=ticket['status']
        if stage=='E30' and row.get('queue_status')=='waiting_for_A_E30':
            row['status']='waiting_for_A_E30'
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
        if 'physical_started_unix_s' not in row and (compact/'index.json').exists():
            entry=next((x for x in read(compact/'index.json').get('attempts',[]) if x.get('attempt')==attempt),{})
            row.update(short(entry,['physical_started_unix_s']))
    if row['status']=='interrupted' and build=='r58' and stage=='E300':
        recovery=WINDOW/'c/r58-e300-recovery-outcome.json'
        if recovery.is_file():
            entry=next(x for x in read(recovery)['cases'] if x['attempt']==attempt)
            row.update(last_confirmed_mpc_s=entry['last_solved_mpc_s'],
                       last_mpc_write_unix_s=entry['last_mpc_write_unix_s'],
                       interruption_evidence=str(recovery.relative_to(ROOT)))
    for receipt_tree in (ROOT,tree,F,MATERIAL,N,R60):
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
            ends=[r.get('physical_ended_unix_s') or r.get('last_mpc_write_unix_s') or (r.get('heartbeat',{}).get('unix_s') if r['status']=='running' else None) for r in (a,b)]
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
    sources={d.stem:str(d.relative_to(ROOT)) for d in sorted(results)}
    gpu=WINDOW/'c/cpu-gpu-real-histories.json'
    if gpu.is_file():
        stats=read(gpu)
        phase['batch_benchmark_unix_s']=[stats['start_unix'],stats['end_unix']]
        phase['two_physics_plus_material_benchmark_overlap_s']=sum(
            max(0.,min(p['end_unix_s'],stats['end_unix'])-max(p['begin_unix_s'],stats['start_unix'])) for p in overlap)
    direct=WINDOW/'c/cpu-gpu-direct-single-launch.json'
    if direct.is_file():
        timing=read(direct)
        left=datetime.fromisoformat(timing['start_utc']).timestamp()
        right=datetime.fromisoformat(timing['end_utc']).timestamp()
        phase['direct_CPU_GPU_measurement_with_two_original_robots_s']=sum(
            max(0.,min(p['end_unix_s'],right)-max(p['begin_unix_s'],left)) for p in overlap)
    receipts={}
    for p in sorted((WINDOW/'delivery').glob('*/delivery.json')):
        j=read(p)
        if j.get('status')=='published':
            for a in j['assets']:
                receipts[a['name']]=a
    publication=dict(release='https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-local-readiness-r12-w56',
        verified_assets=len(receipts),verified_bytes=sum(a['size'] for a in receipts.values()),
        source='W56 verified per-delivery receipts on main')
    proofs={}
    for pattern in ('c/*.json','n/same-input-*.json','n/*original-objective.json','n/queue-*.json',
                    'f/r57-*-actor-selection.json','f/F-*/E*-paired.json','n/N-*/E*-actual-analysis.json','delivery/*/delivery.json'):
        for path in sorted(WINDOW.glob(pattern)):
            proofs[str(path.relative_to(WINDOW))]=str(path.relative_to(ROOT))
    f_closed=all((tree/REL/'w56/f'/label/'queue.json').is_file() and
        read(tree/REL/'w56/f'/label/'queue.json').get('status') in ('ladder_ended','not_run_A_E30_did_not_pass')
        for label,tree,levels in CASES if label.startswith('F-') and not label.endswith('r55'))
    n_queue=N/REL/'w56/n/queue-r59.json'
    all_ended=f_closed and n_queue.is_file() and read(n_queue)['status'] in ('published','physics_ended') and all(
        r.get('publication',{}).get('status')=='published' for r in rows if r['status'] in ('passed','failed') or (r['status']=='interrupted' and r.get('interruption_evidence'))) and all(
        r['status'] in ('passed','failed','interrupted') for r in windows['w55'] if r['level']=='E600')
    execution='declared_ladders_concluded_and_delivered' if all_ended else 'executing'
    newer=windows['w56']
    r57=(ROOT/'experiments/terrain_memory/s1-w56-f-r57.json').is_file()
    started=any(r.get('physical_samples',0)>0 or r.get('heartbeat',{}).get('physical_time_s',0)>0 for r in newer)
    index=dict(window='W56',implementation_base='3578b494f05b6a10c64a75cfceaaf5d3d4bc558c',
        status=execution,snapshot_unix_s=now,publication=publication,evidence=proofs,phase=str(REL/'w56/phase.json'),frozen_r55_scope='Two W44 nominal cases. E600 stages keep their W55 identities.',
        workspaces=dict(integration=str(HOST/'MCP-RL-w55-integration'),F=str(F),C=str(MATERIAL),N=str(N),F_r60=str(R60)),
        original_qualification=summary,baseline_array_checks={p.stem:str(p.relative_to(ROOT)) for p in sorted((WINDOW/'baseline').glob('*.json'))},actor_query_reconstruction=sources,robot_control_implemented=r57,robot_physics_started=started,
        new_PPO_samples=0,new_PPO_updates=0,donor=0,**BACKGROUND)
    save(WINDOW/'index.json',index)
    save(WINDOW/'phase.json',phase)
    current=read(ROOT/REL/'current.json')
    current.update(date=datetime.fromtimestamp(now,timezone.utc).astimezone().isoformat(timespec='seconds'),
        status='W56_execution',latest_executed_window='W56',latest_results=str(REL/'w56/index.json'),
        latest_execution_state=str(REL/'w56/phase.json'),result_source_commit_scope='W55 reviewed baseline. Current W56 per-case results and live process snapshot are in latest_execution_state.')
    if '/releases/tag/tm-s1-go2-local-readiness-r12-w55' in current.get('publication',{}).get('release',''):
        current['W55']['publication']=current['publication']
    current['publication']=publication
    current['W56'].update(status=execution,publication=publication,index=index['phase'].replace('phase.json','index.json'),phase=index['phase'],
        robot_control_implemented=r57,robot_physics_started=started,original_qualification=summary['w56'])
    current['continuation']['D57_tmux']['w56_publication']='s1-w56-publish-results'
    current['continuation']['D57_tmux'].update({r['label']:r['queue_tmux'] for r in newer if r.get('queue_tmux')})
    current['continuation']['lane_workspaces'].update(W56_F=str(F),W56_C=str(MATERIAL),W56_N=str(N),W56_F_r60=str(R60))
    current['continuation']['rule']='The r58 E300 attempts ended after authorized recovery and shutdown. Follow the independent r60 original ladders in the phase records.'
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
        plan['last_actual_F']='S5-r55' if not any(r.get('physical_samples') or r.get('heartbeat') for r in newer if r['label'].startswith('F')) else 'see W56 phase.json by case'
        plan['last_actual_N']='S5-r59' if any(r.get('physical_samples') or r.get('heartbeat') for r in newer if r['label']=='N-A-r59') else 'S5-r56'
        plan['W56_stage_passes']={level:sum(r['level']==level and r['status']=='passed' for r in newer) for level in STAGES}
        plan['legacy_unprefixed_pass_counts_window']='W55'
        plan.update({level+'_passes':count for level,count in w55['original_stage_passes'].items()})
        plan['highest_passed_original_stage']=w55['highest_passed_original_stage']
        save(config,d)
    return index


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(derive()['original_qualification']))
