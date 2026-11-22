"""Merge distinct W57 attempts, material evidence and policy nodes into the compact run index."""
from datetime import datetime
from pathlib import Path
import json
from zoneinfo import ZoneInfo

from publish_s1_go2_w54 import write

ROOT=Path(__file__).resolve().parents[1]
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
R=ROOT/REL/'w57'
RAW=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'


def read(p):
    return json.loads(p.read_text())


def combine():
    rows=[read(p) for p in sorted((R/'attempts').glob('*.json'))]
    starts=[];stops=[];attempts=[]
    for row in rows:
        name=row['attempt']
        data={k:row.get(k) for k in ('attempt','case','arm','material','task_mode','level','controller','pid','tmux',
               'slot','status','simulation_s','reason','source_commit','started_unix_s','ended_unix_s','user_cpu_s',
               'system_cpu_s','physical_samples','physical_started_unix_s','physical_ended_unix_s')}
        live=RAW/name/'live.json'
        if row['status']=='running' and live.exists():
            active=read(live)
            data.update(last_recorded_physical_s=active['physical_time_s'],last_recorded_phase=active['phase'],
                        last_progress_unix_s=active['unix_s'])
        if row.get('physical_started_unix_s') and row.get('physical_ended_unix_s'):
            starts.append((row['physical_started_unix_s'],1));stops.append((row['physical_ended_unix_s'],-1))
        if row.get('mechanism'):
            data['confirmed_revisit_cycles']=row['mechanism']['confirmed_cycle_count']
        data['receipt']=str((REL/'w57/attempts'/(name+'.json')))
        attempts.append(data)
    active=0;last=0.;overlap=0.
    for t,d in sorted(starts+stops):
        if active>=2:overlap+=t-last
        active+=d;last=t
    learning={}
    for p in sorted((R/'r').glob('*/*/update-*.json')):
        metrics=read(p)
        head=metrics['head']
        v=learning.setdefault(head,dict(distinct_in_completed_updates=0,updates=0))
        v['distinct_in_completed_updates']+=metrics['distinct'];v['updates']+=1
    deliveries=[read(p) for p in sorted((R/'delivery').glob('*.json'))]
    mdir=R/'m';pdir=R/'p'
    out=dict(schema='s1_w57_execution_v1',as_of_shanghai=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
             plan_commit='d321d61fd13ba301b477deaf473101debebb88e5',
             execution_workspace=str(ROOT),baseline=dict(original_E600=['r55 A','r55 B','r60 A','r60 B'],
             original_reruns=0,reference=str(REL/'w56/index.json')), background=dict(deadline=None,
             wall_deadline_enabled=False,kill_on_conversation_end=False),attempts=attempts,
             distinct_physics_attempts=len(rows),completed_physics_samples=sum(x.get('physical_samples') or 0 for x in rows),
             completed_process_cpu_s=sum((x.get('user_cpu_s') or 0)+(x.get('system_cpu_s') or 0) for x in rows),
             overlap_wall_s_of_finished_physical_intervals=overlap,
             original_qualifications=[dict(arm=x['arm'],case=x['case'],level=x['level'],attempt=x['attempt'],
                                           controller=x['controller']) for x in rows
                                      if x['task_mode']=='original' and x['status']=='passed'],
             material_evidence=[str(p.relative_to(ROOT)) for p in sorted(mdir.glob('*.json'))],
             planner_evidence=[str(p.relative_to(ROOT)) for p in sorted(pdir.glob('*.json'))
                               if not p.name.startswith('integrator-')],
             timing_learning=learning,frozen_W44_actor_updates=0,
             learning_counts_scope='Closed unique 128+128 event transitions in committed optimizer updates; partial live buffers are separate.',
             verified_release_bundles=[r['label'] for r in deliveries if r.get('status')=='verified'],
             verified_release_assets=sum(len(r.get('assets',[])) for r in deliveries),
             release_bytes=sum(a['size'] for r in deliveries for a in r.get('assets',[])),
             declared_mechanism_episodes=16,additional_matched_blind_if_triggered=4)
    write(R/'index.json',out)
    return out


if __name__=='__main__':
    idx=combine()
    print(json.dumps({k:idx[k] for k in ('as_of_shanghai','distinct_physics_attempts','completed_physics_samples',
                                        'timing_learning','verified_release_assets')}))
