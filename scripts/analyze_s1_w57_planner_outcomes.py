"""Join actual P1 candidate, mapped input and next physical load by published event ID."""
import argparse
from collections import Counter, defaultdict
from bisect import bisect_right
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
LEGS=('FL','FR','RL','RR')


def records(path):
    if path.is_file():
        with path.open() as stream:
            for line in stream:yield json.loads(line)


def q(values):
    return (dict(zip(('min','p50','p95','max'),np.percentile(values,[0,50,95,100]).tolist()))
            if len(values) else None)


def clean(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)


def overlay(raw,columns,leg_index):
    fcols=[columns.index(f'grf_{leg}_z') for leg in LEGS]
    pcols=[columns.index(f'foot_position_{leg}_{i}_m') for leg in LEGS for i in 'xyz']
    @lru_cache(maxsize=2)
    def chunk(i):
        with np.load(raw/f'physical-{i:03d}.npz') as z:
            data=z['physical'];return data[:,0],data[:,fcols],data[:,pcols].reshape(-1,4,3)
    # Full files have 30 s / 150000 physical ticks. The final one can be shorter.
    def vectors(a,b,leg):
        parts=[]
        for i in range(a//150000,b//150000+1):
            t,f,foot=chunk(i);j=max(0,a-i*150000);k=min(b-i*150000+1,len(t))
            if k>j:parts.append(np.c_[t[j:k],f[j:k,leg],foot[j:k,leg]])
        return np.concatenate(parts) if parts else np.empty((0,5))
    return vectors


class CrossFootLoads:
    def __init__(self,raw):
        self.raw=raw;self.index=defaultdict(list);self.loads={};self.releases={}
        for e in records(raw/'material-events.jsonl'):
            if e['type']=='release':
                self.releases[e['previous_load_event_id']]=e
            else:
                self.loads[e['event_id']]=e
                self.index[tuple(e['world_key'])].append(e)
        self.clocks={key:[e['time_s'] for e in events] for key,events in self.index.items()}

    @lru_cache(maxsize=16)
    def feet(self,chunk):
        with np.load(self.raw/f'material-{chunk:04d}.npz') as z:
            return z['foot'][:,6:18].reshape(-1,4,3).copy()

    def foot(self,clock,leg):
        chunk,step=divmod(round(clock/.0002)-1,5000)
        return self.feet(chunk)[step,leg,:2]

    @staticmethod
    def overlap(xys,key):
        half=np.sqrt(.001)/2
        lower=np.maximum.reduce([np.asarray(xy)-half for xy in xys]+[np.asarray(key)*.02])
        upper=np.minimum.reduce([np.asarray(xy)+half for xy in xys]+[(np.asarray(key)+1)*.02])
        return float(np.prod(np.maximum(upper-lower,0.)))

    def matches(self,query,when,commit,confirmation,detached):
        matches=[]
        for cell in query['cells']:
            key=tuple(cell['key'])
            if key not in detached or not cell['ever_loaded'] or cell['loaded'] or key not in self.index:
                continue
            j=bisect_right(self.clocks[key],when+1.e-9)
            if j==len(self.index[key]):continue
            # The first later actual load at this world key must be this published target leg.
            e=self.index[key][j]
            if e['type']!='revisit' or not e['cross_leg'] or e['leg']!=query['leg'] or not (
                    commit[2]-1.e-9 <= e['time_s'] <= confirmation+1.e-9):
                continue
            prev=self.loads.get(e['previous_load_event_id']);release=self.releases.get(e['previous_load_event_id'])
            if prev is None or release is None or release['time_s']>when+1e-9 or release['event_id']>=e['event_id']:
                continue
            if prev['leg']!=cell['last_owner']:continue
            target_xy=self.foot(e['time_s'],e['leg'])
            start=self.foot(prev['time_s'],prev['leg'])
            last=self.foot(release['time_s']-.0002,prev['leg'])
            area=max(self.overlap([start,target_xy,query['xy']],key),self.overlap([last,target_xy,query['xy']],key))
            if area<=0.:continue
            matches.append(dict(world_key=list(key),source_leg=prev['leg'],target_leg=e['leg'],
                observer_source_load_id=cell['source_load_event_id'],observer_release_id=cell['release_event_id'],
                true_source_load_id=prev['event_id'],true_release_id=release['event_id'],true_target_load_id=e['event_id'],
                source_to_reload_s=e['time_s']-prev['time_s'],release_to_reload_s=e['time_s']-release['time_s'],
                query_to_reload_s=e['time_s']-when,triple_geometric_intersection_m2=area))
        return matches


def analyze(attempt):
    raw=ROOT/'outputs'/REL/attempt
    cdir=raw/'w55-f';r=ROOT/'results'/REL/'w57'
    outcome=json.loads((r/'episodes'/(attempt+'.json')).read_text())
    end=float(outcome['physical_samples']*.0002)
    case=json.loads((r/'attempts'/(attempt+'.json')).read_text())
    get=overlay(raw,outcome['physical_columns'],LEGS)
    history=CrossFootLoads(raw)
    first_crossfoot={};first_mechanism={};pairs={};matched_publications=0
    achieved=set((outcome.get('mechanism') or {}).get('achieved_cycles',[]))
    target_pairs={(p['source_id'],p['release_id'],p['target_id']):p['cycle']
                  for p in records(raw/'w57-revisit-pairs.jsonl')
                  if p['condition_met'] and p['exact_overlap_m2']>0 and p['cycle'] in achieved}
    commits={};touch={}
    for event in records(cdir/'contact-events.jsonl'):
        if event['kind']=='liftoff_committed':
            for leg,key in zip(event['legs'],event['target_event_ids']):
                commits[key]=(LEGS.index(leg),event['time_s'],event['fixed_apex_s'])
        if event['kind']=='actual_touchdown' and 'target_event_id' in event:
            touch[event['target_event_id']]=event
    observed={}
    for key,(leg,lo,apex) in commits.items():
        if key not in touch:continue
        event=touch[key];limit=event['time_s']
        samples=get(round(lo/.0002),min(round(limit/.0002),round(end/.0002)-1),leg)
        if not samples.size:continue
        f=samples[:,1]>1.
        edges=np.flatnonzero(f[1:] & ~f[:-1])+1
        edges=edges[samples[edges,0]>=apex-1.e-9]
        if not len(edges):continue
        first=samples[edges[0]]
        future=get(round(first[0]/.0002),min(round((first[0]+.020)/.0002)-1,round(end/.0002)-1),leg)
        observed[key]=dict(leg=leg,first_force_on_s=float(first[0]),confirmation_s=limit,
                           xyz=first[2:].tolist(),peak_next20ms_N=float(future[:,1].max()),
                           impulse_next20ms_Ns=float(future[:,1].sum()*.0002))
    applied={row['call']:row for row in records(cdir/'material-execution.jsonl')}
    statuses=Counter();wins=Counter();physical_wins=Counter();missing=Counter();aligned=defaultdict(list);uniq={}
    times=[];obj=[];force_delta=[];readback=[];read_target=[];read_raw_pd=[];read_actuator=[]
    queries=0;actual_count=0;outside=0
    with np.load(raw/'trajectory.npz') as tr:
        logged_time=tr['time'];logged_target=tr['q_target']
    cols=outcome['physical_columns']
    torque_columns=[cols.index(f'actuator_force_{j}') for j in range(12)]
    pd_columns=[cols.index(f'unclipped_pd_request_{j}') for j in range(12)]
    @lru_cache(maxsize=2)
    def physical_command(chunk):
        with np.load(raw/f'physical-{chunk:03d}.npz') as z:
            x=z['physical'];return x[:,pd_columns],x[:,torque_columns]
    for row in records(cdir/'material-candidates.jsonl'):
        statuses[row.get('status','material_evaluated')]+=1
        if not row.get('data'):continue
        call=row['call'];wins[row['selected']]+=1;times.append(row['candidate_wall_s'])
        if row.get('original_objective') and row['original_objective'][1] is not None:
            obj.append(row['original_objective'][0]-row['original_objective'][1]);force_delta.append(row['max_first_force_difference_N'])
        if call in applied:actual_count+=1
        # Numeric equality of published applied input against independent saved WBC rows.
        step=round(row['time_s']/.002)
        control_file=cdir/f'control-{step//500+1:05d}.npz'
        if call in applied and control_file.exists():
            saved_u,requested=control(control_file)
            applied_u=np.array(applied[call]['mapped_input_rad'])
            readback.append(float(np.max(np.abs(saved_u[step%500]-applied_u))))
            if step<len(logged_time) and abs(logged_time[step]-row['time_s'])<1.e-8:
                read_target.append(float(np.max(np.abs(logged_target[step]-applied_u))))
            ix=step*10
            if ix<round(end/.0002):
                physical_wins[row['selected']]+=1
                pd,actuator=physical_command(ix//150000)
                read_raw_pd.append(float(np.max(np.abs(pd[ix%150000]-requested[step%500]))))
                read_actuator.append(float(np.max(np.abs(actuator[ix%150000]-pd[ix%150000]))))
        current=applied.get(call)
        targets=row.get('target_events',[])
        if row['detached_history']:
            with np.load(cdir/row['data']) as z:
                detached={tuple(key) for key in z['world_keys'][z['detached_history']]}
        else:detached=set()
        for e in targets:
            key=e['target_event_id'];leg=e['leg'];queries+=1
            if not e['within_material_horizon']:
                outside+=1;continue
            if not e['soft']:continue
            if key not in observed:
                why='no_committed_swing' if key not in commits else 'no_confirmed_loaded_target'
                missing[why]+=1;continue
            actual=observed[key];first=actual['first_force_on_s']
            if first < row['time_s']-1.e-9:
                missing['first_physical_contact_already_passed']+=1;continue
            ar=np.maximum(0.,np.sqrt(.001)-np.abs(np.array(e['xy'])-actual['xyz'][:2]))
            overlap=float(np.prod(ar))
            aligned['plan_minus_actual_target_s'].append(e['target_time_s']-first)
            aligned['query_to_first_load_s'].append(first-row['time_s'])
            aligned['plan_xy_error_m'].append(float(np.linalg.norm(np.asarray(e['xy'])-actual['xyz'][:2])))
            aligned['same_sole_overlap_m2'].append(overlap)
            existed=any(h.get('ever_loaded') for h in e['cells'])
            if overlap>0 and existed:aligned['overlapping_prior_history_query_to_load_s'].append(first-row['time_s'])
            matched=history.matches(e,row['time_s'],commits[key],actual['confirmation_s'],detached)
            if matched:
                matched_publications+=1
                for m in matched:
                    m['mechanism_cycle']=target_pairs.get((m['true_source_load_id'],m['true_release_id'],
                                                         m['true_target_load_id']))
                    ident=(m['true_source_load_id'],m['true_target_load_id'],tuple(m['world_key']))
                    if ident not in pairs:
                        pairs[ident]=dict(world_key=m['world_key'],source_leg=m['source_leg'],target_leg=m['target_leg'],
                           target_event_id=key,source_load_id=m['true_source_load_id'],release_id=m['true_release_id'],
                           target_load_id=m['true_target_load_id'],source_load_to_reload_s=m['source_to_reload_s'],
                           source_release_to_reload_s=m['release_to_reload_s'],first_query_s=row['time_s'],
                           first_query_to_reload_s=m['query_to_reload_s'],first_query_call=call,publications=0,
                           max_triple_overlap_m2=0.,mechanism_cycle=m['mechanism_cycle'])
                    item=pairs[ident];item['publications']+=1;item['last_query_s']=row['time_s']
                    item['last_query_to_reload_s']=m['query_to_reload_s'];item['last_query_call']=call
                    item['max_triple_overlap_m2']=max(item['max_triple_overlap_m2'],m['triple_geometric_intersection_m2'])
                    aligned['verified_crossfoot_query_to_true_cell_reload_s'].append(m['query_to_reload_s'])
                sample=dict(target_event_id=key,leg=leg,call=call,query_s=row['time_s'],
                    planned_touch_s=e['target_time_s'],measured_first_touch_s=first,
                    measured_confirmation_s=actual['confirmation_s'],xy=e['xy'],
                    data_file=row['data'],recorded_selected=row['selected'])
                if key not in first_crossfoot:first_crossfoot[key]=dict(sample,matched=matched)
                central=[m for m in matched if m['mechanism_cycle'] is not None]
                if central and key not in first_mechanism:first_mechanism[key]=dict(sample,matched=central)
            # Repeated replans do not become independent contacts; retain first, last and nearest call.
            entry=uniq.setdefault(key,dict(**actual,target_event_id=key,matched_plan_count=0,
                         first_call=call,first_query_s=row['time_s'],first_xy=e['xy'],overlap_queries=0,
                         prior_history_queries=0,first_command_present=(current is not None)))
            entry.update(last_call=call,last_query_s=row['time_s'],last_xy=e['xy'])
            entry['matched_plan_count']+=1;entry['overlap_queries']+=int(overlap>0);entry['prior_history_queries']+=int(existed)
        # At each plan's next 25ms node compare the planned load / loaded sole against actual kinematics at the same time.
        with np.load(cdir/row['data']) as z:
            grid=z['segment_grid_s'];coordinate=z['active'];key='nominal_force' if row['selected']=='nominal' else 'candidate_force'
            displacement=z['full_sole_z_m'] if row['selected']=='nominal' else z['candidate_sole_z_m']
            j=int(np.argmin(abs(grid[1:]-(row['time_s']+.025))))
            when=float(grid[j+1]);at=round(when/.0002)
            if at>=round(end/.0002):continue
            for leg in np.flatnonzero(coordinate[j]):
                actual=get(at,at,leg)
                if not len(actual):continue
                actual=actual[0];pred=z[key][0,leg,2]
                aligned['first_node_force_error_N'].append(float(pred-actual[1]))
                aligned['first_node_force_error_abs_N'].append(float(abs(pred-actual[1])))
                if pred>1 and actual[1]>1:
                    aligned['jointly_loaded_first_node_sole_error_m'].append(float(displacement[j,leg]-(actual[4]-.022)))
                aligned['absolute_clock_error_s'].append(float(abs(actual[0]-when)))
    res=dict(attempt=attempt,case=case['case'],arm=case['arm'],estimator=case['material'],task=case['task_mode'],
        level=case['level'],termination=outcome.get('reason'),physical_observation_s=end,source=str(raw),
        recorded_substep_s=.0002,installed_candidate_count=actual_count,forecast_replan_status=dict(statuses),
        eligible_winners=dict(wins),winners_with_following_physics=dict(physical_wins),
        terminal_WBC_plans_without_following_hold=actual_count-len(read_actuator),candidate_timing_s=q(times),nominal_minus_corrected_objective=q(obj),
        maximum_first_force_difference_N=q(force_delta),installed_input_vs_WBC_max_rad=max(readback) if readback else None,
        applied_input_vs_trajectory_target_max_rad=max(read_target) if read_target else None,
        original_PD_at_first_physics_step_vs_WBC_requested_max_Nm=max(read_raw_pd) if read_raw_pd else None,
        measured_first_actuator_force_vs_raw_PD_max_Nm=max(read_actuator) if read_actuator else None,
        exact_physics_command_records=len(read_actuator),
        published_targets=queries,touchdown_outside_inner_horizon=outside,missing=dict(missing),
        metrics={key:dict(n=len(value),distribution=q(value)) for key,value in aligned.items()},
        distinct_confirmed_physical_contacts=len(uniq),target_events=list(uniq.values()),
        true_released_crossfoot_query_target_publications=matched_publications,
        distinct_world_cell_source_reload_pairs=len(pairs),verified_source_reload_pairs=list(pairs.values()),
        distinct_pair_source_load_to_reload_s=q([m['source_load_to_reload_s'] for m in pairs.values()]),
        distinct_pair_source_release_to_reload_s=q([m['source_release_to_reload_s'] for m in pairs.values()]),
        distinct_pairs_whose_first_verified_query_precedes_reload_by_at_most_0p35s=sum(
            m['first_query_to_reload_s']<=.35+1e-9 for m in pairs.values()),
        first_matching_publication_per_true_crossfoot_target=list(first_crossfoot.values()),
        first_matching_publication_per_verified_mechanism_target=list(first_mechanism.values()),
        scope='At first sampled force >1N after committed apex and release. WBC confirmation is separate. Contacts with no committed swing remain unmatched. Same physical clock 25ms forecasts include subsequent replans/robot tracking error. Multiple publications for the same leg event are correlated.')
    out=r/'p/physical'/(attempt+'.json');out.parent.mkdir(exist_ok=True,parents=True)
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(res,indent=2,allow_nan=False,default=clean)+'\n');temp.replace(out)
    return {k:res[k] for k in ('attempt','installed_candidate_count','distinct_confirmed_physical_contacts','eligible_winners','installed_input_vs_WBC_max_rad')}


@lru_cache(maxsize=2)
def control(file):
    with np.load(file) as z:return z['mapped_u'],z['requested_pd']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('attempt')
    print(json.dumps(analyze(parser.parse_args().attempt)))
