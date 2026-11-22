"""Node diagnostics from on-policy training episodes, saved contacts and the original physical plant."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from s1_w57_learning import STEPS, batch_path, read_rows


def intersection(query, center, key, area=.001, cell=.02):
    half = np.sqrt(area)/2
    lower = np.maximum.reduce([np.asarray(query)-half,np.asarray(center)-half,np.array(key)*cell])
    upper = np.minimum.reduce([np.asarray(query)+half,np.asarray(center)+half,(np.array(key)+1)*cell])
    return float(np.prod(np.maximum(upper-lower,0.)))


def matched_reloads(folder, episode_rows):
    """Offline link through a forecast ID, actual swing/confirmed touchdown and a common world cell."""
    frows = read_rows(folder/'w55-f'/'contact-events.jsonl')
    liftoff = {}
    touch = {}
    for event in frows:
        if event['kind'] == 'liftoff_committed':
            for event_id in event.get('target_event_ids', []):
                liftoff[event_id] = event
        elif event['kind'] == 'actual_touchdown' and 'target_event_id' in event:
            touch[event['target_event_id']] = event
    physics = read_rows(folder/'material-events.jsonl')
    loads = defaultdict(list)
    released = {}
    source_times = {e['event_id']:e['time_s'] for e in physics if e['type'] in ('load','revisit')}
    for event in physics:
        if event['type'] == 'release':
            released[event['previous_load_event_id']] = event
        elif event['type'] == 'revisit' and event.get('unloaded_gap_s') is not None:
            loads[(event['leg'],tuple(event['world_key']))].append(event)
    @lru_cache(maxsize=2)
    def observed_foot_chunk(index):
        path = folder/f'material-{index:04d}.npz'
        if not path.is_file():
            return None
        with np.load(path) as material:
            return material['foot']

    def actual_xy(t,leg):
        # Ground events own the successor material clock. Returned XY is the measured left boundary.
        chunk,offset = divmod(round(t/.0002)-1,5000)
        foot = observed_foot_chunk(chunk)
        if foot is None or offset >= len(foot):
            return None
        if abs(foot[offset,1]-t)>1e-8:
            raise ValueError(f'Plant event {t} does not match recorded material clock {foot[offset,1]}')
        return foot[offset,6+3*leg:8+3*leg]

    wanted = {r['decision_id']:r for r in episode_rows}
    observed = []
    counts = defaultdict(int)
    for decision in read_rows(folder/'timing-decisions.jsonl'):
        if decision['decision_id'] not in wanted:
            continue
        row = wanted[decision['decision_id']]
        for query in decision['queries']:
            counts[query['status']] += 1
            if not query['query_valid'] or not query['soft_mask']:
                continue
            if not any(c['ever_loaded'] and not c['loaded'] for c in query['cells']):
                continue
            event_id = query['target_event_id']
            landing = touch.get(event_id)
            takeoff = liftoff.get(event_id)
            if landing is None or takeoff is None:
                counts['unconfirmed_or_uncommitted_target_at_node'] += 1
                continue
            start = max(decision['sample_time_s'],takeoff['fixed_apex_s'])
            end = landing['time_s']
            for column in query['cells']:
                if not column['ever_loaded'] or column['loaded']:
                    continue
                key = tuple(column['key'])
                for target in loads[(query['leg'],key)]:
                    if not start <= target['time_s'] <= end+1e-9 or target['normal_N']<=0:
                        continue
                    release = released.get(target['previous_load_event_id'])
                    if release is None or release['event_id'] >= target['event_id'] or release['time_s']>decision['sample_time_s']:
                        continue
                    center = actual_xy(target['time_s'],query['leg'])
                    if center is None:
                        counts['physical_boundary_unavailable_at_training_node'] += 1
                        continue
                    shared_area = intersection(query['xy'],center,key)
                    if shared_area<=0:
                        continue
                    # The initial source must agree with the queried past owner. Observer and plant
                    # use different event ID namespaces; never equate those two integers.
                    if column['last_owner'] != target['previous_leg']:
                        continue
                    true_preload = abs(float(target['preload_free_e'][0]))*.05
                    residual = max(abs(h) for h in query['free_recovery_m'])
                    observed.append(dict(attempt=row['attempt'],case=row['case'],decision_id=row['decision_id'],
                        query_time_s=decision['sample_time_s'],leg=query['leg'],key=list(key),target_id=event_id,
                        target_time_s=query['target_time_s'],actual_confirmed_s=end,measured_load_s=target['time_s'],
                        source_observer_load_event_id=column['source_load_event_id'],
                        source_observer_load_s=column['source_load_time_s'],
                        source_plant_load_event_id=target['previous_load_event_id'],
                        source_plant_load_s=source_times[target['previous_load_event_id']],
                        actual_force_boundary_xy=center.tolist(),query_to_target_s=target['time_s']-decision['sample_time_s'],
                        source_to_reload_s=target['time_s']-source_times[target['previous_load_event_id']],
                        plant_release_id=release['event_id'],plant_release_s=release['time_s'],
                        plant_target_id=target['event_id'],unloaded_gap_s=target['unloaded_gap_s'],
                        actual_normal_N=target['normal_N'],true_preload_free_abs_m=true_preload,
                        predicted_free_abs_max_m=residual,intersection_m2=shared_area,
                        cross_foot=bool(target['cross_leg']),timing_actuated=bool(row['execution'] and (
                        row['execution']['effective'] or row['execution']['transfer_fraction_changed']))))
                    break
    return observed,dict(counts)


def node_evidence(run, version, identity_fixed):
    by_attempt = defaultdict(list)
    updates = []
    for v in range(version):
        for case in ('A','B'):
            for row in read_rows(batch_path(run,v,case)):
                by_attempt[row['attempt']].append(row)
        updates.append(json.loads((run/f'update-{v+1:03d}.json').read_text()))
    matches = []
    observed_cases = {}
    counts = defaultdict(int)
    rewards = {}
    for attempt,rows in by_attempt.items():
        folder = Path(rows[0]['source_dir'])
        examples,status = matched_reloads(folder,rows)
        matches.extend(examples)
        for name, number in status.items():
            counts[name] += number
        observed_cases[attempt] = rows[0]['case']
        # Require a completed natural episode inside the training ledger. A frozen continuation
        # following a sample cap contributes neither reward nor assessment data to this statistic.
        if rows[-1]['terminated'] or rows[-1]['truncated']:
            rewards[attempt] = sum(row['reward'] for row in rows)/sum(row['dt_s'] for row in rows)
    # One physical target may have appeared in multiple forecasts. Count it once per attempt.
    unique = {(m['attempt'],m['plant_target_id']):m for m in matches}
    detectable = [m for m in unique.values() if m['predicted_free_abs_max_m']>1.e-9 and
                  m['true_preload_free_abs_m']>1.e-9 and m['cross_foot']]
    acted = {c:sum(bool(r['execution'] and (r['execution']['effective'] or
        r['execution']['transfer_fraction_changed'])) for seq in by_attempt.values() for r in seq
        if r['case']==c) for c in ('A','B')}
    act_timing = {c:sum(bool(r['execution'] and r['execution']['effective'])
        for seq in by_attempt.values() for r in seq if r['case']==c) for c in ('A','B')}
    rewards_by_case = {c:[v for a,v in rewards.items() if observed_cases[a]==c] for c in ('A','B')}
    sem = {c:float(np.std(v,ddof=1)/np.sqrt(len(v))) if len(v)>1 else None for c,v in rewards_by_case.items()}
    trend = {c:float(np.mean(v[len(v)//2:])-np.mean(v[:len(v)//2])) if len(v)>1 else None
             for c,v in rewards_by_case.items()}
    uncertainty = any(s is not None and s>1e-6 for s in sem.values())
    improves = any(s is not None and s>1e-6 for s in trend.values())
    n = sum(len(rows) for rows in by_attempt.values())
    valid = (n==version*2*STEPS and len({r['decision_id'] for seq in by_attempt.values() for r in seq})==n
             and all(x['end_rewards_finite'] and x['optimizer_steps']==16 for x in updates))
    flags = dict(action_authority_both_cases=all(v>0 for v in acted.values()),
        training_cross_foot_history_and_measured_consequence=bool(detectable),credit_counts_valid=valid,
        protocol_identity_fixed=identity_fixed,training_trend_or_episode_uncertainty=bool(uncertainty or improves))
    return dict(distinct=n,head=updates[-1]['head'],episode_attempts=list(by_attempt),
        witnessed_transfer_or_liftoff=acted,witnessed_liftoff=act_timing,
        forecast_status_counts=dict(counts),matched_query_cell_rows=len(matches),
        unique_observed_target_loads=len(unique),unique_cross_foot_exceeding_1nm=len(detectable),
        distinct_matched_cases=sorted({m['case'] for m in detectable}),eligible=all(flags.values()),
        explicit_blind_informative=all(flags[k] for k in ('action_authority_both_cases',
        'training_cross_foot_history_and_measured_consequence','credit_counts_valid','protocol_identity_fixed')),
        eligibility=flags,episode_reward_per_s=rewards,within_case_episode_standard_error=sem,
        first_vs_second_half_return_rate=trend,
        scope='Group gate uses training data only. Physical e before load is offline evidence and never fed to the actor. Observed load/overlap does not establish a counterfactual timing benefit.'),matches



def baseline_handover_windows(root, original_tree):
    """Bound the positive timing-request opportunity on existing original E60 gate traces."""
    rel=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
    cases={}
    for case in ('A','B'):
        attempt=json.loads((root/rel/'w56/f'/f'F-{case}-r60/E60.json').read_text())['attempt']
        directory=original_tree/'outputs/terrain_memory/simulation'/rel.name/attempt
        path=directory/'w55-f/contact-events.jsonl'
        events=read_rows(path)
        committed=[r for r in events if r['kind']=='liftoff_committed']
        delay=np.asarray([r['time_s']-r['nominal_due_s'] for r in committed])
        cases[case]=dict(attempt=attempt,original_handover_events=len(committed),
            recorded_liftoff_lag_ms=dict(zip(('min','q25','median','q75','max'),
                (1e3*np.maximum(0,np.quantile(delay,[0,.25,.5,.75,1]))).tolist())),
            original_handover_precedes_latest_positive_request=int(np.sum(delay < .05-1e-9)),
            original_gate_event_path=str(path),historical_physics_replayed=False)
    return dict(cases=cases,new_training_transitions=0,
        scope='Uses actual r60 gate and request timestamps. The positive 50 ms endpoint would prohibit the earlier recorded handover by time alone. Closed-loop support, ramp, and new action effects require new physical trajectories.')


if __name__=='__main__':
    from s1_w57_learning import ROOT,atomic_json
    rdir=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57/r'
    rdir.mkdir(parents=True,exist_ok=True)
    result=baseline_handover_windows(ROOT,ROOT.parent/'MCP-RL-w56-r60')
    atomic_json(rdir/'baseline-handover-windows.json',result)
    print(json.dumps(result))
