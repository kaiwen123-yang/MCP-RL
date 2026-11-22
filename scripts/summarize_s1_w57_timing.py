"""Read completed frozen W57 episodes and report requested, gated and actual timing effects."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from s1_w57_learning import ROOT, SPEC, atomic_json, read_rows, restore
from s1_w57_timing import STATUSES, blind_observation
from s1_w57_training_evidence import matched_reloads


RUN = 'tm-s1-go2-local-readiness-r12'
SMALL = ROOT/'results/terrain_memory/simulation'/RUN/'w57'
LARGE = ROOT/'outputs/terrain_memory/simulation'/RUN


def distribution(data):
    a = np.asarray(data,dtype=float)
    if not len(a):
        return dict(n=0,mean=None,mean_abs=None,min=None,q05=None,median=None,q95=None,max=None)
    cuts = np.quantile(a,[0,.05,.5,.95,1])
    return dict(n=len(a),mean=float(a.mean()),mean_abs=float(np.abs(a).mean()),
                **{k:float(v) for k,v in zip(('min','q05','median','q95','max'),cuts)})


def unmasked_inputs(event):
    values = np.asarray(event['selected_observation'],dtype=np.float32)
    statuses = values[59:].reshape(4,6)
    for query in event['queries']:
        i = query['leg']
        values[24+3*i:27+3*i] = (np.asarray(query['free_recovery_m'])/.001
                                 if query['query_valid'] else np.zeros(3))
        statuses[i].fill(0.)
        statuses[i,STATUSES.index(query['status'])] = 1.
    return values


@torch.no_grad()
def frozen_actions(model,obs):
    mean = model.actor(torch.from_numpy(obs))[0][0].squeeze(-1)
    return .05*mean.tanh().cpu().numpy()


def summarize_episode(receipt, models):
    info = receipt['settings']
    arm = info['arm']
    case = receipt['case']
    attempt = receipt['attempt']
    folder = LARGE/attempt
    decisions = read_rows(folder/'timing-decisions.jsonl')
    transitions = read_rows(folder/'timing-transitions.jsonl')
    for row in transitions:
        row['case'] = case
    f_events = read_rows(folder/'w55-f/contact-events.jsonl')
    all_gate = read_rows(folder/'w55-f/handover-requests.jsonl')
    sampled_gate = [x for x in all_gate if x['stage']=='request' and x.get('sampled')]
    executed = [x for x in all_gate if x['stage']=='executed' and x.get('sampled')]
    censored = [x for x in all_gate if x['stage']=='censored' and x.get('sampled')]
    decision_by_number = {x['decision_number']:x for x in decisions}
    request_errors = [abs(r['requested_due_s']-decision_by_number[r['decision_id']]['requested_due_s'])
                      for r in sampled_gate if r['decision_id'] in decision_by_number]
    actual_target = {e['target_event_id']:e for e in f_events
                     if e['kind']=='actual_touchdown' and e.get('target_event_id')}
    exposure = Counter()
    target_lags = []
    query_to_touch = []
    publication_ages = []
    actual_targets = set()
    historical_actual_targets = set()
    historical_queries = 0
    state_deltas = []
    query_targets = set()
    for decision in decisions:
        diff = decision['full_minus_blind_deterministic_request_s']
        if diff is not None:
            state_deltas.append(diff*1000)
        for q in decision['queries']:
            exposure[q['status']] += 1
            publication_ages.append(1000*q['observer_publication_age_s'])
            if not q['query_valid']:
                continue
            eid = q['target_event_id']
            query_targets.add(eid)
            relevant = any(c['ever_loaded'] and not c['loaded'] for c in q['cells'])
            historical_queries += int(relevant)
            target = actual_target.get(eid)
            if target and target['time_s'] >= q['query_time_s']:
                actual_targets.add(eid)
                query_to_touch.append(target['time_s']-q['query_time_s'])
                target_lags.append(target['time_s']-q['target_time_s'])
                if relevant:
                    historical_actual_targets.add(eid)
    frequencies = Counter()
    for row in executed+censored:
        frequencies.update(row['gate_blocks'])
    query_rows,physical_counts = matched_reloads(folder,transitions)
    true_targets = {(r['attempt'],r['plant_target_id']):r for r in query_rows}
    true_cross = [r for r in true_targets.values() if r['cross_foot']]
    actual_loads = [r for r in true_cross if min(r['predicted_free_abs_max_m'],r['true_preload_free_abs_m'])>1e-9]
    frozen = {}
    if decisions:
        full_x = np.stack([unmasked_inputs(r) for r in decisions])
        blind_x = np.stack([blind_observation(r) for r in full_x])
        if 'full' in models:
            inspection_full = frozen_actions(models['full'],full_x)
            full_mask = frozen_actions(models['full'],blind_x)
            frozen['same_full_weights_explicit_input_on_minus_off_ms'] = distribution((inspection_full-full_mask)*1000)
            if arm in ('R','PR'):
                installed = .05*np.array([r['action'] for r in decisions])
                frozen['installed_replay_abs_ms'] = distribution(np.abs(inspection_full-installed)*1000)
        if 'blind' in models:
            blind = frozen_actions(models['blind'],blind_x)
            if arm=='blind':
                installed = .05*np.array([r['action'] for r in decisions])
                frozen['installed_replay_abs_ms'] = distribution(np.abs(blind-installed)*1000)
            if 'full' in models:
                frozen['full_vs_independently_trained_blind_at_same_state_ms'] = distribution((inspection_full-blind)*1000)
    status_path=folder/'timing-summary.json'
    summary=json.loads(status_path.read_text()) if status_path.is_file() else None
    status=receipt.get('status')
    return dict(attempt=attempt,task=info['task_mode'],level=receipt['level'],case=case,arm=arm,
        material=info['material'],planner=info['planner'],head=info['timing_head'],status=status,
        natural_end_reason=receipt.get('reason'),physical_s=receipt.get('simulation_s'),
        checkpoint_sha256=receipt.get('timing_checkpoint_sha256'),runtime_timing_summary=summary,
        recorded_decisions=len(decisions),sampled_gate_requests=len(sampled_gate),
        action_gate_join_rows=len(request_errors),max_request_mismatch_s=max(request_errors,default=None),
        nonzero_requests=sum(abs(x['action'])>1e-12 for x in decisions),physical_execution=len(executed),
        all_original_gates_accepted=sum(bool(x['accepted_gate']['ready']) for x in executed),
        censored_unexecuted=len(censored),changed_handover_same_realized_gate_sequence=sum(x['effective'] for x in executed),
        unchanged_handover_ramp_only=sum(x['transfer_fraction_changed'] and not x['effective'] for x in executed),
        unchanged_handover_and_ramp=sum(not(x['effective'] or x['transfer_fraction_changed']) for x in executed),
        gate_block_ticks=dict(frequencies),
        requested_residual_ms=distribution([1000*(x['requested_due_s']-x['base_due_s']) for x in sampled_gate]),
        execution_minus_requested_ms=distribution([1000*(x['actual_liftoff_s']-x['requested_due_s']) for x in executed]),
        execution_minus_first_zero_eligible_ms=distribution([1000*(x['actual_liftoff_s']-x['first_zero_eligible_s'])
            for x in executed if x['first_zero_eligible_s'] is not None]),
        accepted_before_any_zero_gate_opportunity=sum(x['first_zero_eligible_s'] is None for x in executed),
        accepted_event_dt_s=distribution([r['dt_s'] for r in transitions]),
        history_query_foot_counts=dict(exposure),observer_publication_age_ms=distribution(publication_ages),
        past_unloaded_foot_queries=historical_queries,distinct_predicted_touchdown_ids=len(query_targets),
        distinct_confirmed_target_ids=len(actual_targets),distinct_historical_confirmed_target_ids=len(historical_actual_targets),
        query_to_actual_touch_s=distribution(query_to_touch),touch_forecast_signed_error_ms=distribution(np.asarray(target_lags)*1000),
        matched_query_cell_rows=len(query_rows),distinct_true_reloaded_cell_events=len(true_targets),
        unique_crossfoot_phy_loads=len(true_cross),crossfoot_gt_1nm=len(actual_loads),
        true_reload_gap_s=distribution([x['unloaded_gap_s'] for x in true_cross]),
        actual_load_N=distribution([x['actual_normal_N'] for x in true_cross]),
        physical_source_to_reload_s=distribution([x['source_to_reload_s'] for x in true_cross]),
        missing_target_or_clock_at_scoring={k:v for k,v in physical_counts.items() if k not in STATUSES},
        recorded_same_full_weights_explicit_input_on_minus_off_ms=distribution(state_deltas),
        independent_checkpoint_replay=frozen,
        new_PPO_transitions=sum(t['sampling']=='on_policy' for t in transitions))


def summarize(campaign, destination):
    models = {}
    for head in ('full','blind'):
        p = campaign/head/'frozen-evaluation.json'
        if p.is_file():
            checkpoint = json.loads(p.read_text())['checkpoint']
            models[head],_ = restore(Path(checkpoint),'cpu')
    results = []
    for p in sorted((SMALL/'attempts').glob('*.json')):
        receipt = json.loads(p.read_text())
        info = receipt['settings']
        if info['arm'] not in ('R','PR','blind') or info['task_mode']=='training':
            continue
        if receipt['status'] in ('declared','running'):
            continue
        if not (LARGE/receipt['attempt']/'timing-summary.json').is_file():
            results.append(dict(attempt=receipt['attempt'],status=receipt['status'],timing_evidence=None))
            continue
        results.append(summarize_episode(receipt,models))
    if not results:
        raise ValueError('No completed frozen R, PR or blind physical episode is available.')
    output = dict(schema=SPEC['identity'],episodes=results,
        effect_definition='Compare actual committed handover with the first eligible original-time handover on the same realized gate sequence. The matched independent policy replay describes actions at identical observations; outcome differences require separate physical episodes.',
        counts='Foot queries and physical world-cell loads are repeated measurements within one episode. Stable target IDs and actual plant load IDs are counted separately.',
        consequences='Actual cell-load matching uses positive continuous-footprint intersection at measured force boundary and the same confirmed target leg. A nonzero, measured subsequent load is not proof that the timing policy improved that load.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(destination,output)
    return results


def verify_saved_credit(run,version,streams):
    with np.load(run/f'credit-{version:03d}.npz') as data:
        stored_adv = data['advantage']
        stored_returns = data['returns']
        interrupted = data['interrupted']
    actual = np.zeros((128,2))
    returns = np.zeros_like(actual)
    for col,seq in enumerate(streams):
        following = 0.
        for pos in reversed(range(128)):
            row = seq[pos]
            gamma = .99**(row['dt_s']/.5)
            lam = .95**(row['dt_s']/.5)
            bootstrap = 0. if row['terminated'] else row['next_value']
            delta = row['reward']+gamma*bootstrap-row['value']
            if row['terminated'] or row['truncated'] or row['collector_cut'] or interrupted[pos,col]:
                following = 0.
            following = delta+gamma*lam*following
            actual[pos,col]=following
            returns[pos,col]=following+row['value']
    return dict(shape=list(stored_adv.shape),
                max_abs_advantage_vs_independent_f64=float(np.max(np.abs(actual-stored_adv))),
                max_abs_return_vs_independent_f64=float(np.max(np.abs(returns-stored_returns))))


def training_actuals(campaign,output):
    heads={}
    for name in ('full','blind'):
        run=campaign/name
        if not (run/'run.json').is_file():
            continue
        rowids=set()
        duplicates=0
        versions=[]
        totals=Counter()
        samples=Counter()
        sources={}
        costs={}
        cut_intervals=[]
        for version in range(33):
            streams=[read_rows(run/f'batch-{version:03d}-{c}.jsonl') for c in ('A','B')]
            if not any(streams):
                continue
            for c,rows in zip(('A','B'),streams):
                samples[c]+=len(rows)
                for r in rows:
                    event=r['decision_id']
                    duplicates+=int(event in rowids)
                    rowids.add(event)
                    totals['on_policy']+=int(r['sampling']=='on_policy')
                    totals['dt_s']+=r['dt_s']
                    totals['terminated']+=r['terminated']
                    totals['episode_time_limits']+=r['truncated']
                    totals['collector_cuts']+=r['collector_cut']
                    actual=r['execution']
                    classification=('unexecuted' if actual is None else 'liftoff_changed' if actual['effective']
                        else 'ramp_only' if actual['transfer_fraction_changed'] else 'same_handover_and_ramp')
                    totals[classification]+=1
                    sources[r['attempt']]=Path(r['source_dir'])
            update_path=run/f'update-{version+1:03d}.json'
            if update_path.is_file() and all(len(s)==128 for s in streams):
                m=json.loads(update_path.read_text())
                started=(run/'run.json' if version==0 else run/f'update-{version:03d}.json').stat().st_mtime
                elapsed=update_path.stat().st_mtime-started
                versions.append(dict(input_version=version,output_version=version+1,
                    case_counts=[len(s) for s in streams],physical_logdensity_max_error=m['physical_max_logp_error'],
                    latent_logdensity_max_error=m['behavior_max_logp_error'],bounded_action_max_error=m['action_max_error'],
                    resource=m['device'],optimization_wall_s=m['optimization_wall_s'],optimization_cpu_s=m['optimization_cpu_s'],
                    optimizer_steps=m['optimizer_steps'],peak_gpu_allocated_bytes=m['peak_gpu_bytes'],
                    actual_256_row_wall_s=elapsed,distinct_rows_per_wall_h=256*3600/elapsed,
                    **verify_saved_credit(run,version,streams)))
            elif any(streams):
                cut_intervals.append(dict(version=version,case_counts=[len(s) for s in streams],updated=False))
        query_count=Counter()
        query_publication_max=0.
        max_sample_delay=0.
        for attempt,folder in sources.items():
            receipt=SMALL/'attempts'/(attempt+'.json')
            facts=json.loads(receipt.read_text()) if receipt.is_file() else {}
            costs[attempt]=dict(status=facts.get('status'),simulation_s=facts.get('simulation_s'),
                physical_samples=facts.get('physical_samples'),user_cpu_s=facts.get('user_cpu_s'),
                system_cpu_s=facts.get('system_cpu_s'),pid=facts.get('pid'))
            for row in read_rows(folder/'timing-decisions.jsonl'):
                if row['decision_id'] not in rowids:
                    continue
                for q in row['queries']:
                    query_count[q['status']]+=1
                    query_publication_max=max(query_publication_max,q['observer_publication_age_s'])
                    max_sample_delay=max(max_sample_delay,q['publication_time_s']-row['sample_time_s'],
                                         q['observer_publication_s']-row['sample_time_s'])
        heads[name]=dict(real_distinct=len(rowids),repeated_identifiers=duplicates,case_counts=dict(samples),
            exposure_by_foot=dict(query_count),max_observer_age_s=query_publication_max,
            max_positive_future_publication_s=max_sample_delay,transitions=dict(totals),updates=versions,
            not_yet_updated=cut_intervals,immutable_identity=json.loads((run/'run.json').read_text())['identity'],
            natural_physical_attempts=costs,
            final_checkpoint=(json.loads((run/'frozen-evaluation.json').read_text())['checkpoint']
                              if (run/'frozen-evaluation.json').is_file() else None))
    if not any(x['real_distinct'] for x in heads.values()):
        raise ValueError('No genuine timing transition has closed yet.')
    count=sum(x['real_distinct'] for x in heads.values())
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,dict(training_heads=heads,total_distinct=count,old_W44_actor_updates=0,
        measured_throughput='Elapsed file timestamps between successive completed shared updates; includes collecting both real E60 streams and updating the learner. First interval starts at the frozen run declaration.'))
    return count


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--training',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.training:
        output=args.output or SMALL/'r'/args.campaign.name/'training-actuals.json'
        count=training_actuals(args.campaign,output)
        print(json.dumps(dict(real_distinct=count,output=str(output))))
    else:
        output=args.output or SMALL/'r/frozen-episode-actions.json'
        rows=summarize(args.campaign,output)
        print(json.dumps(dict(completed_physical_episodes=len(rows),output=str(output))))

