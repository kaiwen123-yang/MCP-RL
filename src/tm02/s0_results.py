"""Actual task consequences and fixed-budget policies from saved S0 trajectories."""
from collections import Counter, defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .evaluate import write_json
from .s0 import catalog, collect

PEAK = 'max_remaining_pad_compression_m'
WORK = 'positive_remaining_contact_work_J'
TIME = 'completion_time_s'


def csv_rows(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def choose(rows, budget, tolerance, myopic=False):
    feasible = 'myopic_feasible' if myopic else 'remaining_feasible'
    peak = 'current_contact_peak_m' if myopic else PEAK
    work = 'myopic_positive_work_J' if myopic else WORK
    clock = 'myopic_completion_time_s' if myopic else TIME
    eligible = [r for r in rows if r['wait_s'] <= budget and r[feasible]]
    if not eligible:
        return None, [], None
    minimum = min(r[peak] for r in eligible)
    tied = [r for r in eligible if r[peak] <= minimum + tolerance]
    picked = min(tied, key=lambda r: (r[clock], r[work], r['candidate_id']))
    return picked, [r['candidate_id'] for r in tied], minimum


def summarize(config, out, compact):
    rows = collect(config, out)
    contexts, actions = catalog(config)
    tolerance = config['evaluation']['compression_tie_tolerance_m']
    budgets = config['evaluation']['extra_wait_budgets_s']
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['predictor'], r['context_id']].append(r)
    truth = {(r['context_id'], r['candidate_id']): r for r in rows if r['predictor'] == 'reference_continuous'}
    for group in grouped.values():
        eligible = [r for r in group if r['remaining_feasible']]
        for row in group:
            row['pareto'] = bool(row['remaining_feasible'] and not any(
                all(other[k] <= row[k] for k in (TIME, PEAK, WORK)) and
                any(other[k] < row[k] for k in (TIME, PEAK, WORK)) for other in eligible))
            real = truth[row['context_id'], row['candidate_id']]
            row['actual_reference_feasible'] = real['remaining_feasible']
            for key in (PEAK, WORK):
                row['reference_' + key] = real[key]
                row['error_' + key] = row[key] - real[key] if row[key] is not None and real[key] is not None else None
        complete = [r for r in group if r['error_' + PEAK] is not None]
        center_bias = float(np.mean([r['error_' + PEAK] for r in complete])) if complete else None
        for row in group:
            row['candidate_center_bias_m'] = center_bias
            row['centered_action_response_error_m'] = (row['error_' + PEAK] - center_bias
                if row['error_' + PEAK] is not None else None)
    csv_rows(compact / 'candidates.csv', rows)
    csv_rows(compact / 'contexts.csv', contexts)
    csv_rows(compact / 'actions.csv', actions)
    events = json.loads((out / 'events.json').read_text())
    csv_rows(compact / 'events.csv', [dict(context_id=e['context']['context_id'], candidate_id=e['action']['candidate_id'],
              decision_s=e['decision_s'], finish_s=e['finish_s'], **phase) for e in events for phase in e['events']])

    policies = []
    constants = []
    for gamma in config['context_grid']['material_scale_gamma']:
        ids = [c['context_id'] for c in contexts if c['gamma'] == gamma]
        for budget in budgets:
            options = []
            for action in actions:
                group = [truth[c, action['candidate_id']] for c in ids]
                feasible_count = sum(r['remaining_feasible'] and r['wait_s'] <= budget for r in group)
                complete = all(r[PEAK] is not None for r in group)
                item = dict(gamma=gamma, wait_budget_s=budget, candidate_id=action['candidate_id'],
                            allowed=action['wait_s'] <= budget, contexts=len(ids),
                            completed_feasible_contexts=feasible_count,
                            all_contexts_feasible=feasible_count == len(ids),
                            mean_peak_m=float(np.mean([r[PEAK] for r in group])) if complete else None,
                            mean_time_s=float(np.mean([r[TIME] for r in group])),
                            mean_work_J=float(np.mean([r[WORK] for r in group])) if complete else None,
                            selected_retrospective=False)
                constants.append(item)
                if item['all_contexts_feasible']:
                    options.append(item)
            if options:
                minimum = min(r['mean_peak_m'] for r in options)
                tied = [r for r in options if r['mean_peak_m'] <= minimum + tolerance]
                best = min(tied, key=lambda r: (r['mean_time_s'], r['mean_work_J'], r['candidate_id']))
                best['selected_retrospective'] = True
                best_id = best['candidate_id']
            else:
                best_id = None
            for cid in ids:
                true_group = grouped['reference_continuous', cid]
                oracle, oracle_ties, strict_min = choose(true_group, budget, tolerance)
                def append(policy, picked, tied=(), prediction_min=None):
                    actual = None if picked is None else truth[cid, picked['candidate_id']]
                    regret = (actual[PEAK] - strict_min if actual is not None and
                              actual['remaining_feasible'] and strict_min is not None else None)
                    policies.append(dict(context_id=cid, gamma=gamma, wait_budget_s=budget,
                        policy=policy, selected_id=None if picked is None else picked['candidate_id'],
                        decision_failure=picked is None,
                        actual_task_failure=actual is None or not actual['remaining_feasible'],
                        predicted_peak_m=None if picked is None else picked[PEAK],
                        actual_peak_m=None if actual is None else actual[PEAK],
                        actual_time_s=None if actual is None else actual[TIME],
                        actual_positive_work_J=None if actual is None else actual[WORK],
                        actual_signed_work_J=None if actual is None else actual['signed_remaining_contact_work_J'],
                        actual_current_peak_m=None if actual is None else actual['current_contact_peak_m'],
                        actual_hind_peak_m=None if actual is None else actual['hind_contact_peak_m'],
                        actual_return_peak_m=None if actual is None else actual['return_contact_peak_m'],
                        strict_reference_min_peak_m=strict_min, actual_compression_regret_m=regret,
                        regret_excess_over_tolerance_m=None if regret is None else max(0., regret-tolerance),
                        oracle_selected_id=None if oracle is None else oracle['candidate_id'],
                        oracle_near_tie_ids=oracle_ties, predicted_near_tie_ids=list(tied),
                        prediction_selection_minimum_m=prediction_min,
                        actual_trajectory=None if actual is None else actual['trajectory']))
                for predictor in config['predictors']:
                    picked, tied, minimum = choose(grouped[predictor['id'], cid], budget, tolerance)
                    append(predictor['id'], picked, tied, minimum)
                picked, tied, minimum = choose(true_group, budget, tolerance, myopic=True)
                append('myopic_oracle', picked, tied, minimum)
                for action in actions:
                    picked = truth[cid, action['candidate_id']] if action['wait_s'] <= budget else None
                    append(f"constant_{action['candidate_id']:02d}", picked)
                append('retrospective_best_constant', None if best_id is None else truth[cid, best_id])
    csv_rows(compact / 'constants.csv', constants)
    csv_rows(compact / 'policies.csv', policies)
    policy_map = {(p['policy'], p['context_id'], p['wait_budget_s']): p for p in policies}

    summaries = []
    summary_groups = defaultdict(list)
    for row in policies:
        summary_groups[row['gamma'], row['wait_budget_s'], row['policy']].append(row)
    for (gamma, budget, policy), group in summary_groups.items():
        succeeded = [r for r in group if not r['actual_task_failure']]
        summary = dict(gamma=gamma, wait_budget_s=budget, policy=policy, contexts=len(group),
            decision_failures=sum(r['decision_failure'] for r in group),
            actual_task_failures=sum(r['actual_task_failure'] for r in group),
            selected_id_counts=dict(Counter(str(r['selected_id']) for r in group)),
            successful_contexts=len(succeeded), statistics_scope='successful contexts; failures retained separately')
        for key in ('actual_peak_m', 'actual_time_s', 'actual_positive_work_J', 'actual_compression_regret_m'):
            values = [r[key] for r in succeeded if r[key] is not None]
            summary['mean_' + key] = float(np.mean(values)) if values else None
            summary['max_' + key] = float(np.max(values)) if values else None
        pairs = [(r, policy_map['reference_continuous', r['context_id'], budget]) for r in succeeded]
        pairs = [(r, o) for r, o in pairs if not o['actual_task_failure']]
        for key in ('actual_peak_m', 'actual_time_s', 'actual_positive_work_J'):
            differences = [r[key]-o[key] for r, o in pairs]
            summary['mean_excess_over_full_oracle_' + key] = float(np.mean(differences)) if differences else None
            summary['max_excess_over_full_oracle_' + key] = float(np.max(differences)) if differences else None
        summary['different_action_from_full_oracle'] = sum(r['selected_id'] != o['selected_id'] for r, o in pairs)
        summary['strictly_worse_than_full_oracle_beyond_tolerance'] = sum(r['actual_peak_m']-o['actual_peak_m'] > tolerance for r, o in pairs)
        summaries.append(summary)
    csv_rows(compact / 'policy-summary.csv', summaries)

    matched = []
    families = defaultdict(list)
    for ctx in contexts:
        families[ctx['gamma'], ctx['hind_N'], ctx['return_N'], ctx['return_gap_s']].append(ctx)
    matched_policies = [p['id'] for p in config['predictors']] + ['myopic_oracle', 'retrospective_best_constant']
    for family in families.values():
        for c1, c2 in itertools.combinations(family, 2):
            for budget in budgets:
                for policy in matched_policies:
                    a, b = (policy_map[policy, c['context_id'], budget] for c in (c1, c2))
                    row = dict(gamma=c1['gamma'], hind_N=c1['hind_N'], return_N=c1['return_N'],
                        return_gap_s=c1['return_gap_s'], policy=policy, wait_budget_s=budget,
                        context1=c1['context_id'], context2=c2['context_id'],
                        prefix_pad1=c1['prefix_pad'], prefix_pad2=c2['prefix_pad'],
                        prefix_gap1_s=c1['prefix_gap_s'], prefix_gap2_s=c2['prefix_gap_s'],
                        action1=a['selected_id'], action2=b['selected_id'],
                        changed_action=a['selected_id'] != b['selected_id'])
                    for key in ('actual_peak_m', 'actual_time_s', 'actual_positive_work_J', 'actual_compression_regret_m'):
                        row[key + '_difference_2_minus_1'] = b[key] - a[key] if a[key] is not None and b[key] is not None else None
                    # Transfer the action across histories while keeping future demands fixed.
                    # These counterfactuals reuse the same already-computed reference rows.
                    for source_row, destination, suffix in ((a, b, '1_to_2'), (b, a, '2_to_1')):
                        transferred = (None if source_row['selected_id'] is None else
                                       truth[destination['context_id'], source_row['selected_id']])
                        row['transferred_action_' + suffix + '_feasible'] = transferred is not None and transferred['remaining_feasible']
                        for key, reference_key in (('peak_m', PEAK), ('time_s', TIME), ('positive_work_J', WORK)):
                            value = transferred[reference_key] if transferred is not None else None
                            row['transferred_action_' + suffix + '_' + key] = value
                            actual = destination['actual_' + key]
                            row['transferred_action_' + suffix + '_excess_' + key] = value - actual if value is not None and actual is not None else None
                    matched.append(row)
    csv_rows(compact / 'history-matched.csv', matched)
    model_summaries = []
    for predictor in config['predictors']:
        for gamma in config['context_grid']['material_scale_gamma']:
            group = [r for r in rows if r['predictor'] == predictor['id'] and r['gamma'] == gamma]
            complete = [r for r in group if r['error_' + PEAK] is not None]
            model_summaries.append(dict(predictor=predictor['id'], gamma=gamma, tasks=len(group),
                numerical_failures=sum(not r['integration_complete'] for r in group),
                physical_remaining_failures=sum(not r['remaining_feasible'] for r in group),
                pareto_candidates=sum(r['pareto'] for r in group),
                mean_candidate_center_bias_mm=float(np.mean([r['candidate_center_bias_m'] for r in complete])) * 1000,
                peak_prediction_MAE_mm=float(np.mean([abs(r['error_' + PEAK]) for r in complete])) * 1000,
                centered_action_response_MAE_mm=float(np.mean([abs(r['centered_action_response_error_m']) for r in complete])) * 1000,
                worst_peak_prediction_error_mm=float(max(abs(r['error_' + PEAK]) for r in complete)) * 1000))
    csv_rows(compact / 'predictor-summary.csv', model_summaries)
    manifest = json.loads((out / 'run.json').read_text())
    scale_effects = []
    summary_map = {(s['gamma'], s['wait_budget_s'], s['policy']): s for s in summaries}
    for gamma in config['context_grid']['material_scale_gamma']:
        for budget in budgets:
            oracle = summary_map[gamma, budget, 'reference_continuous']
            constant = next((c for c in constants if c['gamma'] == gamma and c['wait_budget_s'] == budget and c['selected_retrospective']), None)
            item = dict(gamma=gamma, wait_budget_s=budget,
                        full_oracle_mean_peak_mm=oracle['mean_actual_peak_m']*1000,
                        full_oracle_mean_time_s=oracle['mean_actual_time_s'],
                        full_oracle_mean_positive_work_J=oracle['mean_actual_positive_work_J'],
                        retrospective_fixed_id=constant['candidate_id'] if constant else None,
                        history_matched_future_families=6,
                        history_families_with_action_changes=sum(len({policy_map['reference_continuous', c['context_id'], budget]['selected_id'] for c in family}) > 1
                            for family in families.values() if family[0]['gamma'] == gamma))
            for other in ('myopic_oracle', 'retrospective_best_constant'):
                summary = summary_map[gamma, budget, other]
                item[other + '_minus_full_mean_peak_mm'] = summary['mean_excess_over_full_oracle_actual_peak_m']*1000
                item[other + '_minus_full_max_peak_mm'] = summary['max_excess_over_full_oracle_actual_peak_m']*1000
                item[other + '_minus_full_mean_time_s'] = summary['mean_excess_over_full_oracle_actual_time_s']
                item[other + '_minus_full_mean_work_J'] = summary['mean_excess_over_full_oracle_actual_positive_work_J']
            cross_history = [r for r in matched if r['gamma'] == gamma and r['wait_budget_s'] == budget and r['policy'] == 'reference_continuous']
            penalties = [r['transferred_action_' + direction + '_excess_peak_m'] for r in cross_history for direction in ('1_to_2', '2_to_1')]
            item['worst_history_action_transfer_peak_penalty_mm'] = max(penalties)*1000
            item['directed_history_action_transfers_exceeding_tolerance'] = sum(p > tolerance for p in penalties)
            item['directed_history_action_transfers'] = len(penalties)
            scale_effects.append(item)
    csv_rows(compact / 'oracle-scale-effects.csv', scale_effects)
    ref_rows = list(truth.values())
    index = dict(run=config['run_id'], kind='Executed S0 frame-supported two-foot/two-coupon contact oracle study',
        scope='Synthetic homogeneous small-strain coupons; not a free quadruped gait, empirical soil calibration, motor energy, or hardware result.',
        complete=True, counts=dict(physical_contexts=len(contexts), candidates_per_context=len(actions),
            complete_two_pad_predictions=len(rows), reference_tasks=len(ref_rows), other_predictions=len(rows)-len(ref_rows),
            pad_histories=2*len(rows), numerical_failures=sum(not r['integration_complete'] for r in rows),
            reference_remaining_physical_failures=sum(not r['remaining_feasible'] for r in ref_rows),
            all_predictor_remaining_physical_failures=sum(not r['remaining_feasible'] for r in rows),
            new_training_runs=0, counting='72 deterministic contexts, not 6048 independent experiments; policy evaluations reuse reference trajectories.'),
        command='work/envs/tm02/bin/python scripts/run_terrain_memory_s0.py --config experiments/terrain_memory/s0-contact-oracle.json --run ' + config['run_id'],
        run_metadata=manifest, compact_directory=str(compact), output_directory=str(out),
        physical_mapping=dict(area_m2=config['physical_scope']['area_m2'], thickness_m=config['physical_scope']['thickness_m'],
            gamma=[1, 0.025], gamma_0025='Declared synthetic scaling psi*=gamma psi and phi*=gamma phi; integrate original sigma/gamma without changing weights or normalization.',
            material_state='Own full observed-prefix state; continuous both pads including unloaded recovery. Reset only in the explicitly marked planning ablation.',
            motion='Continuous piecewise-linear travel; 0.12 s raising/lowering, horizontal travel at <=0.5 m/s. Flight clearance is checked during clear flight/parking, with nonpenetration also checked during transitions.'),
        numerical_evidence=dict(max_force_compatibility_error_N=max(r['max_force_compatibility_error_N'] for r in rows),
            max_boundary_volume_work_difference_J=max(r['max_boundary_volume_work_difference_J'] for r in rows),
            max_reference_compression_mm=max(r['maximum_full_compression_m'] for r in ref_rows)*1000,
            max_horizontal_speed_m_s=max(r['maximum_horizontal_speed_m_s'] for r in rows),
            max_vertical_speed_m_s=max(r['maximum_vertical_speed_m_s'] for r in rows),
            minimum_flight_clearance_m=min(r['minimum_flight_clearance_m'] for r in rows),
            sum_task_worker_elapsed_s=sum(r['elapsed_s'] for r in rows)),
        tie_tolerance_m=tolerance, oracle_scale_effects=scale_effects,
        policy_summary=summaries, predictor_summary=model_summaries,
        observed_conclusions=[
            'All 864 reference candidates satisfy the declared contact/motion limits; there is no success-rate advantage in this grid.',
            'Full-horizon choices vary with history and future demands. Their physical peak-compression advantage over myopic and a single constant is quantified with its actual time/work cost in oracle-scale-effects.csv.',
            'The original-scale effects are sub-micrometre. Synthetic gamma=0.025 increases the observed effect but does not establish real terrain or quadruped locomotion benefit.',
            'The frozen noisy models retain substantial response bias while much of it is common across actions; material prediction error is not itself decision regret.',
            'This fixed S0 budget is complete. No further softening, expanded experiment, new training, RL, or hardware activation was performed.'],
        retrospective_constant_scope='One single ID per gamma and wait budget over all 36 corresponding contexts; chosen retrospectively, optimistic descriptive comparison, no held-out generalization.',
        artifact_files=['contexts.csv', 'actions.csv', 'events.csv', 'candidates.csv', 'constants.csv',
                        'policies.csv', 'policy-summary.csv', 'predictor-summary.csv', 'history-matched.csv', 'oracle-scale-effects.csv'],
        trajectory_schema=dict(time='s, dt=0.01', pads=['P', 'Q'], feet=['front', 'hind'],
            tensor_components=['11', '22', '33', '12', '13', '23'],
            motion_modes={'0':'contact', '1':'raising', '2':'clear flight or parking', '3':'lowering'},
            q='Each predictor own constitutive internal strain state; never injected from reference.',
            work='Trapezoidal force times material compression increments; positive increments summed separately, past prefix and final 2 s excluded from objectives.'),
        failures=[dict(predictor=r['predictor'], context=r['context_id'], candidate=r['candidate_id'],
                       failure=r['failure'], violations=r['violations']) for r in rows if not r['full_history_feasible']])
    if (compact / 'index.json').exists():
        previous = json.loads((compact / 'index.json').read_text())
        if 'figures' in previous:
            index['figures'] = previous['figures']
    write_json(compact / 'index.json', index)
    print(json.dumps(index['counts']), flush=True)
    return index
