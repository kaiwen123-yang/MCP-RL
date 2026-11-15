"""Read committed W49 JSON and derive control-review quantities; no physics or solve."""
from pathlib import Path
import json
import math
from collections import Counter

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('analysis.json'))
args = parser.parse_args()
REPO = args.repo.resolve()
RESULTS = REPO / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
W49 = RESULTS / 'w49'


def read(path):
    return json.loads(path.read_text())


task = read(W49 / 'task-b.json')
deploy = read(W49 / 'deployment.json')
build = read(W49 / 'build.json')
deployment = {x['label']: x for x in deploy['checks']}
finite = []
for event in task['events']:
    guided = event['guided']
    best = min(event['qualified_proposals'], key=lambda x: (x['risk'], x['J']))
    native = [d for d in event['call']['r34_decisions']
              if d['kind'] == 1 and d['source'].startswith('alpha_') and d['alpha'] > 0]
    d = deployment[event['label']]
    attempts = event['call']['search_attempts']
    first_valid = next(a for a in attempts if a['backward_complete'])
    finite.append(dict(
        event=event['label'], origin_s=event['origin_s'],
        baseline_J=guided['baseline_J'], selected_J=best['J'],
        decrease=guided['baseline_J'] - best['J'],
        decrease_percent=100 * (guided['baseline_J'] - best['J']) / guided['baseline_J'],
        selected_alpha=best['alpha'], selected_risk=best['risk'],
        strict_comparison_budget=guided['baseline_budget'] + best['budget'],
        decrease_to_budget=(guided['baseline_J'] - best['J']) / (guided['baseline_budget'] + best['budget']),
        guided_candidates=len(guided['candidates']),
        guided_qualified=len(event['qualified_proposals']),
        native_positive_candidates=len(native), native_positive_eligible=sum(d['eligible'] for d in native),
        native_rollouts=event['native_N_rollouts'], backward_attempts=event['call']['backward_attempts'],
        first_valid_backward_mu=first_valid['lambda_used'], guided_invocation_mu=guided['mu_before'],
        selected_max_action_difference_vs_baseline_rad=best['action_change_vs_baseline_rad'],
        selected_cost_parts=best['J_parts'],
        selected_support_height_cost_share=best['J_parts']['MaterialSupportHeight'] / best['J'],
        historical_exact_solve=event['historical_exact_solve'],
        deployment_duration_s=d['logical_s'],
        deployment_displacement_m=[f - i for f, i in zip(d['final_xyz'], d['initial_xyz'])],
        deployment_response_roundoff_rad=d['response_roundoff_rad'],
        deployment_metric_and_QP_valid=d['metric_and_QP_valid'],
    ))

original = {}
proposed_events = []
for side, ids in [('a', [237, 328]), ('b', [236, 625])]:
    path = next(RESULTS.glob(f'job-*w49-{side}*-resolved.json'))
    resolved = read(path)
    records = resolved['r29_recording']['event_calls']
    last = max(records, key=lambda x: x['call_record']['call_id'])
    b = last['boundary']
    c = last['call_record']
    parts = b['current_solve']['cost_components']
    summary = read(W49 / 'original-evidence' / path.name.replace('-resolved', ''))
    original[side] = dict(
        source=str(path.relative_to(REPO)), retained_event_calls=len(records), total_calls=summary['planner_calls'],
        retained_candidate_source_families=dict(Counter(r['call_record']['candidate_source'].split('_')[0] for r in records)),
        retained_all_metric_valid=all(r['call_record']['metric_valid'] for r in records),
        first_call_source=records[0]['call_record']['candidate_source'],
        first_call_origin_s=records[0]['call_record']['origin'],
        first_call_policy_id=records[0]['call_record']['installed_plan_id'],
        last_call_id=c['call_id'], last_actual_xyz=b['actual_state'][:3],
        last_cost=c['deployed_current_cost'],
        last_support_height_cost=parts['MaterialSupportHeight'],
        last_support_height_cost_share=parts['MaterialSupportHeight'] / c['deployed_current_cost'],
        last_support_height_rms_error_m=.04 * math.sqrt(2 * parts['MaterialSupportHeight']),
        last_position_cost=parts['Position'],
        last_position_cost_share=parts['Position'] / c['deployed_current_cost'],
        last_policy_generation_age_s=c['policy_age_s'],
        last_source=c['candidate_source'], last_stop_reason=c['stop_reason'],
        largest_retained_cost_recompute_error=max(abs(r['boundary']['current_solve']['cost_recompute_error']) for r in records),
        original_result=summary['original_result'],
        positive_installs=summary['positive_installs'],
        positive_same_problem_cost_descent=summary['positive_same_problem_cost_descent'],
        expected_frozen_reference_end_s_from_source=.35,
        expected_reference_age_at_last_call_s_from_source=c['origin'] - .35,
        reference_age_evidence='Source invariant and retained first bootstrap imply this; NPZ arrays not downloaded or directly measured.',
    )
    for event in records:
        call = event['call_record']
        if call['call_id'] not in ids:
            continue
        boundary = event['boundary']
        proposed_events.append(dict(
            label=f'W49-{side.upper()}{call["call_id"]}',
            role='early_support_loss' if call['call_id'] in (237, 236) else 'low_posture_recovery',
            attempt=path.name.replace('-resolved.json', ''),
            call_id=call['call_id'], origin_s=call['origin'], actual_xyz=boundary['actual_state'][:3],
            J=call['deployed_current_cost'], risk=call['deployed_current_trunk_violation'],
            support_height_cost=boundary['current_solve']['cost_components']['MaterialSupportHeight'],
            candidate_source=call['candidate_source'], objects=event['objects'],
            snapshot_scope=boundary['mission_actor']['external_snapshot_scope'],
        ))

out = dict(
    audit_base_commit='de6606dc8363e8ab1c816fa682cace752a06318a',
    comparison_commit='b142001624cfea5877c57e63f45e0f29c9dc4735',
    scope='Committed JSON arithmetic and source review only. Zero new physics/FD/solve/PPO/GPU work; no D57 access; no raw-archive download.',
    finite=finite,
    finite_totals=dict(
        guided_candidates=sum(x['guided_candidates'] for x in finite),
        guided_qualified=sum(x['guided_qualified'] for x in finite),
        native_positive_candidates=sum(x['native_positive_candidates'] for x in finite),
        native_positive_eligible=sum(x['native_positive_eligible'] for x in finite),
        native_solves=task['native_solves'], recorded_work=task['work'],
        deployment_substeps=deploy['physics_substeps'],
        maximum_deployment_roundoff_rad=max(x['deployment_response_roundoff_rad'] for x in finite),
    ),
    build_boundary=dict(finite_binary=task['binary_sha256'], formal_binary=build['binary_sha256'], explanation=build['finite_build_difference']),
    original=original,
    local_reference=dict(maximum_lead_m=.15, maximum_speed_m_s=.25, horizon_s=.35,
        derived_lead_time_s=.15 / .25 - .35,
        at_constant_0215_m_s_origin_lead_m=.215 * (.15 / .25 - .35),
        at_constant_0215_m_s_terminal_lead_m=.215 * (.15 / .25),
        interpretation='Analytic illustration of current r26 local reference, not measured raw W49 packet. The endpoint still bounds the local target.'),
    proposed_w50_events=proposed_events,
    missing_raw_evidence=['W49 complete-policy NPZ arrays', 'W49 all-call native planner-calls.jsonl',
        'W49 per-candidate full horizons and derivative arrays outside retained event identities'],
)
args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
print(json.dumps({'output': str(args.output), 'finite_totals': out['finite_totals'],
                  'selected_events': [e['label'] for e in proposed_events]}, indent=2))
