"""Collect measured W57 saved-event planner, force integration and gait evidence."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57/p'


def lines(pattern):
    return [json.loads(l) for f in sorted(P.glob(pattern)) for l in f.open()]


def quant(values):
    return dict(zip(('min','p50','p95','max'),np.percentile(values,[0,50,95,100]).tolist())) if len(values) else None


def collect():
    cal=lines('contact-splits-*-M1.jsonl')
    measurements={(x['attempt'],x['mpc']):x['withheld'] for x in lines('withheld-boundaries-*-M1.jsonl')}
    valid=[x for x in cal if 'history_at_plant_step' in x]
    active=[x for x in valid if x['detached_keys']]
    diff={}
    for h in ('0.025','0.002'):
        values=[x['coarse_vs_plant_step'][h] for x in valid]
        diff[h]={k:quant([v[k] for v in values]) for k in ['sole_m','tangent_N_m','e','q','first_F_N']}
        diff[h]['winner_changes']=sum(v['winners'][0]!=v['winners'][1] for v in values)
    hist=[x['history_at_plant_step'] for x in active]
    received={h:{g:[] for g in ('all','measured_loaded_gt_1_N','measured_le_1_N')} for h in ('0.025','0.002','0.0002')}
    unique=set()
    for x in valid:
        targets=measurements[x['attempt'],x['mpc']]
        for h,arr in x['subsequent_force_errors_N'].items():
            for e,m in zip(arr,targets,strict=True):
                unique.add((x['attempt'],m['leg'],round(m['boundary_time_s'],7)))
                rec=received[h];rec['all'].append(e)
                rec['measured_loaded_gt_1_N' if m['measured_force_N']>1. else 'measured_le_1_N'].append(e)
    hold={h:{g:dict(n=len(a),absolute_error_N=quant(np.abs(a)),bias_N=float(np.mean(a))) for g,a in v.items()}
          for h,v in received.items()}
    law=dict(source_attempts=sorted({x['attempt'] for x in cal}),source_event_families=sum(len(x['source_events']) for x in cal),
             unique_mpc_states=len(cal),evaluated=len(valid),independent_episodes=2,intra_node_exact_contact_changes=sum(len(x['exact_planned_switches_s']) for x in cal),
             archived_contact_mismatches=sum(x['original_contact_node_mismatches'] for x in cal),
             max_archived_xy_disagreement_m=max(x['max_archived_xy_difference_m'] for x in cal),
             node_step_vs_0p2ms=diff,nonempty_private_interventions=len(active),
             full_minus_private_at_0p2ms={k:quant([v[k] for v in hist]) for k in ['sole_m','tangent_N_m','e','q','first_F_N']},
             private_history_winner_changes=sum(v['winners'][0]!=v['winners'][1] for v in hist),
             withheld_measured_boundary_force=hold,distinct_withheld_physical_boundary_leg_keys=len(unique),
             interpretation='Absolute arrival/response errors include initial state, prescribed future load, changing footprint and imperfect robot tracking. The withheld-force test uses the actual next measurement as an independent endpoint boundary. Solver inverse-to-forward residual is only numerical agreement.')
    modes={}
    for tag,pattern in [('M0','force-execution-?.jsonl'),('M1','force-execution-?-M1.jsonl')]:
        xs=lines(pattern);private=[x for x in xs if x['reset_keys']]
        modes[tag]=dict(states=len(xs),measured_nonempty_reset=len(private),
            original_or_counterfactual_mapped_vs_archived_rad=quant([x['nominal_mapped_vs_archive_rad'] for x in xs]),
            selected_first_mapped_vs_original_rad=quant([x['full']['first_applied_mapped_vs_nominal_rad'] for x in xs]),
            selected_first_mapped_full_vs_private_rad=quant([x['first_applied_full_vs_private_reset_rad'] for x in private]),
            selected_first_force_vs_original_N=quant([x['full']['first_force_vs_nominal_N'] for x in xs]),
            full_winners=dict(Counter(str(x['full']['winner']) for x in xs)),
            distinct_action_since_identical_model='Full vs private changes above repeat error tracked on protected, common initialized WBC; no new robot physics in these saved-input solves.',
            timing=dict(original_qp_s=quant([x['nominal_qp_s'] for x in xs]),
              both_qps_both_branches_s=quant([x['full']['all_candidate_wall_s'] for x in xs]),
              original_and_trial_and_whole_private_comparison_s=quant([x['full']['all_candidate_wall_s']+x['private_reset']['all_candidate_wall_s']-x['nominal_qp_s'] for x in private]),
              final_wbc_s=quant([x['full']['wbc_s'] for x in xs])))
    parallel={}
    for prefix in ('parallel-components','equal-components'):
        xs=lines(prefix+'-*.jsonl')
        times=[sum(x['timing'][w]['wall_s'] for x in xs) for w in ('1','4')]
        parallel[prefix]=dict(states=len(xs),states_and_objective_and_forces_bitwise_equal=sum(x['all_outputs_bitwise_equal'] for x in xs),
            serial_wall_s=times[0],parallel_wall_s=times[1],ratio=times[0]/times[1],
            serial_core_s=sum(x['timing']['1']['cpu_s'] for x in xs),parallel_core_s=sum(x['timing']['4']['cpu_s'] for x in xs))
    xs=lines('long-event-?.jsonl');matched=[x for x in xs if 'full' in x];changed=[x for x in matched if x['selected']!=0]
    low=dict(schema='W57-P2-event-load-distance-v1-saved',horizon_s=2.0,estimator='M0',distinct_material_support_liftoffs=len(xs),
             exactly_reproduced_nominal_gate=sum(x['nominal_gate_on_saved_path_s'] is not None and
               abs(x['nominal_gate_on_saved_path_s']-x['actual_next_liftoff_s'])<1e-8 for x in xs),
             matched_families=len(matched),no_matched_targets=len(xs)-len(matched),
             matched_nominal_minus50_equal_scores=sum(x['full']['scores'][1]['score_m']==x['full']['scores'][0]['score_m'] for x in matched),
             chosen_request_offsets_s=dict(Counter(str(x['offsets_s'][x['selected']]) for x in matched)),
             known_gate_tick_changes=sum(x['changed_handover_on_saved_prefix'] for x in changed),
             protected_private_history_winner_changes=sum(x['full']['winner']!=x['private_reset']['winner'] for x in matched),
             reduction_vs_nominal_on_changed_choices_um=quant([(x['full']['scores'][0]['score_m']-x['full']['scores'][x['selected']]['score_m'])*1e6 for x in changed]),
             scope='A later timing request changes both predicted recovery and world landing XY under the published common-force scenario. P2 has the same request permission as the new timing head. The archived selection effect does not isolate old detached material history. P2 omitted from B/P/R/PR.')
    result=dict(schema='W57_saved_planner_experiments_v1',law=law,force_to_wbc=modes,component_timing=parallel,
                low_frequency_event_layer=low,physical_and_joint_results='R12/w57/attempts, mechanics, paired-mechanisms and p/physical; tracked separately from saved-input experiments.')
    target=P/'saved-planning-summary.json';target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    v=collect();print(v['law']['unique_mpc_states'],v['low_frequency_event_layer']['matched_families'])
