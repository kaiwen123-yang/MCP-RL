"""Summarize family-separated W55 material evidence without assigning robot qualification."""
import json
import numpy as np
from s1_w55_c_data import COMPACT,RAW,ROOT,WEIGHTS,REVIEW,write

def read(name):
    return json.loads((COMPACT/name).read_text())


def key(row):
    return ','.join(str(i) for i in row['key'])


def summary(rows,groups):
    rel=[r['recovery_error_over_peak'] for r in rows if r['unloaded_reference_peak_um']>=1.]
    fast=[r['recovery_error_over_peak'] for r in rows if r['unloaded_reference_peak_um']>=1. and r.get('max_continuous_loading_rate_per_s',0.)>=1.]
    familystats={}
    for r in rows:
        if r['unloaded_reference_peak_um']>=1.:
            fam=tuple(groups.get(key(r),[key(r)]))
            familystats.setdefault(fam,[]).append(r['recovery_error_over_peak'])
    return dict(world_cells=len(rows),stress11_abs_max_MPa=max(r['stress_abs_max_MPa'] for r in rows),
                normal_force_single_cell_abs_max_N=max(r['normal_force_abs_max_N'] for r in rows),
                unloaded_compression_abs_max_um=max(r['unloaded_recovery_error_um'] for r in rows),
                resolved_relative_count=len(rel),resolved_peak_relative_max=float(max(rel)) if rel else None,
                resolved_equal_family_mean_relative=float(np.mean([np.mean(v) for v in familystats.values()])) if familystats else None,
                fast_resolved_count=len(fast),fast_resolved_peak_error_ratio=float(max(fast)) if fast else None)


def main():
    split=read('disjoint-holdouts.json')['cohorts']
    fit=read('fit.json');hist=read('historical-result.json');comparison=read('cpu-gpu-comparison.json')
    cases={}
    for name in ('N44','F51','N48','F52','B46','F53'):
        src=read(name+'-inventory.json')
        hold=split.get(name)
        models={}
        for tag in ('frozen','candidate'):
            calc=read(f'{tag}-{name}.json')
            group_lookup=hold['all_families'] if hold else {}
            allowed=set(hold['kept_complete_history_keys']) if hold else set(group_lookup)
            models[tag]={clock:summary([r for r in calc[clock] if not hold or key(r) in allowed],group_lookup) for clock in ('fine','observer')}
        cases[name]=dict(role=src['role'],ever_loaded=src['ever_loaded'],added_or_held_out_cells=src['new_keys'],
                         previously_reviewed_cells=src['previously_audited'],accepted_last_material_time_s=src['final_applied_tick']*.0002,
                         event_selected=src.get('physical_event_selection',{}),models=models,reference_replay_published_rows=src['original_rows_checked'],
                         max_reference_published_force_N=src['max_original_e_q_stress_loaded_force_error'][-1],
                         **{k:calc[k] for k in ('fine_vs_endpoint_sampling','same_endpoint_2ms_vs_02ms_integration','combined_sampling_and_dt')})
    rows=hist['target_paths'];hist_summary={}
    for family in sorted({r['family'] for r in rows.values()}):
        rr=[r for r in rows.values() if r['family']==family]
        complete=[r for r in rr if r['candidate_value'] is not None]
        hist_summary[family]=dict(n=len(rr),candidate_complete=len(complete),metric=rr[0]['metric'],
            frozen_mean=float(np.mean([r['archived_frozen_value'] for r in rr])),
            candidate_mean=float(np.mean([r['candidate_value'] for r in complete])) if complete else None,
            candidate_improved=sum(r['candidate_value']<r['archived_frozen_value'] for r in complete))
    timing=[d for r in fit['iterations'] for d in r['GPU']]
    r53=read('F53-inventory.json')
    from pathlib import Path
    decisions=json.loads((Path(r53['source'])/'pann-action-dependence.json').read_text())
    output=dict(status='offline_candidate_assessed',full_path_count_W54_preexisting=9,additional_W54_cells=366,
        held_out_W46B_seed2_cells=18,held_out_novel_W55_F53_cells=5,weights_baseline_sha256=fit['source_weights_sha256'],
        weights_candidate_sha256=fit['candidate_SHA256'],candidate_file='W55-short-load-ideal-potentials.npz',
        source_episodes=list(cases),cases=cases,group_split='disjoint-holdouts.json',
        training=dict(real_paths=fit['real_paths'],real_event_families=fit['real_event_families'],synthetic_paths=len(fit['synthetic_conditions']),
                      initial_loss=fit['initial_cost'],final_loss=fit['final_cost'],nfev=fit['optimizer_nfev'],
                      total_FD_objective_evaluations=fit['objective_including_difference_calls'],
                      positive_multipliers=fit['positive_multipliers'],source='fit.json'),
        gpu=dict(device='NVIDIA RTX PRO 6000 Blackwell Server Edition',fit_wall_s=fit['wall_s'],fit_cpu_s=fit['cpu_time_s'],
                 GPU_kernel_total_s=sum(d['gpu_kernel_s'] for d in timing),setup_and_input_s=sum(d['gpu_setup_H2D_s'] for d in timing),
                 output_s=sum(d['gpu_D2H_s'] for d in timing),measured_process_peak_MiB=fit['measured_process_memory_MiB_peak'],
                 max_explicit_allocated_MiB=max(d['explicit_gpu_bytes'] for d in timing)/2**20,usable_batch_sizes=sorted(set(d['paths'] for d in timing)),
                 paired_candidate_CPU_wall_s=comparison['cpu']['wall_s'],paired_candidate_GPU_wall_s=comparison['gpu']['wall_s'],
                 paired_candidate_offline_speedup=comparison['host_over_gpu_wall_speedup'],paired_case='F51',
                 fit_started_unix=fit['start_unix'],fit_ended_unix=fit['end_unix']),
        historical_F1_F6=hist_summary,historical_complete=sum(r['candidate_value'] is not None for r in rows.values()),
        historical_scheduled=len(rows),zero_state_recovery_times_s={r['model_id']:[r['free_recovery_volumetric_time_s'],r['free_recovery_deviatoric_time_s']] for r in hist['zero_state']},
        decision_path=dict(F53_E30_frozen_actor_cap_rows=len(decisions),max_observed_zero_features_instantaneous_delta_m_s=max(abs(r['with_pann_cap']-r['zero_feature_cap']) for r in decisions),
            candidate_used_in_robot=False,candidate_action_counterfactual_measured=False,closed_loop_memory_benefit_evaluated=False),
        old_replay='reviews/W54-review/pann-replay-summary.json', validation='All cell and family maxima are empirical on the indicated full replay inputs. The two endpoint comparisons separate change of boundary from change of numerical step; their maxima need not add. Selection uses no internal-state labels. The real floor and controller never receive the candidate.',
        new_robot_episodes=0,new_PPO_samples=0,new_PPO_updates=0,donor=0)
    write(COMPACT/'summary.json',output)
    print('Completed offline assessment:',len(rows),'historical paths and',len(cases),'episode cohorts.',flush=True)


if __name__=='__main__':main()
