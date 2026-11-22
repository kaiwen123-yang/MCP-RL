"""Apply the predeclared W57 M selection to paired measured episode forecasts."""
from collections import defaultdict
import gzip
import hashlib
import json

import numpy as np

from collect_s1_w57_material import OUT, ART, ROOT


def read(path):
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def metrics(rows,key):
    errors=np.array([x['errors'][key] for x in rows if key in x['errors']])
    if not len(errors):
        return None
    ab=np.abs(errors)
    return dict(n=len(errors),signed_mean=float(errors.mean()),mae=float(ab.mean()),rmse=float(np.sqrt(np.mean(errors**2))),
                abs_p50=float(np.quantile(ab,.50)),abs_p95=float(np.quantile(ab,.95)),abs_max=float(np.max(ab)))


def main():
    cases={c:read(OUT/f'{c}-held-forecasts.json.gz') for c in 'ab'}
    replays={c:{m:read(OUT/f'{c}-{m}-replay.json') for m in ('M0','M1')} for c in 'ab'}
    family_counts={}
    tables={}
    calibrations={}
    evidence={}
    for part in ('rows','release_to_reload'):
        table={}
        standard={}
        for c in ('b','a'):
            table[c]={};family_counts[part,c]={}
            for mode in ('M0','M1'):
                subset=[r for r in cases[c][part] if r['mode']==mode]
                table[c][mode]={}
                for family in sorted({r['family'] for r in subset}):
                    group=[r for r in subset if r['family']==family]
                    family_counts[part,c][family]=len(group)
                    table[c][mode][family]={}
                    for key in sorted({k for r in group for k in r['errors']}):
                        values=metrics(group,key)
                        table[c][mode][family][key]=values
                        if c=='b':
                            standard[mode,family,key]=values['abs_p95']
                        else:
                            band=standard.get((mode,family,key))
                            vals=np.abs([r['errors'][key] for r in group if key in r['errors']])
                            calibrations[part,mode,family,key]=None if band is None else dict(
                                absolute_B95_band=band,A_count=len(vals),A_covered=int((vals<=band).sum()),
                                A_coverage_fraction=float(np.mean(vals<=band)))
        tables[part]=table
    reference={}
    for c in 'ab':
        matches={m:{(r['event_id'],r['leg']):r['wbc'] for r in cases[c]['rows']
                 if r['mode']==m and r['wbc']} for m in ('M0','M1')}
        ids=sorted(matches['M0'].keys() & matches['M1'].keys())
        magnitude=np.array([abs(matches['M1'][i]['predicted_increment_m']-matches['M0'][i]['predicted_increment_m']) for i in ids])
        repeat=np.array([abs(matches['M0'][i]['predicted_increment_m']-matches['M0'][i]['archived_increment_m']) for i in ids])
        reference[c]=dict(count=len(ids),M0_vs_archived_r60_abs_p50_p95_max_m=
               np.quantile(repeat,[.5,.95,1]).tolist() if len(ids) else None,
               M1_M0_abs_p50_p95_max_m=np.quantile(magnitude,[.5,.95,1]).tolist() if len(ids) else None,
               count_above_both_W56_repeatability_and_this_M0_reconstruction=int(np.sum(magnitude>np.maximum(repeat,5.21e-17))))
        evidence[c]={}
        for mode in ('M0','M1'):
            s=[r for r in cases[c]['rows'] if r['mode']==mode]
            released=[r for r in s if r['family']=='release']
            future=[r for r in s if r['family']!='release']
            rr=[r for r in cases[c]['release_to_reload'] if r['mode']==mode]
            evidence[c][mode]=dict(nonrelease_foot_force=metrics(future,'future_force_N'),
                                    released_anchor_depth=metrics(released,'continuously_unloaded_anchor_depth_m'),
                                    future_conditional_sole_depth=metrics(future,'common_sole_sink_m'),
                                    release_to_reload_cell_force=metrics(rr,'next_cell_force_N'),
                                    release_to_reload_preload_depth=metrics(rr,'preload_depth_m'),
                                    next_anchor_tangent=metrics(rr,'next_cell_tangent_N_per_m'),
                                    published_model_error_reference_force=metrics(s,'reference_vs_recorded_force_N'),
                                    total_forward_event_rows=len(s),base_foot_contact_intervals=len({(r['leg'],r['base_foot_interval_tick']) for r in s}),
                                    release_to_reload_rows=len(rr))
    observability={}
    for c in 'ab':
        o=[r['aggregate_force_observability'] for r in cases[c]['rows'] if r['aggregate_force_observability']]
        obs=lambda name: np.quantile([v[name] for v in o],[0,.5,.95,1]).tolist()
        observability[c]=dict(positive_loaded_foot_resultants=len(o),rank_one=sum(v['local_force_rank']==1 for v in o),
            single_force_row_q_dimension_min_p50_p95_max=obs('prior_q_components'),
            local_nullity_min_p50_p95_max=obs('local_nullity'),singular_value_N_per_strain_min_p50_p95_max=obs('singular_value_N_per_unit_internal_strain'))
    a=evidence['a']
    conditions=dict(holdout_forward_foot_force_MAE_lower=a['M1']['nonrelease_foot_force']['mae']<a['M0']['nonrelease_foot_force']['mae'],
                    holdout_released_anchor_depth_MAE_not_higher=a['M1']['released_anchor_depth']['mae']<=a['M0']['released_anchor_depth']['mae'],
                    holdout_finite_future_WBC_reference_change_above_repeatability=reference['a']['count_above_both_W56_repeatability_and_this_M0_reconstruction']>0)
    selected='M1' if all(conditions.values()) else 'M0'
    compact=dict(schema='W57-M-held-forecast-v1',selected=selected,selection_conditions=conditions,
      split='B complete original r60 E60 to calibrate error bands; A complete original r60 E60 held for M0/M1 selection; historical cases, no claim of independent-domain generalization',
      n_independent_robot_episodes=2,reused_episodes={c:cases[c]['episode'] for c in 'ab'},new_robot_episodes=0,new_training_distinct=0,
      no_true_state_in_observer=True,primary_and_secondary=evidence,future_common_input_WBC_reference=reference,
      numerical_local_observability=observability,per_family_count={f'{a}/{b}':v for (a,b),v in family_counts.items()},
      force_rank='one effective shared creep/free duration per eligible owned sole; unobservable individual hidden states retain frozen prior',
      scoped_uncertainty='empirical absolute development-B 95th-percentile bands with held-A coverage by actual event family. No long recovery band available on B.',
      observer_refresh='force innovation in the final measured tick is assimilation fit only; 20.2ms open loop and source-release to real reload are held-future estimands',
      paired_legacy_M0={c:replays[c]['M0']['frozen_r60_nominal_parity'] for c in 'ab'},
      recorded_start_states=str(ART.relative_to(ROOT)),numeric_resources={c:read(OUT/f'{c}-forecast-resources.json') for c in 'ab'},
      replay_total_CPU_core_s=sum(r['cpu_core_s'] for c in replays.values() for r in c.values()),
      all_new_PANN_parameter_updates=0)
    compact['conditional_limits']={
        'heldout_source_release_to_cross_foot_reload_cell_force':{m:tables['release_to_reload']['a'][m]['cross_foot_revisit']['next_cell_force_N'] for m in ('M0','M1')},
        'heldout_short_forecast_cross_foot_whole_foot_force':{m:tables['rows']['a'][m]['cross_foot_revisit']['future_force_N'] for m in ('M0','M1')},
        'heldout_released_depth_tail':{m:{k:a[m]['released_anchor_depth'][k] for k in ('abs_p95','abs_max')} for m in ('M0','M1')},
        'A_coverage_of_B95_M1_cross_foot_force':calibrations['rows','M1','cross_foot_revisit','future_force_N'],
        'interpretation':'M1 improves the predeclared average short-horizon estimands. The source-release to actual cross-foot reload cell force worsens, and released-depth upper tail worsens. The retained A common-input incremental WBC reference changes subnanometrically; closed-loop effects require independent robot outcomes.'
    }
    replay_runs=[r for case in replays.values() for r in case.values()]
    wall=max(r['ended_unix_s'] for r in replay_runs)-min(r['started_unix_s'] for r in replay_runs)
    compact['replay_resources']=dict(comparison_processes=len(replay_runs),original_episode_seconds=120.,
        estimator_episode_seconds_excluding_legacy_parity=240.,additional_legacy_reference_seconds=120.,
        replay_process_span_wall_s=wall,mean_effective_process_cores=compact['replay_total_CPU_core_s']/wall,
        each_M0_process_also_computes_original_r60_for_every_publication=True,
        physical_robot_episodes_that_ran_in_this_material_lane=0)
    compact['initial_state_assets']=[dict(name=path.name,bytes=path.stat().st_size,
              sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(ART.glob('*-starts.json.gz'))]
    (OUT/'selection.json').write_text(json.dumps(compact,indent=2,allow_nan=False)+'\n')
    (OUT/'error-by-event.json').write_text(json.dumps(tables,indent=2,allow_nan=False)+'\n')
    (OUT/'error-band-calibration.json').write_text(json.dumps({'/'.join(k):v for k,v in calibrations.items()},indent=2,allow_nan=False)+'\n')
    print('Selected',selected,conditions)


if __name__=='__main__':
    main()
