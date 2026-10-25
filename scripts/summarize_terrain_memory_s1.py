#!/usr/bin/env python3
"""Summarize the fixed completed S1 matrix, including every failure."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(cohort, source_output_root):
    config = json.loads((ROOT/'experiments/terrain_memory/s1-go2-contact.json').read_text())
    small = ROOT/config['outputs']['compact_root']
    out = ROOT/config['outputs']['large_root']
    rows = [json.loads(p.read_text()) for p in sorted((small/cohort).glob('*-seed-*.json'))]
    envelope={}
    for path in sorted((Path(source_output_root)/'final_reference').glob('*.npz')):
        a=np.load(path)
        family=path.name.split('-')[0]
        item=envelope.setdefault(family,dict(histories=0,max_abs_component_stress_MPa=0.,max_abs_component_strain_rate_s_inv=0.))
        item['histories']+=1
        item['max_abs_component_stress_MPa']=max(item['max_abs_component_stress_MPa'],float(np.abs(a['stress']).max()))
        item['max_abs_component_strain_rate_s_inv']=max(item['max_abs_component_strain_rate_s_inv'],
            float(np.abs(np.diff(a['strain'],axis=0)/np.diff(a['time'])[:,None]).max()))
    stress_envelope=max(v['max_abs_component_stress_MPa'] for v in envelope.values())
    rate_envelope=max(v['max_abs_component_strain_rate_s_inv'] for v in envelope.values())
    for row in rows:
        physics=np.load(ROOT/row['material_trajectory'])['physics_foot_time_bottom_velocity_force_indent']
        row['max_indentation_500Hz_mm']=row.get('max_indentation_500Hz_mm',row['max_indentation_mm'])
        row['max_indentation_mm']=float(physics[:,37:41].max()*1000)
        row['max_geometric_foot_below_virgin_plane_mm']=float(np.maximum(-physics[:,1:13].reshape(-1,4,3)[:,:,2],0).max()*1000)
        row['max_indentation_source']='All 5000 Hz physical substeps; 500 Hz sampled peak is retained separately.'
        foot_velocity=physics[:,13:25].reshape(-1,4,3)
        foot_force=physics[:,25:37].reshape(-1,4,3)
        normal_power=foot_velocity[:,:,2]*foot_force[:,:,2]
        tangential_power=np.sum(foot_velocity[:,:,:2]*foot_force[:,:,:2],axis=2)
        dt=config['resolved_numerics']['physics_dt_s']
        row['positive_normal_ground_contact_work_J']=float(np.maximum(-normal_power,0).sum()*dt)
        row['signed_normal_robot_contact_work_J']=float(normal_power.sum()*dt)
        row['signed_tangential_robot_contact_work_J']=float(tangential_power.sum()*dt)
        if row['controller']=='C4' and row['first_material_call_since_episode_start_wall_s'] is not None:
            a=np.load(ROOT/row['trajectory'])
            cold_start=row['first_material_call_since_episode_start_wall_s']
            cold=(a['wall_start_s']<=cold_start)&(a['wall_start_s']+a['total_s']>=cold_start)
            row['material_cold_control_tick_time_s']=a['time'][cold].tolist()
            warm=~cold & ~a['compile_included']
            plans=json.loads((out/cohort/f"{row['terrain']}-{row['controller']}-seed-{row['seed']}-plans.json").read_text())
            # The last failed decision has no subsequent physics tick. Retain
            # its measured cost when computing steady intent latency.
            upper_warm=np.array([p['wall_s'] for p in plans if not any(abs(p['time_s']-t)<1e-9 for t in a['time'][cold])])
            for part,values,deadline in [('upper_steady',upper_warm,.2),
                                         ('predictor_steady',a['predictor_s'][warm],.002),
                                         ('total_steady',a['total_s'][warm],.002)]:
                if len(values):
                    row['timing'][part]=dict(count=len(values),p50_ms=float(np.median(values)*1000),
                        p95_ms=float(np.percentile(values,95)*1000),max_ms=float(np.max(values)*1000),
                        deadline_misses=int(np.sum(values>deadline)),deadline_s=deadline)
        if row['terrain']=='tm02_anchor':
            row['exceeds_R2_sampled_stress_envelope']=row['max_original_stress_MPa']>stress_envelope
            row['exceeds_R2_sampled_rate_envelope']=row['max_strain_rate_s_inv']>rate_envelope
            row['rate_to_R2_sampled_max_ratio']=row['max_strain_rate_s_inv']/rate_envelope
        name=f"{row['terrain']}-{row['controller']}-seed-{row['seed']}.json"
        (small/cohort/name).write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
    metrics = ['duration_s', 'velocity_rmse_m_s', 'attitude_rms_rad', 'grf_rmse_N', 'max_indentation_mm',
               'joint_positive_mechanical_work_J', 'signed_robot_contact_work_J', 'positive_ground_contact_work_J',
               'positive_normal_ground_contact_work_J','signed_normal_robot_contact_work_J','signed_tangential_robot_contact_work_J',
               'contact_slip_distance_m','angular_velocity_rmse_rad_s','torque_saturation_fraction','wall_s']
    flat = []
    for row in rows:
        record = {k:row[k] for k in ['terrain','controller','seed','success','termination', *metrics]}
        record.update(material_domain_exit=row['material_domain_exit'] is not None,
                      integration_failure=row['integration_error'] is not None,
                      predictor_domain_exit=row['predictor_domain_exit'] is not None,
                      patches=row['patches'], patch_revisits=row['revisits'], trajectory=row['trajectory'], sha256=row['sha256'])
        if row['terrain']=='tm02_anchor':
            record.update({key:row[key] for key in ('max_abs_principal_strain','max_original_stress_MPa','max_strain_rate_s_inv',
                'exceeds_R2_sampled_stress_envelope','exceeds_R2_sampled_rate_envelope','rate_to_R2_sampled_max_ratio')})
        for part, value in row['timing'].items():
            if value:
                for key in ('p50_ms','p95_ms','max_ms','deadline_misses','count'):
                    record[part+'_'+key] = value[key]
        flat.append(record)
    write_csv(small/'episodes.csv', flat)
    write_csv(small/'timing.csv',[dict(terrain=r['terrain'],controller=r['controller'],seed=r['seed'],part=part,**values)
              for r in rows for part,values in r['timing'].items() if values])
    groups = []
    for terrain in [x['id'] for x in config['terrains']]:
        for method in ('C0','C1','C2','C3','C4'):
            group = [r for r in rows if r['terrain']==terrain and r['controller']==method]
            if not group:
                continue
            row = dict(terrain=terrain, controller=method, episodes=len(group), successes=sum(r['success'] for r in group),
                task_failures=sum(not r['success'] and r['material_domain_exit'] is None and r['integration_error'] is None for r in group),
                material_domain_exits=sum(r['material_domain_exit'] is not None for r in group),
                integration_failures=sum(r['integration_error'] is not None for r in group),
                predictor_domain_exits=sum(r['predictor_domain_exit'] is not None for r in group))
            for metric in metrics:
                values = np.array([r[metric] for r in group])
                row[metric+'_mean_all_observed'] = float(values.mean())
                row[metric+'_min'] = float(values.min())
                row[metric+'_max'] = float(values.max())
            row['successful_completion_time_mean_s'] = float(np.mean([r['duration_s'] for r in group if r['success']])) if any(r['success'] for r in group) else None
            groups.append(row)
    write_csv(small/'groups.csv', groups)
    paired = []
    lookup = {(r['terrain'], r['controller'], r['seed']):r for r in rows}
    for terrain in [x['id'] for x in config['terrains']]:
        for a, b in [('C0','C1'), ('C0','C2'), ('C2','C3'), ('C3','C4')]:
            for seed in range(5):
                if (terrain,a,seed) not in lookup or (terrain,b,seed) not in lookup:
                    continue
                ra, rb = lookup[(terrain,a,seed)], lookup[(terrain,b,seed)]
                item = dict(terrain=terrain, reference=a, compared=b, seed=seed,
                    reference_success=ra['success'], compared_success=rb['success'],
                    success_difference=int(rb['success'])-int(ra['success']),
                    reference_failure=ra['termination'], compared_failure=rb['termination'],
                    completion_time_difference_s=rb['duration_s']-ra['duration_s'] if ra['success'] and rb['success'] else None,
                    observed_duration_difference_s=rb['duration_s']-ra['duration_s'])
                for metric in metrics[1:]:
                    item[metric+'_difference_observed'] = rb[metric]-ra[metric]
                paired.append(item)
    write_csv(small/'paired.csv', paired)
    event_rows, coverage_rows = [], []
    for row in rows:
        stem = f"{row['terrain']}-{row['controller']}-seed-{row['seed']}"
        events = json.loads((out/cohort/f'{stem}-events.json').read_text())['reference']
        releases = {}
        for event in events:
            if event['type']=='release':
                releases[event['patch']] = event
            if event['type']=='revisit':
                release = releases.get(event['patch'])
                item = dict(terrain=row['terrain'], controller=row['controller'], seed=row['seed'],
                    time_s=event['time_s'], patch=event['patch'], leg=event['leg'], previous_leg=event['previous_leg'],
                    cross_leg=event['leg']!=event['previous_leg'], unloaded_gap_s=event['unloaded_gap_s'],
                    overlap_m2=event['area_m2'], x_m=event['xy_m'][0], y_m=event['xy_m'][1],
                    normal_state_at_reload=event['e'][0], internal_state_at_reload=event['q'][0],
                    internal_state_at_release=release['q'][0] if release else None,
                    state_unit='strain' if row['terrain']=='tm02_anchor' else 'm' if row['terrain'].startswith('kv') else 'none')
                if item['cross_leg']:
                    event_rows.append(item)
        plan_path = out/cohort/f'{stem}-plans.json'
        if plan_path.exists():
            for decision in json.loads(plan_path.read_text()):
                chosen = decision['candidates'][decision['candidate_id']] if decision['candidate_id'] is not None else None
                predicted = chosen['covered_revisits'] if chosen else []
                matches = 0
                for pe in predicted:
                    target = decision['time_s']+pe['time_s']
                    xy = {tuple(np.round(v, 8)) for v in pe['patch_xy_m']}
                    if any(e['type']=='revisit' and e['leg']==pe['leg'] and abs(e['time_s']-target)<=.15 and
                           tuple(np.round(e['xy_m'], 8)) in xy for e in events):
                        matches += 1
                coverage_rows.append(dict(terrain=row['terrain'], controller=row['controller'], seed=row['seed'],
                    decision_time_s=decision['time_s'], candidate_id=decision['candidate_id'],
                    horizon_s=decision['horizon_s'], predicted_revisit_events=len(predicted),
                    matched_actual_events_within_150ms=matches,
                    selected_candidate_infeasible_steps=chosen['infeasible_steps'] if chosen else None,
                    controller_no_feasible_intent=chosen is None,wall_s=decision['wall_s']))
    if event_rows:
        write_csv(small/'actual-cross-leg-revisits.csv', event_rows)
    if coverage_rows:
        write_csv(small/'event-coverage.csv', coverage_rows)
    index_path = small/'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index.update(run=config['run_id'], cohort=cohort, completed_episodes=len(rows), fixed_budget=85,
        fixed_matrix_complete=len(rows)==85, successes=sum(r['success'] for r in rows), new_training_runs=0,
        hardware_activated=False, NX_timing_measured=False, config='experiments/terrain_memory/s1-go2-contact.json',
        groups=[{k:g[k] for k in ('terrain','controller','episodes','successes','task_failures','material_domain_exits',
            'integration_failures','predictor_domain_exits','successful_completion_time_mean_s','max_indentation_mm_max')} for g in groups],
        group_table='groups.csv',episode_table='episodes.csv', paired_table='paired.csv', actual_cross_leg_revisits='actual-cross-leg-revisits.csv',
        event_coverage='event-coverage.csv', compact_root=str(small), large_root=str(out),
        metric_interpretation='All observed pre-termination data are included. A failed duration is not a completion time; work over a shorter failure trajectory is not an efficiency benefit. Joint mechanical work is not battery consumption.',
        statistical_scope='Five paired optimizer seeds on one fixed scenario per material, not five independent materials or broad terrain generalization.',
        contact_model='World-fixed .02 m cells, exact .001 m2 interior square footprint partition; hard-priority whole-foot boundary assignment in a .0158114 m edge strip; original Go2 spherical geometry retained. Tangential friction is regularized Coulomb, not static friction.',
        physical_state_scope='TM02 gamma .025 is synthetic; KV parameters are published simulation scales, not Go2 calibration. Return-gap recovery does not establish long-horizon RL value.',
        event_scope='The compact CSV retains cross-leg patch revisits. Every load, release and revisit, including same-leg contact chatter, remains in each episode events.json in the trajectory assets. Patch events are not independent task conditions.',
        coverage_matching_window_s=.15,
        source_weights=config['base']['frozen_predictor'], source_release=config['base']['material_release_url'],
        R2_existing_evaluated_component_envelope=envelope,
        envelope_interpretation='Empirical maxima read from the 700 existing frozen R2 reference histories; not a continuous-domain validity guarantee. The separately specified S1 0.02 principal-strain limit is unchanged. No R2 integration or training was rerun.',
        work_quadrature='Contact work uses all 5000 Hz physical foot force/velocity samples. Positive joint work uses the recorded 500 Hz torque/velocity samples, with no battery-energy interpretation.',
        development=[dict(cohort=p.parent.name, terrain=d['terrain'], controller=d['controller'], seed=d['seed'],
                         termination=d['termination'], duration_s=d['duration_s'], trajectory=d['trajectory'], sha256=d['sha256'])
                     for p in sorted(small.glob('development-*/*-seed-*.json')) for d in [json.loads(p.read_text())]],
        episode_records=[str((small/cohort/f"{r['terrain']}-{r['controller']}-seed-{r['seed']}.json").relative_to(small)) for r in rows])
    index_path.write_text(json.dumps(index, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(completed=len(rows), successes=index['successes'], groups=index['groups'])), flush=True)
    return config, small, out, rows


def summarize_repair(config_path,cohort):
    config=json.loads((ROOT/config_path).read_text())
    small=ROOT/config['outputs']['compact_root']
    out=ROOT/config['outputs']['large_root']
    rows=[json.loads(p.read_text()) for p in sorted((small/cohort).glob('*-seed-*.json'))]
    metrics=['duration_s','velocity_rmse_m_s','attitude_rms_rad','grf_rmse_N','max_indentation_mm',
        'joint_positive_mechanical_work_J','signed_robot_contact_work_J','positive_ground_contact_work_J',
        'positive_normal_ground_contact_work_J','signed_normal_robot_contact_work_J','signed_tangential_robot_contact_work_J',
        'contact_slip_distance_m','angular_velocity_rmse_rad_s','torque_saturation_fraction','wall_s']
    flat,event_rows,rate_rows,tracking_rows,forecast_rows,touchdown_rows,rejection_rows,coverage_rows=[],[],[],[],[],[],[],[]
    time_corrections=[]
    dt=config['numerics']['physics_dt_s']
    for row in rows:
        material=np.load(ROOT/row['material_trajectory'])
        physics=material['physics_foot_time_bottom_velocity_force_indent']
        row['max_indentation_500Hz_mm']=row.get('max_indentation_500Hz_mm',row['max_indentation_mm'])
        row['max_indentation_mm']=float(physics[:,37:41].max()*1000)
        velocity=physics[:,13:25].reshape(-1,4,3)
        force=physics[:,25:37].reshape(-1,4,3)
        normal=velocity[:,:,2]*force[:,:,2]
        tangent=np.sum(velocity[:,:,:2]*force[:,:,:2],axis=2)
        row.update(positive_normal_ground_contact_work_J=float(np.maximum(-normal,0).sum()*dt),
            signed_normal_robot_contact_work_J=float(normal.sum()*dt),
            signed_tangential_robot_contact_work_J=float(tangent.sum()*dt))
        stem=f"{row['terrain']}-{row['controller']}-seed-{row['seed']}"
        identity={k:row[k] for k in ('terrain','controller','seed')}
        if not row.get('reused_from'):
            with np.load(ROOT/row['trajectory']) as archive:
                trace={key:archive[key] for key in archive.files}
            if 'stance_tracking_active' in trace:
                target_time=trace['time']+trace['stance_reference_effective_delay_s']
                for leg in range(4):
                    active=trace['stance_tracking_active'][:,leg]&(target_time<=trace['time'][-1])
                    item=dict(**identity,leg=leg,active_reference_samples=int(active.sum()),reference_effective_delay_s=.002)
                    if np.any(active):
                        times=target_time[active]
                        for reference,actual,label,scale in [('stance_reference_position','feet','position',1000.),
                                                          ('stance_reference_velocity','feet_velocity','velocity',1.)]:
                            measured=np.column_stack([np.interp(times,trace['time'],trace[actual][:,leg,j]) for j in range(3)])
                            error=trace[reference][active,leg]-measured
                            unit='mm' if scale==1000 else 'm_s'
                            item[label+'_rmse_'+unit]=float(np.sqrt(np.mean(error**2))*scale)
                            item[label+'_vertical_rmse_'+unit]=float(np.sqrt(np.mean(error[:,2]**2))*scale)
                        residual=trace['stance_inverse_force_residual_N'][active,leg]
                        item['inverse_forward_force_residual_rmse_N']=float(np.sqrt(np.mean(residual**2)))
                        item['inverse_forward_force_residual_max_abs_N']=float(np.abs(residual).max())
                        actual_normal=np.interp(times,physics[:,0],force[:,leg,2])
                        item['planned_actual_normal_force_rmse_N']=float(np.sqrt(np.mean((trace['planned_grf'][active,leg,2]-actual_normal)**2)))
                        for component in ('native_grf','inertia','tracking','bias'):
                            values=trace['stance_'+component+'_torque'][active,leg]
                            item[component+'_torque_rms_Nm']=float(np.sqrt(np.mean(values**2)))
                            item[component+'_torque_max_abs_Nm']=float(np.abs(values).max())
                    tracking_rows.append(item)
        record={k:row.get(k) for k in ('terrain','controller','seed','success','termination',*metrics)}
        record.update(reused=bool(row.get('reused_from')),material_domain_exit=row['material_domain_exit'] is not None,
            requested_stance_reference_domain_failure=row['termination']=='stance_reference_infeasible',
            stance_reference_integration_failure=row['termination']=='stance_reference_integration_failure',
            integration_failure=row['integration_error'] is not None,predictor_domain_exit=row['predictor_domain_exit'] is not None,
            trajectory=row['trajectory'],sha256=row['sha256'],patches=row['patches'],patch_revisits=row['revisits'],
            max_abs_principal_strain=row['max_abs_principal_strain'],max_total_strain_difference_rate_s_inv=row['max_strain_rate_s_inv'])
        for part,values in row['timing'].items():
            if values:
                record.update({part+'_'+key:values[key] for key in ('count','p50_ms','p95_ms','max_ms','deadline_misses')})
        flat.append(record)
        events_path=Path(row.get('events_path',out/cohort/f'{stem}-events.json'))
        events=json.loads(events_path.read_text())['reference']
        actual_cross=[e for e in events if e['type']=='revisit' and e['leg']!=e['previous_leg']]
        for event in actual_cross:
            event_rows.append(dict(**identity,event_id=event.get('event_id'),previous_load_event_id=event.get('previous_load_event_id'),
                time_s=event['time_s'],patch=event['patch'],leg=event['leg'],previous_leg=event['previous_leg'],
                unloaded_gap_s=event['unloaded_gap_s'],area_m2=event['area_m2'],normal_N=event.get('normal_N'),
                x_m=event['xy_m'][0],y_m=event['xy_m'][1],
                e_before_step=json.dumps(event['e_before_step']) if 'e_before_step' in event else None,
                q_before_step=json.dumps(event['q_before_step']) if 'q_before_step' in event else None,
                preload_free_e=json.dumps(event['preload_free_e']) if 'preload_free_e' in event else None,
                preload_free_q=json.dumps(event['preload_free_q']) if 'preload_free_q' in event else None,
                postload_e=json.dumps(event['e']),postload_q=json.dumps(event['q']),
                unique_event_diagnostics_available='event_id' in event))
        for metric,value in row.get('material_rate_extrema',{}).items():
            if value:
                rate_rows.append(dict(**identity,metric=metric,**{k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in value.items()}))
        plan_path=Path(row.get('plans_path',out/cohort/f'{stem}-plans.json'))
        plans=json.loads(plan_path.read_text()) if plan_path.exists() and not row.get('reused_from') else []
        if plans:
            prediction_deadline=1./config['control']['upper_intent_hz']
            row['prediction_first_call_including_possible_compilation_s']=float(plans[0]['wall_s'])
            row['prediction_first_call_physical_time_s']=float(plans[0]['time_s'])
            row['prediction_recording_period_s']=(1./config['resolved_execution']['forecast_nominal_recording_hz']
                if row['controller']=='B1' else prediction_deadline)
            for label,calls in [('prediction_first_call_including_possible_compilation',plans[:1]),
                                ('prediction_after_first_call',plans[1:])]:
                durations=np.array([p['wall_s'] for p in calls],dtype=float)
                values=(dict(count=len(durations),p50_ms=float(np.median(durations)*1000),
                    p95_ms=float(np.percentile(durations,95)*1000),max_ms=float(durations.max()*1000),
                    deadline_misses=int(np.sum(durations>prediction_deadline)),deadline_s=prediction_deadline)
                    if len(durations) else None)
                row['timing'][label]=values
                if values:
                    record.update({label+'_'+key:values[key] for key in ('count','p50_ms','p95_ms','max_ms','deadline_misses')})
            record['prediction_first_call_including_possible_compilation_s']=row['prediction_first_call_including_possible_compilation_s']
            record['prediction_first_call_physical_time_s']=row['prediction_first_call_physical_time_s']
            record['prediction_recording_period_s']=row['prediction_recording_period_s']
            physical_contacts=physics[:,25:37].reshape(-1,4,3)[:,:,2]>0
            physical_touchdowns=[np.flatnonzero(~physical_contacts[:-1,leg]&physical_contacts[1:,leg])+1 for leg in range(4)]
            globally_matched_actual=set()
            prediction_count,evaluable_prediction_count,matched_prediction_count=0,0,0
            actual_by_id={e['event_id']:e for e in actual_cross}
            for di,decision in enumerate(plans):
                candidates=decision.get('candidates',[])
                chosen=candidates[decision['candidate_id']] if decision.get('candidate_id') is not None else None
                forecast=[dict(sample) for sample in decision.get('aligned_trace',chosen.get('aligned_trace',[]) if chosen else [])]
                start=decision.get('start_time_s',decision['time_s'])
                correction=None
                first=chosen.get('first_rejection') if chosen else None
                if forecast and first and first['condition'] in ('material_domain_exit','soft_region_nonfoot_contact','invalid_contact'):
                    last=forecast[-1]
                    grf_time=last['grf_sample_time_s']
                    corrected=grf_time+dt
                    if abs(first['absolute_time_s']-grf_time)<=1e-9 and last['absolute_time_s']>corrected+1e-9:
                        original=last['absolute_time_s']
                        overcount=original-corrected
                        correction=dict(**identity,decision_time_s=start,candidate_id=decision['candidate_id'],
                            first_rejection_condition=first['condition'],first_rejection_absolute_time_s=first['absolute_time_s'],
                            last_grf_sample_time_s=grf_time,original_absolute_time_s=original,corrected_absolute_time_s=corrected,
                            original_predicted_duration_s=chosen['predicted_duration_s'],
                            derived_corrected_predicted_duration_s=chosen['predicted_duration_s']-overcount,
                            final_interval_overcount_s=overcount,original_rejected_cost=chosen['cost'],
                            rejected_cost_final_interval_overcount_uncorrected=True,
                            rejected_cost_used_for_ranking=False,raw_plans_path=str(plan_path))
                        time_corrections.append(correction)
                        last['original_absolute_time_s']=original
                        last['absolute_time_s']=corrected
                for ci,candidate in enumerate(candidates):
                    rejection=candidate.get('first_rejection')
                    if rejection:
                        rejection_rows.append(dict(**identity,decision_time_s=decision['time_s'],candidate_id=ci,
                            action_effective_time_s=candidate.get('action_effective_time_s'),first_rejection=json.dumps(rejection),
                            first_rejection_original_absolute_time_s=rejection['absolute_time_s'],
                            final_pose_time_corrected=bool(correction and ci==decision['candidate_id']),
                            corrected_final_pose_time_s=correction['corrected_absolute_time_s'] if correction and ci==decision['candidate_id'] else None))
                next_time=plans[di+1]['time_s'] if di+1<len(plans) else row['duration_s']
                samples=[s for s in forecast if trace['time'][0]<=s['absolute_time_s']<=trace['time'][-1] and
                    physics[0,0]<=s.get('grf_sample_time_s',s['absolute_time_s'])<=physics[-1,0]]
                matched_touchdowns=set()
                for previous,sample in zip(samples[:-1],samples[1:]):
                    for leg in range(4):
                        if sample['contact'][leg] and not previous['contact'][leg]:
                            when=sample['absolute_time_s']
                            candidates_actual=[int(k) for k in physical_touchdowns[leg] if
                                abs(physics[k,0]-when)<=.15 and (leg,int(k)) not in matched_touchdowns]
                            item=dict(**identity,decision_time_s=start,leg=leg,predicted_time_s=when,
                                forecast_segment=sample.get('prediction_segment'),matched=bool(candidates_actual))
                            if candidates_actual:
                                k=min(candidates_actual,key=lambda j:abs(physics[j,0]-when))
                                matched_touchdowns.add((leg,k))
                                actual_time=float(physics[k,0])
                                actual_foot=np.array([np.interp(actual_time,trace['time'],trace['feet'][:,leg,j]) for j in range(3)])
                                item.update(actual_contact_event_id=f'{leg}:{k}',actual_time_s=actual_time,
                                    time_error_s=when-actual_time,location_error_mm=float(np.linalg.norm(np.array(sample['feet'][leg])-actual_foot)*1000))
                            touchdown_rows.append(item)
                for window,limit in [('common_prefix',min(.2,next_time-start)),('short_from_same_long',.24),('long_tail',2.5)]:
                    subset=[s for s in samples if 0<s['absolute_time_s']-start<=limit+1e-9 and
                        (window!='long_tail' or s['absolute_time_s']-start>.24)]
                    item=dict(**identity,decision_time_s=start,window=window,samples=len(subset),horizon_s=decision['horizon_s'],
                        forecast_failure=bool(decision.get('failure') or (chosen and chosen.get('first_rejection'))),
                        decision_failure=decision.get('failure'),wall_s=decision['wall_s'],
                        derived_timestamp_corrections_used=sum('original_absolute_time_s' in sample for sample in subset))
                    if subset:
                        times=np.array([s['absolute_time_s'] for s in subset])
                        for field,label,scale in [('base_pos','position',1000.),('orientation','attitude',180/np.pi),
                                                 ('feet','feet',1000.),('actual_grf','actual_grf',1.),('planned_grf','planned_grf',1.)]:
                            actual=physics[:,25:37] if field=='actual_grf' else trace[field].reshape(len(trace['time']),-1)
                            measured_times=physics[:,0] if field=='actual_grf' else trace['time']
                            actual_times=np.array([s.get('grf_sample_time_s',s['absolute_time_s']) for s in subset]) if 'grf' in field else times
                            if field=='planned_grf':
                                held=np.searchsorted(measured_times,actual_times,side='right')-1
                                aligned=actual[held]
                            else:
                                aligned=np.column_stack([np.interp(actual_times,measured_times,actual[:,j]) for j in range(actual.shape[1])])
                            predicted=np.array([s[field] for s in subset]).reshape(len(subset),-1)
                            error=predicted-aligned
                            if field=='orientation':
                                error=np.arctan2(np.sin(error),np.cos(error))
                            unit='N' if 'grf' in field else 'deg' if field=='orientation' else 'mm'
                            item[label+'_rmse_'+unit]=float(np.sqrt(np.mean(error**2))*scale)
                            item[label+'_max_abs_'+unit]=float(np.max(np.abs(error))*scale)
                            if field=='base_pos':
                                item['height_rmse_mm']=float(np.sqrt(np.mean(error[:,2]**2))*1000)
                            if field=='orientation':
                                item['roll_rmse_deg']=float(np.sqrt(np.mean(error[:,0]**2))*180/np.pi)
                                item['pitch_rmse_deg']=float(np.sqrt(np.mean(error[:,1]**2))*180/np.pi)
                        item['exact_prefix_samples']=sum(s.get('prediction_segment')=='exact_prefix' for s in subset)
                        item['exact_tail_samples']=sum(s.get('prediction_segment')=='exact_tail' for s in subset)
                    forecast_rows.append(item)
                predicted=chosen.get('covered_revisits',[]) if chosen else []
                matched_here=set()
                evaluable_here=0
                for pe in predicted:
                    prediction_count+=1
                    when=pe.get('absolute_time_s',start+pe.get('time_s',0))
                    if when>row['duration_s']:
                        continue
                    evaluable_here+=1
                    evaluable_prediction_count+=1
                    patches=set(pe.get('patches',[]))
                    xy={tuple(np.round(v,8)) for v in pe.get('patch_xy_m',[])}
                    possible=[e for eid,e in actual_by_id.items() if eid not in matched_here and e['leg']==pe['leg'] and
                        abs(e['time_s']-when)<=.15 and
                        (tuple(np.round(e['xy_m'],8)) in xy if 'patch_xy_m' in pe else e['patch'] in patches)]
                    if possible:
                        match=min(possible,key=lambda e:abs(e['time_s']-when))
                        matched_here.add(match['event_id'])
                        globally_matched_actual.add(match['event_id'])
                        matched_prediction_count+=1
                coverage_rows.append(dict(**identity,decision_time_s=start,predicted_events=len(predicted),
                    observed_time_prediction_denominator=evaluable_here,unobserved_after_termination=len(predicted)-evaluable_here,
                    matched_predictions=len(matched_here),matched_actual_event_ids=json.dumps(sorted(matched_here)),
                    precision=len(matched_here)/evaluable_here if evaluable_here else None))
            coverage_rows.append(dict(**identity,decision_time_s=None,predicted_events=prediction_count,
                observed_time_prediction_denominator=evaluable_prediction_count,
                unobserved_after_termination=prediction_count-evaluable_prediction_count,
                matched_predictions=matched_prediction_count,unique_actual_cross_leg_denominator=len(actual_by_id),
                unique_matched_actual_cross_leg_events=len(globally_matched_actual),
                precision=matched_prediction_count/evaluable_prediction_count if evaluable_prediction_count else None,
                actual_cross_leg_recall=len(globally_matched_actual)/len(actual_by_id) if actual_by_id else None))
        (small/cohort/f'{stem}.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
    write_csv(small/'episodes.csv',flat)
    write_csv(small/'timing.csv',[dict(terrain=r['terrain'],controller=r['controller'],seed=r['seed'],reused=bool(r.get('reused_from')),part=part,**v)
        for r in rows for part,v in r['timing'].items() if v])
    groups=[]
    for terrain,method in sorted({(r['terrain'],r['controller']) for r in rows}):
        group=[r for r in rows if r['terrain']==terrain and r['controller']==method]
        item=dict(terrain=terrain,controller=method,episodes=len(group),successes=sum(r['success'] for r in group),
            reused=sum(bool(r.get('reused_from')) for r in group),
            material_domain_exits=sum(r['material_domain_exit'] is not None for r in group),
            requested_stance_reference_domain_failures=sum(r['termination']=='stance_reference_infeasible' for r in group),
            stance_reference_integration_failures=sum(r['termination']=='stance_reference_integration_failure' for r in group),
            integration_failures=sum(r['integration_error'] is not None for r in group),
            terminations=json.dumps({reason:sum(r['termination']==reason for r in group) for reason in sorted({r['termination'] for r in group})}))
        for metric in metrics:
            values=[r[metric] for r in group if r.get(metric) is not None]
            item[metric+'_mean_all_observed']=float(np.mean(values)) if values else None
            item[metric+'_max']=float(max(values)) if values else None
        successful=[r['duration_s'] for r in group if r['success']]
        item['successful_completion_time_mean_s']=float(np.mean(successful)) if successful else None
        groups.append(item)
    write_csv(small/'groups.csv',groups)
    paired=[]
    lookup={(r['terrain'],r['controller'],r['seed']):r for r in rows}
    for terrain in config['phase_A']['terrain_ids']:
        for a,b in [('B0','B1'),('B1','B2'),('B2','B3')]:
            for seed in config['phase_A']['paired_optimizer_seeds']:
                if (terrain,a,seed) not in lookup or (terrain,b,seed) not in lookup:
                    continue
                ra,rb=lookup[terrain,a,seed],lookup[terrain,b,seed]
                item=dict(terrain=terrain,reference=a,compared=b,seed=seed,reference_success=ra['success'],compared_success=rb['success'],
                    success_difference=int(rb['success'])-int(ra['success']),reference_failure=ra['termination'],compared_failure=rb['termination'],
                    reference_reused=bool(ra.get('reused_from')),compared_reused=bool(rb.get('reused_from')),
                    completion_time_difference_s=rb['duration_s']-ra['duration_s'] if ra['success'] and rb['success'] else None)
                item.update({metric+'_difference_observed':rb[metric]-ra[metric] if rb.get(metric) is not None and ra.get(metric) is not None else None for metric in metrics})
                paired.append(item)
    write_csv(small/'paired.csv',paired)
    for name,data in [('actual-cross-leg-revisits',event_rows),('material-rate-extrema',rate_rows),('stance-tracking',tracking_rows),
                      ('forecast-errors',forecast_rows),('forecast-touchdowns',touchdown_rows),
                      ('forecast-time-corrections',time_corrections),
                      ('first-candidate-rejections',rejection_rows),('event-coverage',coverage_rows)]:
        write_csv(small/(name+'.csv'),data)
    path=small/'index.json'
    index=json.loads(path.read_text())
    phase_a=[r for r in rows if r['controller'] in config['phase_A']['controller_ids']]
    phase_b=[r for r in rows if r['controller'] in config['phase_B']['controller_ids']]
    index.update(config=config_path,cohort=cohort,completed_episodes=len(rows),successes=sum(r['success'] for r in rows),
        reused_episodes=sum(bool(r.get('reused_from')) for r in rows),new_completed_episodes=sum(not bool(r.get('reused_from')) for r in rows),
        phase_A_records=len(phase_a),phase_A_complete=len(phase_a)==config['phase_A']['formal_result_records'],
        phase_B_records=len(phase_b),phase_B_complete=len(phase_b)==config['phase_B']['formal_result_records'],
        groups=groups,group_table='groups.csv',episode_table='episodes.csv',paired_table='paired.csv',timing_table='timing.csv',
        actual_cross_leg_revisits='actual-cross-leg-revisits.csv',material_rate_extrema='material-rate-extrema.csv',
        stance_tracking='stance-tracking.csv',stance_tracking_alignment='References generated at t are compared to actual feet and force at t + 0.002 s; inactive references are excluded and no missing old diagnostics are replaced with zero.',
        forecast_errors='forecast-errors.csv',forecast_touchdowns='forecast-touchdowns.csv',
        forecast_time_corrections='forecast-time-corrections.csv',forecast_time_corrections_count=len(time_corrections),
        first_candidate_rejections='first-candidate-rejections.csv',event_coverage='event-coverage.csv',
        episode_records=[str(p.relative_to(small)) for p in sorted((small/cohort).glob('*-seed-*.json'))],
        metric_interpretation='All observed data end at actual termination. Failed duration/work is not completion time/efficiency. Joint work is not battery consumption. Missing reused diagnostics remain absent.',
        prediction_timing_interpretation='The first saved prediction call includes possible compilation plus actual prediction work; it is not a measurement of compilation alone. Later saved calls provide separate P50/P95/max and deadline misses. The reported deadline is the configured 0.2 s planning target; B1 nominal diagnostic recording actually occurs once per 1 s. These wall times describe synchronous server simulation, not NX real time.',
        forecast_alignment='Interpolate recorded actual values at forecast absolute time; actual GRF uses full-rate physics samples at the explicit forecast GRF timestamp. Planned GRF uses zero-order hold at that timestamp, never interpolation across a command change. Samples beyond observed time are excluded. Short errors reuse the same long prediction. Common prefix ends at 0.2 s or the next planning record; longer tails are separate.',
        forecast_timestamp_correction='For a confirmed early-substep rejection (material_domain_exit, soft_region_nonfoot_contact or invalid_contact), a saved final pose timestamp that extends past the last GRF timestamp plus one 0.2 ms physics step is corrected only in derived comparison samples when the original first-rejection timestamp equals that GRF timestamp. The correction table retains original pose time, predicted duration, rejected cost and final-interval overcount. Raw plans and original rejected costs remain unchanged; missing endpoint fields prevent exact cost reconstruction. These infeasible candidate costs were not ranked. Physical task outcomes and trajectories are unchanged, and no task rerun is required for this metadata correction.',
        event_matching='Each predicted patch event matches at most one actual cross-leg event ID per forecast within 150 ms and the same world patch/leg. Precision uses prediction instances whose event time is observed before actual termination; later predictions remain unobserved. Recall counts unique actual event IDs over the whole episode, never rolling-window repetitions. Long-tail events remain conditional on the forecast future controls.',
        event_state_interpretation='e_before_step and q_before_step are the actual material state preceding the current step. preload_free_e and preload_free_q are the zero-external-stress candidate used by unilateral contact detection before imposing the new load; they are not a second actual integration or automatically an observed residual-memory state, especially during a contact-owner transfer. postload_e and postload_q are the resulting actual loaded state. Missing legacy B0 fields remain unrecorded.',
        material_rate_interpretation='Continuous loaded strain rates, activation/release strain jumps, actual foot vz/H and qdot are separate. Total-strain finite-difference peak is not an impact speed.',
        total_recorded_episode_wall_s=sum(r['wall_s'] for r in rows if not r.get('reused_from')),
        total_new_simulated_duration_s=sum(r['duration_s'] for r in rows if not r.get('reused_from')))
    path.write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=len(rows),successes=index['successes'],groups=groups)),flush=True)
    return config,small,out,rows


def figures(config, small, out, rows, cohort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    colors = ['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']
    figdir = small/'figures'
    figdir.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(figdir/(name+'.png'), dpi=300, bbox_inches='tight')
        fig.savefig(figdir/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
    terrains = [t['id'] for t in config['terrains']]
    fig, axes = plt.subplots(2,2,figsize=(7.2,5.4),constrained_layout=True)
    for ax, terrain in zip(axes.flat, terrains):
        group = [r for r in rows if r['terrain']==terrain]
        methods = sorted(set(r['controller'] for r in group))
        for j, method in enumerate(methods):
            rr = [r for r in group if r['controller']==method]
            for r in rr:
                ax.scatter(j+(r['seed']-2)*.06, r['duration_s'], color=colors[j], s=25,
                           marker='o' if r['success'] else 'x')
            ax.text(j, 61, f"{sum(r['success'] for r in rr)}/{len(rr)}", ha='center', fontsize=8)
        ax.set(xlim=(-.5,len(methods)-.5),ylim=(0,66),xticks=range(len(methods)),xticklabels=methods,
               title=terrain, ylabel='Observed mission duration (s)')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Every paired optimizer seed: circles = completed; crosses = failed\nLabels give completed / attempted; shorter failure is not faster completion', fontsize=10)
    save(fig,'01-all-mission-outcomes')
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cases=sorted({(r['terrain'],r['controller']) for r in rows})
    matrix=np.full((len(cases),5),np.nan)
    durations=np.zeros((len(cases),5))
    for i,(terrain,method) in enumerate(cases):
        for r in [r for r in rows if r['terrain']==terrain and r['controller']==method]:
            code=(0 if r['success'] else 1 if r['termination'].startswith('controller_') else
                  2 if r['material_domain_exit'] is not None else 3 if r['integration_error'] is not None else
                  4 if r['termination']=='task_timeout' else 6 if r['termination']=='stance_reference_infeasible' else
                  7 if r['termination']=='stance_reference_integration_failure' else 5)
            matrix[i,r['seed']]=code
            durations[i,r['seed']]=r['duration_s']
    palette=['#009E73','#CC79A7','#D55E00','#333333','#999999','#E69F00','#56B4E9','#0072B2']
    labels=['Completed','Controller decision failure','Actual material domain','Actual integration failure','Task timeout',
        'Body/contact failure','Stance reference infeasible','Stance inverse integration failure']
    fig,ax=plt.subplots(figsize=(7.2,6.4),constrained_layout=True)
    ax.imshow(matrix,cmap=ListedColormap(palette),vmin=-.5,vmax=7.5,aspect='auto')
    for i,j in np.argwhere(np.isfinite(matrix)):
        ax.text(j,i,f'{durations[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j] in (0,2,3,7) else 'black')
    ax.set(yticks=range(len(cases)),yticklabels=[t+' / '+m for t,m in cases],xticks=range(5),
           xticklabels=[str(i) for i in range(5)],xlabel='Paired optimizer seed',title=f'All {len(rows)} outcomes; labels are observed duration in seconds')
    ax.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(palette,labels)],loc='upper center',
              bbox_to_anchor=(.5,-.09),ncol=3,frameon=False,fontsize=7)
    save(fig,'01b-failure-types')
    for metric, label, name in [('max_indentation_mm','Peak indentation (mm)','02-indentation'),
                                ('joint_positive_mechanical_work_J','Observed positive joint work (J)','03-joint-work')]:
        fig, axes = plt.subplots(2,2,figsize=(7.2,5.2),constrained_layout=True)
        for ax, terrain in zip(axes.flat, terrains):
            group=[r for r in rows if r['terrain']==terrain]
            methods=sorted(set(r['controller'] for r in group))
            for j, method in enumerate(methods):
                for r in [r for r in group if r['controller']==method]:
                    ax.scatter(j+(r['seed']-2)*.06,r[metric],color=colors[j],marker='o' if r['success'] else 'x',s=25)
            ax.set(xticks=range(len(methods)),xticklabels=methods,title=terrain,ylabel=label)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y',alpha=.2)
        fig.suptitle('All five seeds and all failures; each terrain uses its own vertical scale',fontsize=10)
        save(fig,name)
    for terrain in terrains:
        group = [r for r in rows if r['terrain']==terrain and r['seed']==0]
        fig, axes = plt.subplots(4,1,figsize=(7.2,7),sharex=True,constrained_layout=True)
        for r in group:
            a=np.load(ROOT/r['trajectory'])
            j=int(r['controller'][1])
            for ax,y in zip(axes,[a['base_pos'][:,0],np.linalg.norm(a['velocity'][:,:2],axis=1),
                                  np.rad2deg(a['orientation'][:,1]),a['actual_grf'][:,:,2].sum(axis=1)]):
                ax.plot(a['time'][::5],y[::5],label=r['controller'],color=colors[j],lw=1,
                    linestyle=['-','--',':','-.','-'][j],marker='s' if j==4 else None,markevery=200,markersize=2)
        for ax,label in zip(axes,['Body x (m)','Planar speed (m/s)','Pitch (deg)','Total actual normal force (N)']):
            ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].legend(ncol=5,frameon=False,loc='upper right')
        axes[-1].set_xlabel('Physical time (s)')
        fig.suptitle(terrain+' | complete seed-0 records, ending at actual termination',fontsize=10)
        save(fig,'04-'+terrain+'-seed0-time-series')
    material_cases=([(t,m) for t,m in cases if t!='rigid' and m=='B1'] if 'phase_A' in config else
        [('tm02_anchor','C3'),('tm02_anchor','C4'),('kv_soft','C0'),('kv_soft','C3')])
    for terrain,method in material_cases:
        row=next(r for r in rows if r['terrain']==terrain and r['controller']==method and r['seed']==0)
        stem=f'{terrain}-{method}-seed-0'
        events=json.loads(Path(row.get('events_path',out/cohort/(stem+'-events.json'))).read_text())['reference']
        a=np.load(ROOT/row['material_trajectory'])
        material=terrain=='tm02_anchor'
        fig,axes=plt.subplots(4 if material else 2,1,figsize=(7.2,6.3 if material else 3.8),sharex=True,constrained_layout=True)
        loads=[e for e in events if e['type']!='release']
        if not loads:
            axes[0].text(.5,.5,'Episode terminated before material loading',ha='center',va='center',transform=axes[0].transAxes)
        else:
            keys={e['patch'] for e in loads}
            patch=min(keys,key=lambda k:(-len({e['leg'] for e in loads if e['patch']==k}),min(e['time_s'] for e in loads if e['patch']==k),k))
            xy=a['reference_patch_xy_m'][patch]
            def state_series(prefix,idx):
                offsets=a[prefix+'_offsets'];valid=np.diff(offsets)>idx
                times=a[prefix+'_time'][valid]
                flat=offsets[:-1][valid]+idx
                return times,a[prefix+'_e'][flat],a[prefix+'_q'][flat],a[prefix+'_stress_MPa'][flat],a[prefix+'_loaded'][flat]
            t,e,q,s,loaded=state_series('reference',patch)
            axes[0].plot(t,-e[:,0]*(50 if material else 1000),color=colors[0],label='Reference ground',lw=1)
            axes[0].set_ylabel('Surface depression (mm)')
            if material:
                axes[1].plot(t,q[:,0],color=colors[0],lw=1)
                axes[1].set_ylabel('Reference internal strain q11')
                axes[2].plot(t,-s[:,0]*.025*1000,color=colors[0],label='Reference ground',lw=1)
                axes[2].set_ylabel('Normal stress (kPa)')
            axes[-1].step(t,loaded.astype(int),where='post',color=colors[0],lw=.8)
            axes[-1].set(yticks=[0,1],yticklabels=['Free','Loaded'],ylabel='Patch contact')
            if method=='C4' and 'predictor_time' in a.files:
                candidates=np.flatnonzero(np.all(np.isclose(a['predictor_patch_xy_m'],xy,atol=1e-10),axis=1))
                if len(candidates):
                    tp,ep,qp,sp,lp=state_series('predictor',int(candidates[0]))
                    axes[0].plot(tp,-ep[:,0]*50,color=colors[4],ls='--',lw=.9,label='Frozen B own-state observer')
                    axes[2].plot(tp,-sp[:,0]*.025*1000,color=colors[4],ls='--',lw=.9,label='Frozen B own-state observer')
                    axes[2].legend(frameon=False,fontsize=7)
            axes[0].set_title(f'{terrain} / {method} / seed 0 | world patch ({xy[0]:.2f}, {xy[1]:.2f}) m',fontsize=9)
        for ax in axes:ax.grid(alpha=.2)
        axes[-1].set_xlabel('Physical time (s); saved material states at 500 Hz')
        save(fig,'05-'+terrain+'-'+method+'-material-history')
    if 'phase_A' in config:
        for row in [r for r in rows if r['controller']=='B1' and r['seed']==0 and r['terrain']!='rigid']:
            a=np.load(ROOT/row['trajectory'])
            physical=np.load(ROOT/row['material_trajectory'])['physics_foot_time_bottom_velocity_force_indent']
            fig,axes=plt.subplots(4,2,figsize=(9,8),sharex=True,constrained_layout=True)
            for leg in range(4):
                active=a['stance_tracking_active'][:,leg]
                effective=a['time']+a['stance_reference_effective_delay_s']
                reference=np.where(active,a['stance_reference_position'][:,leg,2]*1000,np.nan)
                axes[leg,0].plot(a['time'][::5],a['feet'][::5,leg,2]*1000,color=colors[0],label='Actual',lw=.8)
                axes[leg,0].plot(effective[::5],reference[::5],color=colors[1],ls='--',label='Material reference at effective time',lw=.8)
                axes[leg,1].plot(physical[::50,0],physical[::50,25+leg*3+2],color=colors[0],label='Actual GRF',lw=.8)
                axes[leg,1].plot(a['time'][::5],a['planned_grf'][::5,leg,2],color=colors[1],ls='--',label='MPC planned GRF',lw=.8)
                axes[leg,0].set_ylabel(['FL','FR','RL','RR'][leg]+' centre z (mm)')
                axes[leg,1].set_ylabel('Normal GRF (N)')
                for ax in axes[leg]:ax.grid(alpha=.2)
            for ax in axes[0]:ax.legend(frameon=False,fontsize=6)
            for ax in axes[-1]:ax.set_xlabel('Physical time (s)')
            fig.suptitle(row['terrain']+' / B1 / seed 0 | observed support references and actual execution',fontsize=10)
            save(fig,'06-'+row['terrain']+'-stance-tracking')
        forecast_path=small/'forecast-errors.csv'
        with forecast_path.open(newline='') as f:
            evidence=[r for r in csv.DictReader(f) if r['window']=='common_prefix' and r['controller']=='B1' and r['seed']=='0' and int(r['samples'])]
        if evidence:
            fig,axes=plt.subplots(3,1,figsize=(7.2,6),sharex=True,constrained_layout=True)
            for j,terrain in enumerate(terrains):
                group=[r for r in evidence if r['terrain']==terrain]
                for ax,metric,label in zip(axes,['roll_rmse_deg','height_rmse_mm','actual_grf_rmse_N'],['Roll RMSE (deg)','Height RMSE (mm)','GRF RMSE (N)']):
                    ax.plot([float(r['decision_time_s']) for r in group],[float(r[metric]) for r in group],label=terrain,color=colors[j],lw=.9,marker='.',ms=3)
                    ax.set_ylabel(label);ax.grid(alpha=.2)
            axes[0].legend(frameon=False,ncol=2,fontsize=7)
            axes[-1].set_xlabel('Forecast start in physical mission time (s)')
            fig.suptitle('Nominal-continuation forecast versus actual execution at the same absolute time\nB1 seed 0, common prefix through 0.2 s; every available forecast retained',fontsize=10)
            save(fig,'07-common-prefix-prediction-errors')
    index_path=small/'index.json'
    index=json.loads(index_path.read_text())
    index['figures']=[str(p.relative_to(small)) for p in sorted(figdir.glob('*.png'))]
    index_path.write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='experiments/terrain_memory/s1-control-repair.json')
    p.add_argument('--cohort',default='formal-v1')
    p.add_argument('--figures',action='store_true')
    p.add_argument('--source-output-root',default='/home/fanyuan/projects/kaiwen-learning/terrain-memory-study/outputs/terrain_memory/reproduction/material/tm02-memory-study-20261023')
    args=p.parse_args()
    repair='phase_A' in json.loads((ROOT/args.config).read_text())
    config,small,out,rows=(summarize_repair(args.config,args.cohort) if repair else summarize(args.cohort,args.source_output_root))
    if args.figures:
        figures(config,small,out,rows,args.cohort)
