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
    cases=[(t,m) for t in terrains for m in ('C0','C1','C2','C3')]+[('tm02_anchor','C4')]
    matrix=np.full((17,5),np.nan)
    durations=np.zeros((17,5))
    for i,(terrain,method) in enumerate(cases):
        for r in [r for r in rows if r['terrain']==terrain and r['controller']==method]:
            code=(0 if r['success'] else 1 if r['termination']=='controller_no_feasible_intent' else
                  2 if r['material_domain_exit'] is not None else 3 if r['integration_error'] is not None else
                  4 if r['termination']=='task_timeout' else 5)
            matrix[i,r['seed']]=code
            durations[i,r['seed']]=r['duration_s']
    palette=['#009E73','#CC79A7','#D55E00','#333333','#999999','#E69F00']
    labels=['Completed','No feasible intent','Material domain','Integration failure','Task timeout','Body/contact failure']
    fig,ax=plt.subplots(figsize=(7.2,6.4),constrained_layout=True)
    ax.imshow(matrix,cmap=ListedColormap(palette),vmin=-.5,vmax=5.5,aspect='auto')
    for i,j in np.argwhere(np.isfinite(matrix)):
        ax.text(j,i,f'{durations[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j] in (0,2,3) else 'black')
    ax.set(yticks=range(17),yticklabels=[t+' / '+m for t,m in cases],xticks=range(5),
           xticklabels=[str(i) for i in range(5)],xlabel='Paired optimizer seed',title='All 85 outcomes; cell labels are observed duration in seconds')
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
    for terrain,method in [('tm02_anchor','C3'),('tm02_anchor','C4'),('kv_soft','C0'),('kv_soft','C3')]:
        row=next(r for r in rows if r['terrain']==terrain and r['controller']==method and r['seed']==0)
        stem=f'{terrain}-{method}-seed-0'
        events=json.loads((out/cohort/(stem+'-events.json')).read_text())['reference']
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
    index_path=small/'index.json'
    index=json.loads(index_path.read_text())
    index['figures']=[str(p.relative_to(small)) for p in sorted(figdir.glob('*.png'))]
    index_path.write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',default='formal-v1')
    p.add_argument('--figures',action='store_true')
    p.add_argument('--source-output-root',default='/home/fanyuan/projects/kaiwen-learning/terrain-memory-study/outputs/terrain_memory/reproduction/material/tm02-memory-study-20261023')
    args=p.parse_args()
    config,small,out,rows=summarize(args.cohort,args.source_output_root)
    if args.figures:
        figures(config,small,out,rows,args.cohort)
