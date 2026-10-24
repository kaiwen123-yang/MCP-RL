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


def summarize(cohort):
    config = json.loads((ROOT/'experiments/terrain_memory/s1-go2-contact.json').read_text())
    small = ROOT/config['outputs']['compact_root']
    out = ROOT/config['outputs']['large_root']
    rows = [json.loads(p.read_text()) for p in sorted((small/cohort).glob('*-seed-*.json'))]
    metrics = ['duration_s', 'velocity_rmse_m_s', 'attitude_rms_rad', 'grf_rmse_N', 'max_indentation_mm',
               'joint_positive_mechanical_work_J', 'signed_robot_contact_work_J', 'positive_ground_contact_work_J', 'wall_s']
    flat = []
    for row in rows:
        record = {k:row[k] for k in ['terrain','controller','seed','success','termination', *metrics]}
        record.update(material_domain_exit=row['material_domain_exit'] is not None,
                      integration_failure=row['integration_error'] is not None,
                      predictor_domain_exit=row['predictor_domain_exit'] is not None,
                      patches=row['patches'], patch_revisits=row['revisits'], trajectory=row['trajectory'], sha256=row['sha256'])
        for part, value in row['timing'].items():
            if value:
                for key in ('p50_ms','p95_ms','max_ms','deadline_misses','count'):
                    record[part+'_'+key] = value[key]
        flat.append(record)
    write_csv(small/'episodes.csv', flat)
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
                    normal_strain_at_reload=event['e'][0], internal_normal_at_reload=event['q'][0],
                    internal_normal_at_release=release['q'][0] if release else None)
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
        write_csv(small/'actual-revisits.csv', event_rows)
    if coverage_rows:
        write_csv(small/'event-coverage.csv', coverage_rows)
    index_path = small/'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index.update(run=config['run_id'], cohort=cohort, completed_episodes=len(rows), fixed_budget=85,
        fixed_matrix_complete=len(rows)==85, successes=sum(r['success'] for r in rows), new_training_runs=0,
        hardware_activated=False, NX_timing_measured=False, config='experiments/terrain_memory/s1-go2-contact.json',
        groups=groups, episode_table='episodes.csv', paired_table='paired.csv', actual_revisits='actual-revisits.csv',
        event_coverage='event-coverage.csv', compact_root=str(small), large_root=str(out),
        metric_interpretation='All observed pre-termination data are included. A failed duration is not a completion time; work over a shorter failure trajectory is not an efficiency benefit. Joint mechanical work is not battery consumption.',
        statistical_scope='Five paired optimizer seeds on one fixed scenario per material, not five independent materials or broad terrain generalization.',
        contact_model='World-fixed .02 m cells, exact .001 m2 interior square footprint partition; hard-priority whole-foot boundary assignment in a .0158114 m edge strip; original Go2 spherical geometry retained. Tangential friction is regularized Coulomb, not static friction.',
        physical_state_scope='TM02 gamma .025 is synthetic; KV parameters are published simulation scales, not Go2 calibration. Return-gap recovery does not establish long-horizon RL value.',
        source_weights=config['base']['frozen_predictor'], source_release=config['base']['material_release_url'],
        development=[dict(cohort=p.parent.name, terrain=d['terrain'], controller=d['controller'], seed=d['seed'],
                         termination=d['termination'], duration_s=d['duration_s'], trajectory=d['trajectory'], sha256=d['sha256'])
                     for p in sorted(small.glob('development-*/*-seed-*.json')) for d in [json.loads(p.read_text())]],
        episode_records=[str((small/cohort/f"{r['terrain']}-{r['controller']}-seed-{r['seed']}.json").relative_to(small)) for r in rows])
    index_path.write_text(json.dumps(index, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(completed=len(rows), successes=index['successes'], groups=groups)), flush=True)
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
        fig.savefig(figdir/(name+'.png'), dpi=220, bbox_inches='tight')
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
                ax.plot(a['time'][::5],y[::5],label=r['controller'],color=colors[j],lw=1)
        for ax,label in zip(axes,['Body x (m)','Planar speed (m/s)','Pitch (deg)','Total actual normal force (N)']):
            ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].legend(ncol=5,frameon=False,loc='upper right')
        axes[-1].set_xlabel('Physical time (s)')
        fig.suptitle(terrain+' | complete seed-0 records, ending at actual termination',fontsize=10)
        save(fig,'04-'+terrain+'-seed0-time-series')
    index_path=small/'index.json'
    index=json.loads(index_path.read_text())
    index['figures']=[str(p.relative_to(small)) for p in sorted(figdir.glob('*.png'))]
    index_path.write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',default='formal-v1')
    p.add_argument('--figures',action='store_true')
    args=p.parse_args()
    config,small,out,rows=summarize(args.cohort)
    if args.figures:
        figures(config,small,out,rows,args.cohort)
