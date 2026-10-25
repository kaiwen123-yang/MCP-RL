#!/usr/bin/env python3
"""Derive paired outcomes and contact/query evidence from actual complete tasks."""
import csv
import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'experiments/terrain_memory/s1-contact-transition.json').read_text())
SMALL=ROOT/C['outputs']['compact_root']
INDEX=SMALL/'index.json'
COLORS={'B1':'#0072B2','T1':'#D55E00','T2':'#009E73'}

def csv_write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (SMALL/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def grouped_revisits(row):
    events=json.loads(Path(row['events_path']).read_text())['reference']
    loads={e['event_id']:e for e in events if e['type'] in ('load','revisit')}
    with np.load(ROOT/row['material_trajectory']) as z:
        p=z['physics_foot_time_bottom_velocity_force_indent']
    contact=p[:,25:37].reshape(-1,4,3)[:,:,2]>0
    segment=np.cumsum(contact & ~np.vstack((np.zeros((1,4),bool),contact[:-1])),axis=0)
    grouped=defaultdict(dict)
    for e in events:
        if not (e['type']=='revisit' and e.get('unloaded_gap_s') is not None and e['unloaded_gap_s']>0 and e['normal_N']>0):
            continue
        prior=loads.get(e['previous_load_event_id'])
        if prior is None:
            continue
        now=min(np.searchsorted(p[:,0],e['time_s']),len(p)-1)
        old=min(np.searchsorted(p[:,0],prior['time_s']),len(p)-1)
        key=(int(e['previous_leg']),int(segment[old,e['previous_leg']]),int(e['leg']),int(segment[now,e['leg']]))
        # Multiple grid transitions in the same physical foot contact pair are
        # one pair; a patch contributes its greatest observed overlap once.
        if e['patch'] not in grouped[key] or e['area_m2']>grouped[key][e['patch']]['area_m2']:
            grouped[key][e['patch']]=e
    result=[]
    for (front,previous_segment,leg,current_segment),patches in grouped.items():
        es=list(patches.values())
        result.append(dict(terrain=row['terrain'],controller=row['controller'],seed=row['seed'],
            previous_leg=front,previous_foot_contact_segment=previous_segment,leg=leg,foot_contact_segment=current_segment,
            front_to_hind=front in (0,1) and leg in (2,3),unique_patches=len(es),
            unique_overlap_area_m2=sum(e['area_m2'] for e in es),
            summed_patch_reload_normal_N=sum(e['normal_N'] for e in es),
            force_scope='sum across patch reload events, not an instantaneous total',
            earliest_reload_s=min(e['time_s'] for e in es),latest_reload_s=max(e['time_s'] for e in es),
            minimum_unloaded_gap_s=min(e['unloaded_gap_s'] for e in es),maximum_unloaded_gap_s=max(e['unloaded_gap_s'] for e in es),
            max_abs_before_reload_e=max(max(abs(x) for x in e['e_before_step']) for e in es),
            actual_event_ids=' '.join(str(e['event_id']) for e in es)))
    return result

def main():
    rows=[json.loads(p.read_text()) for p in sorted((SMALL/'formal-v1').glob('*-seed-*.json'))]
    index=json.loads(INDEX.read_text())
    metrics=['duration_s','joint_positive_mechanical_work_J','velocity_rmse_m_s','attitude_rms_rad',
        'contact_slip_distance_m','grf_rmse_N','max_indentation_mm','max_abs_principal_strain','wall_s']
    csv_write('episodes.csv',[{k:r.get(k) for k in ['terrain','controller','seed','success','termination',*metrics,
        'front_to_hind_revisit','precontact_query_used','trajectory','sha256','material_trajectory','material_sha256']} for r in rows])
    paired=[]; legacy=[]; old_refs=[]; timings=[]; queries=[]; revisits=[]; groups=[]
    oldroot=Path(C['resolved_transition']['old_B1_source_root'])
    oldindex=json.loads((oldroot/C['frozen_second_S1']['result_index']).read_text())
    assets=oldindex['publication']['assets']
    for r in rows:
        if r['controller']=='B1':
            oldpath=oldroot/Path(C['frozen_second_S1']['result_index']).parent/'formal-v1'/f"{r['terrain']}-B1-seed-{r['seed']}.json"
            old=json.loads(oldpath.read_text())
            legacy.append(dict(terrain=r['terrain'],seed=r['seed'],old_success=old['success'],corrected_B1_success=r['success'],
                old_termination=old['termination'],corrected_termination=r['termination'],
                old_duration_s=old['duration_s'],corrected_duration_s=r['duration_s'],
                both_completed=old['success'] and r['success'],
                completion_delta_s=r['duration_s']-old['duration_s'] if old['success'] and r['success'] else None))
            old_refs.append(dict(terrain=r['terrain'],seed=r['seed'],source_commit=C['reviewed_source_commit'],
                record=str(oldpath),trajectory_sha256=old['sha256'],material_sha256=old['material_sha256'],
                release=C['frozen_second_S1']['release_url'],
                assets=[a for a in assets if f"{r['terrain']}-B1-" in a['filename']]))
        if r['controller'] in ('T1','T2'):
            baseline='B1' if r['controller']=='T1' else 'T1'
            b=next(b for b in rows if b['terrain']==r['terrain'] and b['controller']==baseline and b['seed']==r['seed'])
            pair=dict(terrain=r['terrain'],baseline=baseline,method=r['controller'],seed=r['seed'],
                baseline_success=b['success'],method_success=r['success'],
                baseline_termination=b['termination'],method_termination=r['termination'],both_completed=b['success'] and r['success'])
            for m in metrics:
                pair[m+'_difference']=r[m]-b[m] if pair['both_completed'] and r.get(m) is not None and b.get(m) is not None else None
            paired.append(pair)
            for q in json.loads(Path(r['queries_path']).read_text()):
                full,virgin=q['T1'],q['T2']
                queries.append(dict(terrain=r['terrain'],controller=r['controller'],seed=r['seed'],query_id=q['query_id'],
                    time_s=q['time_s'],leg=q['leg'],reset_columns=len(virgin['reset_patch_ids']),
                    initial_state_differs=full['initial_e']!=virgin['initial_e'] or full['initial_q']!=virgin['initial_q'],
                    surface_T1_m=full['surface_m'],surface_T2_m=virgin['surface_m'],
                    compression_T1_m=full['target_displacement_m'],compression_T2_m=virgin['target_displacement_m'],
                    speed_T1_m_s=full['approach_speed_m_s'],speed_T2_m_s=virgin['approach_speed_m_s'],
                    action_differs=(full['surface_m']!=virgin['surface_m'] or full['approach_speed_m_s']!=virgin['approach_speed_m_s']),
                    executed_prior=q['selected_prior'],executed_variable=q['executed_variable']))
        revisits.extend(grouped_revisits(r))
        for component,v in r['timing'].items():
            if v:
                timings.append(dict(terrain=r['terrain'],controller=r['controller'],seed=r['seed'],component=component,**v))
    for terrain,method in sorted({(r['terrain'],r['controller']) for r in rows}):
        group=[r for r in rows if r['terrain']==terrain and r['controller']==method]
        good=[r for r in group if r['success']]
        groups.append(dict(terrain=terrain,controller=method,episodes=len(group),successes=len(good),
            failures=dict(Counter(r['termination'] for r in group if not r['success'])),
            successful_completion_time_mean_s=float(np.mean([r['duration_s'] for r in good])) if good else None))
    csv_write('paired.csv',paired);csv_write('shared-correction-vs-frozen-B1.csv',legacy)
    csv_write('timing.csv',timings);csv_write('same-start-queries.csv',queries)
    csv_write('grouped-contact-revisits.csv',revisits)
    index.update(completed_episodes=len(rows),successes=sum(r['success'] for r in rows),groups=groups,
        old_B1_references=old_refs,reused_records=[],old_B1_formal_reuse_count=0,
        development_formal_reuse_count=sum('reused_development_from' in r for r in rows),
        actual_new_formal_results=len(rows),attempt_totals=dict(total=len(index['attempts']),
            development=sum(a['kind']=='development' for a in index['attempts']),
            execution_errors=sum(a['status']=='execution_error' for a in index['attempts']),
            ledger_active_count=sum(a['status']=='started' for a in index['attempts'])),
        data_limits=['Five paired optimizer seeds, not independent material draws.',
            'Timing/completion-work comparisons use only both-completed pairs.',
            'Development v1 lost trajectory arrays at serialization; logs/source retained and attempts counted.',
            'True current constitutive state is a mechanism reference; no observer, PANN/PPO training, hardware or NX validation.'],
        information_chain=dict(queries=len(queries),different_initial_states=sum(q['initial_state_differs'] for q in queries),
            different_selected_actions=sum(q['action_differs'] for q in queries),
            meaning='Same-start full versus virgin queries; this is not an executed T2 mission unless phase B was authorized by readiness.'))
    figure_dir=SMALL/'figures';figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(10,3.6),layout='constrained')
    for ax,terrain in zip(axes,C['phase_A']['terrain_ids']):
        for j,method in enumerate(('B1','T1')):
            for r in rows:
                if r['terrain']==terrain and r['controller']==method:
                    ax.scatter(j+(r['seed']-2)*.055,r['duration_s'],color=COLORS[method],marker='o' if r['success'] else 'x',s=40)
        ax.set(xticks=[0,1],xticklabels=['B1 corrected','T1'],ylim=(0,62),title=terrain,ylabel='Observed task duration (s)')
    fig.suptitle('Every paired seed: circle = completed; cross = failed (shorter failure is not faster)')
    fig.savefig(figure_dir/'mission-outcomes.png',dpi=200);fig.savefig(figure_dir/'mission-outcomes.pdf');plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(10,9),sharex='col',layout='constrained')
    for column,method in enumerate(('B1','T1')):
        r=next(r for r in rows if r['terrain']=='tm02_anchor' and r['controller']==method and r['seed']==0)
        with np.load(ROOT/r['trajectory']) as z:
            h={k:z[k] for k in z.files}
        with np.load(ROOT/r['material_trajectory']) as z:
            physical=z['physics_foot_time_bottom_velocity_force_indent']
        bottom=physical[:,1:13].reshape(-1,4,3);force=physical[:,25:37].reshape(-1,4,3)
        soft=(bottom[:,:,0]>.8+np.sqrt(.001)/2)&(bottom[:,:,0]<2.2-np.sqrt(.001)/2)&(abs(bottom[:,:,1])<.6-np.sqrt(.001)/2)
        hits=np.argwhere(soft & (force[:,:,2]>0))
        if len(hits):
            hit,leg=hits[0];t0=physical[hit,0];caption=f'{method}, first actual soft contact, leg {leg}'
        else:
            t0=min(r['duration_s'],5.54);leg=0;caption=f'{method}, no actual soft contact'
        a=(physical[:,0]>=t0-.09)&(physical[:,0]<=t0+.14)
        b=(h['time']>=t0-.09)&(h['time']<=t0+.14)
        x=physical[a,0]-t0;y=h['time'][b]-t0
        axes[0,column].plot(x,bottom[a,leg,2]*1000,color=COLORS[method]);axes[0,column].axhline(0,color='gray',ls=':')
        axes[0,column].set(title=caption,ylabel='Sole z (mm)')
        axes[1,column].plot(x,physical[a,13:25].reshape(-1,4,3)[:,leg,2],color=COLORS[method]);axes[1,column].set_ylabel('Actual vz (m/s)')
        axes[2,column].plot(x,force[a,leg,2],color=COLORS[method],label='physical')
        axes[2,column].plot(y,h['planned_grf'][b,leg,2],ls='--',color='black',label='MPC')
        if 'executed_grf' in h:
            axes[2,column].plot(y,h['executed_grf'][b,leg,2],ls=':',color='#009E73',label='WB command')
        axes[2,column].set_ylabel('Normal force (N)');axes[2,column].legend(loc='upper left',fontsize=7)
        axes[3,column].plot(y,h['requested_torque'][b,3*leg+2],color='black',ls='--',label='requested')
        axes[3,column].plot(y,h['torque'][b,3*leg+2],color=COLORS[method],label='applied')
        axes[3,column].set(ylabel='Knee torque (Nm)',xlabel='Time from actual soft contact (s)');axes[3,column].legend(fontsize=7)
        for ax in axes[:,column]:ax.axvline(0,color='gray',ls=':')
    fig.savefig(figure_dir/'contact-transition-seed0.png',dpi=200);fig.savefig(figure_dir/'contact-transition-seed0.pdf');plt.close(fig)
    index['figures']=['figures/mission-outcomes.png','figures/contact-transition-seed0.png']
    index['figure_scope']='Actual full-task data, all seeds; manual final visual inspection remains separate from automatic rendering.'
    index['status']='experiments_completed_publication_pending'
    INDEX.write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=len(rows),successes=index['successes'],groups=groups)),flush=True)

if __name__=='__main__':main()
