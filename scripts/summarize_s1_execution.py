#!/usr/bin/env python3
"""Summarize the authorized complete-mission attempts from their saved traces."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
CONFIG='experiments/terrain_memory/s1-contact-execution.json'
C=json.loads((ROOT/CONFIG).read_text())
SMALL=ROOT/C['outputs']['compact_root']
INDEX=SMALL/'index.json'
LEGS=['FL','FR','RL','RR']
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']


def csv_write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (SMALL/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def main():
    index=json.loads(INDEX.read_text())
    attempts=index['attempts']
    rows=[json.loads((ROOT/a['record']).read_text()) for a in attempts if a['status']=='finished']
    traces=[np.load(ROOT/r['trajectory']) for r in rows]
    metrics=['duration_s','joint_positive_mechanical_work_J','contact_slip_distance_m',
        'velocity_rmse_m_s','attitude_rms_rad','grf_rmse_N','wall_s']
    csv_write('episodes.csv',[dict(attempt_id=a['attempt_id'],kind=a['kind'],source_commit=a['source_commit'],
        record=a['record'],**{k:r[k] for k in ['terrain','controller','seed','success','termination',*metrics,
        'trajectory','sha256','material_trajectory','material_sha256']}) for a,r in zip(attempts,rows)])
    paired=[];historical=[];execution=[];events=[];timings=[];queries=[];samples=[]
    sources=[('T1_previous','tm-s1-contact-transition-20261025','T1','f0a1632f445415be6c33a7eac31c3a2c974d1f4a','tm-s1-contact-transition-evidence-20261025'),
        ('B1_third','tm-s1-contact-transition-20261025','B1','f0a1632f445415be6c33a7eac31c3a2c974d1f4a','tm-s1-contact-transition-evidence-20261025'),
        ('B1_second','tm-s1-control-repair-20261024','B1','5af2c25c5223c7b74a473133fe97aed3d9962c34','tm-s1-control-repair-evidence-20261024')]
    for a,r,z in zip(attempts,rows,traces):
        resolved=json.loads((ROOT/a['record']).with_name(f"{r['terrain']}-T1R-resolved.json").read_text())
        upper=resolved['controller_parameters']['grf_max'];mg=resolved['actual_mass_kg']*9.81
        t=z['time'];f=z['commanded_grf'];h=z['held_grf'];support=z['actual_support_available'];alpha=z['transition_load_fraction']
        un=z['planned_contact'].astype(bool)&~support
        dforce=z['command_minus_hold_force_N'];dmoment=z['command_minus_hold_moment_Nm']
        parts=sum(z[k] for k in ['stance_native_grf_torque','stance_inertia_torque','stance_tracking_torque','stance_bias_torque'])
        change=np.any(abs(np.diff(h,axis=0))>1e-10,axis=(1,2))
        dead=un&(np.linalg.norm(z['final_leg_torque_requested'],axis=2)<1e-10)
        force_peak=int(np.argmax(np.linalg.norm(dforce,axis=1)));moment_peak=int(np.argmax(np.linalg.norm(dmoment,axis=1)))
        m=dict(attempt_id=a['attempt_id'],seed=r['seed'],samples=len(t),last_phase=['initial_stand','outbound','turn','return','terminal_stand','complete'][int(z['phase'][-1])],
            unsupported_nominal_leg_samples=int(un.sum()),unsupported_wrong_owner_samples=int((un&(z['transition_mode']!=1)).sum()),
            unsupported_zero_requested_torque_samples=int(dead.sum()),unsupported_zero_requested_torque_after_initial_sample=int(dead[1:].sum()),
            unavailable_command_max_N=float(np.max(abs(f[~support]))),normal_lower_violation_N=float(max(0.,-f[:,:,2].min())),
            normal_upper_violation_N=float(max(0.,(f[:,:,2]-alpha*upper).max())),
            friction_violation_N=float(max(0.,(abs(f[:,:,:2])-.5*f[:,:,2,None]).max())),
            torque_component_sum_max_abs_error_Nm=float(np.nanmax(abs(parts-z['final_leg_torque_requested']))),
            torque_component_nonfinite_values=int((~np.isfinite(parts)).sum()),
            held_force_changes_without_MPC_solve=int((change&~np.isfinite(z['solve_s'][1:])).sum()),
            peak_force_difference_norm_N=float(np.linalg.norm(dforce[force_peak])),peak_force_difference_time_s=float(t[force_peak]),
            peak_moment_difference_norm_Nm=float(np.linalg.norm(dmoment[moment_peak])),peak_moment_difference_time_s=float(t[moment_peak]),
            fraction_control_samples_normal_capacity_below_weight=float(np.mean(alpha.sum(axis=1)*upper<mg)),
            total_weight_N=mg,grf_max_N=upper,torque_saturation_fraction=r['torque_saturation_fraction'],
            prediction_load_alpha_nonfinite=int((~np.isfinite(z['mpc_load_fraction'])).sum()),
            physics_GRf_time_offset_min_s=float((z['actual_grf_post_step_time_s']-t).min()),
            physics_GRf_time_offset_max_s=float((z['actual_grf_post_step_time_s']-t).max()))
        execution.append(m)
        control_clock=np.r_[z['actual_grf_post_step_time_s'][0]-.002,z['actual_grf_post_step_time_s'][:-1]]
        for i in range(len(t)):
            samples.append(dict(attempt_id=a['attempt_id'],seed=r['seed'],time_s=t[i],control_simulator_time_s=control_clock[i],
                actual_grf_at_control_simulator_time_s=control_clock[i] if i else None,
                actual_grf_after_step_simulator_time_s=z['actual_grf_post_step_time_s'][i],base_z_m=z['base_pos'][i,2],
                roll_rad=z['orientation'][i,0],pitch_rad=z['orientation'][i,1],actual_support_legs=int(support[i].sum()),
                held_Fz_N=h[i,:,2].sum(),commanded_Fz_N=f[i,:,2].sum(),actual_Fz_at_control_N=z['actual_grf_at_control'][i,:,2].sum(),
                force_difference_norm_N=np.linalg.norm(dforce[i]),moment_difference_norm_Nm=np.linalg.norm(dmoment[i])))
        for ev in json.loads(Path(r['transitions_path']).read_text()):
            i=min(np.searchsorted(t,ev['time_s']),len(t)-1);leg=ev['leg']
            events.append(dict(attempt_id=a['attempt_id'],seed=r['seed'],time_s=ev['time_s'],leg=LEGS[leg],event=ev['event'],
                mode=int(z['transition_mode'][i,leg]),nominal_contact=bool(z['planned_contact'][i,leg]),actual_support=bool(support[i,leg]),
                alpha=float(alpha[i,leg]),held_Fz_N=float(h[i,leg,2]),commanded_Fz_N=float(f[i,leg,2]),actual_Fz_at_control_N=float(z['actual_grf_at_control'][i,leg,2]),
                requested_torque_norm_Nm=float(np.linalg.norm(z['final_leg_torque_requested'][i,leg])),
                force_difference_norm_N=float(np.linalg.norm(dforce[i])),moment_difference_norm_Nm=float(np.linalg.norm(dmoment[i]))))
        for q in json.loads(Path(r['queries_path']).read_text()):
            full,virgin=q['T1'],q['T2']
            queries.append(dict(attempt_id=a['attempt_id'],seed=r['seed'],query_id=q['query_id'],time_s=q['time_s'],leg=LEGS[q['leg']],
                reason=q.get('reason','descending_swing'),surface_difference_m=full['surface_m']-virgin['surface_m'],
                approach_speed_difference_m_s=full['approach_speed_m_s']-virgin['approach_speed_m_s'],
                executed_approach_speed_m_s=full['approach_speed_m_s'],remaining_s=q['remaining_s']))
        for component,values in r['timing'].items():
            if values:timings.append(dict(seed=r['seed'],component=component,**values))
        for label,run,method,commit,tag in sources:
            oldpath=ROOT/'results/terrain_memory/simulation'/run/'formal-v1'/f"{r['terrain']}-{method}-seed-{r['seed']}.json"
            old=json.loads(oldpath.read_text())
            oldindex=json.loads((oldpath.parent.parent/'index.json').read_text())
            pair=dict(seed=r['seed'],new_kind='development',baseline=label,baseline_commit=commit,new_commit=a['source_commit'],
                baseline_success=old['success'],new_success=r['success'],baseline_termination=old['termination'],new_termination=r['termination'],
                both_completed=bool(old['success'] and r['success']))
            for key in metrics:pair[key+'_difference']=r[key]-old[key] if pair['both_completed'] else None
            paired.append(pair)
            historical.append(dict(id=label,seed=r['seed'],source_commit=commit,record_url=f'https://github.com/kaiwen123-yang/MCP-RL/blob/{commit}/{oldpath.relative_to(ROOT)}',
                record_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),trajectory_sha256=old['sha256'],material_sha256=old['material_sha256'],
                release_url=f'https://github.com/kaiwen123-yang/MCP-RL/releases/tag/{tag}',
                assets=[{k:x[k] for k in ('filename','bytes','sha256','url')} for x in oldindex['publication']['assets'] if f"{r['terrain']}-{method}-" in x['filename']],
                interpretation='Existing evidence only; B1 moving backend differs from T1R. No baseline was rerun.'))
    csv_write('paired-existing-versions.csv',paired);csv_write('contact-execution.csv',execution)
    csv_write('contact-events.csv',events);csv_write('timing.csv',timings);csv_write('query-effects.csv',queries)
    csv_write('control-time-series.csv',samples)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    figs=SMALL/'figures';figs.mkdir(exist_ok=True)
    figure_geometry={}
    def save(fig,name):
        fig.canvas.draw()
        figure_geometry[name]=[list(ax.get_position().bounds) for ax in fig.axes]
        fig.savefig(figs/(name+'.png'),dpi=160)
        fig.savefig(figs/(name+'.pdf'))
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.2));fig.subplots_adjust(left=.18,right=.97,bottom=.2,top=.8)
    ypos=np.arange(len(rows))
    ax.scatter([r['duration_s'] for r in rows],ypos,marker='x',s=85,color='#D55E00')
    for y,r in zip(ypos,rows):ax.annotate(f" {r['duration_s']:.4g} s",(r['duration_s'],y),xytext=(6,0),textcoords='offset points',va='center')
    ax.set(yticks=ypos,yticklabels=[f"Seed {r['seed']}" for r in rows],xlabel='Recorded physical duration before failure (s)',xlim=(0,8),ylim=(-.5,len(rows)-.5))
    ax.invert_yaxis();fig.suptitle('Three original rigid missions started; none completed',fontsize=12)
    fig.text(.5,.88,'Development only. Formal stages were not opened. Short failure duration is not a speed benefit.',ha='center',fontsize=8)
    save(fig,'development-outcomes')
    fig,axes=plt.subplots(4,len(rows),figsize=(12,10),sharex='col',squeeze=False)
    fig.subplots_adjust(left=.08,right=.98,bottom=.07,top=.9,wspace=.33,hspace=.38)
    for col,(r,z) in enumerate(zip(rows,traces)):
        t=z['time'];axes[0,col].plot(t,z['base_pos'][:,2],color='#0072B2')
        axes[0,col].set_title(f"Seed {r['seed']} | failed at {r['duration_s']:.4g} s")
        axes[0,col].set_ylabel('Base height (m)')
        for data,color,ls,label in [(z['held_grf'][:,:,2].sum(axis=1),'#777777','--','Held plan'),(z['commanded_grf'][:,:,2].sum(axis=1),'#D55E00','-','Command'),(z['actual_grf_at_control'][:,:,2].sum(axis=1),'#0072B2',':','Measured')]:
            axes[1,col].plot(t,data,color=color,ls=ls,label=label,lw=1)
        axes[1,col].axhline(execution[col]['total_weight_N'],color='black',lw=.6,ls=':')
        axes[1,col].set_ylabel('Total normal force (N)')
        axes[2,col].plot(t,np.linalg.norm(z['command_minus_hold_force_N'],axis=1),color='#D55E00');axes[2,col].set_ylabel('Force difference norm (N)')
        axes[3,col].plot(t,np.linalg.norm(z['command_minus_hold_moment_Nm'],axis=1),color='#009E73');axes[3,col].set(ylabel='Moment difference norm (Nm)',xlabel='Physical time (s)')
        for a in axes[:,col]:a.axvline(2,color='#777777',ls=':',lw=.7);a.set_xlim(0,r['duration_s'])
    axes[1,0].legend(fontsize=8,loc='upper right')
    fig.suptitle('Held-force execution exposes support deficits during rigid-ground failures',fontsize=13,y=.975)
    fig.text(.5,.94,'Continuous records, all development attempts. Differences use command minus held force at the same base point.',ha='center',fontsize=9)
    save(fig,'continuous-execution')
    oldroot=Path('/home/fanyuan/projects/kaiwen-learning/terrain-memory-s1-transition')
    old=json.loads((oldroot/'results/terrain_memory/simulation/tm-s1-contact-transition-20261025/formal-v1/rigid-T1-seed-4.json').read_text())
    oldz=np.load(oldroot/old['trajectory']);new=traces[0]
    model=mujoco.MjModel.from_binary_path(rows[0]['model_path'])
    actnames=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(j)) for j in model.actuator_trnid[:,0]]
    fig,axes=plt.subplots(2,2,figsize=(8,5.7),sharex=True)
    fig.subplots_adjust(left=.11,right=.98,bottom=.12,top=.82,wspace=.3,hspace=.25)
    for col,leg in enumerate((1,2)):
        ids=[i for i,n in enumerate(actnames) if n.startswith(LEGS[leg])]
        for z,label,color,ls in [(oldz,'Previous T1','#777777','--'),(new,'T1R development','#0072B2','-')]:
            mask=z['time']<=1.344
            axes[0,col].plot(z['time'][mask],(z['feet'][mask,leg,2]-.022)*1000,color=color,ls=ls,label=label)
            axes[1,col].plot(z['time'][mask],np.linalg.norm(z['torque'][mask][:,ids],axis=1),color=color,ls=ls)
        axes[0,col].set_title(LEGS[leg]);axes[0,col].set_ylabel('Sole height (mm)');axes[1,col].set(ylabel='Applied joint-torque norm (Nm)',xlabel='Physical time (s)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Seed4: the initial unsupported-leg torque gap is removed',fontsize=12,y=.965)
    fig.text(.5,.885,'Same optimizer seed; old task failed at 1.344 s. T1R later failed at 2.769 s.',ha='center',fontsize=9)
    save(fig,'seed4-initial-contact')
    index.update(completed_development_records=len(rows),development_successes=sum(r['success'] for r in rows),
        development_failures=dict(Counter(r['termination'] for r in rows if not r['success'])),historical_references=historical,
        execution_metrics=execution,queries=len(queries),query_speed_difference_max_m_s=max(abs(q['approach_speed_difference_m_s']) for q in queries),
        all_new_attempts_preserved=len(rows)==len(attempts),material_missions=0,material_memory_conclusion='Not evaluated: no soft-terrain mission was authorized after failed rigid development.',
        figures=['figures/'+name+'.png' for name in figure_geometry],figure_rendered_axes=figure_geometry,
        figure_scope='Saved full-task trajectories, no smoothing, no resampled physical missions; development outcomes are not formal five-seed results.',
        time_alignment={'time':'Mission-relative control clock; original simulator reset has a 0.0002s clock origin offset.',
            'wrench':'Held and commanded forces use the same control row and recorded base point.',
            'actual_at_control':'Retained measurement from the preceding physics step; first row has no preceding measurement and its measurement timestamp is empty in the CSV.',
            'actual_grf':'Original trajectory field is sampled after advancing physics. The new actual_grf_post_step_time_s records its simulator clock; do not treat it as simultaneous with command time.'},
        publication_decision='Stop incremental T1R development. Reuse an established whole-body Go2 execution basis for future work; TM20 Go2 is the selected integration candidate, not an already reproduced or installed replacement.',
        integration_cost={'reuse':['Original Go2 inertial/actuator/collision model and complete mission definitions','World-fixed material law, 2% domain, recovery, overlap areas and physical force application','Existing continuous trace and video outputs'],
            'required_work':['Map joint-position targets/PD and final clipped torques into the prediction with the retained actuator limits; verify model and joint ordering.',
                'Put persistent material state and physical reaction in each independent prediction branch, including finite-difference perturbations; a Python external force callback alone does not make the material part of the planner state.',
                'Retain soft-region native-contact suppression, history/recovery and failure definitions in both real simulation and candidate clones.',
                'Translate the unchanged route/turn/terminal-stand task into solver targets and record actual execution delay and resource cost; no NX timing is inferred.'],
            'cost_judgment':'Substantial model-and-state integration, not a controller-class replacement. No measured engineering-time estimate is available.',
            'selected_source':'https://github.com/johnzhang3/mujoco_mpc_deploy','solver_source':'https://github.com/johnzhang3/mujoco_mpc/tree/go2',
            'limitations':'Author interface remains marked work in progress and uses motion capture; this round did not install, run or validate it. Frozen second-round B1 remains a local rigid/KV reference, not a demonstrated TM02 memory baseline.'},
        scientific_conclusions={'supported':'Explicit unsupported nominal-stance ownership, zero unavailable-foot command, shared unilateral/friction/loading caps, held force without low-level redistribution, and saved complete-task failures.',
            'remaining_failure':'Repeated loss/reacquisition and low available support occur with body instability; these full-task data establish that the two interface repairs are insufficient. They do not isolate every dynamical cause.',
            'not_established':'Usable rigid execution, TM02 traversal/revisits, material-memory benefit, learning benefit, hardware or NX real time.'})
    INDEX.write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'attempts':len(rows),'successes':sum(r['success'] for r in rows),'formal':0,'execution':execution}),flush=True)

if __name__=='__main__':main()
