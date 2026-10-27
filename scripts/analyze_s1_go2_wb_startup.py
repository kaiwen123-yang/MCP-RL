#!/usr/bin/env python3
"""Derive R6 mission/terminal evidence only from the completed stored trajectories."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
CONFIG='experiments/terrain_memory/s1-go2-wb-startup.json'
c=json.loads((ROOT/CONFIG).read_text())
small=ROOT/c['outputs']['root']
index=json.loads((small/'index.json').read_text())
figures=small/'figures';figures.mkdir(exist_ok=True)
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})

def write_csv(path,header,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

metrics=[]
for attempt in index['attempts']:
    name=attempt['attempt']
    out=ROOT/'outputs/terrain_memory/simulation'/c['run_id']/name
    a=np.load(out/'trajectory.npz');init=np.load(out/'startup.npz')
    result=json.loads((small/f'{name}.json').read_text())
    resolved=json.loads((small/f'{name}-resolved.json').read_text())
    t=a['time'];q=a['qpos'];v=a['qvel'];rpy=a['rpy'];terminal=a['phase']==4
    speed=np.linalg.norm(v[:,:3],axis=1);omega=np.linalg.norm(v[:,3:6],axis=1)
    checks={'all_feet_contact':a['contact'].all(axis=1),
        'origin_distance':np.linalg.norm(q[:,:2],axis=1)<.15,
        'linear_speed':speed<.05,'angular_speed':omega<.1,
        'roll_pitch':np.abs(rpy[:,:2]).max(axis=1)<.15}
    stable=np.logical_and.reduce(list(checks.values())) & terminal
    streak=np.zeros(len(t));since=None
    for j,ok in enumerate(stable):
        since=(t[j] if since is None else since) if ok else None
        streak[j]=t[j]-since if since is not None else 0.
    term_time=t[terminal]
    write_csv(small/f'{name}-terminal-criteria.csv',
        ['time_s','origin_distance_m','linear_speed_m_s','angular_speed_rad_s','max_abs_roll_pitch_rad',
         'all_feet_contact','all_criteria_satisfied','continuous_stable_s'],
        np.c_[term_time,np.linalg.norm(q[terminal,:2],axis=1),speed[terminal],omega[terminal],
              np.abs(rpy[terminal,:2]).max(axis=1),checks['all_feet_contact'][terminal],stable[terminal],streak[terminal]])
    write_csv(small/f'{name}-startup-iterations.csv',
        ['iteration','plant_time_s','planner_s']+[f'best_first_target_{j}_rad' for j in range(12)]+
        [f'feedback_target_{j}_rad' for j in range(12)],
        np.c_[init['iteration'],init['time_s'],init['planner_s'],init['actions'][:,0,:],init['feedback_target']])
    # The compact CSV is explicitly sampled; full 500 Hz observations are in NPZ.
    ix=np.unique(np.r_[np.arange(0,len(t),10),len(t)-1])
    header=['time_s','diagnostic_time_s','phase']+[f'qpos_{j}' for j in range(q.shape[1])]+[f'qvel_{j}' for j in range(v.shape[1])]
    columns=[t[ix],a['diagnostic_time'][ix],a['phase'][ix],q[ix],v[ix]]
    for field in ['rpy','q_target','torque','contact','actual_grf','feet','body_target','heading_target']:
        values=a[field][ix].reshape(len(ix),-1)
        header += [f'{field}_{j}' for j in range(values.shape[1])]
        columns.append(values)
    write_csv(small/f'{name}-continuous-50hz.csv',header,np.concatenate([x.reshape(len(ix),-1) for x in columns],axis=1))
    prediction_rows=[]
    for horizon in [.02,.1,.2,.35]:
        j=round(horizon/.01)
        times=a['plan_times'][:,j]
        valid=times<=t[-1]+1e-9
        times=times[valid];pred=a['plan_states'][valid,j,:]
        realq=np.stack([np.interp(times,t,q[:,k]) for k in range(q.shape[1])],axis=1)
        realv=np.stack([np.interp(times,t,v[:,k]) for k in range(v.shape[1])],axis=1)
        pr=Rotation.from_quat(pred[:,[4,5,6,3]])
        ar=Rotation.from_quat(realq[:,[4,5,6,3]])
        values=np.c_[a['plan_origin'][valid],times,np.full(len(times),horizon),
            np.linalg.norm(pred[:,:3]-realq[:,:3],axis=1),(pr.inv()*ar).magnitude(),
            np.linalg.norm(pred[:,19:22]-realv[:,:3],axis=1),
            np.linalg.norm(pred[:,22:25]-realv[:,3:6],axis=1)]
        prediction_rows.extend(values.tolist())
    prediction_rows=np.asarray(prediction_rows)
    write_csv(small/f'{name}-prediction-alignment.csv',
        ['plan_origin_s','absolute_target_time_s','horizon_s','base_position_error_m','orientation_error_rad',
         'base_linear_velocity_error_m_s','base_angular_velocity_error_rad_s'],prediction_rows)
    physical=a['physical'];grfz=physical[:,[32,35,38,41]].sum(axis=1)
    nominal_init=init['seed_action'];initial=q[0,7:]
    predq=a['plan_states'][:,:,:19].reshape(-1,19)
    predori=Rotation.from_quat(predq[:,[4,5,6,3]]).as_euler('xyz')
    horizon_stats={str(h):dict(samples=int(np.isclose(prediction_rows[:,2],h).sum()),
        mean_position_error_m=float(prediction_rows[np.isclose(prediction_rows[:,2],h),3].mean()),
        mean_orientation_error_rad=float(prediction_rows[np.isclose(prediction_rows[:,2],h),4].mean())) for h in [.02,.1,.2,.35]}
    m=dict(attempt=name,success=result['success'],reason=result['reason'],
        data_profile=dict(control_samples=len(t),physical_samples=len(physical),planning_origins=len(a['plan_origin']),
            prediction_steps=36,full_control_dt_s=.002,compact_csv_stride=10,
            nonfinite_values={k:int((~np.isfinite(a[k])).sum()) for k in ['qpos','qvel','torque','actual_grf','plan_states','plan_actions','physical']}),
        startup=dict(iterations=len(init['iteration']),plant_time_values=np.unique(init['time_s']).tolist(),
            seed_home_max_abs_difference_rad=float(abs(nominal_init-init['home_ctrl']).max()),
            predicted_initial_qpos_max_abs_difference=float(abs(init['states'][:,0,:19]-q[0]).max()),
            final_initialization_vs_executed_action_max_abs_difference=float(abs(init['feedback_target'][-1]-a['q_target'][0]).max()),
            first_actual_max_abs_torque_Nm=float(abs(a['torque'][0]).max()),
            first_2s_peak_total_vertical_grf_N=float(grfz[physical[:,0]<=2].max()),
            first_2s_peak_base_linear_speed_m_s=float(speed[t<=2].max()),
            affine_PD_max_abs_error_Nm=float(abs(a['torque']-(60*(a['q_target']-q[:,7:])-5*v[:,6:])).max())),
        terminal=dict(start_s=float(term_time[0]),duration_s=float(term_time[-1]-term_time[0]),
            criteria_satisfied_fraction={k:float(ok[terminal].mean()) for k,ok in checks.items()},
            all_criteria_satisfied_fraction=float(stable[terminal].mean()),longest_stable_streak_s=float(streak.max()),
            maximum_angular_speed_rad_s=float(omega[terminal].max()),p95_angular_speed_rad_s=float(np.percentile(omega[terminal],95)),
            endpoint_angular_speed_rad_s=float(omega[-1]),failure_condition='No continuous 2 s interval satisfied every original terminal criterion; angular-speed crossings reset the stable timer.',
            causal_limit='Observed stopping condition, not isolation of a unique dynamical cause; no confirmed new implementation error and no identical retry.'),
        best_prediction=dict(min_base_z_m=float(predq[:,2].min()),max_abs_roll_pitch_rad=float(abs(predori[:,:2]).max()),
            aligned_error=horizon_stats,alignment='Best prediction and actual state compared at equal absolute simulation time; no extrapolation past actual termination. This is not a horizon-cost convergence curve.'))
    # Kinematic replay of the saved best states; no physical integration.
    model=mujoco.MjModel.from_binary_path(str(out/'model.mjb'))
    data=mujoco.MjData(model)
    feet=np.array([model.geom(n).id for n in ['FL','FR','RL','RR']])
    radius=model.geom_size[feet,0]
    floor=resolved['model']['flat_ground']['world_height_m']
    gaps=np.empty((len(predq),4))
    for j,state in enumerate(predq):
        data.qpos[:]=state
        mujoco.mj_kinematics(model,data)
        gaps[j]=data.geom_xpos[feet,2]-radius-floor
    pi,fi=np.unravel_index(gaps.argmin(),gaps.shape)
    origin,step=divmod(pi,36)
    m['best_prediction']['ground_geometry']=dict(
        foot_radius_m=radius.tolist(),predicted_min_foot_bottom_gap_m=float(gaps.min()),
        actual_min_foot_bottom_gap_m=float((a['feet'][:,:,2]-radius-floor).min()),
        worst_plan_origin_s=float(a['plan_origin'][origin]),
        worst_horizon_s=float(a['plan_times'][origin,step]-a['plan_origin'][origin]),
        worst_foot=['FL','FR','RL','RR'][fi],
        meaning='Signed sphere-bottom minus fixed plane height; negative means geometric overlap under the retained native contact model. Analytic residual height did not eliminate these overlaps. This is not a new termination rule.')
    np.savez_compressed(out/'prediction-ground-gaps.npz',plan_origin=a['plan_origin'],
        plan_times=a['plan_times'],foot_bottom_gap_m=gaps.reshape(-1,36,4))
    metrics.append(m)
    fig,axes=plt.subplots(3,2,figsize=(10.5,8.2),layout='constrained')
    ax=axes[0,0];ax.plot(t,q[:,0],c=colors[0],label='Actual body');ax.plot(t,a['body_target'][:,0],'--',c=colors[1],label='Target');ax.set(ylabel='World x (m)',title='a  Outbound and return');ax.legend(loc='upper right',fontsize=8)
    ax=axes[0,1];ax.plot(t,np.unwrap(rpy[:,2]),c=colors[0],label='Actual');ax.plot(t,a['heading_target'],'--',c=colors[1],label='Target');ax.set(ylabel='Yaw (rad)',title='b  Turn and heading');ax.legend(loc='lower right',fontsize=8)
    ax=axes[1,0];ax.plot(t,q[:,2],c=colors[0]);ax.axhline(.12,c=colors[1],ls='--',label='Fall threshold');ax.set(ylabel='Base height (m)',title='c  Actual body height',ylim=(0,.34));ax.legend(loc='lower right',fontsize=8)
    ax=axes[1,1];ax.plot(physical[:,0],grfz,c=colors[0],lw=.6,label='Actual total');ax.axhline(resolved['mass_kg']*9.81,c=colors[1],ls='--',label='Body weight');ax.set(ylabel='Vertical ground force (N)',title='d  Measured simulated contact force',ylim=(0,None));ax.legend(loc='upper right',fontsize=8)
    ax=axes[2,0];ax.plot(term_time,omega[terminal],c=colors[0],lw=.65);ax.axhline(.1,c=colors[1],ls='--',label='Required < 0.1 rad/s');ax.set(ylabel='Angular speed (rad/s)',title='e  Terminal angular-speed crossings',ylim=(0,None));ax.legend(loc='upper right',fontsize=8)
    ax=axes[2,1];ax.plot(term_time,streak[terminal],c=colors[0]);ax.axhline(2,c=colors[1],ls='--',label='Required 2 s');ax.set(ylabel='Continuous stable interval (s)',title='f  Original terminal criterion',ylim=(0,2.3));ax.legend(loc='upper right',fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.15)
    for ax in axes[:2,:].flat:
        ax.set_xlim(0, float(t[-1])); ax.set_xticks(np.arange(0,61,10))
    for ax in axes[2,:]:
        ax.set_xlim(35, float(t[-1])); ax.set_xticks(np.arange(35,61,5))
    fig.suptitle(f'{name}: original 60 s mission ended in timeout; one physical start',fontsize=12)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();bounds=fig.bbox
    clipped=[]
    for ax in axes.flat:
        for label in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
            box=label.get_window_extent(renderer)
            if label.get_visible() and label.get_text() and not (box.x0>=0 and box.y0>=0 and box.x1<=bounds.width and box.y1<=bounds.height):
                clipped.append(label.get_text())
    m['figure_layout']={'text_outside_canvas':clipped}
    fig.savefig(figures/f'{name}-mission.png',dpi=160)
    svg=figures/f'{name}-mission.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
index['trajectory_analysis']=metrics
index['execution']['physics_started']=bool(index['attempts'])
index['execution']['status']='physical_execution_and_video_replay_finished'
index['decision']='Close R6 after one full G0 timeout. The route and turn were traversed, but the original continuous 2 s terminal stability condition was not met. No confirmed new implementation error warrants a second identical start. G1 remains unexecuted.'
index['unused_allowances']={'G0':1,'G1_development':3,'G1_formal_conditions':3,'transfer_allowed':False}
index['postprocessing_command']='work/envs/mjpc-go2/bin/python scripts/analyze_s1_go2_wb_startup.py'
index['compact_data_note']='50 Hz CSV is decimated from full 500 Hz trajectory; terminal criteria retain every 500 Hz sample. NPZ contains all actual states, physical forces, joint targets, torques and best predictions.'
index['publication']={'status':'pending','release_tag':'tm-s1-go2-wb-startup-r6','upload_origin':'execution host only; no result download to the user PC'}
(small/'index.json').write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(metrics,indent=2))
