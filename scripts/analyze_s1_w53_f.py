#!/usr/bin/env python3
"""Read W53 F actual logs and saved failed constraints; no solver/model execution."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-local-readiness-r12'

def analyze(attempt):
    small=ROOT/'results/terrain_memory/simulation'/RUN
    out=ROOT/'outputs/terrain_memory/simulation'/RUN/attempt
    folder=out/'w53-f'
    result=json.loads((small/(attempt+'.json')).read_text())
    resolved=json.loads((small/(attempt+'-resolved.json')).read_text())
    trajectory=np.load(out/'trajectory.npz')
    chunks=[np.load(p) for p in sorted(folder.glob('control-*.npz'))]
    ctrl={k:np.concatenate([p[k] for p in chunks]) for k in chunks[0].files}
    failure=json.loads((folder/'solver-failure.json').read_text())
    label=f"failed-mpc-{failure['time_s']:.6f}"
    A=sparse.load_npz(folder/(label+'-A.npz')).toarray()
    bounds=np.load(folder/(label+'.npz'))
    lower,upper=bounds['lower'],bounds['upper']
    cfg=result['w53_f_recording']['config'];N=cfg['mpc']['intervals'];nx=12*(N+1)
    contradictions=[]
    for k in range(N):
        fs=slice(nx+12*k,nx+12*(k+1));row=12+52*k
        force_rows=slice(row+40,row+52);torque_rows=slice(row+28,row+40)
        zero=(lower[force_rows]==0)&(upper[force_rows]==0)
        T=A[torque_rows,fs].copy();T[:,zero]=0
        for j in range(12):
            rr=row+28+j
            if np.linalg.norm(T[j])==0 and (lower[rr]>0 or upper[rr]<0):
                contradictions.append(dict(stage=k,joint_index=j,joint=resolved['joint_names'][j],
                    failed_matrix_row=rr,remaining_row_norm=0.,lower=float(lower[rr]),upper=float(upper[rr]),
                    forced_zero_force_columns_in_stage=np.flatnonzero(zero).tolist()))
    moving=ctrl['time']>=2.002-1e-8
    p=ctrl['planned_contact'][moving];o=ctrl['observed_contact'][moving];a=ctrl['active_wbc_contact'][moving]
    q=trajectory['qpos'][-1,7:];v=trajectory['qvel'][-1,6:]
    urange=np.array(resolved['actuator_ctrlrange']);trange=np.array(resolved['actuator_forcerange'])
    tau_lower=np.maximum(trange[:,0],60*(urange[:,0]-q)-5*v)
    tau_upper=np.minimum(trange[:,1],60*(urange[:,1]-q)-5*v)
    timing=result['w53_f_recording']['timing'];wall=result['wall_s']
    stages=[]
    for begin,end in [(0.,2.),(2.002,2.30),(2.30,2.60),(2.60,result['simulation_s'])]:
        mask=(ctrl['time']>=begin-1e-8)&(ctrl['time']<end-1e-8)
        rows=(trajectory['time']>=begin-1e-8)&(trajectory['time']<=end+1e-8)
        pos=trajectory['qpos'][rows]
        stages.append(dict(start_s=begin,end_s=end,control_samples=int(mask.sum()),
            first_x_m=float(pos[0,0]),last_x_m=float(pos[-1,0]),
            body_z_min_m=float(pos[:,2].min()),body_z_max_m=float(pos[:,2].max()),
            all_feet_loaded_fraction=float(ctrl['observed_contact'][mask].all(1).mean()),
            mean_measured_fz_N=ctrl['measured_grf'][mask,:,2].mean(0).tolist()))
    onset={}
    for z in [.28,.27,.25,.22]:
        ids=np.flatnonzero((trajectory['time']>=2.002-1e-8)&(trajectory['qpos'][:,2]<z))
        onset[str(z)]=None if not len(ids) else float(trajectory['time'][ids[0]])
    touchdowns=[]
    for leg,name in enumerate(('FL','FR','RL','RR')):
        indices=np.flatnonzero((~ctrl['planned_contact'][:-1,leg])&ctrl['planned_contact'][1:,leg])+1
        for i in indices:
            later=np.flatnonzero(ctrl['observed_contact'][i:,leg])
            first=None if not len(later) else i+later[0]
            row=int(np.argmin(abs(trajectory['time']-ctrl['time'][i])))
            touchdowns.append(dict(leg=name,planned_touchdown_s=float(ctrl['time'][i]),
                actual_foot_bottom_m=float(trajectory['feet'][row,leg,2]-.022),
                last_swing_reference_bottom_m=float(ctrl['feet_position_reference'][i-1,leg,2]-.022),
                first_subsequent_observed_load_s=None if first is None else float(ctrl['time'][first]),
                delay_s=None if first is None else float(ctrl['time'][first]-ctrl['time'][i]),
                wbc_mode=int(ctrl['mode'][i,leg])))
    answer=dict(schema='w53_f_actual_failure_analysis_v1',attempt=attempt,
        scope='Read saved trajectory, every WBC control record and original failed constraint arrays; no QP re-solve, physics, native rollout, FD or parameter change.',
        original_stage_status=result['stage_status'],original_reason=result['reason'],
        scientific_classification='controller_model_feasibility_failure_after_real_physical_start_before_material',
        solver_failure=failure,physical_end_s=result['simulation_s'],physical_samples=result['physical_samples'],
        full_mpc_records=result['w53_f_recording']['mpc_calls'],full_wbc_records=len(ctrl['time']),
        material_loaded=result['stage_progress']['actual_material_loaded'],
        body_end_xyz_m=trajectory['qpos'][-1,:3].tolist(),max_x_m=result['max_x_m'],
        body_z_drop_onsets_after_motion_s=onset,windows=stages,
        actual_touchdown_events=touchdowns,
        touchdown_interpretation='Eight completed scheduled touchdown transitions had 2.73-8.04mm actual positive clearance and 74-114ms measured-load delay, exceeding nominal 50ms diagonal overlap. The frozen contact feedback used downward-search mode while the opposite pair began scheduled swing; these observations explain support gaps temporally but do not isolate all causes of foot tracking error.',
        motion=dict(first_moving_control_s=2.002,samples=int(moving.sum()),
            net_dx_m=float(trajectory['qpos'][-1,0]-trajectory['qpos'][np.argmin(abs(trajectory['time']-2.002)),0]),
            swing_still_loaded_fraction=float(np.sum(~p&o)/np.sum(~p)),
            planned_stance_without_observed_load_fraction=float(np.sum(p&~o)/np.sum(p)),
            zero_observed_support_ticks=int(np.sum(o.sum(1)==0)),zero_active_wbc_support_ticks=int(np.sum(a.sum(1)==0)),
            approximate_zero_observed_support_duration_s=float(np.sum(o.sum(1)==0)*.002),
            max_foot_bottom_height_m=np.max(trajectory['feet'][:,:,2]-.022,axis=0).tolist(),
            conclusion='Swing unloading and forward displacement occurred, but support gaps and progressive height loss do not establish restored sustained normal locomotion.'),
        failed_constraint_proof=dict(matrix_shape=list(A.shape),contradictions=contradictions,
            explanation='For scheduled swing FR, its force xyz are fixed to zero and other feet have zero FR-calf joint Jacobian columns. The static torque approximation then fixes FR-calf torque to h. The mapped input lower bound requires a larger torque; six horizon stages have an exactly zero row requiring >=1.7002576406 Nm. This proves infeasibility directly from saved arrays, independent of solver residuals.',
            FR_calf=dict(q_rad=float(q[5]),v_rad_s=float(v[5]),mapped_total_torque_lower_Nm=float(tau_lower[5]),
                total_torque_upper_Nm=float(tau_upper[5]),static_h_Nm=float(tau_upper[5]-contradictions[0]['upper']),
                contradiction_gap_Nm=contradictions[0]['lower']),
            boundary='This identifies the immediate controller termination and limitation of static h-J^T f feasibility for a moving swing leg. It does not by itself isolate the earlier support-gap/height-loss cause or prove all dynamic WBC actions were infeasible.'),
        execution=dict(max_initial_PD_mapping_error_Nm=float(np.max(abs(ctrl['requested_pd']-ctrl['tau_total']))),
            max_floating_dynamics_residual=float(np.max(abs(ctrl['dynamics_residual']))),
            max_total_torque_constraint_violation_Nm=float(max(np.max(ctrl['torque_lower']-ctrl['tau_total']),np.max(ctrl['tau_total']-ctrl['torque_upper']))),
            wbc_solver_all_solved=bool(np.all(ctrl['solver_status']==1)),
            last_active_contact=ctrl['active_wbc_contact'][-1].tolist(),
            last_wbc_total_torque_Nm=ctrl['tau_total'][-1].tolist(),
            no_future_controller_succeeded_claim=True),
        runtime=dict(eval_wall_s=wall,timing=timing,
            exact_model_group_fraction=timing['model']['wall_s']/wall,
            full_centroidal_matrix_assembly_group_fraction=timing['matrix_assembly']['wall_s']/wall,
            combined_group_fraction=(timing['model']['wall_s']+timing['matrix_assembly']['wall_s'])/wall,
            C_coverage_note='These groups contain unported M/h/Jdot and full sparse Python horizon assembly. Their combined share is not the GPU-covered fraction and cannot be multiplied by the C speed ratio to claim online speedup.',
            failed_mpc_solve_timer_note='The 158th matrix assembly is included; the raising failed solve is absent from successful mpc_solve accumulated timing. Its original OSQP solve_time is in solver_failure.',
            retained_F_record_bytes=sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())))
    path=small/'w53/f-actual-analysis.json';path.write_text(json.dumps(answer,indent=2)+'\n')
    print(path)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    analyze(parser.parse_args().attempt)
