"""Re-solve the real force QP and WBC on the complete selected archived event family."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import mujoco
import numpy as np

from qualify_s1_w57_planning import CFG, OUT, ROOT, Saved, solve
from s1_w54_f_r52_backend import assemble_centroidal, assemble_force_horizon
from s1_w56_f_r60_backend import FController
from s1_w57_planning import Forecast, rank


def replay_wbc(saved,plan,states,forces,clock,row,folder):
    model=mujoco.MjModel.from_binary_path(str(saved.root/'model.mjb')); data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id);mujoco.mj_forward(model,data)
    feet=np.array([model.geom(l).id for l in ('FL','FR','RL','RR')])
    folder.mkdir(exist_ok=False,parents=True)
    controller=FController(model,data,feet,model.body('trunk').id,folder,CFG)
    data.qpos[:]=plan['qpos'];data.qvel[:]=plan['qvel'];data.time=clock
    real=saved.material(round(clock/.0002)//5000)[round(clock/.0002)%5000]
    soft=real[2:6].astype(bool)
    model.geom_contype[feet[soft]]=0;model.geom_conaffinity[feet[soft]]=0
    mujoco.mj_forward(model,data)
    info=dict(material_observer=saved.obs,force_boundary_bottom=saved.obs.published_bottom,
              stiffness_N_m=CFG['material_stiffness_N_m'],force=row['measured_grf'])
    controller.current_material_info=info;controller.current_soft=soft
    mq=controller.model_quantities(data)
    mq.update(M=plan['M'],h=plan['h'],J=plan['J'],Jforce=row['J_contact_force'],
              Jroll=row['J_rolling'],contact_force_point=row['contact_force_point'])
    controller.moving=True;controller.anchor=row['stance_anchor'].copy()
    controller.measured=row['observed_contact'];controller.confirmed=row['confirmed_contact']
    controller.load_bearing=row['load_bearing_contact'];controller.event_mode=row['event_mode']
    controller.next_pair=np.setdiff1d(np.arange(4),row['transfer_support_pair'])
    controller.transfer_fraction=float(row['transfer_fraction']);controller.transfer_start=clock if controller.transfer_fraction else None
    # The saved acceleration task contains tracking feedback. Invert that original algebra
    # to restore the instantaneous committed swing command from the same physical record.
    pos=row['feet_position_reference'];vel=row['feet_velocity_reference'];target=row['feet_acceleration_reference']
    swing_acc=target-CFG['wbc']['swing_kp']*(pos-mq['feet'])-CFG['wbc']['swing_kd']*(vel-mq['foot_velocity'])
    controller._swing_reference=lambda t,leg:(pos[leg],vel[leg],swing_acc[leg])
    for leg in np.flatnonzero(~row['observed_contact']):
        controller.swing[leg]=dict(landing=pos[leg])
    blocks=assemble_centroidal(controller.mass,plan['inertia'],plan['com'],plan['feet'][0],
                               plan['x0'][:3],plan['x0'][6:9],friction=CFG['force_pyramid_slope'])
    controller.last_plan=dict(time=clock,forces=forces,state=states,reference=plan['reference'],blocks=blocks)
    start=time.perf_counter()
    action=controller.wbc(clock,data,mq,SimpleNamespace(phase='outbound'),row['planned_contact'],
                          row['observed_contact'],row['measured_grf'],soft,info)
    wall=time.perf_counter()-start
    return action,wall


def run(side):
    saved=Saved(side)
    folder=ROOT/'work/w57-p'/f'wbc-{side}'
    output=OUT/f'force-execution-{side}.jsonl'
    done={json.loads(line)['mpc'] for line in output.open()} if output.exists() else set()
    for i,events in saved.groups():
        if i in done:continue
        with np.load(saved.raw/f'mpc-{i:06d}.npz') as z:plan=dict(z)
        t=float(plan['times'][0]);saved.state(t);tick=round(t/.002)
        with np.load(saved.raw/f'control-{tick//500+1:05d}.npz') as z:
            row={k:z[k][tick%500] for k in z.files}
        fc=Forecast(saved.obs.ground,plan['times'],plan['times'],plan['foot_center_plan'][:-1,:,:2],
                    plan['contact'],saved.obs.published_bottom,row['observed_contact'])
        if not len(fc.keys):continue
        record=dict(attempt=saved.root.name,mpc=i,published_s=t,events=events,reset_keys=int(fc.reset.sum()),
                    clock='Archived 25ms kinematic contact/footprint table; both QPs actually solved, .2ms private material without diagnostic tangents.')
        inputs={k:plan[k] for k in ('x0','reference','last_force','contact','feet','force_normal_upper','stage_A','stage_B','stage_c')}
        s=time.perf_counter();z,st=solve(assemble_force_horizon(inputs,CFG));qp0=time.perf_counter()-s
        nx=inputs['reference'].size;u=z[nx:].reshape(plan['forces'].shape);x=z[:nx].reshape(inputs['reference'].shape)
        record.update(nominal_qp_s=qp0,repeated_nominal_force_N=float(np.abs(plan['forces']-u).max()))
        baseline,wall=replay_wbc(saved,plan,x,u,t,row,folder/f'{i:06}-nominal')
        record.update(nominal_wbc_s=wall,nominal_mapped_vs_archive_rad=float(np.abs(baseline-row['mapped_u']).max()))
        commands={};rankings={}
        for mode in ('full','private_reset'):
            if mode=='private_reset' and not fc.reset.any():
                commands[mode]=commands['full'];record[mode]=dict(reused='full',reason='empty_intervention');continue
            elapsed=[];solve_time=[]
            def propagate(f,h,history,tangents):
                start=time.perf_counter();value=Forecast.propagate(fc,f,h,history,tangents)
                elapsed.append(time.perf_counter()-start);return value
            fc.propagate=propagate
            def time_solver(qp):
                start=time.perf_counter();val=solve(qp);solve_time.append(time.perf_counter()-start);return val
            start=time.perf_counter();c=time.process_time()
            ranked=rank(fc,inputs,plan['inertia'],CFG,u,time_solver,.0002,mode)
            ranking_cpu=time.process_time()-c;ranking_wall=time.perf_counter()-start
            winner=ranked['winner'];rankings[mode]=ranked
            us=ranked['corrected'] if winner else u;xs=ranked['qp_states'] if winner else x
            commands[mode],wb=replay_wbc(saved,plan,xs,us,t,row,folder/f'{i:06}-{mode}')
            record[mode]=dict(winner=winner,objectives=ranked['cost'],two_forecasts_s=elapsed,corrected_qp_solver_s=solve_time[0],
                all_candidate_wall_s=qp0+ranking_wall,ranking_cpu_after_nominal_s=ranking_cpu,wbc_s=wb,
                first_applied_mapped_vs_nominal_rad=float(np.abs(commands[mode]-baseline).max()),
                first_force_vs_nominal_N=float(np.abs(us[0]-u[0]).max()))
        record['first_applied_full_vs_private_reset_rad']=float(np.abs(commands['full']-commands['private_reset']).max())
        with output.open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        print(i,record['nominal_mapped_vs_archive_rad'],record['full']['first_applied_mapped_vs_nominal_rad'],
              record['first_applied_full_vs_private_reset_rad'],flush=True)


if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('case',choices=['a','b']);a=arg.parse_args();run(a.case)
