"""W61 rigid stride task, on the read-only prepared G1 plant and the r55 controller.

Additional controllers are Python mixins. A physical step always comes from the
same MuJoCo model and command hold. Each mixin may only alter the controller.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import osqp
import scipy
from scipy.spatial.transform import Rotation
from s1_go2_r11_runtime import contact_state
from s1_w55_f_r55_backend import FController as StableF
from s1_w54_f_r52_backend import ControllerNumericalFailure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path('/root/autodl-tmp/MCP-RL-w57/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57-m1-b-a-original-e30-a01/model/G1-project-go2.xml')
LEGS = ['FL','FR','RL','RR']
DT = .002


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {'file':path.name, 'sha256':digest(path), 'commit':subprocess.check_output(
        ['git','-C',str(path.parent),'rev-parse','HEAD'],text=True).strip()}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def percentiles(values):
    return np.percentile(values,[50,95,99]).tolist() if len(values) else None


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = json.loads(args.contract.read_text())
    case = spec['cases'][args.case]
    conf = json.loads((ROOT/'experiments/terrain_memory/s1-w55-f-r55.json').read_text())
    conf['w61'] = dict(spec.get('controller',{}), case=case, case_id=args.case,
                       role=args.role, artifact=str(args.fit) if args.fit else None, out=str(out))
    mixins, sources = [], []
    for item in args.mixin:
        module, cls = item.split(':')
        mod = importlib.import_module(module)
        mixins.append(getattr(mod,cls))
        sources.append(identity(mod.__file__))
    controller_cls = type('InstalledF',tuple([*mixins,StableF]),{})
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
    initial_yaw = np.deg2rad(case['yaw_deg'])
    data.qpos[3:7] = [np.cos(initial_yaw/2),0,0,np.sin(initial_yaw/2)]
    mujoco.mj_forward(model,data)
    base = model.body('trunk').id
    feet = np.array([model.geom(n).id for n in LEGS])
    allowed = {0} | {i for i in range(1,model.nbody) if i!=base and model.body_mocapid[i]<0}
    ctl = controller_cls(model,data,feet,base,out,conf)
    mission = SimpleNamespace(task={'turnaround_xy_m':[3,0],'return_xy_m':[0,0]}, phase='initial_stand',
                              v=np.zeros(3),w=np.zeros(3),heading=0.,body_target=np.array([0.,0.,.27]))
    trajectory = []
    timings,sections,events,clocks = [],[],[],[]
    torque_violation,nonfoot_peak,impact,work,slip,length = 0.,0.,0.,0.,0.,0.
    last_pos = data.qpos[:2].copy()
    last_ev = 0
    physical = float(model.opt.timestep)
    substeps = round(DT/physical)
    if not np.isclose(physical,.0002) or substeps!=10:
        raise ValueError('W61 requires the fixed G1 hold/physics ratio')
    runtime = {'status':'running','contract_sha256':digest(args.contract),'case':args.case,
               'run_id':out.name, 'role':args.role, 'mixins':sources,
               'runner':identity(__file__),'baseline':identity(sys.modules[StableF.__module__].__file__),
               'baseline_config_sha256':digest(ROOT/'experiments/terrain_memory/s1-w55-f-r55.json'),
               'fit_sha256':digest(args.fit) if args.fit else None,
               'plant_xml':str(args.model),'plant_xml_sha256':digest(args.model),
               'physics_dt_s':physical,'feedback_dt_s':DT, 'python':platform.python_version(),
               'mujoco':mujoco.__version__, 'numpy':np.__version__,'scipy':scipy.__version__,'osqp':osqp.__version__,
               'cpu_affinity':sorted(os.sched_getaffinity(0)),'deadline':None,'wall_deadline_enabled':False,
               'kill_on_conversation_end':False,'command':sys.argv,'gpu_used':False,'gpu_utilization':None,
               'gpu_peak_bytes':0,'created_unix':time.time()}
    save(out/'runtime.json',runtime)
    start, cpu = time.perf_counter(),time.process_time()
    reason = 'horizon'
    num = round(case['duration_s']/DT)
    program_time = spec['task']['start_move_s']
    for n in range(num):
        t = float(data.time)
        begin = time.perf_counter()
        mujoco.mj_forward(model,data)
        contact, grf, invalid, _, nonfoot,_ = contact_state(model,data,feet,allowed,True)
        phase = 'initial_stand' if n < round(program_time/DT) else 'outbound'
        switch = phase != mission.phase
        mission.phase = phase
        speed = case['speed_m_s'] if t < case['switch_s'] else case['target_speed_m_s']
        mission.v[0] += np.clip((speed if phase=='outbound' else 0.)-mission.v[0],-.5*DT,.5*DT)
        mission.v[1] = np.clip(-.8*data.qpos[1],-.1,.1)
        yaw = Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler('xyz')[2]
        mission.w[2] = np.clip(-1.5*yaw,-.5,.5)
        mission.body_target[:2] += mission.v[:2]*DT
        data.xfrc_applied[base] = 0.
        if case['push_start_s'] <= t < case['push_start_s']+case['push_duration_s']:
            data.xfrc_applied[base,1] = case['lateral_push_N']
        old_event = ctl.event_number
        try:
            action = ctl.update(t,data,mission,contact,grf,np.zeros(4,bool),
                                material_info={},replan=(n%10==0 or switch))
        except ControllerNumericalFailure as err:
            reason = str(err)
            break
        ctrl_elapsed = time.perf_counter()-begin
        data.ctrl[:] = action
        for event in ctl.events[last_ev:]:
            events.append(event)
            if event['kind']=='liftoff_committed' and event['legs']==['FR','RL']:
                R = Rotation.from_euler('z',yaw).as_matrix()
                body_v = R.T @ data.qvel[:3]
                tilt = Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler('xyz')[:2]
                sections.append(dict(t=t,state=[float(body_v[0]),float(body_v[1]),float(tilt[0]),float(tilt[1])],
                    qpos_shape=np.r_[data.qpos[2],tilt,data.qpos[7:]].tolist(),
                    body_linear_velocity=body_v.tolist(),body_angular_velocity=data.qvel[3:6].tolist(),
                    joint_velocity=data.qvel[6:].tolist(),requested_speed=float(mission.v[0]),
                    chart=float(getattr(ctl,'chart_value',0.)), action=float(getattr(ctl,'section_action',0.))))
        last_ev = len(ctl.events)
        body_orient = Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler('xyz')
        trajectory.append(dict(time=t,qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=action.copy(),
            foot_xyz=data.geom_xpos[feet].copy(),contact=contact.copy(),force=grf.copy(),
            planned_force=ctl.last_force.copy(),planned_contact=(ctl.event_mode!='swing').copy(),
            foot_target=np.asarray([ctl._swing_reference(t,i)[0] if ctl.event_mode[i]=='swing' else ctl.anchor[i] for i in range(4)]),
            requested_velocity=mission.v.copy(),body_rpy=body_orient,actuator_force=data.actuator_force.copy(),
            scheduled_torso=ctl.last_plan['reference'][0].copy(),
            slip_velocity=ctl.current_mq['contact_velocity'][:,:2].copy(), ctrl_wall_s=ctrl_elapsed))
        hold_work=0.
        hold_torque=[]
        for _ in range(substeps):
            mujoco.mj_step(model,data)
            real, f, invalid, _, nf,_ = contact_state(model,data,feet,allowed,True)
            actual = data.actuator_force
            torque_violation=max(torque_violation,float(np.max(np.maximum(
                actual-model.actuator_forcerange[:,1],model.actuator_forcerange[:,0]-actual))))
            hold_torque.append(actual.copy())
            hold_work+=float(np.abs(actual*data.qvel[ctl.va]).sum())*physical
            impact=max(impact,float(f[:,2].max()))
            nonfoot_peak=max(nonfoot_peak,float(np.linalg.norm(nf)))
            if invalid:
                reason='invalid_ground_contact'
                break
        work += hold_work
        if hasattr(ctl,'after_physics'):
            end_state=Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler('xyz')
            physical_end=reason != 'horizon' or max(abs(end_state[:2]))>1.2 or data.qpos[2]<.12
            ctl.after_physics(t=t, data=data, mission=mission, command=action.copy(),
                              grf=f.copy(), contact=real.copy(), torque=np.asarray(hold_torque),
                              work_J=hold_work, terminal=bool(physical_end or n+1==num))
        length+=float(np.linalg.norm(data.qpos[:2]-last_pos))
        last_pos=data.qpos[:2].copy()
        slip+=float((np.linalg.norm(ctl.current_mq['contact_velocity'][:,:2],axis=1)*contact).sum())*DT
        elapsed = time.perf_counter()-begin
        timings.append([ctrl_elapsed,elapsed])
        if n%250==0:
            clocks.append([t,time.time()])
        if reason != 'horizon' or np.max(np.abs(body_orient[:2]))>1.2 or data.qpos[2]<.12:
            if reason=='horizon':
                reason='body_fall'
            break
    if hasattr(ctl,'at_rollout_end'):
        ctl.at_rollout_end(t=float(data.time), data=data, mission=mission, reason=reason)
    ctl.flush()
    arrays={k:np.asarray([r[k] for r in trajectory]) for k in trajectory[0]} if trajectory else {}
    np.savez_compressed(out/'trajectory.npz',**arrays)
    save(out/'sections.json',sections)
    cost=None
    if trajectory:
        vel=arrays['qvel'][:,:2]; ref=arrays['requested_velocity'][:,:2]
        cost=float(np.square(ref-vel).sum()*DT)
    timing=np.asarray(timings)
    wall=time.perf_counter()-start
    stats=dict(runtime, status='ended', end_reason=reason,physical_s=float(data.time),planned_s=case['duration_s'],
        control_ticks=len(timings), completed=bool(reason=='horizon' and len(timings)==num),
        x_m=float(data.qpos[0]),y_m=float(data.qpos[1]),travelled_m=length,
        requested_tracking_ISE=cost, mechanical_abs_work_J=work, slip_integral_m=slip,
        maximum_vertical_foot_force_N=impact, nonfoot_force_max_N=nonfoot_peak,
        torque_violation_Nm=torque_violation, section_count=len(sections),
        consecutive_returns=max(0,len(sections)-1), clocks=clocks,wall_s=wall,
        cpu_h=(time.process_time()-cpu)/3600, effective_physics_steps_s=(data.time/physical)/wall,
        effective_control_solves_s=len(timings)/wall,mpc_solves=ctl.plan_count,
        controller_ms_p50_p95_p99=percentiles(timing[:,0]*1000) if len(timing) else None,
        feedback_and_plant_ms_p50_p95_p99=percentiles(timing[:,1]*1000) if len(timing) else None,
        feedback_overdue_fraction=float((timing[:,0]>DT).mean()) if len(timing) else None,
        feedback_and_plant_overdue_fraction=float((timing[:,1]>DT).mean()) if len(timing) else None)
    if hasattr(ctl,'w61_d_stats'):
        stats['wbc']=ctl.w61_d_stats()
    save(out/'result.json',stats)
    print(json.dumps({k:stats[k] for k in ('run_id','end_reason','physical_s','x_m','section_count','wall_s','requested_tracking_ISE')}),flush=True)
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mixin',action='append',default=[])
    p.add_argument('--role',default='fixed')
    p.add_argument('--fit',type=Path)
    run(p.parse_args())
