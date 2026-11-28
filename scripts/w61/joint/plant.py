"""Short Go2 forefoot-entry support/query in the existing reference TM02 strip."""
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, os, resource, subprocess, time
import mujoco
import numpy as np
from s1_go2_joint_material import JointMaterial
from s1_go2_r11_runtime import contact_state
from s1_w54_f_r52_backend import ControllerNumericalFailure,quintic
from s1_w55_f_r55_backend import FController
from tm02.s1_material import ReferenceColumns
from w61.material.coupons import conditions
from w61.joint.heads import MaterialHead,proposal,NA
from w61_gait_rollout import DEFAULT_MODEL

ROOT=Path(__file__).resolve().parents[3]
MATERIAL=Path('/root/autodl-tmp/MCP-RL-w61-material/work/w61/material/weights.pt')
CONTRACT=ROOT/'experiments/terrain_memory/w61/joint/forefoot.json'
G1=DEFAULT_MODEL
DT=.002
TICK=.0002
LEGS=('FL','FR','RL','RR')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,obj):
    temp=p.with_suffix('.writing')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(p)


class Controller(FController):
    def _body_reference(self,t,mq,mission):
        measured,ref=super()._body_reference(t,mq,mission)
        ref[:,5]+=self.adjust['height']
        return measured,ref

    def _start_swing(self,t,leg,data,mq,mission,soft,info):
        super()._start_swing(t,leg,data,mq,mission,soft,info)
        sw=self.swing[leg]
        if self.adjust['dx']==0:return
        start=mq['feet'][leg].copy();v0=mq['foot_velocity'][leg].copy()
        landing=sw['landing'].copy();landing[:2]+=mq['R'][:2,0]*self.adjust['dx']
        mid=(start+landing)*.5;mid[2]=max(start[2],landing[2])+.03
        vmid=(landing-start)/.20;vmid[2]=0.
        sw.update(landing=landing,rise=quintic(start,v0,np.zeros(3),mid,vmid,np.zeros(3),.10),
                  fall=quintic(mid,vmid,np.zeros(3),landing,np.zeros(3),np.zeros(3),.10))


def rollout(folder,theta,case,noise_seed,arm='joint',record=True,head_factory=MaterialHead):
    spec=json.loads(CONTRACT.read_text())
    folder.mkdir(parents=True,exist_ok=False)
    wall=time.monotonic(); cpu=time.process_time()
    model=mujoco.MjModel.from_xml_path(str(G1));data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
    data.qpos[0]+=case['x_m']
    mujoco.mj_forward(model,data)
    feet=np.array([model.geom(s).id for s in LEGS]);trunk=model.body('trunk').id
    allowed={0}|{b for b in range(1,model.nbody) if b!=trunk and model.body_mocapid[b]<0}
    conf=json.loads((ROOT/'experiments/terrain_memory/s1-w55-f-r55.json').read_text())
    ctl=Controller(model,data,feet,trunk,folder,conf)
    fixed=dict(mode=0,dx=0.,height=0.,ramp=.050,candidates=0)
    ctl.adjust=fixed.copy()
    mat=JointMaterial(ROOT,{},full_recording=record)
    laws=json.loads((ROOT/'experiments/terrain_memory/w61/material/family-v1.json').read_text())
    params,_=conditions(case['specimen'],laws)
    mat.ground.law=ReferenceColumns(params)
    # Native contact masks are disabled only on material soles by JointMaterial.
    mat.attach(model,data,feet,allowed,folder,model)
    mission=SimpleNamespace(task={'turnaround_xy_m':[case['x_m']+3,0.],'return_xy_m':[case['x_m'],0.]},
        phase='initial_stand',v=np.zeros(3),w=np.zeros(3),heading=0.,body_target=np.array([case['x_m'],0.,.27]))
    head=head_factory(MATERIAL,spec['frozen_fit_seed'])
    noise_rng=np.random.default_rng(noise_seed)
    k=spec['reference_k_N_m']; cal=None;code=None
    support=[];trace=[];plans=[];timing=[];grf_ticks=[];end_reason='horizon'
    energy=0.;slip=0.;track=0.;query_steps=0;prefix_x=float(data.qpos[0]);pretime=0.
    support_ticks=round(spec['support_s']/DT)
    total=support_ticks+round(spec['query_s']/DT)
    commit=os.environ.get('W61_F_RUNTIME_COMMIT') or subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    identity=dict(case=case,noise_seed=int(noise_seed),theta=np.asarray(theta).tolist(),arm=arm,
                  source=commit,plant_runner_sha256=digest(__file__),head_sha256=digest(Path(__file__).with_name('heads.py')),contract_sha256=digest(CONTRACT),plant_sha256=digest(G1),
                  A_checkpoint=digest(MATERIAL),cpu_affinity=sorted(os.sched_getaffinity(0)),
                  pid=os.getpid(),mujoco=mujoco.__version__,physics_step_s=TICK,control_s=DT,
                  deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
    write(folder/'identity.json',identity)
    for tick in range(total):
        begin=time.perf_counter();t=float(data.time)
        mujoco.mj_forward(model,data)
        on,grf,invalid=mat.contacts(contact_state(model,data,feet,allowed,True))
        if tick==support_ticks:
            saved=[(s['bottom'],s['fz'],s['soft']) for s in support]
            np.savez_compressed(folder/'support.npz',bottom=np.stack([r[0] for r in saved]),fz=np.stack([r[1] for r in saved]),
                                soft=np.stack([r[2] for r in saved]),xy=np.stack([r['xy'] for r in support]),dt_s=np.array([s['dt'] for s in support]))
            flag=mujoco.mjtState.mjSTATE_INTEGRATION
            state=np.empty(mujoco.mj_stateSize(model,flag));mujoco.mj_getState(model,data,state,flag)
            np.savez_compressed(folder/'support-end.npz',state=state,ground_e=mat.ground.e,ground_q=mat.ground.q,
                keys=np.array(list(mat.ground.ids)),ground_loaded=mat.ground.loaded,ground_owner=mat.ground.last_owner,
                modes=ctl.event_mode,anchor=ctl.anchor,confirmed=ctl.confirmed,last_force=ctl.last_force)
            adapt_start=time.perf_counter()
            code,identified_k,cal=head.adapt(theta[:NA],support,spec['inner_steps'])
            if arm not in ('frozen_A','frozen_both','independent_C'):k=identified_k
            pretime=time.perf_counter()-adapt_start;prefix_x=float(data.qpos[0])
            mission.phase='outbound'
        if tick>=support_ticks:
            mission.v[0]=min(case['speed_m_s'],mission.v[0]+.5*DT)
            mission.v[1]=np.clip(-.8*data.qpos[1],-.1,.1)
            mission.body_target[:2]+=mission.v[:2]*DT
            if (tick-support_ticks)%spec['decision_ticks']==0:
                noise=noise_rng.standard_normal((spec['search_rounds'],2,spec['candidates'],2))
                a=proposal(theta[NA:],float(data.qvel[0]),case['speed_m_s'],grf[:,2],k,noise)
                ctl.adjust=fixed.copy() if arm in ('frozen_C','frozen_both') else a
                ctl.cfg['contact_events']['load_transfer_ramp_s']=ctl.adjust['ramp']
                plans.append(dict(t=t,stiffness=k,applied=ctl.adjust,search=a))
        try:
            action=ctl.update(t,data,mission,on,grf,mat.soft,dict(stiffness_N_m=k,force=grf),replan=tick%10==0 or tick==support_ticks)
        except ControllerNumericalFailure as err:
            end_reason=str(err);break
        data.ctrl[:]=action
        row=[t,*data.qpos,*data.qvel,*grf.ravel(),*on,*action,*ctl.last_force.ravel()]
        trace.append(row)
        for micro in range(10):
            if not mat.before_physics(TICK,mission.phase):
                end_reason=mat.reason;break
            if case['lateral_N'] and spec['support_s']+.20 <= t+micro*TICK < spec['support_s']+.26:
                data.xfrc_applied[trunk,1]+=case['lateral_N']
            mujoco.mj_step(model,data)
            footmask,force,invalid=mat.contacts(contact_state(model,data,feet,allowed,True))
            mat.after_physics(TICK,force)
            if tick>=support_ticks:
                query_steps+=1
                energy+=TICK*float(np.abs(data.actuator_force*data.qvel[ctl.va]).sum())
                slip+=TICK*float((np.linalg.norm(mat.velocity[:,:2],axis=1)*footmask).sum())
                track+=TICK*float((data.qvel[0]-mission.v[0])**2+data.qvel[1]**2)
                grf_ticks.append(float(force[:,2].max()))
            if invalid or data.qpos[2]<.12:
                end_reason='nonfoot_or_body_fall';break
        if tick<support_ticks and (tick+1)%spec['support_decimation']==0:
            support.append(dict(dt=DT*spec['support_decimation'],bottom=mat.force_boundary_bottom[:,2].copy(),
                    xy=mat.force_boundary_bottom[:,:2].copy(),fz=mat.force[:,2].copy(),soft=mat.soft.copy()))
        timing.append((time.perf_counter()-begin)*1000)
        if end_reason!='horizon':break
    ctl.flush()
    mat.flush();mat.events_file.close();mat.failures_file.close()
    np.savez_compressed(folder/'trajectory.npz',state=np.asarray(trace),peak_force_per_query_step=grf_ticks)
    write(folder/'plans.json',plans)
    progress=float(data.qpos[0]-prefix_x)
    complete=(query_steps==round(spec['query_s']/TICK))
    target=case['speed_m_s']*(spec['query_s']-case['speed_m_s'])  # area of 0.5 m/s² ramp from rest
    force95=float(np.percentile(grf_ticks,95)) if grf_ticks else 0.
    physical_cost=(track/.015+(max(0.,target-progress)/.08)**2+.07*energy/20+
                   .08*slip/.06+.05*(force95/120)**2+5.*(1-query_steps/(spec['query_s']/TICK)))
    stats=dict(**identity,reason=end_reason,completed=complete,support_samples=len(support),query_steps=query_steps,
        physical_s=float(data.time),progress_m=progress,target_displacement_m=target,tracking_integral=track,
        slip_m=slip,mechanical_abs_work_J=energy,vertical_foot_force_p95_N=force95,
        physical_cost=physical_cost,calibration_rmse_N=cal,
        joint_cost=None if cal is None else physical_cost+spec['calibration_weight']*(cal/40)**2,
        material_code=None if code is None else code.tolist(),stiffness_N_m=k,decisions=len(plans),
        contact_events=len(ctl.events),cpu_seconds=time.process_time()-cpu,wall_s=time.monotonic()-wall,
        maximum_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,adapt_s=pretime,
        latency_ms_p50_p95_p99=np.percentile(timing,[50,95,99]).tolist() if timing else [])
    write(folder/'result.json',stats)
    return stats
