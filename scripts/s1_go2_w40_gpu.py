#!/usr/bin/env python3
"""W40 independent, pinned FP64 MJX predictor/FD adaptation on saved W39 inputs.
This is a new declared fixture, not a replay of missing historic K/H/task state.
No PPO, no real TM02 ground state input, and no main-controller replacement.
"""
import argparse,json,time,os,subprocess,traceback,resource
from pathlib import Path
import numpy as np
import mujoco
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import forward,smooth,collision_driver,constraint,solver,support
ROOT=Path(__file__).resolve().parents[1]
LARGE=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
OUT=LARGE/'gpu/w40-gpu'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w40-gpu'
HALF=.015811388300841896
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def numeric(m,name):
    i=m.numeric(name).id
    return np.array(m.numeric_data[m.numeric_adr[i]:m.numeric_adr[i]+m.numeric_size[i]])
def inside(p,xp=np):
    return (p[...,0]>=.8+HALF)&(p[...,0]<=2.2-HALF)&(xp.abs(p[...,1])<=.6-HALF)
def fixtures():
    specs=[('a','rigid_normal',1.),('a','first_large_feedback',9.224),
           ('a','first_load',9.850),('b','nominal_saturation',14.734),
           ('b','first_large_feedback',15.044),('b','pre_load_peak',15.264),
           ('b','first_load',15.382),('domain','last_domain_boundary',14.3648)]
    rows=[]
    for stream,label,t in specs:
        name=('train-job-20261105T032354Z-w39-a-PANN_PPO_S5-s1-ep00260' if stream=='domain' else
              f'job-20261105T032354Z-w39-{stream}-eval-PANN_PPO_S5-s{1 if stream=="a" else 2}-E30')
        folder=LARGE/name
        with np.load(folder/'trajectory.npz') as a:
            # Terminal state may be newer than its copied action diagnostics.
            times=a['diagnostic_time'];i=int(np.argmin(abs(times-t)))
            while i>0 and a['time'][i]!=times[i]:i-=1
            rows.append(dict(name=stream+'_'+label,source=str(folder.relative_to(ROOT)),
                index=i,requested_time_s=t,time=float(a['time'][i]),qpos=a['qpos'][i].copy(),
                qvel=a['qvel'][i].copy(),ctrl=a['q_target'][i].copy(),
                mocap_pos=a['mocap_pos'][i].copy(),mocap_quat=a['mocap_quat'][i].copy(),
                userdata=a['support_reference_userdata'][i].copy(),
                task_phase=int(a['phase'][i]),task_history='saved phase only; private transitions/K/H not reconstructed'))
    return rows
class Predictor:
    def __init__(self,model_path):
        self.m=m=mujoco.MjModel.from_binary_path(str(model_path))
        # Match Agent::PlanIteration, not the plant MJB's stored Euler option.
        m.opt.timestep=float(numeric(m,'agent_timestep')[0])
        m.opt.integrator=int(numeric(m,'agent_integrator')[0])
        m.jnt_solimp[:,0]=0;m.geom_solimp[:,0]=0;m.pair_solimp[:,0]=0
        m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        self.feet=np.array([m.geom(n).id for n in ('FL','RL','FR','RR')])
        self.bodies=m.geom_bodyid[self.feet]
        floor=m.geom('floor').id;trunk=m.body('trunk').id
        self.trunk=np.array([i for i in range(m.ngeom) if m.geom_bodyid[i]==trunk and
            ((m.geom_contype[i]&m.geom_conaffinity[floor]) or (m.geom_contype[floor]&m.geom_conaffinity[i]))])
        self.floor=float(numeric(m,'s1_flat_ground_height')[0])
        self.stiff=float(numeric(m,'r10_local_contact')[0])
        self.domain=numeric(m,'r11_domain_margin');self.impact=numeric(m,'r26_impact_margin')
        self.band=float(numeric(m,'r11_landing_approach_band')[0])
        self.clear=float(numeric(m,'r12_trunk_clearance')[0])
        self.margins=np.maximum(m.geom_margin[floor],m.geom_margin[self.trunk])
        # _full_compat permits USER residual sensors; they are computed explicitly
        # below, not silently treated as stock MJX sensors or as a full task cost.
        self.mx=mjx.put_model(m,_full_compat=True)
        # MJX carries mesh/input arrays that may originate as float32.
        # Preserve their values but perform every floating calculation in FP64.
        fp64=lambda x:x.astype(jp.float64) if hasattr(x,'dtype') and np.issubdtype(x.dtype,np.floating) else x
        self.mx=jax.tree.map(fp64,self.mx)
        self.template=jax.tree.map(fp64,mjx.make_data(self.mx))
        self.joints=m.actuator_trnid[:,0];self.qa=m.jnt_qposadr[self.joints];self.va=m.jnt_dofadr[self.joints]
    def cpu_filter(self,m,d,g1,g2):
        return int(any(g in self.feet and inside(d.geom_xpos[g]) for g in (g1,g2)))
    def cpu_passive(self,m,d):
        for g in self.feet:
            if not inside(d.geom_xpos[g]):continue
            p=d.geom_xpos[g].copy();p[2]-=m.geom_size[g,0]
            patch=d.userdata[62:78].reshape(4,4)
            overlap=np.maximum(1-np.abs(p[0]-patch[:,1])/.03162277660168379,0)*np.maximum(1-np.abs(p[1]-patch[:,2])/.03162277660168379,0)
            rest=np.sum(patch[:,0]*overlap*patch[:,3])
            fz=self.stiff*max(self.floor+rest-p[2],0)
            vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_GEOM,int(g),vel,0)
            vx=vel[3]-m.geom_size[g,0]*vel[1];vy=vel[4]+m.geom_size[g,0]*vel[0]
            slip=np.sqrt(vx*vx+vy*vy+.01**2)
            mujoco.mj_applyFT(m,d,np.array([-.5*fz*vx/slip,-.5*fz*vy/slip,fz]),
                np.zeros(3),p,int(m.geom_bodyid[g]),d.qfrc_passive)
    def cpu_data(self,row,qpos=None,qvel=None,ctrl=None):
        d=mujoco.MjData(self.m)
        d.qpos[:]=row['qpos'] if qpos is None else qpos
        d.qvel[:]=row['qvel'] if qvel is None else qvel
        d.ctrl[:]=row['ctrl'] if ctrl is None else ctrl
        d.time=row['time'];d.mocap_pos[:]=row['mocap_pos'];d.mocap_quat[:]=row['mocap_quat'];d.userdata[:]=row['userdata']
        return d
    def gpu_data(self,row):
        return self.template.replace(qpos=jp.array(row['qpos'],dtype=jp.float64),qvel=jp.array(row['qvel'],dtype=jp.float64),
            ctrl=jp.array(row['ctrl'],dtype=jp.float64),time=jp.array(row['time']),mocap_pos=jp.array(row['mocap_pos'],dtype=jp.float64),
            mocap_quat=jp.array(row['mocap_quat'],dtype=jp.float64),userdata=jp.array(row['userdata'],dtype=jp.float64))
    def step_gpu(self,d):
        m=self.mx
        d=smooth.kinematics(m,d);d=smooth.com_pos(m,d);d=smooth.camlight(m,d)
        d=smooth.tendon(m,d);d=smooth.crb(m,d);d=smooth.factor_m(m,d)
        d=collision_driver.collision(m,d)
        # Exactly the r10 contactfilter: suppress every contact involving a
        # material-foot geom, preserve all other original collision geometry.
        soft=inside(d.geom_xpos[self.feet],jp)
        filtered=jp.any((d.contact.geom[:,:,None]==jp.array(self.feet)[None,None,:])&soft[None,None,:],axis=(1,2))
        d=d.replace(contact=d.contact.replace(dist=jp.where(filtered,1e10,d.contact.dist)))
        d=constraint.make_constraint(m,d);d=smooth.transmission(m,d)
        d=forward.fwd_velocity(m,d)
        extra=jp.zeros(m.nv);patch=d.userdata[62:78].reshape(4,4)
        for k,g in enumerate(self.feet):
            point=d.geom_xpos[g].at[2].add(-self.m.geom_size[g,0])
            overlap=jp.maximum(1-jp.abs(point[0]-patch[:,1])/.03162277660168379,0)*jp.maximum(1-jp.abs(point[1]-patch[:,2])/.03162277660168379,0)
            rest=jp.sum(patch[:,0]*overlap*patch[:,3])
            fz=self.stiff*jp.maximum(self.floor+rest-point[2],0)*soft[k]
            jacp,_=support.jac(m,d,point,int(self.bodies[k]))
            velocity=d.qvel@jacp;slip=jp.sqrt(velocity[0]**2+velocity[1]**2+.01**2)
            force=jp.array([-.5*fz*velocity[0]/slip,-.5*fz*velocity[1]/slip,fz])
            extra+=jacp@force
        d=d.replace(qfrc_passive=d.qfrc_passive+extra)
        d=forward.fwd_actuation(m,d);d=forward.fwd_acceleration(m,d)
        d=solver.solve(m,d) if d.efc_J.size else d.replace(qacc=d.qacc_smooth)
        return forward.implicit(m,d)
    def risk_gpu(self,d):
        m=self.mx;d=smooth.kinematics(m,d);d=smooth.com_pos(m,d);d=smooth.com_vel(m,d)
        p=d.geom_xpos[self.feet];r=jp.array(self.m.geom_size[self.feet,0])
        over=self.floor+r-p[:,2]-self.domain[0]
        domain=inside(p,jp)*.5*(over+jp.sqrt(over**2+1e-12))/self.domain[1]
        def ss(u):
            v=jp.clip(u,0,1);return v**3*(10-15*v+6*v*v)
        gate=ss((p[:,0]-(.8+HALF)+self.band)/self.band)*ss(((2.2-HALF)+self.band-p[:,0])/self.band)*ss(((.6-HALF)+self.band-jp.abs(p[:,1]))/self.band)
        vz=[]
        for k,g in enumerate(self.feet):
            jac,_=support.jac(m,d,d.geom_xpos[g],int(self.bodies[k]));vz.append((d.qvel@jac)[2])
        down=jp.maximum(-jp.array(vz),0);z=p[:,2]-r-self.floor
        over=.002*down+down*down/(2*abs(self.m.opt.gravity[2]))-(z+self.impact[0])
        impact=gate*.5*(over+jp.sqrt(over**2+1e-12))/self.impact[1]
        clear=[]
        for k,g in enumerate(self.trunk):
            size=self.m.geom_size[g];typ=int(self.m.geom_type[g]);axis=d.geom_xmat[g][2]
            if typ==6:extent=jp.sum(jp.abs(axis)*size)
            elif typ==3:extent=size[0]+size[1]*jp.abs(axis[2])
            elif typ==2:extent=size[0]
            else:raise NotImplementedError('trunk geometry '+str(typ))
            distance=jp.minimum(d.geom_xpos[g,2]-extent-self.floor,1.)
            over=self.margins[k]-distance;clear.append(.5*(over+jp.sqrt(over**2+1e-12))/self.clear)
        return jp.concatenate([domain,jp.array(clear),impact])
    def cpu_risk(self,d):
        m=self.m;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d)
        out=[]
        for g in self.feet:
            over=self.floor+m.geom_size[g,0]-d.geom_xpos[g,2]-self.domain[0]
            out.append(float(inside(d.geom_xpos[g]))*.5*(over+np.sqrt(over**2+1e-12))/self.domain[1])
        for k,g in enumerate(self.trunk):
            distance=mujoco.mj_geomDistance(m,d,m.geom('floor').id,int(g),1.,None)
            over=self.margins[k]-distance;out.append(.5*(over+np.sqrt(over**2+1e-12))/self.clear)
        for g in self.feet:
            p=d.geom_xpos[g]
            def ss(u):v=np.clip(u,0,1);return v**3*(10-15*v+6*v*v)
            gate=ss((p[0]-(.8+HALF)+self.band)/self.band)*ss(((2.2-HALF)+self.band-p[0])/self.band)*ss(((.6-HALF)+self.band-abs(p[1]))/self.band)
            vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_GEOM,int(g),vel,0)
            down=max(-vel[5],0);z=p[2]-m.geom_size[g,0]-self.floor
            over=.002*down+down*down/(2*abs(m.opt.gravity[2]))-(z+self.impact[0])
            out.append(gate*.5*(over+np.sqrt(over**2+1e-12))/self.impact[1])
        return np.array(out)
def state(d):return np.r_[np.asarray(d.qpos),np.asarray(d.qvel)]
def diff(m,a,b):
    dq=np.empty(m.nv);mujoco.mj_differentiatePos(m,dq,1.,a[:m.nq],b[:m.nq])
    return np.r_[dq,b[m.nq:]-a[m.nq:]]
def sync(x):jax.block_until_ready(x);return x
def main():
    OUT.mkdir(parents=True,exist_ok=True);SMALL.mkdir(parents=True,exist_ok=True)
    clockpath=OUT/'clock.json'
    if clockpath.exists():clock=json.loads(clockpath.read_text())
    else:
        now=time.time();clock=dict(started_unix_s=now,deadline_unix_s=now+28*3600);write(clockpath,clock)
    report=dict(task='w40-gpu',status='preparing_saved_fixtures',clock=clock,
        version='r28 predictor versus same r28 predictor, fixed held target; main uses r29',
        versions=dict(mujoco=mujoco.__version__,mjx='3.2.3',jax=jax.__version__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        host_affinity=sorted(os.sched_getaffinity(0)),dtype='float64',jax_devices=[str(x) for x in jax.devices()],
        tolerance=dict(position_rad_or_m=5e-7,velocity=1e-5,AB_max_abs=5e-3,risk=1e-5),
        matched=dict(dt=.0002,hold_substeps=10,nu=12,nq=19,nv=18,FD_eps=1e-6,FD='one_sided',
                     MakeDifferentiable='joint/geom/pair solimp0=0',coldstart=True,
                     contact='R10 native-foot filter + four frozen userdata patches',
                     task_private_state='not used by fixed-target dynamics; full cost/planner remains unimplemented'),
        results=[],remaining=['full task residual/cost and private phase reconstruction','feedback candidate ranking','whole planner integration'],
        formal_qualification=False,main_hot_swap=False)
    def publish(status):
        report['status']=status;report['updated_unix_s']=time.time()
        report['max_host_rss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['gpu_memory']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
        write(OUT/'result.json',report);write(SMALL/'result.json',report)
        print(json.dumps(dict(status=status,time=time.time(),completed=len(report['results']))),flush=True)
    try:
        rows=fixtures()
        report['fixtures']=[{k:v for k,v in x.items() if not isinstance(v,np.ndarray)} for x in rows]
        np.savez_compressed(OUT/'fixtures.npz',**{k:np.stack([x[k] for x in rows]) for k in ('qpos','qvel','ctrl','userdata','mocap_pos','mocap_quat')})
        pred=Predictor(ROOT/rows[0]['source']/'model.mjb');m=pred.m
        mujoco.set_mjcb_contactfilter(pred.cpu_filter);mujoco.set_mjcb_passive(pred.cpu_passive)
        gpurows=[pred.gpu_data(x) for x in rows]
        transfer_start=time.perf_counter()
        batch=jax.tree.map(lambda *x:jp.stack(x),*gpurows);sync(batch)
        report['H2D_batch_construction_sync_s']=time.perf_counter()-transfer_start
        step=jax.jit(jax.vmap(pred.step_gpu))
        risk=jax.jit(jax.vmap(pred.risk_gpu))
        publish('compiling_exact_model_one_substep')
        t=time.perf_counter();one=sync(step(batch));jit1=time.perf_counter()-t
        report['one_step_JIT_and_first_s']=jit1
        publish('running_one_and_ten_substeps')
        cpu=[];cpurisk=[]
        t=time.perf_counter()
        for row in rows:
            d=pred.cpu_data(row);seq=[];rr=[]
            for j in range(10):
                rr.append(pred.cpu_risk(d));mujoco.mj_step(m,d);seq.append(state(d))
            cpu.append(seq);cpurisk.append(rr)
        report['CPU_serial_ten_substeps_s']=time.perf_counter()-t
        gpu=[];gr=[];d=batch
        report['first_risk_call_includes_JIT']=True
        risk_jit_start=time.perf_counter();sync(risk(batch))
        report['risk_JIT_and_first_s']=time.perf_counter()-risk_jit_start
        for j in range(10):
            gr.append(np.asarray(sync(risk(d))));d=sync(step(d))
            gpu.append(np.c_[np.asarray(d.qpos),np.asarray(d.qvel)])
        gpu=np.swapaxes(np.array(gpu),0,1);gr=np.swapaxes(np.array(gr),0,1);cpu=np.array(cpu);cpurisk=np.array(cpurisk)
        err=np.array([[diff(m,cpu[i,j],gpu[i,j]) for j in range(10)] for i in range(len(rows))])
        timings=[]
        for _ in range(10):
            t=time.perf_counter();sync(step(batch));timings.append(time.perf_counter()-t)
        report['one_step_batch_steady_s']=dict(p50=float(np.percentile(timings,50)),p95=float(np.percentile(timings,95)),samples=timings)
        for i,row in enumerate(rows):
            report['results'].append(dict(fixture=row['name'],position_max=float(abs(err[i,:,:18]).max()),
              velocity_max=float(abs(err[i,:,18:]).max()),risk_max=float(abs(gr[i]-cpurisk[i]).max()),
              earliest_peak_cpu=np.argmax(cpurisk[i],axis=0).tolist(),earliest_peak_gpu=np.argmax(gr[i],axis=0).tolist()))
        np.savez_compressed(OUT/'one-ten-substeps.npz',cpu=cpu,gpu=gpu,state_error=err,cpu_risk=cpurisk,gpu_risk=gr)
        publish('running_independent_target_FD_prefix_chain')
        # Batch49 independent state/action inputs at EACH nominal prefix state.
        # Compose original A_j/B_j with P,Q; do not redifferentiate a closed policy.
        fdout=[]
        for ri,row in enumerate(rows):
            if time.time()>=clock['deadline_unix_s']:raise TimeoutError('original GPU deadline')
            d=pred.cpu_data(row);P=np.eye(36);Q=np.zeros((36,12));Pg=P.copy();Qg=Q.copy()
            prefixes=[];gprefixes=[];local=[];nominal_states=[]
            for j in range(10):
                q=d.qpos.copy();v=d.qvel.copy();u=d.ctrl.copy()
                nominal_states.append((q.copy(),v.copy(),float(d.time)))
                qs=[];vs=[];us=[]
                for k in range(49):
                    qq=q.copy();vv=v.copy();uu=u.copy()
                    if 1<=k<=18:
                        tangent=np.zeros(18);tangent[k-1]=1e-6;mujoco.mj_integratePos(m,qq,tangent,1.)
                    elif 19<=k<=36:vv[k-19]+=1e-6
                    elif k>=37:uu[k-37]+=1e-6
                    qs.append(qq);vs.append(vv);us.append(uu)
                fixture=dict(row,time=float(d.time),qpos=q,qvel=v,ctrl=u)
                dd=pred.gpu_data(fixture)
                bd=jax.tree.map(lambda x:jp.broadcast_to(x,(49,)+x.shape),dd)
                bd=bd.replace(qpos=jp.array(qs),qvel=jp.array(vs),ctrl=jp.array(us))
                gt=sync(step(bd));ys=np.c_[np.asarray(gt.qpos),np.asarray(gt.qvel)]
                cg=[]
                for k in range(49):
                    cd=pred.cpu_data(fixture,qs[k],vs[k],us[k]);mujoco.mj_step(m,cd);cg.append(state(cd))
                cg=np.array(cg)
                AB=np.stack([diff(m,cg[0],cg[k])/1e-6 for k in range(1,49)],axis=1)
                ABg=np.stack([diff(m,ys[0],ys[k])/1e-6 for k in range(1,49)],axis=1)
                prefixes.append((P.copy(),Q.copy()));gprefixes.append((Pg.copy(),Qg.copy()))
                P,Q=AB[:,:36]@P,AB[:,:36]@Q+AB[:,36:]
                Pg,Qg=ABg[:,:36]@Pg,ABg[:,:36]@Qg+ABg[:,36:]
                local.append(float(abs(AB-ABg).max()));mujoco.mj_step(m,d)
            peaks=np.argmax(cpurisk[ri],axis=0);nr=cpurisk.shape[-1]
            C=np.zeros((nr,36));D=np.zeros((nr,12));Cg=C.copy();Dg=D.copy()
            for j in np.unique(peaks):
                q,v,tt=nominal_states[j]
                fixture=dict(row,time=tt,qpos=q,qvel=v)
                qs=[];vs=[];cv=[]
                for k in range(37):
                    qq=q.copy();vv=v.copy()
                    if 1<=k<=18:
                        tangent=np.zeros(18);tangent[k-1]=1e-6;mujoco.mj_integratePos(m,qq,tangent,1.)
                    elif k>=19:vv[k-19]+=1e-6
                    qs.append(qq);vs.append(vv)
                    cv.append(pred.cpu_risk(pred.cpu_data(fixture,qq,vv)))
                dd=pred.gpu_data(fixture)
                bd=jax.tree.map(lambda x:jp.broadcast_to(x,(37,)+x.shape),dd)
                bd=bd.replace(qpos=jp.array(qs),qvel=jp.array(vs))
                gv=np.asarray(sync(risk(bd)));cv=np.array(cv)
                rx=((cv[1:]-cv[0])/1e-6).T;rxg=((gv[1:]-gv[0])/1e-6).T
                selected=np.where(peaks==j)[0]
                C[selected]=rx[selected]@prefixes[j][0];D[selected]=rx[selected]@prefixes[j][1]
                Cg[selected]=rxg[selected]@gprefixes[j][0];Dg[selected]=rxg[selected]@gprefixes[j][1]
            # Both prefix chains were linearized at IDENTICAL CPU nominal states.
            fdout.append(dict(fixture=row['name'],local_AB_max=max(local),
                              composed_A_max=float(abs(P-Pg).max()),composed_B_max=float(abs(Q-Qg).max()),
                              active_branch_C_max=float(abs(C-Cg).max()),active_branch_D_max=float(abs(D-Dg).max())))
            np.savez_compressed(OUT/(row['name']+'-derivatives.npz'),A=P,B=Q,A_gpu=Pg,B_gpu=Qg,C=C,D=D,C_gpu=Cg,D_gpu=Dg,
              prefix_P=np.array([x[0] for x in prefixes]),prefix_Q=np.array([x[1] for x in prefixes]),
              prefix_P_gpu=np.array([x[0] for x in gprefixes]),prefix_Q_gpu=np.array([x[1] for x in gprefixes]))
            report['FD_results']=fdout;publish('FD_prefix_fixture_completed')
        report['completed_layer']='FP64 batch custom predictor, one/ten steps, independent-target per-substep A/B and composed P/Q'
        report['risk_derivatives']='Original earliest active pre-step risk branch, direct-u zero, geometry FD composed through P/Q; peak switches remain nonsmooth.'
        report['aligned']=all(x['position_max']<=5e-7 and x['velocity_max']<=1e-5 and x['risk_max']<=1e-5 for x in report['results']) and all(x['local_AB_max']<=5e-3 for x in fdout)
        report['next_layer']='Fixed feedback candidates/full planner requires remaining interfaces; no hot swap or physical qualification.'
        publish('completed_batch_layer')
    except Exception:
        report['error']=traceback.format_exc();publish('technical_interruption')
        raise
if __name__=='__main__':main()
