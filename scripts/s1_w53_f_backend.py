#!/usr/bin/env python3
"""W53 F: independent centroidal force MPC and exact-model inverse-dynamics WBC.

No plant stepping, native iLQG, finite-difference dynamics, old B/K/H or fallback
controller occurs here. All feedback torque is solved in WBC, then expressed
through the unchanged affine 60/5 actuator. CPU is the frozen online backend.
"""
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

LEGS = ('FL', 'FR', 'RL', 'RR')
STAND_PHASES = ('initial_stand', 'terminal_stand', 'complete', 'far_hold', 'home_hold')


def skew(x):
    x, y, z = x
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def euler_world_rate(rpy):
    """World angular velocity -> XYZ Euler rates at the frozen orientation."""
    _, pitch, yaw = rpy
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c/np.cos(pitch), s/np.cos(pitch), 0.],
                     [-s, c, 0.], [c*np.tan(pitch), s*np.tan(pitch), 1.]])


def assemble_centroidal(mass, inertia, com, feet, joint_jacobians, h_joint,
                        q_joint, v_joint, torque_range, ctrl_range, rpy, omega,
                        dt=.025, friction=.5):
    """Pure FP64 model/constraint blocks shared with the offline C comparison.

    x=[rpy,COM_world,omega_world,COM_velocity_world]; f=FL,FR,RL,RR xyz.
    h_joint is bias-passive. Static torque inequality is an MPC approximation;
    WBC below constrains exact M*qacc+h-J.T*f and the mapped PD input.
    Returns A,B,c,friction_C,torque_C,torque_l,torque_u. Contact-specific force
    equalities are added by the actual schedule, never by changing measured force.
    """
    E = euler_world_rate(np.asarray(rpy))
    G = np.concatenate([np.linalg.solve(inertia, skew(p-com)) for p in feet], axis=1)
    L = np.tile(np.eye(3)/mass, (1, 4))
    A = np.eye(12)
    A[:3, 6:9] = dt*E
    A[3:6, 9:12] = dt*np.eye(3)
    B = np.zeros((12, 12))
    B[:3] = .5*dt*dt*E@G
    B[3:6] = .5*dt*dt*L
    B[6:9] = dt*G
    B[9:12] = dt*L
    alpha_bias = -np.linalg.solve(inertia, np.cross(omega, inertia@omega))
    c = np.r_[.5*dt*dt*E@alpha_bias, [0.,0.,-.5*9.81*dt*dt],
              dt*alpha_bias, [0.,0.,-9.81*dt]]
    F = np.zeros((16, 12))
    for leg in range(4):
        F[4*leg:4*leg+4, 3*leg:3*leg+3] = [
            [1.,0.,-friction],[-1.,0.,-friction],
            [0.,1.,-friction],[0.,-1.,-friction]]
    T = -np.asarray(joint_jacobians).reshape(12, 12).T
    lo = np.maximum(torque_range[:,0],60.*(ctrl_range[:,0]-q_joint)-5.*v_joint)
    hi = np.minimum(torque_range[:,1],60.*(ctrl_range[:,1]-q_joint)-5.*v_joint)
    return dict(A=A, B=B, c=c, friction_C=F, torque_C=T,
                torque_l=lo-h_joint, torque_u=hi-h_joint)


def quintic(p0, v0, a0, p1, v1, a1, duration):
    """Vector quintic coefficients with actual boundary p/v/a."""
    c = np.zeros((6, 3))
    c[:3] = p0, v0, .5*a0
    T = duration
    mat = np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
    rhs = np.array([p1-(c[0]+c[1]*T+c[2]*T*T),v1-(c[1]+2*c[2]*T),a1-2*c[2]])
    c[3:] = np.linalg.solve(mat, rhs)
    return c


def curve(c, t):
    return (np.array([1.,t,t*t,t**3,t**4,t**5])@c,
            np.array([0.,1.,2*t,3*t*t,4*t**3,5*t**4])@c,
            np.array([0.,0.,2.,6*t,12*t*t,20*t**3])@c)


class ControllerNumericalFailure(RuntimeError):
    pass


class FController:
    def __init__(self, model, data, feet, base, out, config=None):
        if config is None:
            config=json.loads((Path(__file__).resolve().parents[1]/'experiments/terrain_memory/s1-w53-f.json').read_text())
        self.cfg=config
        self.model=model
        self.feet=np.asarray(feet,dtype=int)
        self.base=int(base)
        self.out=Path(out)
        self.record_dir=self.out/'w53-f'
        self.record_dir.mkdir(exist_ok=False)
        self.joint_ids=model.actuator_trnid[:,0]
        self.qa=model.jnt_qposadr[self.joint_ids]
        self.va=model.jnt_dofadr[self.joint_ids]
        self.body_ids=np.array([b for b in range(model.nbody) if self._in_subtree(b)])
        self.mass=float(model.body_mass[self.body_ids].sum())
        self.home_q=data.qpos[self.qa].copy()
        self.home_com_height=float(data.subtree_com[self.base,2])
        self.home_root_height=float(data.qpos[2])
        self.nominal_feet_body=(data.geom_xpos[self.feet]-data.xpos[self.base])@data.xmat[self.base].reshape(3,3)
        self.radius=model.geom_size[self.feet,0].copy()
        self.anchor=data.geom_xpos[self.feet].copy()
        self.last_scheduled=np.ones(4,dtype=bool)
        self.swing=[None]*4
        self.walk_origin=None
        self.stop_time=None
        self.last_plan=None
        self.latest_plan=None
        self.last_force=np.zeros((4,3))
        self.rows=[]
        self.plan_count=0
        self.control_count=0
        self.chunk_count=0
        self.timing={k:dict(wall_s=0.,cpu_s=0.,calls=0) for k in ['model','matrix_assembly','mpc_solve','wbc_assembly','wbc_solve','record_serialization_io']}
        self.manifest=dict(schema='w53_f_records_v1',backend='cpu_mujoco_3.2.3_osqp_1.0.5',
            config=config,legs=LEGS,model_mass_kg=self.mass,model_body_ids=self.body_ids.tolist(),
            command='total feedback torque solved by WBC; u=q+(tau_total+5*v)/60; no added joint PD term',
            model_convention='h=qfrc_bias-qfrc_passive; material/native support counted once as contact-force variables; no qfrc_applied injection',
            contact='plan never changes measured contact; scheduled swing has zero planned force; unobserved planned support uses downward search without WBC reaction',
            material='measured force, local causal stiffness anchor and current normal velocity; no future material state propagation',
            prediction='centroidal state and explicit kinematic foot references; no fabricated articulated rollout',
            full_mpc_files=[],control_chunks=[],timing=self.timing)
        self._write_manifest()

    def _in_subtree(self,b):
        while b and b!=self.base:
            b=int(self.model.body_parentid[b])
        return b==self.base

    def _time(self,key,w,c):
        d=self.timing[key]
        d['wall_s']+=time.perf_counter()-w
        d['cpu_s']+=time.process_time()-c
        d['calls']+=1

    def _write_manifest(self):
        (self.record_dir/'manifest.json').write_text(json.dumps(self.manifest,indent=2)+'\n')

    def model_quantities(self,data):
        w,c=time.perf_counter(),time.process_time()
        m=self.model
        M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,M,data.qM)
        h=data.qfrc_bias.copy()-data.qfrc_passive.copy()
        J=np.zeros((4,3,m.nv));Jdot=np.zeros_like(J)
        for leg,gid in enumerate(self.feet):
            mujoco.mj_jacGeom(m,data,J[leg],None,int(gid))
            mujoco.mj_jacDot(m,data,Jdot[leg],None,data.geom_xpos[gid],int(m.geom_bodyid[gid]))
        Jcom=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,data,Jcom,self.base)
        Jcomdot=np.zeros_like(Jcom)
        inertia=np.zeros((3,3));com=data.subtree_com[self.base].copy()
        for b in self.body_ids:
            mass=float(m.body_mass[b]);rel=data.xipos[b]-com
            R=data.ximat[b].reshape(3,3)
            inertia+=R@np.diag(m.body_inertia[b])@R.T+mass*(np.dot(rel,rel)*np.eye(3)-np.outer(rel,rel))
            jd=np.zeros((3,m.nv))
            mujoco.mj_jacDot(m,data,jd,None,data.xipos[b],int(b))
            Jcomdot+=mass/self.mass*jd
        Jr=np.zeros((3,m.nv));Jrdot=np.zeros_like(Jr)
        mujoco.mj_jacBody(m,data,None,Jr,self.base)
        mujoco.mj_jacDot(m,data,None,Jrdot,data.xpos[self.base],self.base)
        R=data.xmat[self.base].reshape(3,3)
        rpy=Rotation.from_matrix(R).as_euler('xyz')
        out=dict(M=M,h=h,J=J,Jdotv=np.einsum('lij,j->li',Jdot,data.qvel),
                 feet=data.geom_xpos[self.feet].copy(),foot_velocity=np.einsum('lij,j->li',J,data.qvel),
                 com=com,com_velocity=Jcom@data.qvel,Jcom=Jcom,Jcomdotv=Jcomdot@data.qvel,
                 inertia=inertia,Jr=Jr,Jrdotv=Jrdot@data.qvel,omega=Jr@data.qvel,rpy=rpy,R=R)
        self._time('model',w,c)
        return out

    def schedule(self,t):
        if self.walk_origin is None or t<self.walk_origin:
            return np.ones(4,dtype=bool)
        if self.stop_time is not None and t>=self.stop_time:
            return np.ones(4,dtype=bool)
        # First diagonal FL/RR stance .30 s, then swing .20 s. The other pair
        # starts at phase .25: both pairs remain stance for the first .05 s.
        phase=(t-self.walk_origin+np.array([0.,.25,.25,0.]))%.5
        return phase < .3-1e-10

    def _swing_reference(self,t,leg):
        s=self.swing[leg]
        if s is None:
            return self.anchor[leg].copy(),np.zeros(3),np.zeros(3)
        tau=np.clip(t-s['start'],0.,.20)
        if tau<=.10:return curve(s['rise'],tau)
        return curve(s['fall'],tau-.10)

    def _start_swing(self,t,leg,data,mq,mission,soft,material_info):
        p0=mq['feet'][leg].copy();v0=mq['foot_velocity'][leg].copy()
        # A single fixed Raibert touchdown is committed for this entire swing.
        # The absolute end/apex times never move on subsequent MPC updates.
        R=mq['R'];hip_nom=R@self.nominal_feet_body[leg]
        vf=mission.v[:2]
        landing=data.xpos[self.base].copy()+hip_nom
        landing[:2]+= .20*vf+.15*vf+.04*(mq['com_velocity'][:2]-vf)
        landing[2]=self.radius[leg]
        if bool(soft[leg]):
            # At a material foot, observed loaded surface is the only available
            # local surface sample; no future/rest state is queried from plant.
            k=float((material_info or {}).get('stiffness_N_m',1227272.7272727273))
            force=np.asarray((material_info or {}).get('force',np.zeros((4,3))))
            landing[2]=p0[2]+max(0.,force[leg,2])/k
        mid=.5*(p0+landing);mid[2]=max(p0[2],landing[2])+.03
        vmid=(landing-p0)/.20;vmid[2]=0.
        self.swing[leg]=dict(start=float(t),end=float(t+.20),apex=float(t+.10),
            landing=landing,rise=quintic(p0,v0,np.zeros(3),mid,vmid,np.zeros(3),.10),
            fall=quintic(mid,vmid,np.zeros(3),landing,np.zeros(3),np.zeros(3),.10))

    def _update_schedule(self,t,data,mq,mission,soft,material_info):
        moving=mission.phase not in STAND_PHASES
        if moving and self.walk_origin is None:
            self.walk_origin=float(t)
        if not moving and self.walk_origin is not None and self.stop_time is None:
            # Complete any already committed swing; future cycles stop at its
            # fixed touchdown. No reset or moving apex is introduced.
            self.stop_time=max([t]+[s['end'] for s in self.swing if s and s['end']>t])
        if moving and self.stop_time is not None and t>=self.stop_time:
            self.walk_origin=float(t);self.stop_time=None
        contact=self.schedule(t)
        for i in range(4):
            if self.last_scheduled[i] and not contact[i]:
                self._start_swing(t,i,data,mq,mission,soft,material_info)
            if not self.last_scheduled[i] and contact[i]:
                self.anchor[i]=mq['feet'][i]
            if contact[i] and self.walk_origin is None:
                self.anchor[i,:]=mq['feet'][i]
        self.last_scheduled=contact.copy()
        return contact

    def _solve(self,P,q,A,lo,hi,kind,t,warm=None):
        solver=osqp.OSQP()
        solver.setup(P=sparse.triu(sparse.csc_matrix(P),format='csc'),q=q,
            A=sparse.csc_matrix(A),l=lo,u=hi,verbose=False,
            eps_abs=self.cfg['qp']['eps_abs'],eps_rel=self.cfg['qp']['eps_rel'],
            max_iter=self.cfg['qp']['max_iter'],polishing=True,
            adaptive_rho=True,adaptive_rho_interval=50,check_termination=25,
            warm_starting=True)
        if warm is not None:solver.warm_start(x=warm)
        sol=solver.solve(raise_error=False)
        status=dict(status=sol.info.status,status_val=int(sol.info.status_val),
            iterations=int(sol.info.iter),primal_residual=float(sol.info.prim_res),
            dual_residual=float(sol.info.dual_res),objective=float(sol.info.obj_val),
            setup_time_s=float(sol.info.setup_time),solve_time_s=float(sol.info.solve_time))
        if sol.info.status_val not in (1,2):
            sparse.save_npz(self.record_dir/f'failed-{kind}-{t:.6f}-P.npz',sparse.csc_matrix(P))
            sparse.save_npz(self.record_dir/f'failed-{kind}-{t:.6f}-A.npz',sparse.csc_matrix(A))
            np.savez(self.record_dir/f'failed-{kind}-{t:.6f}.npz',q=q,lower=lo,upper=hi)
            (self.record_dir/'solver-failure.json').write_text(json.dumps(dict(time_s=t,kind=kind,**status),indent=2)+'\n')
            raise ControllerNumericalFailure(f'{kind}: {sol.info.status} at physical time {t:.6f}')
        return np.asarray(sol.x),status

    def _body_reference(self,t,mq,mission):
        n=self.cfg['mpc']['intervals'];dt=self.cfg['mpc']['dt_s']
        x0=np.r_[mq['rpy'],mq['com'],mq['omega'],mq['com_velocity']]
        ref=np.tile(x0,(n+1,1))
        desired_v=mission.v.copy()
        desired_w=mission.w.copy()
        for k in range(1,n+1):
            ref[k]=ref[k-1]
            ref[k,9:12]+=np.clip(desired_v-ref[k,9:12],-.5*dt,.5*dt)
            ref[k,6:9]+=np.clip(desired_w-ref[k,6:9],-1.*dt,1.*dt)
            ref[k,3:6]+=.5*dt*(ref[k-1,9:12]+ref[k,9:12])
            ref[k,:3]+=.5*dt*euler_world_rate(ref[k-1,:3])@(ref[k-1,6:9]+ref[k,6:9])
            ref[k,:2]=0.
            # Retain the actual nominal COM height, not a different robot's
            # hard-coded body-origin height. Soft stance motion is independent.
            ref[k,5]=self.home_com_height
            ref[k,11]=0.
            if mission.phase in STAND_PHASES:
                hold_key='turnaround_xy_m' if mission.phase=='far_hold' else 'return_xy_m'
                ref[k,3:5]=np.array(mission.task[hold_key])+(mq['com']-self.model_data_root)[:2]
                ref[k,8]=0.
        return x0,ref

    def _future_feet(self,t,times,mq,ref,mission):
        feet=np.tile(mq['feet'],(len(times),1,1))
        for k,tk in enumerate(times):
            for leg in range(4):
                s=self.swing[leg]
                if s is not None and t>=s['start']-1e-9 and t<s['end']:
                    feet[k,leg]=self._swing_reference(tk,leg)[0] if tk<=s['end'] else s['landing']
                else:
                    feet[k,leg]=self.anchor[leg]
                if self.walk_origin is None or (self.stop_time is not None and tk>=self.stop_time):continue
                off=0. if leg in (0,3) else .25
                phase=(tk-self.walk_origin+off)%.5
                if phase>=.3 and not (s is not None and s['start']<=tk<=s['end']):
                    start=tk-(phase-.3);end=start+.20
                    if start>=t-1e-9:
                        target=self.model_data_root+mq['R']@self.nominal_feet_body[leg]
                        target[:2]+=mission.v[:2]*(end-t+.15)
                        target[2]=self.radius[leg]
                        begin=self.anchor[leg];mid=.5*(begin+target);mid[2]=max(begin[2],target[2])+.03
                        vmid=(target-begin)/.20;vmid[2]=0.
                        tau=tk-start
                        co=quintic(begin,np.zeros(3),np.zeros(3),mid,vmid,np.zeros(3),.10) if tau<=.10 else quintic(mid,vmid,np.zeros(3),target,np.zeros(3),np.zeros(3),.10)
                        feet[k,leg]=curve(co,tau if tau<=.10 else tau-.10)[0]
        return feet

    def replan(self,t,data,mq,mission):
        w,c=time.perf_counter(),time.process_time()
        n=self.cfg['mpc']['intervals'];dt=self.cfg['mpc']['dt_s']
        x0,ref=self._body_reference(t,mq,mission)
        times=t+np.arange(n+1)*dt
        contact=np.array([self.schedule(tk) for tk in times[:-1]])
        foot_plan=self._future_feet(t,times,mq,ref,mission)
        blocks=assemble_centroidal(self.mass,mq['inertia'],mq['com'],mq['feet'],
            mq['J'][:,:,self.va],mq['h'][self.va],data.qpos[self.qa],data.qvel[self.va],
            self.model.actuator_forcerange,self.model.actuator_ctrlrange,mq['rpy'],mq['omega'],
            dt,self.cfg['friction_coefficient'])
        nx=(n+1)*12;nu=n*12;nz=nx+nu
        weights=np.asarray(self.cfg['mpc']['state_weights'])
        qdiag=np.tile(weights,n+1);qdiag[-12:]*=2.
        R=self.cfg['mpc']['force_regularization'];D=self.cfg['mpc']['force_difference_weight']
        P=sparse.diags(np.r_[2*qdiag,np.full(nu,2*R)],format='lil')
        q=np.r_[-2*qdiag*ref.ravel(),np.zeros(nu)]
        # Penalize variation of the force sequence, using prior applied WBC
        # force only for its first knot. No iLQG incumbent is involved.
        for k in range(n):
            fsl=slice(nx+12*k,nx+12*(k+1))
            P[fsl,fsl]+=2*D*sparse.eye(12)
            if k==0:q[fsl]-=2*D*self.last_force.ravel()
            else:
                psl=slice(nx+12*(k-1),nx+12*k)
                P[psl,psl]+=2*D*sparse.eye(12)
                P[psl,fsl]+=-2*D*sparse.eye(12)
                P[fsl,psl]+=-2*D*sparse.eye(12)
        nr=12+n*(12+16+12+12)
        A=sparse.lil_matrix((nr,nz));lo=np.full(nr,-np.inf);hi=np.full(nr,np.inf)
        A[:12,:12]=np.eye(12);lo[:12]=hi[:12]=x0;row=12
        for k in range(n):
            xs=slice(12*k,12*(k+1));xn=slice(12*(k+1),12*(k+2));fs=slice(nx+12*k,nx+12*(k+1))
            # Frozen measured inertia/J; force moment arms follow this same
            # committed/predicted foot plan and continuous body reference.
            stage=assemble_centroidal(self.mass,mq['inertia'],ref[k,3:6],foot_plan[k],
                mq['J'][:,:,self.va],mq['h'][self.va],data.qpos[self.qa],data.qvel[self.va],
                self.model.actuator_forcerange,self.model.actuator_ctrlrange,mq['rpy'],mq['omega'],dt,self.cfg['friction_coefficient'])
            A[row:row+12,xn]=np.eye(12);A[row:row+12,xs]=-stage['A'];A[row:row+12,fs]=-stage['B']
            lo[row:row+12]=hi[row:row+12]=stage['c'];row+=12
            A[row:row+16,fs]=stage['friction_C'];hi[row:row+16]=0.;row+=16
            A[row:row+12,fs]=stage['torque_C'];lo[row:row+12]=stage['torque_l'];hi[row:row+12]=stage['torque_u'];row+=12
            A[row:row+12,fs]=np.eye(12)
            for leg in range(4):
                ss=slice(row+3*leg,row+3*leg+3)
                if contact[k,leg]:lo[row+3*leg+2]=0.
                else:lo[ss]=hi[ss]=0.
            row+=12
        self._time('matrix_assembly',w,c)
        w,c=time.perf_counter(),time.process_time()
        z,status=self._solve(P.tocsc(),q,A.tocsc(),lo,hi,'mpc',t)
        self._time('mpc_solve',w,c)
        states=z[:nx].reshape(n+1,12);forces=z[nx:].reshape(n,4,3)
        self.last_plan=dict(time=t,times=times,state=states,reference=ref,forces=forces,
            feet=foot_plan,contact=contact,blocks=blocks,status=status)
        self.latest_plan=dict(times=times,query_feet=foot_plan.copy(),source='F centroidal and same-schedule foot reference, not a native rollout')
        self.plan_count+=1
        w,c=time.perf_counter(),time.process_time()
        path=self.record_dir/f'mpc-{self.plan_count:06d}.npz'
        np.savez_compressed(path,times=times,states=states,state_reference=ref,forces=forces,
            feet=foot_plan,contact_schedule=contact,inertia=mq['inertia'],com=mq['com'],
            mass=np.array(self.mass),M=mq['M'],h=mq['h'],J=mq['J'],
            **{'stage_'+key:value for key,value in blocks.items()})
        with (self.record_dir/'mpc-solves.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=self.plan_count,time_s=t,file=path.name,**status))+'\n')
        self.manifest['full_mpc_files'].append(dict(call=self.plan_count,time_s=t,path=path.name,size=path.stat().st_size))
        self._time('record_serialization_io',w,c)

    def wbc(self,t,data,mq,mission,scheduled,observed,grf,soft,material_info):
        w,c=time.perf_counter(),time.process_time()
        m=self.model;nv=m.nv;nz=nv+12
        plan=self.last_plan
        stage=min(int(max(0.,t-plan['time'])/self.cfg['mpc']['dt_s']),self.cfg['mpc']['intervals']-1)
        forces=plan['forces'][stage].copy()
        active=scheduled & observed
        # Dynamics includes only currently observed planned stance reactions.
        # Measured unwanted swing force remains untouched in plant/logging.
        force_target=forces.copy();force_target[~active]=0.
        M=mq['M'];h=mq['h'];J=mq['J'].reshape(12,nv)
        D=np.concatenate((M,-J.T),axis=1)
        Dt=D[self.va]
        P=np.eye(nz)*2e-5;q=np.zeros(nz)
        def task(mat,target,weight):
            nonlocal P,q
            P+=2*weight*(mat.T@mat);q-=2*weight*(mat.T@target)
        sample=np.clip(t-plan['time'],0.,self.cfg['mpc']['dt_s'])/self.cfg['mpc']['dt_s']
        xd=(1-sample)*plan['state'][0]+sample*plan['state'][1]
        lin_acc=force_target.sum(axis=0)/self.mass+np.array([0.,0.,-9.81])
        com_acc=lin_acc+self.cfg['wbc']['body_position_kp']*(xd[3:6]-mq['com'])+self.cfg['wbc']['body_velocity_kd']*(xd[9:12]-mq['com_velocity'])
        rot_des=Rotation.from_euler('xyz',xd[:3]).as_matrix()
        rot_error=Rotation.from_matrix(rot_des@mq['R'].T).as_rotvec()
        angular_acc=np.linalg.solve(mq['inertia'],sum((np.cross(mq['feet'][i]-mq['com'],force_target[i]) for i in range(4)),np.zeros(3))-np.cross(mq['omega'],mq['inertia']@mq['omega']))
        angular_acc+=self.cfg['wbc']['orientation_kp']*rot_error+self.cfg['wbc']['angular_velocity_kd']*(xd[6:9]-mq['omega'])
        Ab=np.zeros((3,nz));Ab[:,:nv]=mq['Jcom']
        task(Ab,com_acc-mq['Jcomdotv'],self.cfg['wbc']['body_weight'])
        Ar=np.zeros((3,nz));Ar[:,:nv]=mq['Jr']
        task(Ar,angular_acc-mq['Jrdotv'],self.cfg['wbc']['orientation_weight'])
        pref=np.zeros((4,3));vref=np.zeros((4,3));aref=np.zeros((4,3))
        mode=np.zeros(4,dtype=np.int8)
        k=float((material_info or {}).get('stiffness_N_m',1227272.7272727273))
        for leg in range(4):
            pos=mq['feet'][leg];vel=mq['foot_velocity'][leg]
            if not scheduled[leg]:
                pref[leg],vref[leg],aref[leg]=self._swing_reference(t,leg)
                target=aref[leg]+self.cfg['wbc']['swing_kp']*(pref[leg]-pos)+self.cfg['wbc']['swing_kd']*(vref[leg]-vel)
                weight=self.cfg['wbc']['swing_weight'];mode[leg]=0
            elif not observed[leg]:
                # A fixed downward search until real touchdown, with no
                # fictitious ground reaction before measured support exists.
                pref[leg]=self.anchor[leg];vref[leg,2]=-.03
                target=self.cfg['wbc']['stance_kp']*(pref[leg]-pos)-self.cfg['wbc']['stance_kd']*vel
                target[2]=self.cfg['wbc']['stance_kd']*(-.03-vel[2])
                weight=self.cfg['wbc']['stance_weight'];mode[leg]=2
            else:
                pref[leg]=self.anchor[leg]
                target=self.cfg['wbc']['stance_kp']*(pref[leg]-pos)-self.cfg['wbc']['stance_kd']*vel
                if soft[leg]:
                    # Local causal force anchor rest=p_now+F_now/k. The desired
                    # normal position follows planned load; matching observed
                    # normal velocity avoids a world-fixed material z constraint.
                    rest=pos[2]+max(0.,grf[leg,2])/k
                    pref[leg,2]=rest-max(0.,force_target[leg,2])/k
                    vref[leg,2]=vel[2]
                    target[2]=self.cfg['wbc']['material_normal_kp']*(pref[leg,2]-pos[2])
                    mode[leg]=3
                else:mode[leg]=1
                weight=self.cfg['wbc']['stance_weight']
            aref[leg]=target
            Af=np.zeros((3,nz));Af[:,:nv]=mq['J'][leg]
            task(Af,target-mq['Jdotv'][leg],weight)
        Ap=np.zeros((12,nz));Ap[np.arange(12),self.va]=1.
        posture=self.cfg['wbc']['posture_kp']*(self.home_q-data.qpos[self.qa])-self.cfg['wbc']['posture_kd']*data.qvel[self.va]
        task(Ap,posture,self.cfg['wbc']['posture_weight'])
        Af=np.zeros((12,nz));Af[:,nv:]=np.eye(12)
        task(Af,force_target.ravel(),self.cfg['wbc']['force_weight'])
        # Hard floating dynamics, unilateral/friction forces, complete torque
        # and mapped input ranges, and one-hold joint reachability.
        rows=[D[:6]];lower=[-h[:6]];upper=[-h[:6]]
        tau_lo=np.maximum(m.actuator_forcerange[:,0],60*(m.actuator_ctrlrange[:,0]-data.qpos[self.qa])-5*data.qvel[self.va])
        tau_hi=np.minimum(m.actuator_forcerange[:,1],60*(m.actuator_ctrlrange[:,1]-data.qpos[self.qa])-5*data.qvel[self.va])
        rows.append(Dt);lower.append(tau_lo-h[self.va]);upper.append(tau_hi-h[self.va])
        friction=np.zeros((16,nz));friction[:,nv:]=plan['blocks']['friction_C']
        rows.append(friction);lower.append(np.full(16,-np.inf));upper.append(np.zeros(16))
        force_id=np.zeros((12,nz));force_id[:,nv:]=np.eye(12)
        flo=np.full(12,-np.inf);fhi=np.full(12,np.inf)
        for leg in range(4):
            if active[leg]:flo[3*leg+2]=0.
            else:flo[3*leg:3*leg+3]=fhi[3*leg:3*leg+3]=0.
        rows.append(force_id);lower.append(flo);upper.append(fhi)
        dt=.002;ranges=m.jnt_range[self.joint_ids]
        qfree=data.qpos[self.qa]+dt*data.qvel[self.va]
        rows.append(Ap);lower.append(2*(ranges[:,0]-qfree)/dt**2);upper.append(2*(ranges[:,1]-qfree)/dt**2)
        A=np.vstack(rows);lo=np.concatenate(lower);hi=np.concatenate(upper)
        self._time('wbc_assembly',w,c)
        w,c=time.perf_counter(),time.process_time()
        z,status=self._solve(P,q,A,lo,hi,'wbc',t)
        self._time('wbc_solve',w,c)
        qacc=z[:nv];force=z[nv:].reshape(4,3)
        tau=(D@z+h)[self.va]
        action=data.qpos[self.qa]+(tau+5*data.qvel[self.va])/60.
        dyn=D[:6]@z+h[:6]
        self.last_force=force.copy()
        self.rows.append(dict(time=np.array(t),plan_call=np.array(self.plan_count),
            qacc=qacc,force=force,mpc_force=forces,planned_contact=scheduled,observed_contact=observed,
            active_wbc_contact=active,mode=mode,feet_position_reference=pref,feet_velocity_reference=vref,
            feet_acceleration_reference=aref,body_state_reference=xd,body_acceleration_reference=com_acc,
            angular_acceleration_reference=angular_acc,measured_grf=grf.copy(),foot_velocity=mq['foot_velocity'],
            tau_total=tau,mapped_u=action,q_joint=data.qpos[self.qa].copy(),v_joint=data.qvel[self.va].copy(),
            requested_pd=60*(action-data.qpos[self.qa])-5*data.qvel[self.va],
            dynamics_residual=dyn,mpc_age=np.array(t-plan['time']),
            solver_status=np.array(status['status_val']),solver_iterations=np.array(status['iterations']),
            solver_primal_residual=np.array(status['primal_residual']),solver_dual_residual=np.array(status['dual_residual']),
            torque_lower=tau_lo,torque_upper=tau_hi,
            contact_mode_mismatch=scheduled!=observed))
        self.control_count+=1
        if len(self.rows)>=500:self.flush()
        return action

    def update(self,t,data,mission,contact,grf,soft,material_info=None,replan=False):
        material_info=dict(material_info or {})
        material_info['force']=np.asarray(grf)
        material_info['stiffness_N_m']=float(material_info.get('instantaneous_stiffness_N_m',material_info.get('stiffness_N_m',self.cfg['material_stiffness_N_m'])))
        mq=self.model_quantities(data)
        self.model_data_root=data.xpos[self.base].copy()
        scheduled=self._update_schedule(t,data,mq,mission,soft,material_info)
        if replan or self.last_plan is None:
            self.replan(t,data,mq,mission)
        observed=np.asarray(contact,dtype=bool) & (np.asarray(grf)[:,2]>1.)
        return self.wbc(t,data,mq,mission,scheduled,observed,np.asarray(grf),np.asarray(soft),material_info)

    def flush(self):
        w,c=time.perf_counter(),time.process_time()
        if self.rows:
            self.chunk_count+=1
            path=self.record_dir/f'control-{self.chunk_count:05d}.npz'
            arrays={k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
            np.savez_compressed(path,**arrays)
            self.manifest['control_chunks'].append(dict(path=path.name,samples=len(self.rows),
                start_s=float(self.rows[0]['time']),end_s=float(self.rows[-1]['time']),size=path.stat().st_size))
            self.rows.clear()
        self._time('record_serialization_io',w,c)
        self.manifest.update(mpc_calls=self.plan_count,control_calls=self.control_count)
        self._write_manifest()

    def close(self):
        self.flush()

    def finish(self, final_time, reason):
        self.manifest.update(final_physical_time_s=float(final_time),final_reason=str(reason))
        self.close()
        return self.manifest
