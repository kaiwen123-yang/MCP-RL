#!/usr/bin/env python3
"""W54 F/r52: independent centroidal force MPC and exact-model inverse-dynamics WBC.

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


def assemble_centroidal(mass, inertia, com, feet, rpy, omega, dt=.025, friction=.5/np.sqrt(2.)):
    """SRB dynamics; friction argument is the frozen inner-square axis slope.

    Physical Coulomb mu remains .5; slope=mu/sqrt(2) ensures every allowed
    tangential vector lies inside the plant elliptic cone. Full WBC retains
    the same original articulated bounds.
    """
    E=euler_world_rate(np.asarray(rpy))
    G=np.concatenate([np.linalg.solve(inertia,skew(p-com)) for p in feet],axis=1)
    L=np.tile(np.eye(3)/mass,(1,4))
    A=np.eye(12);A[:3,6:9]=dt*E;A[3:6,9:12]=dt*np.eye(3)
    B=np.zeros((12,12));B[:3]=.5*dt*dt*E@G;B[3:6]=.5*dt*dt*L
    B[6:9]=dt*G;B[9:12]=dt*L
    alpha=-np.linalg.solve(inertia,np.cross(omega,inertia@omega))
    c=np.r_[.5*dt*dt*E@alpha,[0.,0.,-.5*9.81*dt*dt],dt*alpha,[0.,0.,-9.81*dt]]
    F=np.zeros((16,12))
    for leg in range(4):
        F[4*leg:4*leg+4,3*leg:3*leg+3]=[[1,0,-friction],[-1,0,-friction],[0,1,-friction],[0,-1,-friction]]
    return dict(A=A,B=B,c=c,friction_C=F)


def assemble_force_horizon(inputs,cfg):
    """Complete CPU QP, ordered [x_0..x_N, f_0..f_N-1], shared with C."""
    As,Bs,cs=(np.asarray(inputs[k]) for k in ('stage_A','stage_B','stage_c'))
    ref=np.asarray(inputs['reference']);x0=np.asarray(inputs['x0'])
    contact=np.asarray(inputs['contact'],bool);n=len(contact)
    nx=(n+1)*12;nu=n*12;nz=nx+nu
    weights=np.asarray(cfg['mpc']['state_weights'],dtype=float)
    qdiag=np.tile(weights,n+1);qdiag[-12:]*=2
    R=cfg['mpc']['force_regularization'];D=cfg['mpc']['force_difference_weight']
    P=sparse.diags(np.r_[2*qdiag,np.full(nu,2*R)],format='lil')
    q=np.r_[-2*qdiag*ref.ravel(),np.zeros(nu)]
    for k in range(n):
        fs=slice(nx+12*k,nx+12*(k+1));P[fs,fs]+=2*D*sparse.eye(12)
        if k==0:q[fs]-=2*D*np.asarray(inputs['last_force']).ravel()
        else:
            ps=slice(nx+12*(k-1),nx+12*k)
            P[ps,ps]+=2*D*sparse.eye(12);P[ps,fs]-=2*D*sparse.eye(12);P[fs,ps]-=2*D*sparse.eye(12)
    F=np.zeros((16,12));mu=cfg['force_pyramid_slope']
    for leg in range(4):
        F[4*leg:4*leg+4,3*leg:3*leg+3]=[[1,0,-mu],[-1,0,-mu],[0,1,-mu],[0,-1,-mu]]
    nr=12+n*(12+16+12)
    A=sparse.lil_matrix((nr,nz));lo=np.full(nr,-np.inf);hi=np.full(nr,np.inf)
    A[:12,:12]=np.eye(12);lo[:12]=hi[:12]=x0;row=12
    caps=np.asarray(inputs['force_normal_upper'])
    for k in range(n):
        xs=slice(12*k,12*(k+1));xn=slice(12*(k+1),12*(k+2));fs=slice(nx+12*k,nx+12*(k+1))
        A[row:row+12,xn]=np.eye(12);A[row:row+12,xs]=-As[k];A[row:row+12,fs]=-Bs[k]
        lo[row:row+12]=hi[row:row+12]=cs[k];row+=12
        A[row:row+16,fs]=F;hi[row:row+16]=0;row+=16
        A[row:row+12,fs]=np.eye(12)
        for leg in range(4):
            sl=slice(row+3*leg,row+3*leg+3)
            if contact[k,leg]:lo[row+3*leg+2]=0;hi[row+3*leg+2]=caps[k,leg]
            else:lo[sl]=hi[sl]=0
        row+=12
    return dict(P=P.tocsc(),q=q,A=A.tocsc(),lower=lo,upper=hi)


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
            config=json.loads((Path(__file__).resolve().parents[1]/'experiments/terrain_memory/s1-w54-f-r52.json').read_text())
        self.cfg=config
        self.model=model
        self.feet=np.asarray(feet,dtype=int)
        self.base=int(base)
        self.out=Path(out)
        self.record_dir=self.out/'w54-f'
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
        self.initial_landing=self.anchor.copy();self.initial_landing[:,2]=self.radius
        self.last_scheduled=np.ones(4,dtype=bool)
        self.swing=[None]*4
        self.walk_origin=None
        self.stop_time=None
        self.last_plan=None
        self.latest_plan=None
        self.last_force=np.zeros((4,3))
        self.event_mode=np.full(4,'stance',dtype='<U8')
        self.confirmed=np.zeros(4,dtype=bool)
        self.load_bearing=np.zeros(4,dtype=bool)
        self.touch_count=np.zeros(4,dtype=int);self.release_count=np.zeros(4,dtype=int)
        self.load_count=np.zeros(4,dtype=int)
        self.next_pair=np.array([1,2]);self.last_liftoff=None;self.next_liftoff=None
        self.transfer_start=None;self.transfer_fraction=0.
        self.event_number=0;self.events=[];self.last_body_residual=np.inf
        self.last_angular_residual=np.inf
        self.measured=np.zeros(4,dtype=bool)
        self.handover_diagnostics={}
        self.moving=False
        self.rows=[]
        self.plan_count=0
        self.control_count=0
        self.chunk_count=0
        self.timing={k:dict(wall_s=0.,cpu_s=0.,calls=0) for k in ['model','matrix_assembly','mpc_solve','wbc_assembly','wbc_solve','record_serialization_io']}
        self.manifest=dict(schema='w54_f_records_v1',backend='cpu_mujoco_3.2.3_osqp_1.0.5',
            config=config,legs=LEGS,model_mass_kg=self.mass,model_body_ids=self.body_ids.tolist(),
            command='full WBC solves mean torque over10force-evaluation instants; one original2ms position input; tau_start and every predicted substep obey original bounds',
            model_convention='h=qfrc_bias-qfrc_passive+bounded_reference_direction_dry_friction; contact counted once; no qfrc_constraint copy',
            contact='native current contact or causal material force bottom supplies reaction Jacobian; sphere rolling contact task; actual touchdown anchor retained; measured support and load-bearing handover separated',
            material='measured force, local causal stiffness anchor and current normal velocity; no future material state propagation',
            prediction='centroidal state and explicit kinematic foot references; measured current reaction independent of touchdown/load debounce; no fabricated articulated rollout',
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
        # Point-foot kinematics at the sphere center are not the native sphere
        # contact constraint. A rolling sphere has v_contact=v_center+omega x r.
        # r=-radius*n is constant in the world for this flat TM02/floor scene.
        # Therefore d(v_contact)/dt=Jroll*qacc+Jdot_center*v-skew(r)*Jdot_omega*v.
        # In particular there is no attached-material-point centripetal term.
        Jomega=np.zeros_like(J);Jomegadot=np.zeros_like(J)
        force_point=out['feet'].copy();r=np.zeros((4,3));r[:,2]=-self.radius
        point_source=np.zeros(4,dtype=np.int8)
        Jforce=np.zeros_like(J);Jroll=np.zeros_like(J);Jrolldotv=np.zeros((4,3))
        for leg,gid in enumerate(self.feet):
            body=int(m.geom_bodyid[gid])
            mujoco.mj_jacBody(m,data,None,Jomega[leg],body)
            mujoco.mj_jacDot(m,data,None,Jomegadot[leg],out['feet'][leg],body)
            force_point[leg]=out['feet'][leg]+r[leg]
            # Native rigid contacts expose the actual current point. Soft force
            # is applied at the causal force-boundary bottom recorded by runtime.
            contacts=[data.contact[k] for k in range(data.ncon)
                      if int(gid) in (data.contact[k].geom1,data.contact[k].geom2)]
            if contacts:
                force_point[leg]=contacts[0].pos
                point_source[leg]=1
            if getattr(self,'current_soft',np.zeros(4,bool))[leg]:
                bottom=getattr(self,'current_material_info',{}).get('force_boundary_bottom')
                if bottom is not None:
                    force_point[leg]=np.asarray(bottom)[leg];point_source[leg]=2
            mujoco.mj_jac(m,data,Jforce[leg],None,force_point[leg],body)
            Jroll[leg]=J[leg]-skew(r[leg])@Jomega[leg]
            Jrolldotv[leg]=out['Jdotv'][leg]-skew(r[leg])@(Jomegadot[leg]@data.qvel)
        omega_foot=np.einsum('lij,j->li',Jomega,data.qvel)
        out.update(Jforce=Jforce,Jroll=Jroll,Jrolldotv=Jrolldotv,
            contact_velocity=np.einsum('lij,j->li',Jroll,data.qvel),
            contact_force_point=force_point,contact_point_source=point_source,
            foot_omega=omega_foot,rolling_center_velocity=-np.cross(omega_foot,r))
        self._time('model',w,c)
        return out

    def schedule(self,t):
        if t<=getattr(self,'control_time',t)+1e-10:
            return self.event_mode!='swing'
        return self._future_contact_feet(self.control_time,np.array([t]),self.current_mq,self.current_mission)[0][0]

    def _event(self,t,kind,legs,**extra):
        self.event_number+=1
        event=dict(event=self.event_number,time_s=float(t),kind=kind,legs=[LEGS[int(i)] for i in legs],**extra)
        self.events.append(event)
        with (self.record_dir/'contact-events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')

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

    def _update_schedule(self,t,data,mq,mission,soft,material_info,observed,grf):
        cfg=self.cfg['contact_events'];dt=self.cfg['wbc']['dt_s']
        self.moving=mission.phase not in STAND_PHASES
        self.measured=np.asarray(observed,dtype=bool).copy()
        for leg in range(4):
            s=self.swing[leg]
            eligible=self.event_mode[leg]!='swing' or (s is not None and t>=s['apex'])
            touching=bool(observed[leg] and grf[leg,2]>=cfg['touch_on_N'] and eligible)
            self.touch_count[leg]=self.touch_count[leg]+1 if touching else 0
            lost=(not observed[leg]) or grf[leg,2]<cfg['touch_off_N']
            self.release_count[leg]=self.release_count[leg]+1 if lost else 0
            if self.release_count[leg]*dt>=cfg['release_confirm_s']:
                self.confirmed[leg]=False;self.load_bearing[leg]=False
            if touching and self.touch_count[leg]*dt>=cfg['touch_confirm_s'] and not self.confirmed[leg]:
                self.confirmed[leg]=True
                self.anchor[leg]=mq['feet'][leg].copy()
                prior=str(self.event_mode[leg]);self.event_mode[leg]='stance'
                self._event(t,'actual_touchdown',[leg],prior_mode=prior,
                    anchor=self.anchor[leg].tolist(),measured_Fz_N=float(grf[leg,2]))
            bearing=self.confirmed[leg] and grf[leg,2]>=cfg['load_on_bodyweight_fraction']*self.mass*9.81
            self.load_count[leg]=self.load_count[leg]+1 if bearing else 0
            self.load_bearing[leg]=bool(self.load_count[leg]*dt>=cfg['load_confirm_s'])
            if self.event_mode[leg]=='swing' and t>=s['end']:
                self.event_mode[leg]='search'
                self._event(t,'touchdown_search',[leg],landing=s['landing'].tolist(),
                    planned_end_s=s['end'],remaining_gap_m=float(mq['feet'][leg,2]-s['landing'][2]))
        if self.moving and self.walk_origin is None:
            self.walk_origin=float(t);self.next_liftoff=float(t+.05)
            self.transfer_start=float(t);self._event(t,'walking_requested',range(4))
        if not self.moving:
            self.transfer_start=None;self.transfer_fraction=0.
            return self.event_mode!='swing'
        if self.next_liftoff is None:
            self.next_liftoff=float(t+.05)
        support=np.array([i for i in range(4) if i not in self.next_pair])
        all_ground=bool(np.all(self.confirmed) and np.all(self.event_mode=='stance'))
        if all_ground and self.transfer_start is None:
            self.transfer_start=max(float(t),float(self.next_liftoff-.05))
        if self.transfer_start is not None:
            z=np.clip((t-self.transfer_start)/cfg['load_transfer_ramp_s'],0.,1.)
            self.transfer_fraction=float(z*z*z*(10.-15.*z+6.*z*z))
        else:self.transfer_fraction=0.
        required=self.mass*9.81
        measured_pair=float(grf[support,2].sum())
        solved_pair=float(self.last_force[support,2].sum())
        released_pair=float(self.last_force[self.next_pair,2].sum())
        ready=(np.all(self.event_mode=='stance') and np.all(self.confirmed[support]) and np.all(self.load_bearing[support])
            and measured_pair>=cfg['handover_measured_bodyweight_fraction']*required
            and solved_pair>=cfg['handover_solved_bodyweight_fraction']*required
            and released_pair<=cfg['release_solved_bodyweight_fraction']*required
            and self.last_body_residual<=cfg['handover_body_accel_residual_m_s2']
            and self.last_angular_residual<=cfg['handover_angular_accel_residual_rad_s2']
            and self.transfer_fraction>=1.-1e-10)
        self.handover_diagnostics=dict(ready=bool(ready),support_legs=support.tolist(),
            earliest_liftoff_s=self.next_liftoff,measured_support_N=measured_pair,
            previous_solved_support_N=solved_pair,previous_released_pair_N=released_pair,
            previous_body_accel_residual=self.last_body_residual,
            previous_angular_accel_residual=self.last_angular_residual)
        if t>=self.next_liftoff-1e-10 and ready:
            pair=self.next_pair.copy()
            for leg in pair:
                self._start_swing(t,int(leg),data,mq,mission,soft,material_info)
                self.event_mode[leg]='swing';self.confirmed[leg]=False;self.load_bearing[leg]=False
                self.touch_count[leg]=0;self.load_count[leg]=0
            self._event(t,'liftoff_committed',pair,
                nominal_due_s=float(self.next_liftoff),handover=self.handover_diagnostics,
                fixed_apex_s=float(t+.10),fixed_touchdown_s=float(t+.20))
            self.last_liftoff=float(t);self.next_liftoff=float(t+.25)
            self.next_pair=support;self.transfer_start=None;self.transfer_fraction=0.
        return self.event_mode!='swing'

    def _future_contact_feet(self,t,times,mq,mission):
        contact=np.tile(self.measured & (self.event_mode!='swing'),(len(times),1))
        stance_origin=mq['feet'].copy()
        feet=np.tile(stance_origin,(len(times),1,1));events=[[] for _ in range(4)]
        available=np.full(4,t)
        search_speed=self.cfg['contact_events']['search_speed_m_s']
        for leg in range(4):
            s=self.swing[leg]
            if self.event_mode[leg]=='swing':
                events[leg].append((s['start'],s['end'],s))
                available[leg]=s['end']+self.cfg['contact_events']['touch_confirm_s']
            elif not self.confirmed[leg] and self.measured[leg]:
                # Current measured reaction is usable before the load-bearing
                # debounce. The anchor is still established only on confirmation.
                available[leg]=t+self.cfg['contact_events']['touch_confirm_s']
                events[leg].append((t,t,dict(landing=mq['feet'][leg].copy())))
            elif not self.confirmed[leg]:
                gap=max(0.,mq['feet'][leg,2]-(s['landing'][2] if s is not None else self.radius[leg]))
                available[leg]=t+gap/search_speed+self.cfg['contact_events']['touch_confirm_s']
                target=s['landing'] if s is not None else self.initial_landing[leg]
                events[leg].append((t,available[leg],dict(search=True,landing=target.copy())))
        if self.moving and self.next_liftoff is not None:
            pair=self.next_pair.copy()
            due=max(self.next_liftoff,float(available.max()+self.cfg['contact_events']['load_transfer_ramp_s']))
            if np.all(self.confirmed) and self.transfer_start is not None:
                due=max(self.next_liftoff,self.transfer_start+self.cfg['contact_events']['load_transfer_ramp_s'])
            while due<=float(times[-1])+1e-9:
                for leg in pair:
                    begin=events[leg][-1][2]['landing'].copy() if events[leg] else stance_origin[leg].copy()
                    target=self.model_data_root+mq['R']@self.nominal_feet_body[leg]
                    target[:2]+=mission.v[:2]*(due+.20-t+.15)
                    target[2]=self.radius[leg]
                    mid=.5*(begin+target);mid[2]=max(begin[2],target[2])+.03
                    vm=(target-begin)/.20;vm[2]=0.
                    ss=dict(start=due,end=due+.20,landing=target,
                        rise=quintic(begin,np.zeros(3),np.zeros(3),mid,vm,np.zeros(3),.10),
                        fall=quintic(mid,vm,np.zeros(3),target,np.zeros(3),np.zeros(3),.10))
                    events[leg].append((due,due+.20,ss))
                pair=np.array([i for i in range(4) if i not in pair]);due+=.25
        for leg in range(4):
            for k,tk in enumerate(times):
                pos=stance_origin[leg].copy();ct=bool(self.measured[leg] and self.event_mode[leg]!='swing')
                for start,end,event in events[leg]:
                    if tk<start:break
                    if tk<end:
                        ct=False
                        if event.get('search'):
                            pos=mq['feet'][leg].copy();pos[2]-=search_speed*(tk-t)
                        else:
                            tau=np.clip(tk-start,0.,.20)
                            pos=curve(event['rise'] if tau<=.10 else event['fall'],tau if tau<=.10 else tau-.10)[0]
                        break
                    pos=event['landing'].copy();ct=True
                feet[k,leg]=pos;contact[k,leg]=ct
        boundary=np.isclose(times,t,atol=1e-9,rtol=0)
        contact[boundary]=self.measured & (self.event_mode!='swing')
        for k in np.flatnonzero(boundary):
            feet[k,self.measured]=mq['feet'][self.measured]
        return contact,feet

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
        # x0 remains measured. Desired height/upright attitude do not inherit
        # the support-deficient optimized SRB state or reset to a falling body.
        ref[0,5]=self.home_com_height;ref[0,11]=0.;ref[0,:2]=0.
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
        return self._future_contact_feet(t,times,mq,mission)[1]

    def replan(self,t,data,mq,mission):
        w,c=time.perf_counter(),time.process_time()
        n=self.cfg['mpc']['intervals'];dt=self.cfg['mpc']['dt_s']
        x0,ref=self._body_reference(t,mq,mission)
        times=t+np.arange(n+1)*dt
        contact,foot_plan=self._future_contact_feet(t,times,mq,mission)
        force_foot_plan=foot_plan.copy();force_foot_plan[:,:,2]-=self.radius[None,:]
        force_foot_plan[0]=mq['contact_force_point']
        stages=[assemble_centroidal(self.mass,mq['inertia'],ref[k,3:6],force_foot_plan[k],
                mq['rpy'],mq['omega'],dt,self.cfg['force_pyramid_slope']) for k in range(n)]
        inputs=dict(x0=x0,reference=ref,last_force=self.last_force.copy(),contact=contact[:-1],
            feet=force_foot_plan,force_normal_upper=np.full((n,4),np.inf),
            stage_A=np.array([x['A'] for x in stages]),stage_B=np.array([x['B'] for x in stages]),
            stage_c=np.array([x['c'] for x in stages]))
        qp=assemble_force_horizon(inputs,self.cfg)
        self._time('matrix_assembly',w,c);w,c=time.perf_counter(),time.process_time()
        z,status=self._solve(qp['P'],qp['q'],qp['A'],qp['lower'],qp['upper'],'mpc',t)
        self._time('mpc_solve',w,c)
        nx=(n+1)*12;states=z[:nx].reshape(n+1,12);forces=z[nx:].reshape(n,4,3)
        self.last_plan=dict(time=t,times=times,state=states,reference=ref,forces=forces,
            feet=foot_plan,contact=contact[:-1],blocks=stages[0],status=status)
        self.latest_plan=dict(times=times,query_feet=foot_plan.copy(),source='F causal event horizon; predicted touchdown is not future observation')
        self.plan_count+=1;w,c=time.perf_counter(),time.process_time()
        path=self.record_dir/f'mpc-{self.plan_count:06d}.npz'
        np.savez_compressed(path,times=times,states=states,state_reference=ref,forces=forces,
            contact_schedule=contact[:-1],inertia=mq['inertia'],com=mq['com'],
            mass=np.array(self.mass),M=mq['M'],h=mq['h'],J=mq['J'],
            foot_center_plan=foot_plan,J_contact_force=mq['Jforce'],J_rolling=mq['Jroll'],
            contact_point_source=mq['contact_point_source'],qpos=data.qpos.copy(),qvel=data.qvel.copy(),
            event_mode=self.event_mode.copy(),confirmed_contact=self.confirmed.copy(),
            load_bearing_contact=self.load_bearing.copy(),stance_anchor=self.anchor.copy(),
            event_number=np.array(self.event_number),**inputs)
        with (self.record_dir/'mpc-solves.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=self.plan_count,time_s=t,file=path.name,**status))+'\n')
        self.manifest['full_mpc_files'].append(dict(call=self.plan_count,time_s=t,path=path.name,size=path.stat().st_size))
        self._time('record_serialization_io',w,c)

    def _redistribute(self,forces,mq,active,required=None):
        if required is None:
            required=np.r_[forces.sum(axis=0),sum((np.cross(mq['contact_force_point'][i]-mq['com'],forces[i]) for i in range(4)),np.zeros(3))]
        weights=active.astype(float)
        if self.transfer_start is not None and np.all(active):
            weights[self.next_pair]*=(1.-self.transfer_fraction)
            support=np.array([i for i in range(4) if i not in self.next_pair])
            weights[support]*=(1.+self.transfer_fraction)
        G=np.vstack([np.tile(np.eye(3),(1,4)),np.concatenate([skew(p-mq['com']) for p in mq['contact_force_point']],axis=1)])
        scale=np.repeat(weights,3)
        f=scale*np.linalg.lstsq(G*scale[None,:],required,rcond=None)[0]
        return f.reshape(4,3),required

    def soft_reference(self, t, leg, pos_z, measured, target, material_info):
        stiffness = material_info['stiffness_N_m']
        return (pos_z + measured/stiffness) - target/stiffness

    def wbc(self,t,data,mq,mission,scheduled,observed,grf,soft,material_info):
        w,c=time.perf_counter(),time.process_time()
        m=self.model;nv=m.nv;nz=nv+12
        plan=self.last_plan
        stage=min(int(max(0.,t-plan['time'])/self.cfg['mpc']['dt_s']),self.cfg['mpc']['intervals']-1)
        forces=plan['forces'][stage].copy()
        active=scheduled & observed
        # Dynamics includes only currently observed planned stance reactions.
        # Measured unwanted swing force remains untouched in plant/logging.

        M=mq['M'];h=mq['h'];J=mq['Jforce'].reshape(12,nv)
        D=np.concatenate((M,-J.T),axis=1)
        Dt=D[self.va]
        P=np.eye(nz)*2e-5;q=np.zeros(nz)
        def task(mat,target,weight):
            nonlocal P,q
            P+=2*weight*(mat.T@mat);q-=2*weight*(mat.T@target)
        sample=np.clip(t-plan['time'],0.,self.cfg['mpc']['dt_s'])/self.cfg['mpc']['dt_s']
        task_goal=(1-sample)*plan['reference'][0]+sample*plan['reference'][1]
        # The independent task remains the MPC objective. WBC executes the
        # optimized motion, not the unoptimized request and not masked GRFs.
        xd=(1-sample)*plan['state'][0]+sample*plan['state'][1]
        lin_acc=(plan['state'][1,9:12]-plan['state'][0,9:12])/self.cfg['mpc']['dt_s']
        com_acc=lin_acc+self.cfg['wbc']['body_position_kp']*(xd[3:6]-mq['com'])+self.cfg['wbc']['body_velocity_kd']*(xd[9:12]-mq['com_velocity'])
        rot_des=Rotation.from_euler('xyz',xd[:3]).as_matrix()
        rot_error=Rotation.from_matrix(rot_des@mq['R'].T).as_rotvec()
        angular_acc=(plan['state'][1,6:9]-plan['state'][0,6:9])/self.cfg['mpc']['dt_s']
        angular_acc+=self.cfg['wbc']['orientation_kp']*rot_error+self.cfg['wbc']['angular_velocity_kd']*(xd[6:9]-mq['omega'])
        required_wrench=np.r_[self.mass*(com_acc+np.array([0.,0.,9.81])),
            mq['inertia']@angular_acc+np.cross(mq['omega'],mq['inertia']@mq['omega'])]
        force_target,_=self._redistribute(forces,mq,active,required_wrench)
        # A measured target-support contact must be loaded before it can pass
        # confirmation/load-bearing thresholds. Do not wait for confirmation
        # before ever asking this real contact to carry a positive force.
        # The requested preload uses the existing per-foot bearing threshold;
        # measured forces alone still determine confirmation and liftoff.
        support_pair=np.array([i for i in range(4) if i not in self.next_pair])
        preload=active & np.isin(np.arange(4),support_pair) & ~self.load_bearing & self.moving
        preload_normal=self.cfg['contact_events']['load_on_bodyweight_fraction']*self.mass*9.81
        force_target[preload,2]=np.maximum(force_target[preload,2],preload_normal)
        Ab=np.zeros((3,nz));Ab[:,:nv]=mq['Jcom']
        task(Ab,com_acc-mq['Jcomdotv'],self.cfg['wbc']['body_weight'])
        Ar=np.zeros((3,nz));Ar[:,:nv]=mq['Jr']
        task(Ar,angular_acc-mq['Jrdotv'],self.cfg['wbc']['orientation_weight'])
        pref=np.zeros((4,3));vref=np.zeros((4,3));aref=np.zeros((4,3))
        mode=np.zeros(4,dtype=np.int8)
        task_J=mq['J'].copy();task_Jdotv=mq['Jdotv'].copy()
        for leg in range(4):
            pos=mq['feet'][leg];vel=mq['foot_velocity'][leg]
            if not scheduled[leg]:
                pref[leg],vref[leg],aref[leg]=self._swing_reference(t,leg)
                target=aref[leg]+self.cfg['wbc']['swing_kp']*(pref[leg]-pos)+self.cfg['wbc']['swing_kd']*(vref[leg]-vel)
                weight=self.cfg['wbc']['swing_weight'];mode[leg]=0
            elif not observed[leg]:
                if self.swing[leg] is None and mission.phase=='initial_stand':
                    # The original initial pose is slightly above the floor.
                    # Do not retract the legs to hold airborne feet fixed while
                    # the unsupported base falls: keep the nominal joint posture.
                    pref[leg]=pos;vref[leg]=vel;target=np.zeros(3);weight=0.;mode[leg]=5
                else:
                    pref[leg]=self.swing[leg]['landing'] if self.swing[leg] is not None else self.initial_landing[leg]
                    vref[leg,2]=-self.cfg['contact_events']['search_speed_m_s']
                    target=self.cfg['wbc']['stance_kp']*(pref[leg]-pos)-self.cfg['wbc']['stance_kd']*vel
                    target[2]=self.cfg['wbc']['stance_kd']*(vref[leg,2]-vel[2])
                    weight=self.cfg['wbc']['stance_weight'];mode[leg]=2
            elif not self.confirmed[leg]:
                pref[leg]=pos
                vref[leg]=mq['rolling_center_velocity'][leg]
                target=-self.cfg['wbc']['stance_kd']*mq['contact_velocity'][leg]
                task_J[leg]=mq['Jroll'][leg];task_Jdotv[leg]=mq['Jrolldotv'][leg]
                weight=self.cfg['wbc']['stance_weight'];mode[leg]=4
            else:
                pref[leg]=self.anchor[leg]
                # Preserve the measured touchdown anchor as an event record.
                # Tangential no-slip is imposed at the instantaneous sphere
                # contact, not by dragging its rolling center back to that anchor.
                pref[leg,:2]=pos[:2]
                vref[leg]=mq['rolling_center_velocity'][leg]
                target=-self.cfg['wbc']['stance_kd']*mq['contact_velocity'][leg]
                target[2]+=self.cfg['wbc']['stance_kp']*(pref[leg,2]-pos[2])
                task_J[leg]=mq['Jroll'][leg];task_Jdotv[leg]=mq['Jrolldotv'][leg]
                if soft[leg]:
                    # Local causal force anchor rest=p_now+F_now/k. The desired
                    # normal position follows planned load; matching observed
                    # normal velocity avoids a world-fixed material z constraint.
                    pref[leg,2]=self.soft_reference(t,leg,pos[2],max(0.,grf[leg,2]),
                                                    max(0.,force_target[leg,2]),material_info)
                    vref[leg,2]=vel[2]
                    target[2]=self.cfg['wbc']['material_normal_kp']*(pref[leg,2]-pos[2])
                    mode[leg]=3
                else:mode[leg]=1
                weight=self.cfg['wbc']['stance_weight']
            aref[leg]=target
            Af=np.zeros((3,nz));Af[:,:nv]=task_J[leg]
            task(Af,target-task_Jdotv[leg],weight)
        # Reference-direction dry friction enters h and all torque/input bounds.
        # Viscous damping already belongs to passive; only dry rows are recorded.
        vfoot=vref+self.cfg['friction_model']['position_direction_gain_s_inv']*(pref-mq['feet'])
        Jjoint=mq['J'][:,:,self.va].reshape(12,12)
        vjoint_ref=np.linalg.lstsq(Jjoint,vfoot.ravel()-mq['J'][:,:,:6].reshape(12,6)@data.qvel[:6],rcond=None)[0]
        dry=self.model.dof_frictionloss[self.va]*np.tanh(vjoint_ref/self.cfg['friction_model']['reference_velocity_scale_rad_s'])
        h=mq['h'].copy();h[self.va]+=dry
        actual_dry=np.zeros(m.nv)
        if data.nefc:
            selected=np.zeros(data.nefc)
            mask=np.asarray(data.efc_type[:data.nefc])==int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
            selected[mask]=data.efc_force[:data.nefc][mask]
            mujoco.mj_mulJacTVec(m,data,actual_dry,selected)
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
        # Dynamics uses mean torque over the actual ten force-evaluation
        # boundaries 0,.2,...,1.8 ms. The plant keeps the same position u for2ms.
        # Constant local qacc with semi-implicit Euler gives
        # q_k=q0+t_k*v0+.5*t_k*(t_k+plant_dt)*a, v_k=v0+t_k*a.
        # Thus tau_k=tau0-60*t_k*v0-[5*t_k+30*t_k*(t_k+plant_dt)]*a.
        # This is a local held-input approximation, not exact nonlinear safety.
        dt=self.cfg['wbc']['dt_s'];plant_dt=float(m.opt.timestep)
        hold_times=np.arange(int(round(dt/plant_dt)))*plant_dt
        velocity_coeff=60.*hold_times
        accel_coeff=5.*hold_times+30.*hold_times*(hold_times+plant_dt)
        mean_vcoeff=float(velocity_coeff.mean());mean_acoeff=float(accel_coeff.mean())
        Tstart=Dt+mean_acoeff*Ap
        hstart=h[self.va]+mean_vcoeff*data.qvel[self.va]
        # Preserve the original full-dynamics total torque range as well as
        # original mapped-u range at the actual hold boundary.
        rows.append(Dt);lower.append(m.actuator_forcerange[:,0]-h[self.va]);upper.append(m.actuator_forcerange[:,1]-h[self.va])
        rows.append(Tstart);lower.append(tau_lo-hstart);upper.append(tau_hi-hstart)
        for vc,ac in zip(velocity_coeff[1:],accel_coeff[1:]):
            Tk=Tstart-ac*Ap;hk=hstart-vc*data.qvel[self.va]
            rows.append(Tk);lower.append(m.actuator_forcerange[:,0]-hk);upper.append(m.actuator_forcerange[:,1]-hk)
        friction=np.zeros((16,nz));friction[:,nv:]=plan['blocks']['friction_C']
        rows.append(friction);lower.append(np.full(16,-np.inf));upper.append(np.zeros(16))
        force_id=np.zeros((12,nz));force_id[:,nv:]=np.eye(12)
        flo=np.full(12,-np.inf);fhi=np.full(12,np.inf)
        for leg in range(4):
            if active[leg]:flo[3*leg+2]=preload_normal if preload[leg] else 0.
            else:flo[3*leg:3*leg+3]=fhi[3*leg:3*leg+3]=0.
        rows.append(force_id);lower.append(flo);upper.append(fhi)
        dt=.002;ranges=m.jnt_range[self.joint_ids]
        qfree=data.qpos[self.qa]+dt*data.qvel[self.va]
        rows.append(Ap);lower.append(2*(ranges[:,0]-qfree)/dt**2);upper.append(2*(ranges[:,1]-qfree)/dt**2)
        # Keep the former continuous one-hold reachability and add the matching
        # semi-implicit plant boundary; neither original joint range is relaxed.
        reach_coeff=.5*dt*(dt+plant_dt)
        rows.append(Ap);lower.append((ranges[:,0]-qfree)/reach_coeff);upper.append((ranges[:,1]-qfree)/reach_coeff)
        A=np.vstack(rows);lo=np.concatenate(lower);hi=np.concatenate(upper)
        self._time('wbc_assembly',w,c)
        w,c=time.perf_counter(),time.process_time()
        z,status=self._solve(P,q,A,lo,hi,'wbc',t)
        self._time('wbc_solve',w,c)
        qacc=z[:nv];force=z[nv:].reshape(4,3)
        tau_mean=(D@z+h)[self.va]
        tau=(Tstart@z+hstart)
        predicted_hold_torque=tau[None,:]-velocity_coeff[:,None]*data.qvel[self.va][None,:]-accel_coeff[:,None]*qacc[self.va][None,:]
        action=data.qpos[self.qa]+(tau+5*data.qvel[self.va])/60.
        dyn=D[:6]@z+h[:6]
        self.last_force=force.copy()
        actual_foot_acc=np.einsum('lij,j->li',task_J,qacc)+task_Jdotv
        actual_center_acc=np.einsum('lij,j->li',mq['J'],qacc)+mq['Jdotv']
        actual_body_acc=mq['Jcom']@qacc+mq['Jcomdotv']
        actual_angular_acc=mq['Jr']@qacc+mq['Jrdotv']
        self.last_body_residual=float(np.linalg.norm(actual_body_acc-com_acc))
        self.last_angular_residual=float(np.linalg.norm(actual_angular_acc-angular_acc))
        realized_wrench=np.r_[force.sum(axis=0),sum((np.cross(mq['contact_force_point'][i]-mq['com'],force[i]) for i in range(4)),np.zeros(3))]
        self.rows.append(dict(time=np.array(t),plan_call=np.array(self.plan_count),
            qacc=qacc,force=force,mpc_force=forces,planned_contact=scheduled,observed_contact=observed,
            active_wbc_contact=active,mode=mode,feet_position_reference=pref,feet_velocity_reference=vref,
            feet_acceleration_reference=aref,body_state_reference=xd,body_acceleration_reference=com_acc,
            independent_task_goal=task_goal,mpc_body_acceleration_feedforward=lin_acc,
            mpc_angular_acceleration_feedforward=(plan['state'][1,6:9]-plan['state'][0,6:9])/self.cfg['mpc']['dt_s'],
            angular_acceleration_reference=angular_acc,measured_grf=grf.copy(),foot_velocity=mq['foot_velocity'],
            tau_total=tau,tau_start=tau,tau_mean=tau_mean,
            predicted_hold_torque=predicted_hold_torque,hold_force_evaluation_offsets_s=hold_times,
            hold_torque_mean_error=predicted_hold_torque.mean(0)-tau_mean,
            mapped_u=action,q_joint=data.qpos[self.qa].copy(),v_joint=data.qvel[self.va].copy(),
            requested_pd=60*(action-data.qpos[self.qa])-5*data.qvel[self.va],
            dynamics_residual=dyn,mpc_age=np.array(t-plan['time']),
            solver_status=np.array(status['status_val']),solver_iterations=np.array(status['iterations']),
            solver_primal_residual=np.array(status['primal_residual']),solver_dual_residual=np.array(status['dual_residual']),
            torque_lower=tau_lo,torque_upper=tau_hi,
            contact_mode_mismatch=scheduled!=observed,
            confirmed_contact=self.confirmed.copy(),load_bearing_contact=self.load_bearing.copy(),
            event_mode=self.event_mode.copy(),event_number=np.array(self.event_number),
            stance_anchor=self.anchor.copy(),transfer_fraction=np.array(self.transfer_fraction),
            transfer_support_pair=np.array([i for i in range(4) if i not in self.next_pair]),
            feet_Jdotv=task_Jdotv,feet_acceleration_achieved=actual_foot_acc,
            foot_center_acceleration_achieved=actual_center_acc,
            contact_force_point=mq['contact_force_point'],contact_point_source=mq['contact_point_source'],
            rolling_contact_velocity=mq['contact_velocity'],rolling_center_velocity=mq['rolling_center_velocity'],
            foot_angular_velocity=mq['foot_omega'],J_contact_force=mq['Jforce'],J_rolling=mq['Jroll'],
            foot_task_is_contact=observed & scheduled,
            feet_acceleration_residual=actual_foot_acc-aref,
            body_acceleration_achieved=actual_body_acc,body_acceleration_residual=actual_body_acc-com_acc,
            angular_acceleration_achieved=actual_angular_acc,angular_acceleration_residual=actual_angular_acc-angular_acc,
            required_wrench=required_wrench,realized_wrench=realized_wrench,wrench_residual=realized_wrench-required_wrench,
            redistributed_force_target=force_target,friction_reference_joint_velocity=vjoint_ref,
            measured_touch_preload=preload,preload_normal_lower=np.where(preload,preload_normal,0.),
            contact_confirmation_count=self.touch_count.copy(),load_confirmation_count=self.load_count.copy(),
            friction_compensation=dry,actual_friction_constraint=actual_dry[self.va],
            inverse_dynamics_h=h))
        self.control_count+=1
        if len(self.rows)>=500:self.flush()
        return action

    def update(self,t,data,mission,contact,grf,soft,material_info=None,replan=False):
        material_info=dict(material_info or {})
        material_info['force']=np.asarray(grf)
        material_info['stiffness_N_m']=float(material_info.get('instantaneous_stiffness_N_m',material_info.get('stiffness_N_m',self.cfg['material_stiffness_N_m'])))
        self.current_soft=np.asarray(soft,dtype=bool);self.current_material_info=material_info
        mq=self.model_quantities(data);self.current_mq=mq;self.current_mission=mission;self.control_time=float(t)
        self.model_data_root=data.xpos[self.base].copy()
        observed=np.asarray(contact,dtype=bool) & (np.asarray(grf)[:,2]>self.cfg['contact_events']['touch_off_N'])
        scheduled=self._update_schedule(t,data,mq,mission,soft,material_info,observed,np.asarray(grf))
        if replan or self.last_plan is None:self.replan(t,data,mq,mission)
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
