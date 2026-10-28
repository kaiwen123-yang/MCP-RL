"""Actual TM02 strip for the frozen S3 raw-MuJoCo runner."""
import json
import time
import numpy as np
import mujoco
from tm02.s1_contact import Ground

class JointMaterial:
    def __init__(self,root,contract,cap=.25,profile='constant',full_recording=True,terrain='tm02_anchor'):
        self.root,self.contract,self.cap=root,contract,cap
        self.nominal_cap,self.profile=cap,profile
        self.cap_decisions=[];self.last_actor_tick=-1
        pars=json.loads((root/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
        self.ground=Ground({'id':terrain},pars)
        self.soft=np.zeros(4,bool);self.force=np.zeros((4,3))
        self.bottom=np.zeros((4,3));self.velocity=np.zeros((4,3))
        self.force_boundary_bottom=np.zeros((4,3))
        self.material_time=0.;self.material_wall=0.
        self.records=[];self.cells=[];self.chunks=[];self.physics_steps=0
        self.reason='';self.group='DEVELOPMENT_ALLOWED_INTENT'
        self.route_progress=0.;self.indent_integral=0.;self.slip_integral=0.
        self.applied_cap=cap
        self.full_recording=full_recording

    def attach(self,model,data,feet,allowed,out,native_model):
        self.model,self.data,self.feet,self.allowed=model,data,feet,allowed
        self.out,self.native_model=out,native_model
        self.native_data=mujoco.MjData(native_model)
        self.original_type=model.geom_contype[feet].copy()
        self.original_affinity=model.geom_conaffinity[feet].copy()
        model.geom_contype[feet]=self.original_type
        model.geom_conaffinity[feet]=self.original_affinity
        data.qfrc_applied[:]=0;data.xfrc_applied[:]=0
        self.floor_z=float(data.geom_xpos[model.geom('floor').id,2])
        self.radii=model.geom_size[feet,0].copy()
        self.events_file=(out/'material-events.jsonl').open('w')
        self.failures_file=(out/'material-failures.jsonl').open('w')

    def native_reference_state(self):
        d,s=self.native_data,self.data
        d.qpos[:]=s.qpos;d.qvel[:]=s.qvel;d.ctrl[:]=s.ctrl;d.time=s.time
        d.mocap_pos[:]=s.mocap_pos;d.mocap_quat[:]=s.mocap_quat;d.userdata[:]=s.userdata
        mujoco.mj_forward(self.native_model,d)
        return self.native_model,d

    def kinematics(self):
        self.bottom=self.data.geom_xpos[self.feet].copy()
        self.bottom[:,2]-=self.radii+self.floor_z
        for i,foot in enumerate(self.feet):
            spatial=np.zeros(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_GEOM,int(foot),spatial,0)
            self.velocity[i]=spatial[3:]+np.cross(spatial[:3],[0,0,-self.radii[i]])
        return self.bottom,self.velocity

    def contacts(self,native,full=False):
        contact,force,invalid,total,nonfoot,count=native
        contact=contact.copy();force=force.copy();total=total.copy()
        total-=force[self.soft].sum(axis=0)
        force[self.soft]=self.force[self.soft]
        contact[self.soft]=self.force[self.soft,2]>0
        total+=force[self.soft].sum(axis=0)
        return (contact,force,invalid,total,nonfoot,count) if full else (contact,force,invalid)

    def started(self,small,c,index_file):
        with index_file(small,c) as idx:
            if 'campaign_started_unix_s' not in idx:
                now=time.time()
                idx.update(campaign_started_unix_s=now,campaign_deadline_unix_s=now+48*3600,
                    campaign_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            self.deadline=idx['campaign_deadline_unix_s']

    def before_physics(self,dt,phase):
        begin=time.perf_counter();m,d=self.model,self.data
        d.qfrc_applied[:]=0.;d.xfrc_applied[:]=0.
        mujoco.mj_forward(m,d)
        bottom,velocity=self.kinematics()
        self.soft=np.array([self.ground.soft(p[:2]) for p in bottom])
        m.geom_contype[self.feet]=np.where(self.soft,0,self.original_type)
        m.geom_conaffinity[self.feet]=np.where(self.soft,0,self.original_affinity)
        t=float(d.time)
        force,indent,maps=self.ground.step(bottom,velocity,dt,t+dt,log=self.full_recording)
        self.force=force;self.material_time=t+dt
        self.force_boundary_bottom=bottom.copy()
        self.material_wall+=time.perf_counter()-begin
        if self.full_recording:
            self.records.append(np.r_[t,t+dt,self.soft,bottom.ravel(),velocity.ravel(),force.ravel(),
                indent,self.ground.max_strain,self.ground.max_residual])
        if self.ground.domain_exit:
            self.reason='material_domain_exit'
            self.failures_file.write(json.dumps(dict(reason=self.reason,**self.ground.domain_exit,
                bottom_m=bottom.tolist(),velocity_m_s=velocity.tolist(),force_N=force.tolist(),
                physical_state_time_s=t,material_state_time_s=t+dt))+'\n')
            return False
        for i,foot in enumerate(self.feet):
            if self.soft[i]:
                point=bottom[i].copy();point[2]+=self.floor_z
                mujoco.mj_applyFT(m,d,force[i],np.zeros(3),point,int(m.geom_bodyid[foot]),d.qfrc_applied)
        loaded=(force[:,2]>1)&self.soft
        if loaded.any():
            self.indent_integral+=float(np.mean((np.maximum(-bottom[loaded,2],0)/.001)**2))*dt
        self.before_x=float(d.qpos[0]);self.phase=phase
        return True

    def after_physics(self,dt,grf):
        self.physics_steps+=1
        if self.phase in ('outbound','return'):
            self.route_progress+=(float(self.data.qpos[0])-self.before_x)*(1 if self.phase=='outbound' else -1)/6
        loaded=grf[:,2]>1
        if loaded.any():
            self.slip_integral+=float(np.mean(np.sum(self.velocity[loaded,:2]**2,axis=1)/.25**2))*dt
        if self.full_recording and self.physics_steps%10==0 and len(self.ground.e):
            g=self.ground;n=len(g.e)
            self.cells.append(np.c_[np.full(n,self.material_time),np.asarray(list(g.ids)),g.e,g.q,g.stress,
                g.loaded,g.last_owner,g.last_area,g.last_normal])
        if self.full_recording and self.physics_steps%5000==0:self.flush()

    def control(self,t,mission):
        tick=int(np.floor((t+1e-8)/.5))
        if tick!=self.last_actor_tick:
            if self.profile=='switch_2s':
                self.cap=.18 if (tick//4)%2==0 else .25
            elif self.profile=='switch_05s':
                self.cap=.18 if tick%2==0 else .25
            elif self.profile=='sinusoid_4s':
                self.cap=.215+.035*np.sin(2*np.pi*tick/8)
            self.cap_decisions.append(dict(time_s=t,tick=tick,requested_cap_m_s=float(self.cap),
                phase_before_update=mission.phase))
            self.last_actor_tick=tick
        mission.task['command_speed_m_s']=float(self.cap)

    def flush(self):
        if self.records or self.cells or self.ground.rate_hist:
            path=self.out/('material-%04d.npz'%len(self.chunks))
            np.savez_compressed(path,foot=np.asarray(self.records),
                cells=np.concatenate(self.cells) if self.cells else np.empty((0,13)),
                rates=np.asarray(self.ground.rate_hist))
            self.chunks.append(path.name)
            self.records.clear();self.cells.clear();self.ground.rate_hist.clear()
        for event in self.ground.events:
            event['world_key']=list(map(int,np.floor(np.asarray(self.ground.xy[event['patch']])/self.ground.cell)))
            self.events_file.write(json.dumps(event)+'\n')
        self.events_file.flush();self.ground.events.clear()

    def finish(self,result):
        self.flush();self.events_file.close();self.failures_file.close();g=self.ground
        np.savez_compressed(self.out/'material-final.npz',keys=np.asarray(list(g.ids),int),
            e=g.e,q=g.q,stress=g.stress,loaded=g.loaded,ever_loaded=g.ever_loaded,
            last_owner=g.last_owner,last_release=g.last_release)
        (self.out/'cap-decisions.json').write_text(json.dumps(self.cap_decisions,indent=2)+'\n')
        result.update(group=self.group,requested_cap_m_s=self.nominal_cap,cap_profile=self.profile,
            development_intent='Predeclared allowed cap; no policy or training',
            material=dict(truth='ReferenceColumns',dt_s=.0002,observer_enabled=False,
                max_abs_principal_strain=g.max_strain,max_stress_original_MPa=g.max_stress,
                max_residual=g.max_residual,domain_exit=g.domain_exit,rate_extrema=g.rate_extrema,
                world_cells=len(g.e),loaded_ever_cells=int(g.ever_loaded.sum()),
                events=g.event_serial,wall_s=self.material_wall,chunks=self.chunks,
                foot_columns=['boundary_time_s','material_state_time_s']+['soft_'+str(i) for i in range(4)]+
                    [p+'_'+str(i)+'_'+a for p in ['bottom','velocity','force'] for i in range(4) for a in 'xyz']+
                    ['indent_'+str(i) for i in range(4)]+['max_strain','max_residual'],
                cell_columns=['time_s','ix','iy','e11','e22','q11','q22','sigma11','sigma22','loaded','owner','area_m2','normal_N'],
                cell_rate_Hz=500,foot_force_rate_Hz=5000,
                contact_sampling='Pre-integration material force and native forces from the same MuJoCo step.',
                mask_policy='Only soft foot masks disabled; non-foot masks retained. Independent frozen native MPC/FK/reference model.',
                integral_indentation_sq_s=self.indent_integral,integral_slip_sq_s=self.slip_integral,
                signed_route_fraction=self.route_progress))
        return result

