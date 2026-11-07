#!/usr/bin/env python3
"""W41 fixed-input native held-dynamics evidence; no policy or plant rollout.
CPU predictor semantics are reused from the retained W40 implementation.
The source states are real W40 events; hidden TM02 state is not reconstructed.
"""
import json,time,sys,os
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
LARGE=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
OUT=LARGE/'w41-dynamics/native-v2'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w41-dynamics'
HALF=.015811388300841896
def numeric(m,name):
    i=m.numeric(name).id
    return np.array(m.numeric_data[m.numeric_adr[i]:m.numeric_adr[i]+m.numeric_size[i]])
def inside(p,xp=np):
    return (p[...,0]>=.8+HALF)&(p[...,0]<=2.2-HALF)&(xp.abs(p[...,1])<=.6-HALF)
def state(d):return np.r_[np.asarray(d.qpos),np.asarray(d.qvel)]
def diff(m,a,b):
    dq=np.empty(m.nv);mujoco.mj_differentiatePos(m,dq,1.,a[:m.nq],b[:m.nq])
    return np.r_[dq,b[m.nq:]-a[m.nq:]]
class CPUHold:
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

def event_rows():
    rows=[]
    cases=[('a','train-job-20261106T103028Z-w40-a-PANN_PPO_S5-s1-ep00314'),
           ('b','job-20261106T103028Z-w40-b-eval-PANN_PPO_S5-s2-E30')]
    for stream,name in cases:
        folder=LARGE/name
        with np.load(folder/'trajectory.npz') as z:
            a={k:z[k] for k in z.files}
        physical=np.load(folder/'physical-000.npz')['physical']; n=len(physical)//10
        m=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
        raw=physical[:10*n,49:61].reshape(n,10,12)
        over=np.maximum(raw-m.actuator_forcerange[:,1],m.actuator_forcerange[:,0]-raw).max(axis=(1,2))
        contact_change=np.where(np.any(a['contact'][1:n]!=a['contact'][:n-1],axis=1))[0]+1
        loaded=np.where(np.any(inside(a['feet'][:n]) & (a['actual_grf'][:n,:,2]>0),axis=1))[0]
        saturation=np.where(over>1e-4)[0]
        seeds=[(500,'rigid'),(int(saturation[0]),'first_hold_saturation'),
               (int(over.argmax()),'max_hold_saturation'),(int(loaded[0]),'first_material_load'),
               (int(np.argmin(abs(a['qpos'][:n,0]-.8))),'strip_boundary'),
               (int(np.argmin(a['qpos'][:n,2])),'lowest_trunk')]
        events=[]
        for i,label in seeds:
            for offset in [-1,0,1]:
                j=int(np.clip(i+offset,0,n-1))
                if j not in [x[0] for x in events]:events.append((j,label+f'_{offset:+d}'))
        # Six event families, then preserve only sixteen source states per case.
        events=events[:16]
        for i,label in events:
            row=dict(name=stream+'_'+label,source=str(folder.relative_to(ROOT)),index=i,
                time=float(a['time'][i]),qpos=a['qpos'][i],qvel=a['qvel'][i],
                ctrl=a['q_target'][i].astype(float),mocap_pos=a['mocap_pos'][i],mocap_quat=a['mocap_quat'][i],
                userdata=a['support_reference_userdata'][i],task_phase=int(a['phase'][i]),
                task_history='recorded phase and causal userdata; private task state not required for fixed u',
                recorded_raw_pd=raw[i],recorded_force=physical[i*10:(i+1)*10,14:26],
                recorded_next_state=np.r_[a['qpos'][i+1],a['qvel'][i+1]])
            rows.append(row)
    return rows

def contact_record(m,d):
    return [dict(geom=[int(c.geom1),int(c.geom2)],dist=float(c.dist),pos=c.pos.tolist(),
        frame=c.frame.tolist(),dim=int(c.dim),efc_address=int(c.efc_address)) for c in d.contact]

def branch(m,d):
    q=m.jnt_qposadr[m.actuator_trnid[:,0]];v=m.jnt_dofadr[m.actuator_trnid[:,0]]
    raw=d.actuator_force
    return (tuple(sorted((int(c.geom1),int(c.geom2),int(c.dim)) for c in d.contact if c.efc_address>=0)),
            tuple(np.where(raw<=m.actuator_forcerange[:,0],-1,np.where(raw>=m.actuator_forcerange[:,1],1,0))),
            tuple(d.efc_state.tolist()))

def perturb(m,row,k,eps=1e-6):
    r=dict(row,qpos=row['qpos'].copy(),qvel=row['qvel'].copy(),ctrl=row['ctrl'].copy())
    if k<18:
        tangent=np.zeros(18);tangent[k]=eps;mujoco.mj_integratePos(m,r['qpos'],tangent,1)
    elif k<36:r['qvel'][k-18]+=eps
    else:
        j=k-36
        step=eps if (not m.actuator_ctrllimited[j] or r['ctrl'][j]+eps<=m.actuator_ctrlrange[j,1]) else -eps
        r['ctrl'][j]+=step
    return r

def measure(pred,row):
    m=pred.m;d=pred.cpu_data(row);P=np.eye(36);Q=np.zeros((36,12))
    records=[];states=[];As=[];Bs=[];endpoints=[];starts=[];same=[];errors=[]
    for j in range(10):
        prefix=dict(row,qpos=d.qpos.copy(),qvel=d.qvel.copy(),time=float(d.time))
        starts.append(np.r_[d.qpos,d.qvel,d.ctrl]);A=np.zeros((36,36));B=np.zeros((36,12))
        mujoco.mjd_transitionFD(m,d,1e-6,0,A,B,None,None)
        raw=60*(d.ctrl-d.qpos[pred.qa])-5*d.qvel[pred.va]
        ds=pred.cpu_data(prefix);mujoco.mj_step(m,ds);base=state(ds);br=branch(m,ds)
        ys=[base];branches=[]
        for k in range(48):
            dd=pred.cpu_data(perturb(m,prefix,k));mujoco.mj_step(m,dd)
            ys.append(state(dd));branches.append(branch(m,dd)==br)
        signs=np.array([1. if k<36 or not m.actuator_ctrllimited[k-36] or prefix['ctrl'][k-36]+1e-6<=m.actuator_ctrlrange[k-36,1] else -1. for k in range(48)])
        AB=np.stack([diff(m,base,y)/1e-6 for y in ys[1:]],axis=1)*signs
        errors.append(np.max(abs(AB-np.c_[A,B]),axis=0));same.append(branches)
        P,Q=A@P,A@Q+B;As.append(A);Bs.append(B);endpoints.append(ys)
        mujoco.mj_step(m,d);states.append(state(d))
        records.append(dict(substep=j,time=prefix['time'],contact=contact_record(m,d),
            solver_niter=d.solver_niter.tolist(),solver_nnz=d.solver_nnz.tolist(),efc_state=d.efc_state.tolist(),efc_type=d.efc_type.tolist(),
            nefc=d.nefc,qfrc_passive=d.qfrc_passive.tolist(),raw_pd=raw.tolist(),
            actuator_force=d.actuator_force.tolist(),torque=d.qfrc_actuator.tolist(),
            clamp_error=float(abs(d.actuator_force-np.clip(raw,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])).max())))
    held=[];hold_branches=[]
    for k in range(49):
        dd=pred.cpu_data(row if k==0 else perturb(m,row,k-1));bb=[]
        for j in range(10):mujoco.mj_step(m,dd);bb.append(branch(m,dd))
        held.append(state(dd));hold_branches.append(bb)
    direct=np.stack([diff(m,held[0],y)/1e-6 for y in held[1:]],axis=1)*signs
    same_hold=np.array([x==hold_branches[0] for x in hold_branches[1:]])
    err=np.array(errors);sm=np.array(same);composed=np.c_[P,Q]
    summary=dict(fixture=row['name'],source=row['source'],index=row['index'],time_s=row['time'],
        clamp_error=max(x['clamp_error'] for x in records),
        recorded_plant_clamp_error=float(abs(row['recorded_force']-np.clip(row['recorded_raw_pd'],m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])).max()),
        local_native_vs_independent_FD_max=float(err.max()),
        local_same_branch_max=float(err[sm].max()) if np.any(sm) else None,
        local_branch_changes=int((~sm).sum()),held_branch_changes=int((~same_hold).sum()),
        composed_vs_held_FD_max=float(abs(composed-direct).max()),
        composed_same_branch_max=float(abs((composed-direct)[:,same_hold]).max()) if np.any(same_hold) else None,
        predictor_vs_recorded_plant_endpoint=float(abs(diff(m,row['recorded_next_state'],held[0])).max()),
        predictor_plant_difference_scope='Different native-vs-MakeDifferentiable/integrator/proxy contexts, not a matching-plant model-error certificate')
    np.savez_compressed(OUT/(row['name']+'.npz'),starts=starts,states=states,A=As,B=Bs,P=P,Q=Q,
        local_FD_endpoints=endpoints,held_endpoints=held,held_direct_AB=direct,local_same_branch=sm,
        local_native_FD_error=err,held_same_branch=same_hold,userdata=row['userdata'],mocap_pos=row['mocap_pos'],
        mocap_quat=row['mocap_quat'],recorded_raw_pd=row['recorded_raw_pd'],recorded_force=row['recorded_force'])
    (OUT/(row['name']+'-contacts.json')).write_text(json.dumps(records,indent=2)+'\n')
    return summary

def main():
    OUT.mkdir(parents=True,exist_ok=True);SMALL.mkdir(parents=True,exist_ok=True)
    rows=event_rows();meta=[{k:v for k,v in r.items() if not isinstance(v,np.ndarray)} for r in rows]
    np.savez_compressed(OUT/'fixtures.npz',**{k:np.stack([r[k] for r in rows]) for k in
        ('qpos','qvel','ctrl','userdata','mocap_pos','mocap_quat')},time=[r['time'] for r in rows])
    (SMALL/'fixtures.json').write_text(json.dumps(meta,indent=2)+'\n')
    result=dict(task='W41_same_input_hold',deadline_unix_s=None,status='running',results=[],
        original8='Retained in gpu/w40-gpu; not recalculated',new_events=len(rows),
        model_identity='S5-r29-qp2 same held dynamics as r29/qp1; fixed independent target; FP64 one-sided1e-6')
    begin=time.monotonic()
    for row in rows:
        pred=CPUHold(ROOT/row['source']/'model.mjb')
        mujoco.set_mjcb_contactfilter(pred.cpu_filter);mujoco.set_mjcb_passive(pred.cpu_passive)
        out=measure(pred,row);result['results'].append(out);result['wall_s']=time.monotonic()-begin
        for path in [OUT/'result.json',SMALL/'result.json']:path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(out),flush=True)
        mujoco.set_mjcb_contactfilter(None);mujoco.set_mjcb_passive(None)
    result['status']='completed_same_input_checks'
    for path in [OUT/'result.json',SMALL/'result.json']:path.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
