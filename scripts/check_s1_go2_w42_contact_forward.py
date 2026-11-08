#!/usr/bin/env python3
"""W42 checks on four saved W41 inputs; no new task episode or policy training."""
import json,os,time,importlib.util
from pathlib import Path
import numpy as np
import mujoco
from mujoco_mpc.agent import Agent
from s1_go2_w41_dynamics import CPUHold,state,diff,branch,contact_record
from s1_go2_r29_records import R29Records
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/root/autodl-tmp/MCP-RL')
REL=Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
OUT=ROOT/REL/'w42-contact-forward'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w42-contact-forward'
spec=importlib.util.spec_from_file_location('w41_input_audit',ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/controller-input-audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def make(path,role):
    mujoco.set_mjcb_contactfilter(None);mujoco.set_mjcb_passive(None)
    original=mujoco.MjModel.from_binary_path(str(path))
    pred=CPUHold(path)
    if role!='soft':
        pred.m.geom_solimp[:]=original.geom_solimp
        pred.m.pair_solimp[:]=original.pair_solimp
    if role=='plant_reconstructed':
        pred.m.opt.integrator=original.opt.integrator
        pred.m.jnt_solimp[:]=original.jnt_solimp
        pred.m.opt.disableflags=original.opt.disableflags
    return pred
def activate(pred):
    mujoco.set_mjcb_contactfilter(pred.cpu_filter);mujoco.set_mjcb_passive(pred.cpu_passive)
def fz(m,d):
    total=0.
    for j,c in enumerate(d.contact):
        f=np.empty(6);mujoco.mj_contactForce(m,d,j,f)
        total+=(c.frame.reshape(3,3).T@f[:3])[2]
    return float(total)
def held(pred,row,du=None):
    activate(pred);d=pred.cpu_data(row,ctrl=row['ctrl']+(0 if du is None else du))
    states=[];forces=[];branches=[];contacts=[]
    for j in range(10):
        mujoco.mj_step(pred.m,d);states.append(state(d));forces.append(fz(pred.m,d))
        branches.append(branch(pred.m,d));contacts.append(contact_record(pred.m,d))
    return np.array(states),np.array(forces),branches,contacts
def local_fd(soft,row,independent=False):
    activate(soft);m=soft.m;d=soft.cpu_data(row)
    A=np.zeros((2*m.nv,2*m.nv));B=np.zeros((2*m.nv,m.nu))
    mujoco.mjd_transitionFD(m,d,1e-6,0,A,B,None,None)
    if not independent:return A,B,None
    base=soft.cpu_data(row);mujoco.mj_step(m,base);baseline=state(base);fd=[]
    for k in range(2*m.nv+m.nu):
        dd=soft.cpu_data(row);eps=1e-6
        if k<m.nv:
            delta=np.zeros(m.nv);delta[k]=eps;mujoco.mj_integratePos(m,dd.qpos,delta,1.)
        elif k<2*m.nv:dd.qvel[k-m.nv]+=eps
        else:
            a=k-2*m.nv
            if m.actuator_ctrllimited[a] and dd.ctrl[a]+eps>m.actuator_ctrlrange[a,1]:eps=-eps
            dd.ctrl[a]+=eps
        mujoco.mj_step(m,dd);fd.append(diff(m,baseline,state(dd))/eps)
    return A,B,float(abs(np.c_[A,B]-np.array(fd).T).max())
def anchored(native,soft,row):
    P=np.eye(36);Q=np.zeros((36,12));xs=[];As=[];Bs=[];local_errors=[]
    current=dict(row)
    for j in range(10):
        A,B,error=local_fd(soft,current,True);As.append(A);Bs.append(B);local_errors.append(error)
        activate(native);d=native.cpu_data(current);pre=state(d);mujoco.mj_step(native.m,d)
        xs.append(np.r_[current['time'],pre,state(d)])
        P=A@P;Q=A@Q+B
        current=dict(row,qpos=d.qpos.copy(),qvel=d.qvel.copy(),time=float(d.time))
    return np.array(xs),np.array(As),np.array(Bs),P,Q,max(local_errors)
def cpp_saved_input(folder,row,resolved,model_path,soft,native):
    import xml.etree.ElementTree as ET
    dest=folder/'cpp';dest.mkdir(exist_ok=True)
    env=dict(os.environ,MJPC_R29_SNAPSHOT=str(dest/'r29-current-plan.bin'),
        MJPC_R27_PLAN_RECORD=str(dest/'planner-records.jsonl'),
        MJPC_R29_CALL_RECORD=str(dest/'planner-calls.jsonl'),MJPC_R27_ACTION_RECORD=str(dest/'feedback-actions.jsonl'),
        LD_LIBRARY_PATH=str(ROOT/'work/r11-frozen/S5-r31/lib'),
        MJPC_TASKS_DIR=str(ROOT/'work/r11-frozen/S5-r31/runtime-tasks'))
    mujoco.set_mjcb_contactfilter(None);mujoco.set_mjcb_passive(None)
    model=mujoco.MjModel.from_xml_path(str(model_path))
    settings=resolved['applied_settings'][-1]
    if not (dest/'planner-calls.jsonl').exists():
        with (dest/'agent-server.log').open('w') as log,Agent(task_id='Quadruped Flat',
            server_binary_path=ROOT/'work/r11-frozen/S5-r31/agent_server',extra_flags=['--mjpc_workers=2'],
            subprocess_kwargs=dict(stdout=log,stderr=log,env=env),run_init=False) as agent:
            agent.model=model;agent.init(task_id='Quadruped Flat',model=model,send_as='xml')
            agent.set_mode('Quadruped');agent.set_task_parameters({'Gait switch':'Manual','Gait':'Trot'})
            params=dict(time=row['time'],qpos=row['qpos'],qvel=row['qvel'],act=[],
                        mocap_pos=row['mocap_pos'],mocap_quat=row['mocap_quat'],userdata=row['userdata'])
            agent.set_state(**params)
            agent.set_task_parameters(settings['task_parameters'])
            agent.set_cost_weights(settings['cost_weights']);agent.set_state(**params)
            agent.planner_step()
            weights=agent.get_cost_weights()
    else:
        weights=settings['cost_weights']
    call=json.loads((dest/'planner-calls.jsonl').read_text().splitlines()[-1])
    if not call['policy_updated']:
        return dict(binary='S5-r31',status='no_policy_installed',call=call,
            replay_scope='Saved real input; fresh home policy has no historical warm-start/K or private gait-phase state. Failed backward is retained without retry or fallback.',
            accepted_J=None,linearization_snapshot='Unavailable because no policy was installed; same-input held evidence is retained separately.')
    records=R29Records(dest,model);records.plan(row['time'])
    pid=max(records.ring);z=records.ring[pid]['data'];records.save(pid);records.actions.close()
    T,ns,nu,nx,nres=map(int,z['header'][3:8])
    sub=z['r31_native_soft_substeps'].reshape(T-1,10,1+3*ns+nu)
    risk=z['r31_native_risk_peaks'].reshape(T-1,-1,2)
    ab=z['r31_first_soft_AB'].reshape(10,nx*nx+nx*nu)
    native_err=[];soft_err=[];risk_vals=[];cpp_local=[];P=np.eye(nx);Q=np.zeros((nx,nu))
    for j in range(10):
        rr=dict(row,time=sub[0,j,0],qpos=sub[0,j,1:20],qvel=sub[0,j,20:38],ctrl=sub[0,j,38:50],userdata=z['userdata'])
        for pred,offset,errors in [(native,50,native_err),(soft,87,soft_err)]:
            activate(pred);d=pred.cpu_data(rr);mujoco.mj_step(pred.m,d)
            errors.append(float(abs(diff(pred.m,sub[0,j,offset:offset+ns],state(d))).max()))
        A,B,_=local_fd(soft,rr);cpp_local.append(float(abs(np.r_[A.ravel(),B.ravel()]-ab[j]).max()))
        P=A@P;Q=A@Q+B
        activate(native);risk_vals.append(native.cpu_risk(native.cpu_data(rr)))
    risk_vals=np.array(risk_vals)
    # Full native held endpoints connect to each next nominal linearization point.
    continuity=float(abs(sub[:-1,-1,50:87]-sub[1:,0,1:38]).max())
    recomputed,terms=audit.cost_terms(ET.parse(model_path).getroot(),weights,z['accepted_residual'])
    recorded=float(z['header'][15]);recomputed=float(recomputed)
    planner=json.loads((dest/'planner-records.jsonl').read_text().splitlines()[-1])
    return dict(binary='S5-r31',model_roles=planner['local_model'],snapshot_plan_id=pid,
        native_step_replay_max=max(native_err),soft_step_replay_max=max(soft_err),
        cpp_soft_AB_recompute_max=max(cpp_local),
        composed_A_max=float(abs(P-z['r31_held_proxy_A'][0]).max()),
        composed_B_max=float(abs(Q-z['r31_held_proxy_B'][0]).max()),
        native_nominal_continuity_max=continuity,
        native_peak_value_error=float(abs(risk_vals.max(axis=0)-risk[0,:,0]).max()),
        native_peak_substep_mismatch=int(np.sum(risk_vals.argmax(axis=0)!=risk[0,:,1])),
        native_accepted_J=recorded,independent_J=recomputed,J_error=abs(recorded-recomputed),
        cost_terms=terms,candidates=planner['candidates'],actual_cost_decrease=planner['cost_improvement'],
        proxy_expected_decrease=planner['proxy_expected_decrease'],cost_ratio_valid=planner['cost_ratio_valid'],
        cost_ratio=planner['cost_surprise'],regularization=planner['regularization_used'],
        no_improvement_reason=planner['no_improvement_reason'],
        replay_scope='One optimization at saved real state with saved task parameters/weights/userdata. Private old task-phase/warm-start history not reconstructed; no new physical episode.')
def main():
    OUT.mkdir(parents=True,exist_ok=True);SMALL.mkdir(parents=True,exist_ok=True)
    report=dict(task='W42_CONTACT_FORWARD_JOINT',artifact='S5-r31',deadline_unix_s=None,
        scope='Four saved W41 inputs only. Same recorded held u for native/soft/reconstructed-plant comparisons.',
        limitations='Actual first-substep post-state and qacc warmstart were not saved; plant replay uses zero initial solver history. Recorded2ms endpoint/force are separate. Material proxy remains approximate; no full TM02 truth claim.',
        results=[],status='running')
    if (SMALL/'result.json').exists():
        report=json.loads((SMALL/'result.json').read_text())
    start=time.monotonic()
    for label,s,job,seed,pid in [
        ('A590','a','job-20261107T100149Z-w41-a',1,590),
        ('A635','a','job-20261107T100149Z-w41-a',1,635),
        ('B712','b','job-20261107T093906Z-w41-b',2,712),
        ('B713','b','job-20261107T093906Z-w41-b',2,713)]:
        if any(x['fixture']==label for x in report['results']):continue
        source=OLD/REL/f'{job}-eval-PANN_PPO_S5-s{seed}-E30'
        folder=OUT/label;folder.mkdir(exist_ok=True)
        with np.load(source/f'accepted-reference-gain-H-plan{pid:06d}.npz') as z:plan={k:z[k] for k in z.files}
        origin=float(plan['header'][2])
        with np.load(source/'trajectory.npz') as z:
            i=int(np.argmin(abs(z['time']-origin)))
            row=dict(time=float(z['time'][i]),qpos=z['qpos'][i],qvel=z['qvel'][i],ctrl=z['q_target'][i].astype(float),
                mocap_pos=z['mocap_pos'][i],mocap_quat=z['mocap_quat'][i],userdata=z['support_reference_userdata'][i])
            actual_next=np.r_[z['qpos'][i+1],z['qvel'][i+1]]
        physical=np.load(source/'physical-000.npz')['physical']
        ids=np.where((physical[:,0]>=origin-1e-8)&(physical[:,0]<origin+.002-1e-8))[0]
        actual_fz=physical[ids,44]
        oldprefix=list(audit.prefix_rows(plan['accepted_prefix_substeps']))[:10]
        native=make(source/'model.mjb','native');soft=make(source/'model.mjb','soft')
        plant=make(source/'model.mjb','plant_reconstructed')
        n,nf,nb,nc=held(native,row);ss,sf,sb,sc=held(soft,row);p,pf,pb,pc=held(plant,row)
        xs,As,Bs,P,Q,fd_error=anchored(native,soft,row)
        direction=-np.linalg.solve(plan['H'][0],plan['base_Qu'][0])
        direction/=np.linalg.norm(direction)
        pert,_,pert_branch,_=held(native,row,1e-6*direction)
        measured=diff(native.m,n[-1],pert[-1])/1e-6;proxy=Q@direction
        cosine=float(np.dot(measured,proxy)/(np.linalg.norm(measured)*np.linalg.norm(proxy)))
        resolved=json.loads((OLD/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/f'{source.name}-resolved.json').read_text())
        cpp=cpp_saved_input(folder,row,resolved,source/'model/G1-project-go2.xml',soft,native)
        result=dict(fixture=label,source=str(source),origin_s=origin,exact_held_target=True,
            first_Fz_N=dict(recorded=float(actual_fz[0]),native=float(nf[0]),soft=float(sf[0]),reconstructed_plant=float(pf[0])),
            hold_force_error_max_N=dict(native=float(abs(nf-actual_fz).max()),soft=float(abs(sf-actual_fz).max())),
            recorded_hold_joint_velocity_error=dict(native=float(abs(n[-1,25:]-actual_next[25:]).max()),
                soft=float(abs(ss[-1,25:]-actual_next[25:]).max())),
            reconstructed_plant_state_error_02ms=dict(native=float(abs(diff(native.m,p[0],n[0])).max()),soft=float(abs(diff(native.m,p[0],ss[0])).max())),
            reconstructed_plant_state_error_2ms=dict(native=float(abs(diff(native.m,p[-1],n[-1])).max()),soft=float(abs(diff(native.m,p[-1],ss[-1])).max())),
            soft_soft_independent_FD_max=fd_error,zero_order_native_anchor_max=float(abs(xs[:,-37:]-n).max()),
            native_direction=dict(epsilon=1e-6,source='normalized negative saved regularized Newton feedforward -H^-1 Q_u; old plan, not a new optimization claim',
                absolute_error=float(abs(measured-proxy).max()),relative_l2_error=float(np.linalg.norm(measured-proxy)/np.linalg.norm(measured)),
                cosine=cosine,contact_branch_changes=sum(a!=b for a,b in zip(nb,pert_branch))),
            cpp=cpp)
        np.savez_compressed(folder/'same-input.npz',native=n,soft=ss,reconstructed_plant=p,recorded_next=actual_next,
            native_fz=nf,soft_fz=sf,recorded_fz=actual_fz,linearization=xs,A=As,B=Bs,P=P,Q=Q,
            native_direction_fd=measured,proxy_direction=proxy,direction=direction,**row)
        (folder/'contacts.json').write_text(json.dumps(dict(native=nc,soft=sc,reconstructed_plant=pc),indent=2)+'\n')
        report['results'].append(result);report['wall_s']=time.monotonic()-start
        for dest in [OUT/'result.json',SMALL/'result.json']:dest.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    report['status']='completed_targeted_saved_input_checks'
    for dest in [OUT/'result.json',SMALL/'result.json']:dest.write_text(json.dumps(report,indent=2)+'\n')
    mujoco.set_mjcb_contactfilter(None);mujoco.set_mjcb_passive(None)
if __name__=='__main__':main()
