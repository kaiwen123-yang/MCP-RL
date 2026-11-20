"""Replay the W55 slot-8 witness, later infeasible step and original 350ms records."""
from pathlib import Path
import json,subprocess
import numpy as np
import mujoco
from scipy.optimize import minimize_scalar
from analyze_s1_w55_n import records,read_policy
from s1_go2_w56_n import SIZE,SURFACE,MOTION
from s1_go2_w59_n import PrecontactE
from publish_s1_go2_w54 import ROOT,write

RUN='terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
OLD=Path('/root/autodl-tmp/MCP-RL-w55-n-r56')
SMALL=ROOT/'results'/RUN/'w56/n'
WORK=ROOT/'work/w56-n-r59'

def original_phase(m,p):
    # Infer the offset of the frozen 2Hz residual clock directly from the
    # saved four gait residuals and foot heights; only affects private replay.
    t=p['accepted_times'][:174];r=p['accepted_residual'][:174,7:11]
    d=mujoco.MjData(m);ids=[m.geom(n).id for n in ('FL','RL','FR','RR')]
    heights=[]
    for x in p['accepted_states'][:174]:
        d.qpos[:]=x[:19];mujoco.mj_kinematics(m,d)
        heights.append(d.geom_xpos[ids,2]-d.geom_xpos[m.geom('floor').id,2]-.02)
    heights=np.array(heights)
    def objective(phi):
        angle=(4*np.pi*(t-t[0])[:,None]+phi-np.array([0,np.pi,np.pi,0])+np.pi)%(2*np.pi)-np.pi
        step=.03*np.maximum(0.,np.cos(np.clip(angle/(2*.55),-np.pi/2,np.pi/2)))
        predicted=np.where(step>3e-8,heights-step,0.)
        return np.square(predicted-r).mean()
    # Locate the periodic minimum of one recorded phase, without adjusting
    # any control parameter, policy action or physical bound.
    samples=np.linspace(-np.pi,np.pi,65);loc=samples[np.argmin([objective(a) for a in samples])]
    fit=minimize_scalar(objective,bounds=(loc-.10,loc+.10),method='bounded')
    return fit.x,float(np.sqrt(fit.fun))

def main():
    old=OLD/'work/w55-n-r56';run=json.loads((OLD/'results'/RUN/'w55/n/N-A-r56/E30.json').read_text())
    ep=OLD/'outputs'/RUN/run['attempt'];native=ep/'w54-n-native';xml=ep/'model/G1-project-go2.xml'
    m=mujoco.MjModel.from_xml_path(str(xml));m.opt.timestep=.001
    val=WORK/'validation';val.mkdir(exist_ok=True);exe=val/'replay'
    lib=ROOT/'work/r11-frozen/S5-r59/lib';deps=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2/build-autodl/_deps'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(WORK),'-I'+str(deps/'mujoco-src/include'),str(ROOT/'scripts/w59_n_replay.cc'),
                     '-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lw55_support','-l:libmujoco.so.3.2.3','-o',str(exe)],check=True)
    ctx={r['call_id']:r for r in records(ep/'w54-n-context.jsonl') if r['call_id'] in (809,856)}
    comparison=[];witness=[]
    for call in (809,856):
        p=read_policy(native/f'policy-{call:06d}.bin');context=ctx[call]
        u=np.zeros(SIZE);u[:SURFACE]=context['userdata_prefix'];u[MOTION:]=context['epoch_direction_command']
        grid=u[SURFACE:MOTION].reshape(70,60)
        for ix,iy,h in context['world_surface_m']:grid[ix-40,iy+30]=h
        packet=np.r_[np.ravel(context['mocap_pos']),np.ravel(context['mocap_quat']),u]
        def pack(path,p,k,horizon):
            values=np.r_[p['accepted_times'][k],p['accepted_states'][k],p['accepted_actions' if horizon else 'applied_actions'][k],packet]
            if horizon:values=np.r_[values,p['accepted_actions'].ravel()]
            values.tofile(path)
        phi,residual=original_phase(m,p)
        for mode in (0,1):
            inp=val/f'call-{call}.bin';out=val/f'call-{call}-{mode}.txt';pack(inp,p,0,True)
            subprocess.run([str(exe),str(xml),str(inp),str(out),str(phi),str(mode)],check=True)
            a=np.loadtxt(out);bad=np.flatnonzero(np.any(a[:,[3,4,21,22]]!=0,axis=1));k=int(bad[0]) if len(bad) else None
            comparison.append(dict(call=call,mode=['recorded_raw','trot_support_handoff'][mode],phase_rad=phi,gait_reconstruction_rms_m=residual,
                first_failed_node=k,first_failure_fields=['generator','execution','native_substep_1','native_substep_2'],
                first_codes=(a[k,[3,4,21,22]].astype(int).tolist() if k is not None else None),dx_m=float(a[-1,-2]-a[0,1]),
                dx_before_first_failure_m=float(a[k,1]-a[0,1]) if k is not None else None,
                min_base_z_m=float(np.min(a[:,2])),final_height_m=float(a[-1,-1]),foot_transitions=np.count_nonzero(np.diff(a[:,5:9],axis=0),axis=0).tolist(),
                four_bearing_fraction=float(a[:,5:9].all(axis=1).mean())))
        if call==809:
            p8=read_policy(native/'call-000809-candidate-08.bin')
            for t in [15.993,15.994,15.995,15.996,15.997]:
                k=np.searchsorted(p8['accepted_times'],t-1e-8)-1
                sub=round((t-p8['accepted_times'][k])/.001);inp=val/'witness.bin';pack(inp,p8,k,False)
                code=int(subprocess.check_output([str(exe),str(xml),str(inp),str(sub)],text=True))
                witness.append(dict(t_s=t,new_code=code,original_code=7,original_highs_status=0 if t<15.997 else 2,rows_at_first_rejection=275 if t==15.993 else None))
    # Reconfirm the mapped PD counterexample with the actual wheel library and
    # the native command at this saved input (different MuJoCo 3.2.3 ABIs).
    path=old/'validation/RR_calf_upper_counterexample-input.bin';z=np.fromfile(path);d=mujoco.MjData(m);d.time=z[0]
    d.qpos[:]=z[1:20];d.qvel[:]=z[20:38];d.ctrl[:]=z[38:50];offset=50
    for field in (d.mocap_pos,d.mocap_quat,d.userdata):field[:]=z[offset:offset+field.size].reshape(field.shape);offset+=field.size
    applied,dg=PrecontactE(m).apply(d,d.ctrl)
    out=val/'mapping.txt';subprocess.run([str(exe),str(xml),str(path),str(out),'0','0'],check=True)
    mapping=dict(native_wheel_delta_rad=float(np.max(abs(np.loadtxt(out)[0,9:21]-applied))),
                 maximum_command_bound_excess_rad=max(0.,float(np.max(np.maximum(m.actuator_ctrlrange[:,0]-applied,applied-m.actuator_ctrlrange[:,1])))),
                 input=str(path))
    prior=json.loads((WORK/'first-proposal.json').read_text()) if (WORK/'first-proposal.json').is_file() else None
    report=dict(first_handoff_revision=(prior['comparison'] if prior else None),
        revision_basis='Initially front contacts transferred but both hind feet remained loaded throughout. Model the unilateral releasing hard foot only after its opposite diagonal bears, verify positive separating acceleration, and track the six floating-base responses under the same physical constraints.',
        source_job=run['job'],source_attempt=run['attempt'],source_levels='offline native saved-input comparisons',
        matrix_witness=witness,comparison=comparison,mapping=mapping,new_robot_episodes=0)
    write(SMALL/'same-input-r59.json',report);print(json.dumps(report,indent=2),flush=True)
    if [r['new_code'] for r in witness]!=[0,0,0,0,7] or mapping['native_wheel_delta_rad']>1e-8 or mapping['maximum_command_bound_excess_rad']>0:
        raise RuntimeError('r59 original support/mapping witness failed')

if __name__=='__main__':main()
