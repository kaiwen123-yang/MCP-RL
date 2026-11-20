"""Saved-state cross-check of actual E, native rollout and the complete transition FD."""
from pathlib import Path
import argparse, ctypes, hashlib, json, subprocess, xml.etree.ElementTree as ET
import mujoco
import numpy as np
from s1_go2_w55_n import PrecontactE, ROOT, SIZE
from s1_go2_r10_observer import MaterialObserver

def main(reuse_inputs):
    out=ROOT/'work/w55-n-r54/validation';out.mkdir(exist_ok=True)
    old=next(Path('/root/autodl-tmp/MCP-RL-w54/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12').glob('*n-a-r48*E30*'))
    deps=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2/build-autodl/_deps'
    lib=ROOT/'work/r11-frozen/S5-r54/lib'
    exe=out/'same-input'
    subprocess.run(['g++','-O2','-std=c++17','-I'+str(ROOT/'work/w55-n-r54/source'),'-I'+str(deps/'mujoco-src/include'),
        '-I'+str(deps/'abseil-cpp-src'),str(ROOT/'scripts/w55_n_same_input.cc'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),
        '-lw55_support','-l:libmujoco.so.3.2.3','-o',str(exe)],check=True)
    tree=ET.parse(old/'model/G1-project-go2.xml');root=tree.getroot()
    root.find('size').set('nuserdata',str(SIZE));root.find("custom/numeric[@name='w54_precontact_E']").set('data','1')
    xml=out/'model.xml';tree.write(xml,encoding='unicode')
    m=mujoco.MjModel.from_xml_path(str(xml));m.opt.timestep=.001;m.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_WARMSTART
    m.jnt_solimp[:,0]=0;m.geom_solimp[:,0]=0;m.pair_solimp[:,0]=0
    E=PrecontactE(m);E.lib.w55_native_contact_filter()
    step=E.lib.w55_native_step;step.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_bool];step.restype=None
    z=np.load(old/'trajectory.npz');acts=[json.loads(x)['raw_action'] for x in (old/'w54-n-execution.jsonl').read_text().splitlines()]
    params=json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
    observer=None if reuse_inputs else MaterialObserver(params,ROOT/'work/assets/B_random1000--ideal-potentials.npz')
    wanted=[5.88,6.58,6.66];indices={int(np.argmin(abs(z['time']-w))) for w in wanted}
    t=z['time'];feet=z['feet'];fv=z['feet_velocity'];radii=m.geom_size[[m.geom(k).id for k in ('FL','FR','RL','RR')],0]
    last_index=max(indices);rows=[]
    sample_cells={}
    if reuse_inputs:
        saved=json.loads((out/'input-histories.json').read_text())
        sample_cells={int(k):v for k,v in saved.items()}
    for i in (sorted(indices) if reuse_inputs else range(last_index+1)):
        if observer is not None:
            bottom=feet[i].copy();bottom[:,2]-=radii
            observer.observe(t[i],bottom,fv[i])
        if i not in indices:continue
        d=mujoco.MjData(m);d.time=t[i];d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=acts[i]
        d.mocap_pos[:]=z['mocap_pos'][i];d.mocap_quat[:]=z['mocap_quat'][i];d.userdata[:62]=z['support_reference_userdata'][i,:62]
        inp=out/f'input-{i}.bin';op=out/f'output-{i}.bin'
        if observer is not None:
            E.capture(d,z['contact'][i].astype(bool),z['actual_grf'][i],observer)
            np.concatenate([[d.time],d.qpos,d.qvel,d.ctrl,d.mocap_pos.ravel(),d.mocap_quat.ravel(),d.userdata]).tofile(inp)
            sample_cells[i]=len(observer.ground.ids)
        else:
            d.userdata[:]=np.fromfile(inp)[-m.nuserdata:]
        subprocess.run([str(exe),str(xml),str(inp),str(op)],check=True)
        native=np.fromfile(op);ap=native[:12];dg=native[12:108].reshape(4,24);q=native[108:127];v=native[127:145];A=native[145:1441].reshape(36,36);B=native[1441:].reshape(36,12)
        raw=d.ctrl.copy();origq=d.qpos.copy();origv=d.qvel.copy()
        outu=np.empty(12);diagnostic=np.empty((4,24))
        status=E.fn(m._address,d._address,raw,outu,diagnostic)
        du=np.cos(np.arange(12)+1);du/=np.linalg.norm(du)
        vdir=np.sin(np.arange(36)+1);vdir/=np.linalg.norm(vdir)
        for perturbed in (False,True):
            d.qpos[:]=origq;d.qvel[:]=origv;d.time=t[i]
            if perturbed:
                mujoco.mj_integratePos(m,d.qpos,vdir[:18],1e-6);d.qvel[:]+=1e-6*vdir[18:]
            u=np.empty(12);diag=np.empty((4,24));E.fn(m._address,d._address,np.ascontiguousarray(raw+int(perturbed)*1e-6*du),u,diag);d.ctrl[:]=u
            step(m._address,d._address,True);step(m._address,d._address,True)
            if not perturbed:q0=d.qpos.copy();v0=d.qvel.copy()
        dq=np.empty(18);mujoco.mj_differentiatePos(m,dq,1e-6,q0,d.qpos)
        observed=np.r_[dq,(d.qvel-v0)/1e-6];linear=A@vdir+B@du
        rows.append(dict(source_time_s=float(t[i]),projection_status=status,finite=np.isfinite(native).all().item(),
            E_python_native_max_rad=float(np.max(abs(outu-ap))),diagnostic_max=float(np.max(abs(diagnostic-dg))),
            held_transition_max=float(max(np.max(abs(q-q0)),np.max(abs(v-v0)))),
            fd_mixed_direction_relative=float(np.linalg.norm(observed-linear)/max(np.linalg.norm(observed),1e-12)),
            raw_to_applied_rad=float(np.max(abs(outu-raw))),bearing=dg[:,10].tolist(),brake_gate=dg[:,4].tolist(),
            baseline_normal_N=dg[:,16].tolist(),braking_normal_N=dg[:,17].tolist(),
            captured_world_cells=sample_cells[i],nonzero_rest_heights=int(np.count_nonzero(d.userdata[79:])),
            baseline_body_acc_m_s2=dg[0,14],braking_body_acc_m_s2=dg[0,15],max_brake_shortfall_m_s2=max(dg[:,13])))
    (out/'input-histories.json').write_text(json.dumps(sample_cells)+'\n')
    def digest(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    result=dict(source=str(old),rows=rows,physical_steps=0,native_executable_sha256=digest(lib.parent/'agent_server'),
        native_operator_sha256=digest(lib/'libw55_support.so'),plant_abi_operator_sha256=digest(lib/'libw55_support_python.so'),
        saved_input_sha256={str(i):digest(out/f'input-{i}.bin') for i in sorted(indices)},evidence='saved r48 states; independent causal 2 ms PANN replay and unchanged input to C ABI vs native wrapper + two predictor substeps and exact FD perturbations; mixed directional error measures differentiability, each perturbation uses the same model')
    dest=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/n/same-input-r54.json'
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not all(x['finite'] and x['E_python_native_max_rad']<1e-8 and x['diagnostic_max']<1e-7 and x['held_transition_max']<1e-7 for x in rows):
        raise RuntimeError('Saved identical inputs disagree across E/rollout paths')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reuse-inputs',action='store_true')
    main(p.parse_args().reuse_inputs)
