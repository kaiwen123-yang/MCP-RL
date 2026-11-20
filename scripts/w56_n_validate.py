"""The actual counterexample and one frozen recorded full-horizon ordinary comparison."""
from pathlib import Path
import ctypes,json,subprocess,xml.etree.ElementTree as ET
import mujoco,numpy as np
from run_s1_w55_n import ROOT,SMALL,write
from analyze_s1_w55_n import read_policy,records
from s1_go2_w56_n import PrecontactE,SIZE,SURFACE,MOTION

def main():
    parent=Path('/root/autodl-tmp/MCP-RL-w55-n')
    run='tm-s1-go2-local-readiness-r12';compact=parent/f'results/terrain_memory/simulation/{run}/w55/n'
    case=json.loads((compact/'N-A-r54/E30.json').read_text());e=json.loads((compact/'r54-material-entry-candidates.json').read_text())
    o=parent/f'outputs/terrain_memory/simulation/{run}'/case['attempt']
    native=o/'w54-n-native';call=e['source_key_call'];ctx=next(r for r in records(o/'w54-n-context.jsonl') if r['call_id']==call)
    work=ROOT/'work/w55-n-r56';dest=work/'validation';dest.mkdir(exist_ok=True)
    deps=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2/build-autodl/_deps';lib=ROOT/'work/r11-frozen/S5-r56/lib'
    exe=dest/'same-input';subprocess.run(['g++','-std=c++17','-O2','-I'+str(work/'source'),'-I'+str(deps/'mujoco-src/include'),
        '-I'+str(deps/'abseil-cpp-src'),str(ROOT/'scripts/w56_n_same_input.cc'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),
        '-lw55_support','-l:libmujoco.so.3.2.3','-o',str(exe)],check=True)
    tree=ET.parse(o/'model/G1-project-go2.xml');tree.getroot().find('size').set('nuserdata',str(SIZE))
    xml=dest/'model.xml';tree.write(xml,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(xml));m.opt.timestep=.001
    E=PrecontactE(m);array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    progress=E.lib.w56_progress;progress.argtypes=[ctypes.c_void_p,ctypes.c_void_p,array,array];progress.restype=ctypes.c_int
    rows=[]
    for name,slot,k in [('stall_nominal',e['installed_policy']['slot'],0),('RR_calf_upper_counterexample',7,145)]:
        p=read_policy(native/f'call-{call:06d}-candidate-{slot:02d}.bin')
        d=mujoco.MjData(m);d.time=p['accepted_times'][k];d.qpos[:]=p['accepted_states'][k,:19];d.qvel[:]=p['accepted_states'][k,19:]
        d.ctrl[:]=p['accepted_actions'][k];d.mocap_pos[:]=ctx['mocap_pos'];d.mocap_quat[:]=ctx['mocap_quat']
        d.userdata[:SURFACE]=ctx['userdata_prefix'];grid=d.userdata[SURFACE:MOTION].reshape(70,60)
        for ix,iy,h in ctx['world_surface_m']:grid[ix-40,iy+30]=h
        d.userdata[MOTION:]=ctx['command_velocity'];fixed_schedule=p['accepted_actions'][np.minimum(np.arange(k,k+176),175)]
        inputs=dest/(name+'-input.bin');output=dest/(name+'-output.bin')
        np.concatenate([[d.time],d.qpos,d.qvel,d.ctrl,d.mocap_pos.ravel(),d.mocap_quat.ravel(),d.userdata,fixed_schedule.ravel()]).tofile(inputs)
        subprocess.run([str(exe),str(xml),str(inputs),str(output)],check=True)
        values=np.fromfile(output);raw=d.ctrl.copy();applied=np.empty(12);dg=np.empty((4,24));gen=np.empty(12)
        status=E.fn(m._address,d._address,raw,applied,dg);code=progress(m._address,d._address,raw,gen)
        old=p['applied_actions'][k];bounds=m.actuator_ctrlrange
        def excess(a):return max(0.,float(np.max(np.maximum(bounds[:,0]-a,a-bounds[:,1]))))
        row=dict(source=name,node=k,time_s=d.time,old_mapping_excess_rad=excess(old),new_mapping_excess_rad=excess(applied),
            correction_max_rad=float(np.max(np.abs(old-applied))),projection_status=status,matching_native_status=int(values[0])==status,
            matching_progress_status=int(values[109])==code,progress_generator_status=code,E_python_native_max_rad=float(np.max(abs(values[1:13]-applied))),
            E_diagnostic_max=float(np.max(abs(values[13:109]-dg.ravel()))),progress_python_native_max_rad=float(np.max(abs(values[110:122]-gen))))
        def summary(x):return dict(dx_m=x[0],final_z_m=x[1],mean_joint_torque_sq_sum_Nm2=x[2],first_constraint_code=int(x[3]),changed_nodes=int(x[4]),min_material_gap_m=x[5],min_arrest_margin_m=x[6])
        row['frozen_recorded_schedule']=summary(values[122:129]);row['supported_progress']=summary(values[129:136]);rows.append(row)
    receipt=dict(r54_attempt=case['attempt'],source_call=call,operator_identity=json.loads((lib.parent/'identity.json').read_text())['binary_sha256'],rows=rows,
        scope='Original same state and 20ms epoch. Two private native 350ms paths per record use the saved raw schedule; on the second late-horizon node the final raw sample is held past the recorded horizon. The live planner evaluates its state-dependent policy and original J before installation.',robot_physics_steps=0)
    write(SMALL/'same-input-r56.json',receipt);print(json.dumps(receipt,indent=2))
    if any(not r['matching_native_status'] or not r['matching_progress_status'] or r['E_python_native_max_rad']>1e-8 or r['progress_python_native_max_rad']>1e-8 for r in rows):raise RuntimeError('W56 shared operators differ on the saved state')
    if rows[1]['projection_status']!=0 or rows[1]['new_mapping_excess_rad']!=0 or rows[1]['old_mapping_excess_rad']<=0:raise RuntimeError('Original actuator counterexample remains')
if __name__=='__main__':main()
