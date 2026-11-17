#!/usr/bin/env python3
"""One finite offline pass over W52 states for W53 F model/matrix coverage.

No plant step, native planner rollout, numerical derivative, or policy update.
CPU references are exact frozen MuJoCo geometry plus F's actual matrix builder.
"""
import argparse
import hashlib
import inspect
import json
import os
import resource
import time
from pathlib import Path
import mujoco
import numpy as np
import torch
from s1_w53_gpu_model import Go2BatchModel, assemble_centroidal as gpu_assemble

ROOT=Path(__file__).resolve().parents[1]
REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
LARGE=ROOT/'outputs'/REL
SMALL=ROOT/'results'/REL
SOURCE_NAMES=[f'job-20261116T110800Z-w52-{arm}-eval-PANN_PPO_S5-s1-D8-attempt01' for arm in ('d2','d3')]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def native_quantities(model, data, qpos, qvel, geom_ids, body_ids):
    native_start=time.perf_counter()
    data.qpos[:]=qpos
    data.qvel[:]=qvel
    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)
    mujoco.mj_comVel(model,data)
    mujoco.mj_passive(model,data)
    h=np.empty(model.nv)
    mujoco.mj_rne(model,data,0,h)
    h-=data.qfrc_passive
    h_elapsed=time.perf_counter()-native_start
    geom_start=time.perf_counter()
    mass=float(sum(model.body_mass[body_ids]))
    positions=data.xipos[body_ids]
    com=np.sum(positions*model.body_mass[body_ids,None],axis=0)/mass
    inertia=np.zeros((3,3))
    for b in body_ids:
        R=data.ximat[b].reshape(3,3)
        r=data.xipos[b]-com
        inertia+=(R*model.body_inertia[b])@R.T+model.body_mass[b]*(r@r*np.eye(3)-np.outer(r,r))
    J=np.zeros((4,3,model.nv))
    for i,g in enumerate(geom_ids):
        mujoco.mj_jacGeom(model,data,J[i],None,int(g))
    return dict(com=com,inertia=inertia,feet=data.geom_xpos[geom_ids].copy(),jacobian=J,
                root_rotation=data.xmat[1].reshape(3,3).copy(),h=h,
                cpu_h_prerequisite_s=h_elapsed,cpu_geometry_s=time.perf_counter()-geom_start)


def update_error(stats, name, expected, actual, sample_ids):
    diff=np.abs(actual-expected)
    absolute=float(np.max(diff))
    flat=int(np.argmax(diff));idx=np.unravel_index(flat,diff.shape)
    row=stats.setdefault(name,dict(max_abs_error=0.,max_relative_to_sample_scale=0.,
                                   squared_error_sum=0.,element_count=0,worst_sample=None))
    scale=np.maximum(1.,np.max(np.abs(expected).reshape(len(expected),-1),axis=1))
    relative=float(np.max(diff.reshape(len(expected),-1)/scale[:,None]))
    if absolute>row['max_abs_error']:
        row['max_abs_error']=absolute;row['worst_sample']=sample_ids[idx[0]]
    row['max_relative_to_sample_scale']=max(relative,row['max_relative_to_sample_scale'])
    row['squared_error_sum']+=float(np.sum(diff*diff));row['element_count']+=diff.size


def model_description(model,batch_model,xml_path):
    fields=('body_parentid','body_jntadr','body_jntnum','body_pos','body_quat','body_ipos',
            'body_iquat','body_mass','body_inertia','jnt_type','jnt_bodyid','jnt_pos','jnt_axis',
            'jnt_qposadr','jnt_dofadr','qpos0','geom_bodyid','geom_pos',
            'actuator_trnid','actuator_ctrlrange','actuator_forcerange')
    return dict(model_xml=str(xml_path),model_xml_sha256=digest(xml_path),
       mujoco_version=mujoco.__version__,robot_mass_kg=batch_model.mass,
       body_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) for b in range(model.nbody)],
       joint_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(model.njnt)],
       foot_names=['FL','FR','RL','RR'],foot_geom_ids=batch_model.feet,
       physical_subtree_body_ids=batch_model.body_ids,
       free_base='qpos world xyz + wxyz quaternion; qvel world translation + body-frame angular velocity; 12 hinge angles/rates follow in compiled joint order',
       inertia='all compiled physical-subtree body COM masses and principal inertias, rotated to world and parallel-axis shifted about instantaneous subtree COM',
       quantity_units=dict(com='m',feet='m',inertia='kg m^2',jacobian='m per rad for rotational columns; dimensionless for translational columns'),
       compiled_parameters={k:np.asarray(getattr(model,k)).tolist() for k in fields})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    started=time.time();wall0=time.perf_counter();usage0=resource.getrusage(resource.RUSAGE_SELF)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=LARGE/'w53-C';out.mkdir(exist_ok=False)
    source_metadata=[];arrays=[]
    for name in SOURCE_NAMES:
        path=LARGE/name/'trajectory.npz'
        with np.load(path) as z:
            a={k:z[k].copy() for k in ('time','qpos','qvel','rpy','contact','actual_grf',
                                      'command_velocity','command_omega','phase','heading_target')}
        arrays.append(a)
        source_metadata.append(dict(name=name,path=str(path),sha256=digest(path),bytes=path.stat().st_size,
                                    states=len(a['time']),first_time_s=float(a['time'][0]),last_time_s=float(a['time'][-1])))
    resolved=json.loads((SMALL/(SOURCE_NAMES[0]+'-resolved.json')).read_text())
    xml=Path(resolved['model_path']);model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    geom_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('FL','FR','RL','RR')]
    inputs={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    sample_ids=[dict(source=x['name'],row=i,time_s=float(t)) for x,a in zip(source_metadata,arrays) for i,t in enumerate(a['time'])]
    init0=time.perf_counter();torch.cuda.init();torch.cuda.synchronize()
    batch_model=Go2BatchModel(model,geom_ids);torch.cuda.synchronize()
    init_s=time.perf_counter()-init0
    torch.cuda.reset_peak_memory_stats()
    write(out/'model.json',model_description(model,batch_model,xml))
    # Matrix assembler is imported after F's interface has been frozen.
    import s1_w53_f_backend as f_module
    from s1_w53_f_backend import assemble_centroidal
    stage_source=''.join(inspect.getsource(getattr(f_module,k))+'\n' for k in ('skew','euler_world_rate','assemble_centroidal'))
    (out/'f-stage-functions.py').write_text(stage_source)
    from scipy.spatial.transform import Rotation
    f_source=ROOT/'scripts/s1_w53_f_backend.py'
    model_source=ROOT/'scripts/s1_w53_gpu_model.py'
    timing=dict(cpu_h_prerequisite_s=0.,cpu_geometry_s=0.,cpu_matrix_assembly_s=0.,
                gpu_initialization_and_model_upload_s=init_s,gpu_h2d_s=0.,gpu_compute_s=0.,gpu_d2h_s=0.,
                comparison_and_retained_examples_s=0.)
    errors={};batches=[];retained={};effective_errors={}
    N=len(sample_ids)
    representatives={0,N-1}
    start=0
    for source,a in zip(source_metadata,arrays):
        representatives.update((start,start+len(a['time'])-1,start+int(np.argmin(np.abs(a['time']-2.14)))))
        start+=len(a['time'])
    torque_range=torch.tensor(model.actuator_forcerange.copy(),device='cuda',dtype=torch.float64)
    ctrl_range=torch.tensor(model.actuator_ctrlrange.copy(),device='cuda',dtype=torch.float64)
    torch.cuda.synchronize()
    ranges=[(0,1)]+[(i,min(i+args.batch_size,N)) for i in range(1,N,args.batch_size)]
    for begin,end in ranges:
        cpu_rows=[];hs=[]
        for i in range(begin,end):
            q,v=inputs['qpos'][i],inputs['qvel'][i]
            quantities=native_quantities(model,data,q,v,geom_ids,batch_model.body_ids)
            timing['cpu_h_prerequisite_s']+=quantities.pop('cpu_h_prerequisite_s')
            timing['cpu_geometry_s']+=quantities.pop('cpu_geometry_s')
            h=quantities.pop('h');hs.append(h[6:])
            rpy=Rotation.from_matrix(quantities['root_rotation']).as_euler('xyz')
            omega=quantities['root_rotation']@v[3:6]
            matrix_start=time.perf_counter()
            blocks=assemble_centroidal(batch_model.mass,quantities['inertia'],quantities['com'],
                quantities['feet'],quantities['jacobian'][:,:,6:],h[6:],q[7:],v[6:],
                model.actuator_forcerange,model.actuator_ctrlrange,rpy,omega)
            timing['cpu_matrix_assembly_s']+=time.perf_counter()-matrix_start
            cpu_rows.append(dict(**quantities,**blocks))
        expected={k:np.stack([x[k] for x in cpu_rows]) for k in cpu_rows[0]}
        trans0=time.perf_counter()
        q=torch.tensor(inputs['qpos'][begin:end],device='cuda',dtype=torch.float64)
        v=torch.tensor(inputs['qvel'][begin:end],device='cuda',dtype=torch.float64)
        h=torch.tensor(np.array(hs),device='cuda',dtype=torch.float64)
        torch.cuda.synchronize();h2d=time.perf_counter()-trans0
        compute0=time.perf_counter()
        with torch.no_grad():
            quantities=batch_model.evaluate(q)
            omega=(quantities['root_rotation']@v[:,3:6,None]).squeeze(-1)
            blocks=gpu_assemble(batch_model.mass,quantities['inertia'],quantities['com'],quantities['feet'],
                quantities['jacobian'],h,q[:,7:],v[:,6:],torque_range,ctrl_range,quantities['root_rotation'],omega)
            gpu_result=dict(**quantities,**blocks)
        torch.cuda.synchronize();compute=time.perf_counter()-compute0
        copy0=time.perf_counter()
        actual={k:x.cpu().numpy() for k,x in gpu_result.items()}
        torch.cuda.synchronize();d2h=time.perf_counter()-copy0
        timing['gpu_h2d_s']+=h2d;timing['gpu_compute_s']+=compute;timing['gpu_d2h_s']+=d2h
        compare0=time.perf_counter()
        for key in expected:update_error(errors,key,expected[key],actual[key],sample_ids[begin:end])
        force=inputs['actual_grf'][begin:end].reshape(end-begin,12)
        for key in ('B','friction_C','torque_C'):
            delta=np.einsum('bij,bj->bi',actual[key]-expected[key],force)
            effective_errors[key]=max(effective_errors.get(key,0.),float(np.max(np.abs(delta))))
        for i in representatives.intersection(range(begin,end)):
            for key in expected:
                retained[f'state{i:06d}_cpu_{key}']=expected[key][i-begin]
                retained[f'state{i:06d}_gpu_{key}']=actual[key][i-begin]
            retained[f'state{i:06d}_qpos']=inputs['qpos'][i]
            retained[f'state{i:06d}_qvel']=inputs['qvel'][i]
            retained[f'state{i:06d}_h_joint_cpu_input']=hs[i-begin]
        timing['comparison_and_retained_examples_s']+=time.perf_counter()-compare0
        batches.append(dict(begin=begin,end=end,states=end-begin,h2d_s=h2d,compute_s=compute,d2h_s=d2h,
                            inclusive_gpu_path_s=h2d+compute+d2h))
        if len(batches)%20==0:print(json.dumps(dict(status='running',completed_states=end,total_states=N)),flush=True)
    for e in errors.values():e['rms_error']=(e.pop('squared_error_sum')/e['element_count'])**.5
    io0=time.perf_counter();np.savez_compressed(out/'retained-examples.npz',**retained)
    timing['final_examples_write_s']=time.perf_counter()-io0
    masks=inputs['contact'].astype(int)@np.array([1,2,4,8])
    active_modes={str(k):int(np.sum(masks==k)) for k in np.unique(masks)}
    cpu_full=timing['cpu_h_prerequisite_s']+timing['cpu_geometry_s']+timing['cpu_matrix_assembly_s']
    gpu_marginal=timing['gpu_h2d_s']+timing['gpu_compute_s']+timing['gpu_d2h_s']
    hybrid=timing['cpu_h_prerequisite_s']+gpu_marginal
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result=dict(schema='w53_C_exact_model_centroidal_stage_v1',status='completed',started_unix_s=started,ended_unix_s=time.time(),
      deadline=None,wall_deadline_enabled=False,states=N,finite_source_passes=1,sources=source_metadata,
      source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),model_source,f_source)},
      stage_function_snapshot=str((out/'f-stage-functions.py').relative_to(ROOT)),
      stage_function_sha256=hashlib.sha256(stage_source.encode()).hexdigest(),
      compilation=dict(backend='PyTorch eager FP64 CUDA operations using precompiled kernels',custom_code_generation_or_JIT=False,custom_compile_s=0.,first_use_library_initialization_included=True),
      model=str((out/'model.json').relative_to(ROOT)),robot_mass_kg=batch_model.mass,
      implemented=['exact compiled Go2 world foot geometry','full18-column translational foot Jacobians',
                   'physical-subtree COM','articulated world composite inertia about COM',
                   'actual F centroidal A/B/c including gyroscopic affine term',
                   'actual F friction and static-torque/mapped-input stage inequalities'],
      unported=['bias/passive h remains exact same-model CPU input','full articulated mass matrix M',
                'WBC acceleration terms and Jacobian derivatives','full-horizon sparse packing and objective',
                'MPC/WBC QP solve','TM02 material dynamics','contact integration','native iLQG/FD'],
      runtime_installation=False,actual_F_backend='CPU remains frozen',errors=errors,
      actual_saved_force_error_effects=dict(max_abs_B_delta_times_saved_force=effective_errors['B'],
           max_abs_friction_lhs_change_N=effective_errors['friction_C'],max_abs_static_torque_lhs_change_Nm=effective_errors['torque_C'],
           interpretation='Uses each saved W52 actual GRF as a physical scale probe; no new QP solution and no claim about F optimal-action differences.'),
      coverage=dict(actual_W52_contact_masks_FL_FR_RL_RR=active_modes,
                    source_phase_counts={str(k):int(np.sum(inputs['phase']==k)) for k in np.unique(inputs['phase'])},
                    actual_material_loaded=False,
                    schedule_branch='stage matrices do not branch on contact; no new F schedule rollout',
                    terminal_state='final rows contain actual qpos/qvel at termination; copied terminal contact diagnostics are historical and not a fresh contact evaluation'),
      timing=timing,batches=batches,performance=dict(cpu_reference_total_s=cpu_full,
        hybrid_total_s_excluding_initialization=hybrid,
        hybrid_total_s_including_initialization=hybrid+init_s,
        offline_speed_ratio_excluding_initialization=cpu_full/hybrid,
        offline_speed_ratio_including_initialization=cpu_full/(hybrid+init_s),
        first_single_state_gpu_path_s=batches[0]['inclusive_gpu_path_s'],
        online_gain_established=False,
        interpretation='One finite offline comparison; single-state first-use and actual batch timing retained without warmup repeats. CPU h prerequisites included in hybrid total. No full controller speedup or online qualification.'),
      resources=dict(affinity=sorted(os.sched_getaffinity(0)),math_threads=1,device=torch.cuda.get_device_name(),
        peak_tensor_allocated_bytes=torch.cuda.max_memory_allocated(),peak_tensor_reserved_bytes=torch.cuda.max_memory_reserved(),
        process_cpu_s=usage.ru_utime+usage.ru_stime-usage0.ru_utime-usage0.ru_stime,
        process_maxrss_KiB=usage.ru_maxrss,wall_s=time.perf_counter()-wall0),
      work=dict(plant_steps=0,native_rollouts=0,finite_differences=0,PPO_samples=0,PPO_updates=0),
      artifacts=[str((out/'retained-examples.npz').relative_to(ROOT)),str((out/'model.json').relative_to(ROOT)),str((out/'f-stage-functions.py').relative_to(ROOT))])
    write(out/'result.json',result);write(SMALL/'w53/task-c.json',result)
    print(json.dumps({k:result[k] for k in ('status','states','performance','resources','work')}),flush=True)


if __name__=='__main__':
    main()
