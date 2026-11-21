"""Summarize sampled D57 evidence. Keep the recorded pose/plan and observation clocks separate."""
from pathlib import Path
import json
import hashlib
import subprocess
import numpy as np
from capture import HERE, M

def read(name):
    return json.loads((HERE/name).read_text())

before=read('sample1.json')
after=read('sample2.json')
report=dict(schema='s1_w56_r58_original_stall_diagnosis',snapshot_beijing=after['captured_beijing'],
    sampled_from_beijing=before['captured_beijing'],direct_device_sample_before=before['nvidia_process_sample'],
    direct_device_sample_after=after['nvidia_process_sample'],cases=[])
for x,y in zip(before['cases'],after['cases'],strict=True):
    p=Path(y['root'])
    plans=[json.loads(s) for s in (p/'w55-f/mpc-solves.jsonl').open()]
    lo=119. if y['case']=='A' else 78.
    tail=[s for s in plans if s['time_s']>=lo-1.e-8]
    feet=[]
    for s in tail:
        with np.load(p/'w55-f'/s['file']) as a:
            feet.append(a['foot_center_plan'][0,:,:2])
    with np.load(p/'w55-f'/plans[-1]['file']) as a:
        modes=a['event_mode'].tolist()
        confirmed=a['confirmed_contact'].tolist()
    last_pub=y['heartbeat']['unix_s']
    result={k:v for k,v in y.items() if k!='proc'}
    result['observed_worktree_head']=result.pop('original_commit')
    compact=M/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56/f/r58'
    resolved=json.loads((compact/(y['attempt']+'-resolved.json')).read_text())
    result['resolved_source_commit']=resolved.get('source_commit')
    result['execution_fingerprint']=resolved.get('execution_fingerprint')
    dx=x['proc'];dy=y['proc']
    last_refs=[{k:r[k] for k in ('leg','consume_s','publication_s','input_start_s','input_last_s','target_N','measured_N','measured_residual_N','target_residual_N') if k in r}
               for r in (json.loads(line) for line in (p/'w55-f/normal-execution.jsonl').open())
               if r['consume_s']==y['last_soft_reference_s']]
    row=json.loads((p.parent/'evaluation-policy'/p.name/'transitions.jsonl').read_text().splitlines()[-1])
    result.update(last_logged_soft_ref_returns=last_refs,last_actor_transition_s=row['time_s'],
       simultaneous_gpu_work='Observed in both live processes via NVIDIA per-PID monitoring. Device utilization does not identify a kernel.',
       native_threads_and_device_stacks='Unavailable: normal GDB attach refused on D57 container; no signal sent',
       proc_live_measurement=dict(begin_unix_s=dx['unix_s'],end_unix_s=dy['unix_s'],cpu_delta_s=(dy['sched_runtime_ns']-dx['sched_runtime_ns'])/1e9,
          io_delta={k:dy['io'][k]-dx['io'][k] for k in dx['io']},tracer_pid_after=dy['tracer']),
       unchanged_mpc_file=x['mpc_packet_sha256']==y['mpc_packet_sha256'] and x['last_mpc_file_write_unix_s']==y['last_mpc_file_write_unix_s'],
       unchanged_heartbeat=x['heartbeat']==y['heartbeat'],last_plan_foot_x_min_max_m=np.array([np.min(feet,axis=0)[:,0],np.max(feet,axis=0)[:,0]]).tolist(),
       plan_samples_since_final_material_packet=len(tail),last_plan_contact_modes=modes,last_plan_contacts_confirmed=confirmed,
       unrecorded_interval_upper_s_if_next_regular_replan=plans[-1]['time_s']+.02)
    report['cases'].append(result)
report['read_only_attach']=dict(container_yama_ptrace_scope=1,cap_sys_ptrace_in_effective_set=False,seccomp_mode=2,
    native_gdb='No thread selected; could not attach; ptrace: Inappropriate ioctl for device on both attempts',
    preconfigured_user_triggered_cuda_core_dump=False,kernel_journal_inside_container='no journal files',
    device_reset_performed=False,original_workers_signalled=False,original_control_modified=False)
report['localized_path']=dict(common_gpu='Between the recorded MPC calls and the next expected MPC record, the 2 Hz actor is not scheduled. FP64 learned-column work runs on the GPU; actual plant reference, force MPC and WBC QP run on CPU.',
    observer='WindowObserver.observe -> MaterialObserver.observe -> Ground.step -> LearnedColumns.step -> self.solve and NumPy device result readback',
    A='All 37 stored plans from 119.000 to 119.700 put every current planned foot outside the strip. The observer free-traction path is the strongest candidate.',
    B='At 78.140 leg 1 and leg 2 each finished both normal-reference inverses. A later observation or the next not-yet-recorded soft reference also reaches LearnedColumns.step.',
    boundary='Last MPC time denotes a successful, durably recorded solve. Original host PC, exact substep/Newton iteration, pending device function and driver-level cause were not sampled.')
a,b=report['cases']
split=b['last_mpc_file_write_unix_s']
ahead=[json.loads(line) for line in (Path(a['root'])/'w55-f/mpc-solves.jsonl').open()]
after_split=[i for i in ahead if (Path(a['root'])/'w55-f'/i['file']).stat().st_mtime>split]
report['A_after_B_last_plan']=dict(later_A_MPC_records=len(after_split),first_and_last_A_physical_s=[after_split[0]['time_s'],after_split[-1]['time_s']])
report['read_only_cuda_checkpoint_state']=read('cuda-process-state.json')
report['cuda_process_state_interpretation']='Both original contexts returned CUDA_SUCCESS and CU_PROCESS_STATE_RUNNING via cuCheckpointProcessGetState. There is no previously suspended CUDA checkpoint to restore or unlock.'
report['checkpoint_sources']=dict(original_evaluation='Original evaluator has t=0 staged qualification. No restartable coupled plant/controller/observer checkpoint found at these control ticks.',evidence='The MPC NPZ stores plans, qpos and qvel. The last complete measured material blocks end at 119.000 and 78.000. The evaluation-policy folders contain transition and learning logs.')
report['limited_lock_unlock_on_originals']=read('quiescence.json')
report['submission_pause_observation']=read('submission-boundary.json')
report['peer_GPU_exclusion_trials']=read('exclusive.json')
report['peer_GPU_exclusion_result']='Each original alone retained measurable device utilization; neither advanced to another solved MPC record or physical heartbeat. Both peer locks were released successfully.'
report['checkpoint_restore_disposable_process']=read('rehearsal.json')
report['original_checkpoint_restore_attempted']=False
report['preservation_decision']='User requested preservation of both original GPU contexts; checkpoint and restore tested only on the independent B actor/material process.'
report['selected_cell_replay']=read('replay.json')
report['complete_packet_reconstruction']=read('full-window.json')
report['toolchain']=dict(jax='0.6.2',jaxlib='0.6.2',jax_cuda12_plugin='0.6.2',torch='2.8.0+cu128',nvidia_driver='595.71.05',
    causal_boundary_interval_s=.0002,control_publication_interval_s=.002,
    original_worktree=str(M),diagnostic_workspace=str(HERE),
    replay_source_files=['scripts/w56_r58_stall/selected_cells.py','scripts/w56_r58_stall/full_window.py','scripts/w56_r58_stall/columns.cc'])
report['original_background']=dict(deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
report['outcome']=dict(original_attempts_preserved=True,physical_clock_recovered=all(
    y['last_mpc']['time_s']>x['last_mpc']['time_s'] and y['heartbeat']['physical_time_s']>x['heartbeat']['physical_time_s']
    for x,y in zip(before['cases'],after['cases'],strict=True)),
    live_host_instruction_pointer_known=False,live_CUDA_kernel_known=False,exact_root_cause_confirmed=False,
    new_original_starts=0,new_PPO_samples=0,new_PPO_updates=0,ETA=None)
(HERE/'diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
