"""Use one observed r56 stall to test the forward-impulse duration hypothesis."""
import json, shutil,subprocess
from pathlib import Path
import numpy as np
from analyze_s1_w55_n import records,read_policy
from run_s1_w56_n import ROOT,SMALL,RESULTS,LARGE,write
from s1_go2_w56_n import SIZE,SURFACE,MOTION

def main():
    case=json.loads((SMALL/'N-A-r56/E30.json').read_text());orig=LARGE/case['attempt'];native=orig/'w54-n-native'
    calls=list(records(native/'calls.jsonl'))
    key=next(r for r in calls if r['full_candidates'] and r['time_s']>12.5)
    call=key['call_id'];t=key['time_s'];ctx=next(r for r in records(orig/'w54-n-context.jsonl') if r['call_id']==call)
    work=ROOT/'work/w55-n-r56';dest=work/'prefix-evidence';dest.mkdir(exist_ok=True)
    # The currently rescored zero-feedback nominal reference. Private comparison
    # intentionally fixes the saved raw schedule to isolate this one switch.
    slot=next(r['slot'] for r in key['candidates'] if r['generator']=='support_recovery')-1
    reference=read_policy(native/f'call-{call:06d}-candidate-{slot:02d}.bin')
    users=np.zeros(SIZE);users[:SURFACE]=ctx['userdata_prefix'];grid=users[SURFACE:MOTION].reshape(70,60)
    for ix,iy,h in ctx['world_surface_m']:grid[ix-40,iy+30]=h
    users[MOTION:]=ctx['epoch_direction_command']
    inputs=dest/'sample.bin';np.concatenate([[t],reference['accepted_states'][0],reference['accepted_actions'][0],
         np.ravel(ctx['mocap_pos']),np.ravel(ctx['mocap_quat']),users,reference['accepted_actions'].ravel()]).tofile(inputs)
    model=orig/'model/G1-project-go2.xml';deps=ROOT/'work/terrain-memory/upstreams/mujoco_mpc-go2/build-autodl/_deps'
    active_lib=ROOT/'work/r11-frozen/S5-r56/lib';c=(work/'support.cc').read_text()
    condition='norm<1.e-8 || d->time-d->userdata[W55Epoch]>=.02-1.e-9'
    assert c.count(condition)==1
    (dest/'support.cc').write_text(c.replace(condition,'norm<1.e-8'))
    shutil.copy2(work/'support.h',dest/'support.h')
    adv=dest/'liblonger.so';subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC','-I'+str(deps/'mujoco-src/include'),
        str(dest/'support.cc'),'-L'+str(active_lib),'-Wl,-rpath,'+str(active_lib),'-l:libmujoco.so.3.2.3','-o',str(adv)],check=True)
    cmd=['g++','-std=c++17','-O2','-I'+str(work/'source'),'-I'+str(deps/'mujoco-src/include'),
        '-I'+str(deps/'abseil-cpp-src'),str(ROOT/'scripts/w56_n_same_input.cc'),'-L'+str(dest),'-Wl,-rpath,'+str(dest),
        '-L'+str(active_lib),'-Wl,-rpath,'+str(active_lib),'-llonger','-l:libmujoco.so.3.2.3','-o',str(dest/'longer')]
    subprocess.run(cmd,check=True)
    outcomes={}
    for mode,exe in [('frozen_20ms_prefix',work/'validation/same-input'),('sustained_350ms_impulse',dest/'longer')]:
        out=dest/(mode+'.bin'); subprocess.run([str(exe),str(model),str(inputs),str(out)],check=True)
        x=np.fromfile(out);end=x[129:136];outcomes[mode]=dict(dx_m=end[0],final_z_m=end[1],mean_joint_torque_sq_sum_Nm2=end[2],first_constraint_code=int(end[3]),changed_nodes=int(end[4]),min_material_gap_m=end[5],min_arrest_margin_m=end[6])
    sample=[r for r in calls if 12.5<=r['time_s']<=t+1e-8]
    installed={};positive={}
    for row in sample:
        k=row['selection_source'];installed[k]=installed.get(k,0)+1
        for cand in row['candidates']:
            if cand['dx_m']>0:
                cause=str(cand['first_constraint_code'])
                positive[cause]=positive.get(cause,0)+1
    report=dict(source_attempt=case['attempt'],source_call=call,time_s=t,original_status='ongoing_at_capture',
        original_installed_source=key['selection_source'],original_winner=key['winner'],
        actual_observed_interval_s=[12.5,t],installed_source_counts=installed,positive_dx_by_first_constraint=positive,
        actual_selected=[{k:c[k] for k in ('slot','generator','J','dx_m','first_constraint_code','first_constraint_time_s')} for c in key['candidates']],
        private_candidate_comparison=outcomes,robot_physics_steps=0,offline_native_predictor_steps=4*350,
        interpretation='Sole changed condition is whether the already support-constrained candidate asks for longitudinal impulse after 20ms. The 350ms horizon, raw reference, measured packet and whole-body checks are identical. These open-loop private paths have no formal robot qualification or live score.')
    write(SMALL/'r56-material-entry-prefix-evidence.json',report);print(json.dumps(outcomes,indent=2))
if __name__=='__main__':main()
