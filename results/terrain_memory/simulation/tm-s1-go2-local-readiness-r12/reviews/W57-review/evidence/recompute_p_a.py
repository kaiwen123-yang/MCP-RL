"""Read-only independent checks of the published P/A/E30 original recording."""
from pathlib import Path
import argparse, hashlib, json, zipfile
from collections import Counter
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--media-dir',type=Path,required=True)
HERE=parser.parse_args().media_dir.resolve()
RAW = HERE / 'raw-p-a-e30/episode'
LEGS = ['FL', 'FR', 'RL', 'RR']
archive_path=HERE/'w57-m1-p-a-original-e30-a01-originals-001.zip'
expected='74061506a1f6e42efc469a705a3a3976ba36a023f32cb90bc075dac1f86e9915'
archive_hash=hashlib.sha256(archive_path.read_bytes()).hexdigest()
assert archive_hash==expected, 'Published source archive SHA256 mismatch'
with zipfile.ZipFile(archive_path) as source:
    for name in ('episode/trajectory.npz','episode/physical-000.npz','episode/w55-f/contact-events.jsonl','episode/w55-f/manifest.json'):
        target=HERE/'raw-p-a-e30'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read(name))
z = np.load(RAW/'trajectory.npz', allow_pickle=False)
p = np.load(RAW/'physical-000.npz', allow_pickle=False)['physical']
manifest = json.loads((RAW/'w55-f/manifest.json').read_text())
events = [json.loads(line) for line in (RAW/'w55-f/contact-events.jsonl').read_text().splitlines()]
cols = ['force_evaluation_time_s','post_integration_time_s']
cols += [f'torque_{i}' for i in range(12)] + [f'actuator_force_{i}' for i in range(12)]
cols += [f'contact_{l}' for l in LEGS] + [f'grf_{l}_{a}' for l in LEGS for a in 'xyz']
cols += [f'total_grf_{a}' for a in 'xyz'] + [f'nonfoot_grf_{a}' for a in 'xyz'] + ['nonfoot_ground_contacts']
cols += [f'unclipped_pd_request_{i}' for i in range(12)]
cols += [f'foot_position_{l}_{a}_m' for l in LEGS for a in 'xyz']
cols += [f'foot_velocity_{l}_{a}_m_s' for l in LEGS for a in 'xyz']
assert len(cols)==p.shape[1]==85
fz = p[:,[cols.index(f'grf_{l}_z') for l in LEGS]]
pos = p[:,[cols.index(f'foot_position_{l}_{a}_m') for l in LEGS for a in 'xyz']].reshape(-1,4,3)
vel = p[:,[cols.index(f'foot_velocity_{l}_{a}_m_s') for l in LEGS for a in 'xyz']].reshape(-1,4,3)
t, q, phase = z['time'], z['qpos'], z['phase']
dt = np.diff(t,append=t[-1])
distance = np.r_[np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1),0.]
power = z['actuator_force']*z['qvel'][:,6:]
mass = manifest['model_mass_kg']
travel = np.isin(phase,[1,3])
strip = (q[:,0]>=.8)&(q[:,0]<=2.2)
groups = {}
for label,mask in {'all':np.ones(len(t),bool),'straight_strip':travel&strip,'straight_rigid':travel&~strip,'turn':np.isin(phase,[2,8])}.items():
    work = float(np.sum(np.maximum(power[mask],0)*dt[mask,None]))
    path = float(distance[mask].sum())
    info = {'duration_s':float(dt[mask].sum()),'distance_xy_m':path,
            'min_base_height_m':float(q[mask,2].min()),
            'max_abs_roll_pitch_deg':float(np.degrees(np.abs(z['rpy'][mask,:2])).max()),
            'positive_joint_work_J':work,
            'positive_mechanical_CoT': work/(mass*9.81*path) if label.startswith('straight') else None,
            'duty_Fz_over_1N':(np.sum((z['actual_grf'][mask,:,2]>1)*dt[mask,None],axis=0)/dt[mask].sum()).tolist(),
            'periods_s_by_leg':{}}
    for leg in LEGS:
        liftoffs = [ev['time_s'] for ev in events if ev['kind']=='liftoff_committed' and leg in ev['legs']]
        vals=[]
        for a,b in zip(liftoffs[:-1],liftoffs[1:]):
            ids=np.searchsorted(t,[a-1e-8,b-1e-8])
            if b-a<1.5 and ids[-1]<len(t) and mask[ids[0]:ids[-1]+1].all(): vals.append(b-a)
        info['periods_s_by_leg'][leg]={'n':len(vals),'median':float(np.median(vals)) if vals else None,
            'CV':float(np.std(vals)/np.mean(vals)) if vals else None}
    groups[label]=info

# A separate implementation of W57's stated selected-small-force-off interval definition.
# Classification below joins only the particular small intervals found in this new E30.
archive=zipfile.ZipFile(HERE/'w57-m1-p-a-original-e30-a01-originals-001.zip')
cache={}
def control(index):
    chunk=index//500+1; row=index%500
    if chunk not in cache:
        name=f'episode/w55-f/control-{chunk:05d}.npz'
        target=HERE/'raw-p-a-e30'/name
        target.write_bytes(archive.read(name))
        with np.load(target,allow_pickle=False) as c:
            cache[chunk]={key:c[key] for key in ('time','event_mode','confirmed_contact','load_bearing_contact','planned_contact','force','feet_position_reference','measured_grf','contact_confirmation_count','feet_velocity_reference')}
    return {k:v[row] for k,v in cache[chunk].items()}

off_records=[]
for li,leg in enumerate(LEGS):
    mask=fz[:,li]>1.
    boundaries=np.r_[0,np.nonzero(np.diff(mask.astype(int)))[0]+1,len(mask)]
    starts=[ev for ev in events if ev['kind']=='liftoff_committed' and leg in ev['legs']]
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        if left==0 or right==len(mask) or mask[left]: continue
        duration=(right-left)*.0002
        if duration<.04-1e-10: continue
        apex=float(np.max(pos[left:right,li,2]-.022))
        if duration>.05+1e-10 and apex>.005: continue
        onset=left-1
        while onset and mask[onset-1]: onset-=1
        # Recover the first impact in the committed swing, retaining at most one-tick recrossings.
        ev=next((ev for ev in reversed(starts) if ev['time_s']<=p[onset,0]+1e-8),None)
        full_onset=onset
        if ev:
            j=int(np.searchsorted(p[:,0],ev['fixed_apex_s']-1e-8))
            candidates=np.flatnonzero(mask[j:left])
            if len(candidates): full_onset=j+int(candidates[0])
        lo=max(0,full_onset//10);hi=min(len(t)-1,left//10)
        ctrl=[control(i) for i in range(lo,hi+1)]
        bearing=any(bool(c['confirmed_contact'][li] and c['load_bearing_contact'][li] and c['planned_contact'][li]) for c in ctrl)
        swing=any(str(c['event_mode'][li])=='swing' for c in ctrl)
        early=ev is not None and p[full_onset,0]<ev['fixed_touchdown_s']-1e-9
        reversal=vel[full_onset,li,2]<0<vel[left,li,2]
        rec={'leg':leg,'force_on_s':float(p[full_onset,0]),'loss_s':float(p[left,0]),'reload_s':float(p[right,0]),
             'duration_s':duration,'apex_bottom_mm':1000*apex,
             'impact_minus_planned_td_ms':1000*(float(p[full_onset,0])-ev['fixed_touchdown_s']) if ev else None,
             'impact_vz_m_s':float(vel[full_onset,li,2]),'loss_vz_m_s':float(vel[left,li,2]),
             'peak_prior_force_N':float(fz[full_onset:left,li].max()),'confirmed_bearing_before_loss':bearing,
             'control_at_impact_mode':str(ctrl[0]['event_mode'][li]),
             'control_at_impact_planned_stance':bool(ctrl[0]['planned_contact'][li]),
             'classification':'early_swing_contact_rebound' if early and swing and not bearing and reversal else 'unresolved',
             'classification_inputs':{'early':bool(early),'swing':bool(swing),'reversal':bool(reversal)}}
        off_records.append(rec)

timeline=[]
for when in [9.650,9.652,9.654,9.656,9.664,9.666,9.668,9.676,9.756,9.758,9.764,9.770]:
    c=control(round(when/.002))
    timeline.append({'time_s':float(c['time']),'event_mode':str(c['event_mode'][1]),
        'planned_contact':bool(c['planned_contact'][1]),'confirmed_contact':bool(c['confirmed_contact'][1]),
        'load_bearing_contact':bool(c['load_bearing_contact'][1]),
        'observed_Fz_N':float(c['measured_grf'][1,2]),'solved_Fz_N':float(c['force'][1,2]),
        'contact_confirmation_count':int(c['contact_confirmation_count'][1]),
        'desired_foot_vz_m_s':float(c['feet_velocity_reference'][1,2])})
limits=np.array([23.7,23.7,45.43]*4)
actual_tau=p[:,[cols.index(f'actuator_force_{i}') for i in range(12)]]
out={'attempt':'w57-m1-p-a-original-e30-a01',
     'repository_snapshot':'e2d38923a32b0ea4d622001e963789203529de0f',
     'controller_identity':manifest['config']['identity'],
     'source_archive_sha256':archive_hash,
     'source_code_for_column_order':'scripts/s1_go2_r11_runtime.py physical_columns',
     'physical_rows':len(p),'trajectory_rows':len(t),'physical_finite':bool(np.isfinite(p).all()),
     'max_dt_5khz_error_s':float(np.max(np.abs(np.diff(p[:,0])-.0002))),
     'clock_note':'Raw force is recorded after the next before_physics/material application and mj_step, with both force-evaluation and post-integration times. The 500Hz trajectory is captured before those substeps; adjacent boundary Fz need not be numerically equal. Physical peaks and dropouts use raw 5kHz rows only.',
     'max_nonfoot_ground_contacts':int(np.max(p[:,cols.index('nonfoot_ground_contacts')])),
     'max_nonfoot_force_norm_N':float(np.max(np.linalg.norm(p[:,[cols.index(f'nonfoot_grf_{a}') for a in 'xyz']],axis=1))),
     'physical_peak_Fz_N_by_leg':np.max(fz,axis=0).tolist(),
     'physical_peak_actual_torque_fraction':float(np.max(np.abs(actual_tau)/limits)),
     'groups':groups,
     'selected_force_off_scope':'Closed gaps Fz<=1N >=40ms, excluding duration>50ms with sole apex>5mm; raw 5kHz + original WBC joins, P/A/E30 only.',
     'selected_force_off_count':len(off_records),'selected_force_off_classification':dict(Counter(r['classification'] for r in off_records)),
     'selected_force_off_records':off_records,
     'representative_FR_impact_control_timeline':timeline}
(HERE/'p-a-e30-independent-metrics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2,ensure_ascii=False))
