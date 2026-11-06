#!/usr/bin/env python3
"""Audit saved W39 control data. Never calls a dynamics integrator or planner.

Usage: python control-analysis.py --raw /path/to/w39-raw [--all-attempts]
The raw directory contains root review's raw-index.json and verified downloads.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'AGENTS.md').is_file())
SMALL = ROOT / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'


def scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def stat(a):
    a = np.asarray(a)
    return None if not len(a) else dict(n=len(a), min=float(a.min()),
        median=float(np.median(a)), p95=float(np.quantile(a, .95)), max=float(a.max()))


def bytes_for(r):
    parts = []
    for p in r['parts']:
        path = Path(p['local_path'])
        if not path.exists() or path.stat().st_size != p['size']:
            raise FileNotFoundError(path)
        parts.append(path.read_bytes())
    b = b''.join(parts)
    if r['codec'] == 'gzip':
        b = gzip.decompress(b)
    if len(b) != r['source_size']:
        raise ValueError(f"Decompressed source size differs for {r['source_path']}")
    return b


def available(r):
    return all(Path(p['local_path']).exists() and
        Path(p['local_path']).stat().st_size == p['size'] for p in r['parts'])


def load(r, hashes):
    b = bytes_for(r)
    hashes.append(dict(name=r['name'], source_path=r['source_path'],
        source_bytes=len(b), sha256=hashlib.sha256(b).hexdigest(),
        release_parts=[p['name'] for p in r['parts']]))
    if r['name'].endswith('.jsonl'):
        return [json.loads(x) for x in b.splitlines() if x.strip()]
    if r['name'].endswith('.json'):
        return json.loads(b)
    if r['name'].endswith('.npz'):
        with np.load(io.BytesIO(b), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return b


def state_diff(ref, actual):
    """Go2's 37 storage / 36 tangent convention; no physics calculation.

    Free-joint orientation increment is log(conj(q_ref) * q_actual).
    Later validated against every logged state_delta, not assumed equivalent.
    """
    ref, actual = np.asarray(ref), np.asarray(actual)
    q0, q1 = ref[3:7], actual[3:7]
    rot0 = Rotation.from_quat(q0[[1, 2, 3, 0]])
    rot1 = Rotation.from_quat(q1[[1, 2, 3, 0]])
    return np.r_[actual[:3]-ref[:3], (rot0.inv()*rot1).as_rotvec(),
        actual[7:19]-ref[7:19], actual[19:37]-ref[19:37]]


def lerp(times, values, t, normalize=False):
    n = len(values)
    if t <= times[0]:
        y = values[0].copy()
    elif t >= times[n-1]:
        y = values[n-1].copy()
    else:
        hi = int(np.searchsorted(times[:n], t, side='right'))
        lo = hi-1
        w = (t-times[lo])/(times[hi]-times[lo])
        y = (1-w)*values[lo]+w*values[hi]
    if normalize:
        y[3:7] /= np.linalg.norm(y[3:7])
    return y


def analyze(attempt, files, detailed):
    hashes=[]
    traj=load(files['trajectory.npz'], hashes)
    acts=load(files['feedback-actions.jsonl'], hashes)
    plans=load(files['planner-records.jsonl'], hashes)
    events=load(files['material-events.jsonl'], hashes) if 'material-events.jsonl' in files and available(files['material-events.jsonl']) else []
    result=json.loads((SMALL/(attempt+'.json')).read_text())
    resolved=json.loads((SMALL/(attempt+'-resolved.json')).read_text())
    urange=np.asarray(resolved['actuator_ctrlrange']); frange=np.asarray(resolved['actuator_forcerange'])
    t=traj['time'];computed=np.isclose(t,traj['diagnostic_time'],rtol=0,atol=1e-10)
    # A horizon row may contain a fresh RPC target, but runtime breaks before
    # any following physics. A natural-failure terminal row copies old
    # diagnostics. Neither is an actually applied new control interval.
    valid=computed&(t<float(result['simulation_s'])-1e-10)
    bytime={float(np.float32(t[i])):int(i) for i in np.flatnonzero(valid)}
    executed={}; excluded=[]
    for a in acts:
        i=bytime.get(a['time'])
        if i is not None and np.array_equal(np.asarray(a['action_after_clip'],dtype=np.float32),traj['q_target'][i]):
            executed[i]=a
        else:excluded.append(a)
    ii=np.array(sorted(executed));aa=[executed[i] for i in ii];tt=t[ii]
    nom=np.clip(np.asarray([a['nominal_unclipped'] for a in aa]),urange[:,0],urange[:,1])
    before=np.array([a['action_before_clip'] for a in aa]);after=np.array([a['action_after_clip'] for a in aa])
    kk=np.array([a['raw_K_delta_x'] for a in aa]);dd=np.array([a['state_delta'] for a in aa]);ss=np.array([a['feedback_scaling'] for a in aa])
    raw_nom=60*(nom-traj['qpos'][ii,7:19])-5*traj['qvel'][ii,6:18]
    raw_exec=60*(traj['q_target'][ii]-traj['qpos'][ii,7:19])-5*traj['qvel'][ii,6:18]
    sat_nom=(raw_nom<frange[:,0])|(raw_nom>frange[:,1]);sat_exec=(raw_exec<frange[:,0])|(raw_exec>frange[:,1])
    maxk=np.max(abs(kk),axis=1);clip=np.max(abs(before-after),axis=1)
    first_load=next((e['time_s'] for e in events if e['type']=='load'),None)
    outbound=np.flatnonzero(traj['phase']==1);outbound=float(t[outbound[0]]) if len(outbound) else None
    def event(mask):
        ix=np.flatnonzero(mask)
        if not len(ix):return None
        j=int(ix[0]);i=int(ii[j]);a=aa[j]
        return dict(time_s=float(t[i]),plan_id=a['plan_id'],phase=int(traj['phase'][i]),
            max_raw_K_delta_x_rad=float(maxk[j]),max_target_clip_rad=float(clip[j]),
            nominal_saturated_actuators=np.flatnonzero(sat_nom[j]).tolist(),
            executed_saturated_actuators=np.flatnonzero(sat_exec[j]).tolist(),
            feedback_newly_saturated_actuators=np.flatnonzero(sat_exec[j]&~sat_nom[j]).tolist(),
            nominal_raw_pd_Nm=raw_nom[j].tolist(),executed_raw_pd_Nm=raw_exec[j].tolist(),
            base_xyz_m=traj['qpos'][i,:3].tolist(),rpy_rad=traj['rpy'][i].tolist(),
            qvel=traj['qvel'][i].tolist(),state_delta=dd[j].tolist(),raw_K_delta_x=kk[j].tolist(),
            contact=traj['contact'][i].tolist(),actual_grf=traj['actual_grf'][i].tolist())
    maxim=event(np.arange(len(ii))==int(np.argmax(maxk)))
    pre_load=np.flatnonzero(tt<first_load) if first_load is not None else np.arange(len(tt))
    pre_max=event(np.arange(len(ii))==int(pre_load[np.argmax(maxk[pre_load])])) if len(pre_load) else None
    classes=Counter('restored_to_feasible' if p['nominal_trunk_violation_m']>0 and p['accepted_trunk_violation_m']==0 else
        'infeasible_restoration' if p['accepted_trunk_violation_m']>0 else 'feasible_to_feasible' for p in plans)
    out=dict(attempt=attempt,new_physics_steps=0,reason=result['reason'],simulation_s=result['simulation_s'],
        sources=hashes,coverage=dict(trajectory_rows=len(t),executed_control_rows=len(ii),
            excluded_or_superseded_startup_action_records=len(acts)-len(ii),strictly_nonmatching_action_records=len(excluded),unmatched_valid_trajectory_rows=len(set(np.flatnonzero(valid))-set(ii)),
            feedback_records=len(acts),planner_records=len(plans),plans_with_nodes=sum(bool(p['nodes']) for p in plans),
            terminal_row_diagnostics_stale=bool(not computed[-1]),fresh_terminal_control_targets_not_integrated=int(np.sum(computed&~valid)),trajectory_fields={k:list(v.shape) for k,v in traj.items()},
            all_trajectory_arrays_finite=all(np.isfinite(v).all() for v in traj.values())),
        timing=dict(first_outbound_s=outbound,first_material_load_s=first_load,
            control_dt_s=stat(np.diff(t)),rpc_clock_quantization_max_abs_s=float(np.max(abs(tt-np.array([a['time'] for a in aa]))))),
        action_identity=dict(feedback_scaling_values=np.unique(ss).tolist(),
            equation_max_abs_error_rad=float(np.max(abs(before-(np.array([a['nominal_unclipped'] for a in aa])+ss[:,None]*kk)))),
            target_clip_max_abs_error_rad=float(np.max(abs(after-np.clip(before,urange[:,0],urange[:,1])))),
            protobuf_float32_roundoff_max_abs_rad=float(np.max(abs(after-traj['q_target'][ii]))),
            recomputed_raw_PD_vs_logged_max_abs_Nm=float(np.max(abs(raw_exec-traj['unclipped_pd_request'][ii]))),
            recomputed_clipped_PD_vs_control_force_max_abs_Nm=float(np.max(abs(np.clip(raw_exec,frange[:,0],frange[:,1])-traj['actuator_force'][ii])))),
        first_events=dict(large_raw_K_delta_x=event(maxk>.5),target_clipping=event(clip>0),
            nominal_PD_saturation=event(np.any(sat_nom,axis=1)),executed_PD_saturation=event(np.any(sat_exec,axis=1)),
            feedback_newly_saturated=event(np.any(sat_exec&~sat_nom,axis=1))),
        raw_K_max_event=maxim,raw_K_max_before_first_material_load=pre_max,
        control_counts=dict(large_K_gt_05=int(np.sum(maxk>.5)),target_clip_exact_nonzero=int(np.sum(clip>0)),target_clip_gt_1e_10_rad=int(np.sum(clip>1e-10)),
            nominal_sat=int(np.any(sat_nom,axis=1).sum()),executed_sat=int(np.any(sat_exec,axis=1).sum()),
            feedback_new_sat=int(np.any(sat_exec&~sat_nom,axis=1).sum()),large_K_gt_05_before_first_load=int(np.sum(maxk[pre_load]>.5))),
        plans=dict(classes=dict(classes),all_accepted_trunk_feasible=all(p['accepted_trunk_violation_m']==0 for p in plans),
            feasible_candidates_histogram=dict(Counter(p['feasible_candidates'] for p in plans)),
            regularization_at_min=sum(p['regularization_used']==1e-6 for p in plans),
            restoration_records=[{k:p[k] for k in ['origin','plan_id','nominal_trunk_violation_m','accepted_trunk_violation_m','acceptance_mode','cost_improvement','cost_ratio_valid','winner','feasible_candidates']} for p in plans if p['nominal_trunk_violation_m']>0 or p['accepted_trunk_violation_m']>0]))
    expected_calls=result['timing']['planning']['count']+result['startup']['iterations']
    out['plan_call_coverage']=dict(total_planner_calls_including_startup=expected_calls,
        emitted_plan_records=len(plans),calls_without_new_plan_record=expected_calls-len(plans))
    if expected_calls>len(plans) and available(files['reference-packets.json']):
        all_packets=load(files['reference-packets.json'],hashes)['packets']
        origins={round(p['origin'],8) for p in plans};gaps=[]
        for packet in all_packets:
            tm=packet['new_optimized_action_time_s']
            if round(tm,8) in origins:continue
            previous=max((p for p in plans if p['origin']<tm),key=lambda p:p['origin'])
            following=next((p for p in plans if p['origin']>tm),None)
            a=min(aa,key=lambda a:abs(a['time']-tm))
            gaps.append(dict(call_time_s=tm,wall_s=packet['planning_wall_s'],reused_policy_id=a['plan_id'],
                previous={k:previous[k] for k in ['origin','plan_id','regularization_after','accepted_cost','accepted_trunk_violation_m']},
                following=None if following is None else {k:following[k] for k in ['origin','plan_id','regularization_before','regularization_used','accepted_cost','accepted_trunk_violation_m']}))
        out['plan_call_coverage']['calls_without_new_plan']=gaps
        out['plan_call_coverage']['limitation']='Planner has early returns for backward-pass failure and no valid action rollout, before its success-only record. Missing status does not identify every failure path; reuse is directly established by feedback plan_id.'
    if not detailed:return out
    pmap={p['plan_id']:p for p in plans};origin_checks=[];endpoint=[];action_recon=[];gain_nodes=[]
    for p in plans:
        for n in p['nodes']:
            x=np.asarray(n['accepted_state']);ref=np.asarray(n['feedback_reference_state']);K=np.asarray(n['K']).reshape(12,36)
            row,col=np.unravel_index(np.argmax(abs(K)),K.shape)
            gain_nodes.append(dict(plan_id=p['plan_id'],origin=p['origin'],node=n['node'],time_s=n['t'],
                max_abs_K=float(abs(K[row,col])),max_element_action_index=int(row),max_element_tangent_state_index=int(col),
                mixed_coordinate_spectral_norm=float(np.linalg.svd(K,compute_uv=False)[0]),
                K_q_mixed_coordinate_spectral_norm=float(np.linalg.svd(K[:,:18],compute_uv=False)[0]),
                K_v_mixed_coordinate_spectral_norm=float(np.linalg.svd(K[:,18:],compute_uv=False)[0])))
            un=np.asarray(n['feedback_reference_nominal']);recon=np.clip(un+K@state_diff(ref,x),urange[:,0],urange[:,1])
            action_recon.append(float(np.max(abs(recon-n['accepted_action']))))
            i=int(np.argmin(abs(t-n['t'])))
            if abs(t[i]-n['t'])>1e-8:continue
            actual=np.r_[traj['qpos'][i],traj['qvel'][i]]
            if n['node']==0:
                j=int(np.argmin(abs(tt-p['origin'])))
                origin_checks.append(dict(origin=p['origin'],plan_id=p['plan_id'],
                    state_storage_max_abs_error=float(np.max(abs(actual-x))),
                    action_max_abs_error_rad=float(np.max(abs(traj['q_target'][i]-n['accepted_action']))),
                    raw_K_delta_x_max_abs_rad=float(maxk[j])))
            else:
                dq=state_diff(x,actual);v=dq[18:];joint=v[6:];j=int(np.argmax(abs(joint)))
                endpoint.append(dict(origin=p['origin'],time_s=n['t'],plan_id=p['plan_id'],node=n['node'],
                    delta_q_tangent=dq[:18].tolist(),delta_v=v.tolist(),delta_base_z_mm=float(dq[2]*1e3),
                    mixed_velocity_error_norm=float(np.linalg.norm(v)),joint_velocity_error_norm=float(np.linalg.norm(joint)),
                    max_joint_velocity_error_abs_rad_s=float(abs(joint[j])),max_joint_velocity_error_actuator=j,
                    actual_contact=traj['contact'][i].tolist(),accepted_action_reconstruction_max_abs_error_rad=action_recon[-1]))
    out['accepted_policy_checks']=dict(origin_checks=origin_checks,endpoint_closed_loop_discrepancies=endpoint,
        logged_node_action_reconstruction_max_abs_error_rad=max(action_recon,default=0),
        comparison_scope='Same stored initial q/v and controller law; predictor and plant have different dynamics and auxiliary state. Actual inputs diverge under feedback after states diverge. This is not a fixed-input system-identification residual.')
    large_gain_events=[]
    for j in np.flatnonzero(maxk>.5):
        dxnorm=float(np.linalg.norm(dd[j]));dunorm=float(np.linalg.norm(kk[j]))
        large_gain_events.append(dict(time_s=float(tt[j]),plan_id=aa[j]['plan_id'],
            max_raw_K_delta_x_rad=float(maxk[j]),state_position_tangent_mixed_norm=float(np.linalg.norm(dd[j,:18])),
            state_velocity_mixed_norm=float(np.linalg.norm(dd[j,18:])),raw_feedback_l2_norm=dunorm,
            live_K_mixed_coordinate_spectral_norm_lower_bound=dunorm/dxnorm if dxnorm>0 else None))
    out['feedback_gain_diagnostics']=dict(coordinate_warning='Spectral norms mix position/angle/linear and angular velocity coordinates; not dimensionless stability certificates. Column 2 is base z in metres (gain rad/m).',
        missing_actual_node_K='Only node 0 and +20ms are stored. A +2/+4ms execution row cannot be split into true K_q*dq and K_v*dv from these endpoints.',
        recorded_gain_nodes=gain_nodes,large_feedback_events=large_gain_events)
    if 'reference-packets.json' in files and available(files['reference-packets.json']):
        packets=load(files['reference-packets.json'],hashes)['packets']; errors=[];future=[]
        for r in packets:
            i=int(np.argmin(abs(t-r['new_optimized_action_time_s'])))
            errors.append(float(np.max(abs(traj['support_reference_userdata'][i,84:96]-r['packet']))))
            if r['actor_decision_time_s'] is not None and r['actor_decision_time_s']>r['new_optimized_action_time_s']+1e-9:future.append(r)
        out['reference_packets']=dict(count=len(packets),recorded_userdata_max_abs_error=max(errors,default=0),future_decisions=len(future),
            logical_latency_s=stat([r['reference_to_optimized_action_logical_s'] for r in packets]),
            actor_to_reference_latency_s=stat([r['actor_to_reference_latency_s'] for r in packets if r['actor_to_reference_latency_s'] is not None]))
    phyfiles=[r for n,r in files.items() if n.startswith('physical-') and available(r)]
    if phyfiles:
        ph=np.concatenate([load(r,hashes)['physical'] for r in sorted(phyfiles,key=lambda r:r['name'])])
        raw=ph[:,49:61];sat=(raw<frange[:,0])|(raw>frange[:,1]);inds=np.flatnonzero(sat.any(axis=1))
        j=int(inds[0]) if len(inds) else None
        nn=np.flatnonzero(ph[:,47]>0)
        out['physical']=dict(rows=len(ph),clock_start_s=float(ph[0,0]),clock_end_s=float(ph[-1,1]),
            all_arrays_finite=bool(np.isfinite(ph).all()),saturated_substeps=int(np.any(sat,axis=1).sum()),
            first_saturated_substep=None if j is None else dict(time_s=float(ph[j,0]),post_time_s=float(ph[j,1]),raw_pd_Nm=raw[j].tolist(),actuator_force_Nm=ph[j,14:26].tolist()),
            clipped_raw_PD_vs_actuator_force_max_abs_Nm=float(np.max(abs(np.clip(raw,frange[:,0],frange[:,1])-ph[:,14:26]))),
            first_nonfoot_load_s=float(ph[nn[0],0]) if len(nn) else None)
        # Saved records contain all ten actual .2 ms substeps for each 2 ms
        # target hold. This is an actual saturation count, not a QP replay.
        if len(ph)%10==0 and np.max(abs(ph[::10,0]-t[:len(ph)//10]))<1e-8:
            hold=sat.reshape(-1,10,12);late=~np.any(hold[:,0],axis=1)&np.any(hold,axis=(1,2));late_ix=np.flatnonzero(late)
            out['physical']['control_holds_feasible_at_start_but_saturating_later']=int(late.sum())
            out['physical']['first_such_hold_s']=float(ph[10*late_ix[0],0]) if len(late_ix) else None
            out['physical']['all_saturating_control_holds']=int(np.any(hold,axis=(1,2)).sum())
    pred_names=[n for n,r in files.items() if n.startswith('prediction-') and available(r)]
    if pred_names:
        pp=[load(files[n],hashes) for n in sorted(pred_names)];pred={k:np.concatenate([a[k] for a in pp]) for k in pp[0]}
        predmap={round(float(v),8):j for j,v in enumerate(pred['plan_origin'])}
        referr=[];nomerr=[];time_delta=[];nodeerr=[];diag=[]
        for j,(i,a) in enumerate(zip(ii,aa)):
            p=pmap.get(a['plan_id']);pi=None if p is None else predmap.get(round(p['origin'],8))
            if pi is None:continue
            times=pred['plan_times'][pi];states=pred['plan_states'][pi];actions=pred['plan_actions'][pi]
            ref=lerp(times,states,a['time'],True);ref64=lerp(times,states,t[i],True);un=lerp(times,actions,a['time'])
            actual=np.r_[traj['qpos'][i],traj['qvel'][i]];dx=state_diff(ref,actual)
            referr.append(float(np.max(abs(dx-a['state_delta']))));nomerr.append(float(np.max(abs(un-a['nominal_unclipped']))))
            time_delta.append(dict(time_s=float(t[i]),ref_velocity_quantization_error_norm=float(np.linalg.norm(ref[19:]-ref64[19:])),
                ref_joint_quantization_max_abs_rad=float(np.max(abs(ref[7:19]-ref64[7:19]))),
                logged_velocity_error_norm=float(np.linalg.norm(dd[j,18:]))))
            if maxk[j]>.5 and (len(diag)<30 or maxk[j]>5):diag.append(dict(plan_id=a['plan_id'],dx_reconstruction_max_abs_error=referr[-1],nominal_reconstruction_max_abs_error=nomerr[-1],**time_delta[-1]))
        for p in plans:
            pi=predmap.get(round(p['origin'],8))
            if pi is None:continue
            for n in p['nodes']:
                nodeerr.append(float(np.max(abs(pred['plan_states'][pi,n['node']]-n['feedback_reference_state']))))
        out['full_feedback_reference']=dict(arrays={k:list(v.shape) for k,v in pred.items()},
            all_arrays_finite=all(np.isfinite(v).all() for v in pred.values()),matched_controls=len(referr),
            state_delta_reconstruction_max_abs_error=max(referr,default=0),nominal_action_reconstruction_max_abs_error_rad=max(nomerr,default=0),
            planner_node_vs_prediction_reference_state_max_abs_error=max(nodeerr,default=0),
            rpc_time_reference_velocity_error_norm=stat([r['ref_velocity_quantization_error_norm'] for r in time_delta]),
            large_K_diagnostics=diag)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--all-attempts',action='store_true');args=parser.parse_args()
    index=json.loads((args.raw/'raw-index.json').read_text());groups={}
    for r in index:groups.setdefault(r['attempt'],{})[r['name']]=r
    records=[];missing=[]
    for attempt,files in sorted(groups.items()):
        if attempt == 'training':
            continue  # Upload inventory metadata, not a physical attempt.
        detailed='-eval-' in attempt
        if not detailed and not args.all_attempts:continue
        required=['trajectory.npz','feedback-actions.jsonl','planner-records.jsonl']
        not_ready=[n for n in required if n not in files or not available(files[n])]
        if not_ready:missing.append(dict(attempt=attempt,missing=not_ready));continue
        records.append(analyze(attempt,files,detailed))
        print(json.dumps(dict(attempt=attempt,rows=records[-1]['coverage']['executed_control_rows'],first_events={k:v['time_s'] if v else None for k,v in records[-1]['first_events'].items()},maxK=records[-1]['raw_K_max_event']['max_raw_K_delta_x_rad']),ensure_ascii=False),flush=True)
    output=HERE/('control-all-attempts.json' if args.all_attempts else 'control-e30-analysis.json')
    output.write_text(json.dumps(dict(baseline_commit='07b0ee3e15c9f10e9b1a6a1de3997bec36f39556',new_physics_steps=0,records=records,missing=missing),ensure_ascii=False,indent=2,default=scalar)+'\n')


if __name__=='__main__':main()
