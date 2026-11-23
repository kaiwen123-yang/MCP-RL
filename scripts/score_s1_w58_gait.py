"""Score every committed landing and every physical support pattern from one complete or failed episode."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from s1_w58_gait import ContactReference, clearance_correction

LEGS = ('FL', 'FR', 'RL', 'RR')
GROUPS = ('rigid_straight', 'material_straight', 'turn', 'hold')
PHYS = .0002
CTRL = .002


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def first_at_or_after(t, when):
    return np.searchsorted(t, when - 1.e-8)


def quant(values):
    a = [v for v in values if v is not None]
    if not a:
        return None
    return dict(zip(('min', 'median', 'p95', 'max'), np.percentile(a, [0, 50, 95, 100]).tolist()))


def phase_groups(phase, xy):
    material = (xy[:, 0] >= .8) & (xy[:, 0] <= 2.2) & (np.abs(xy[:, 1]) <= .6)
    return np.select([np.isin(phase, (1, 3)) & ~material,
                      np.isin(phase, (1, 3)) & material, np.isin(phase, (2, 8))], [0, 1, 2], default=3)


def physical_arrays(raw, cols):
    names = ['force_evaluation_time_s'] + [f'grf_{l}_z' for l in LEGS] + [
        f'foot_position_{l}_{axis}_m' for l in LEGS for axis in 'xyz'] + [
        f'foot_velocity_{l}_z_m_s' for l in LEGS] + [f'contact_{l}' for l in LEGS]
    index = [cols.index(name) for name in names]
    chunks = []
    for path in sorted(raw.glob('physical-*.npz')):
        with np.load(path) as z:
            p = z['physical']
            if len(p):
                chunks.append(p[:, index])
    a = np.concatenate(chunks)
    return a[:, 0], a[:, 1:5], a[:, 5:17].reshape(-1, 4, 3), a[:, 17:21], a[:, 21:25].astype(bool)


def control_arrays(raw):
    keys = ('time', 'feet_position_reference', 'feet_velocity_reference', 'feet_acceleration_reference',
            'confirmed_contact', 'load_bearing_contact', 'event_mode', 'measured_grf', 'observed_contact',
            'foot_velocity', 'rolling_contact_velocity')
    data = {k: [] for k in keys}
    for path in sorted((raw / 'w55-f').glob('control-*.npz')):
        with np.load(path) as z:
            for k in keys:
                data[k].append(z[k])
    return {k: np.concatenate(v) for k, v in data.items()}


def group_metrics(mask_p, mask_t, traj, ct, qref, pt, support, geom_contact, mass):
    t, xy, qvel, torque = (traj[k] for k in ('time', 'qpos', 'qvel', 'actuator_force'))
    dt = np.diff(t, append=t[-1])
    path = np.r_[np.linalg.norm(np.diff(xy[:, :2], axis=0), axis=1), 0.]
    work = (np.maximum(0., torque * qvel[:, 6:]) * dt[:, None]).sum(axis=1)
    dist = float(path[mask_t].sum())
    cm = mask_t[np.minimum(first_at_or_after(t, ct), len(t) - 1)]
    slip = np.linalg.norm(qref['rolling_contact_velocity'][:, :, :2], axis=2)
    control_dt = np.maximum(0., np.minimum(CTRL, t[-1] - ct))
    slip = (slip * (qref['measured_grf'][:, :, 2] > 1.) * control_dt[:, None])[cm].sum(axis=0)
    bits = np.array([1, 2, 4, 8])
    def histogram(states):
        return np.bincount(states[mask_p].astype(int) @ bits, minlength=16).tolist()
    raw_edges = np.diff(support.astype(np.int8), axis=0, prepend=np.zeros((1, 4), np.int8)) == 1
    sample_count = int(mask_p.sum())
    low = int(((support.sum(axis=1) < 2) & mask_p).sum())
    return dict(physical_samples=sample_count, force_on_edges_by_leg=raw_edges[mask_p].sum(0).tolist(),
                contact_bits_FL1_FR2_RL4_RR8_samples=histogram(geom_contact),
                Fz_gt_1N_bits_FL1_FR2_RL4_RR8_samples=histogram(support),
                fewer_than_two_supports_s=low * PHYS,
                fewer_than_two_supports_fraction=low / sample_count if sample_count else None,
                observed_duration_s=float(dt[mask_t].sum()), base_xy_path_m=dist,
                positive_joint_work_J=float(work[mask_t].sum()),
                positive_CoT=float(work[mask_t].sum() / (mass * 9.81 * dist)) if dist else None,
                loaded_rolling_contact_slip_path_by_leg_m=slip.tolist())


def readback_approach(lift, leg, swings, result, trajectory, ctrl, manifest):
    """Replay the proposed intervention on an unchanged committed W57 trajectory."""
    if manifest.get('w58_gait') or not swings.size:
        return {}
    ct = ctrl['time']
    apex = lift['fixed_apex_s']
    end = lift['fixed_touchdown_s']
    # Vertical committed fall has zero endpoint/apex velocities and accelerations.
    # Recover its declared height from the archived middle fall P/V, away from endpoints.
    mid = swings[np.abs(ct[swings] - .5 * (apex + end)).argmin()]
    tau = (ct[mid] - apex) / (end - apex)
    if not .05 < tau < .95:
        return {'opportunity_status': 'no_interior_archived_fall'}
    h = tau**3 * (10. - 15.*tau + 6.*tau*tau)
    dh = 30.*tau*tau*(1.-tau)**2 / (end-apex)
    pref, vref = (ctrl[k] for k in ('feet_position_reference', 'feet_velocity_reference'))
    landing = pref[mid, leg, 2] + (1.-h) * vref[mid, leg, 2] / dh
    cfg = manifest['config']['wbc']
    for k in swings:
        i = int(first_at_or_after(trajectory['time'], ct[k]))
        pos = trajectory['feet'][i, leg]
        vel = ctrl['foot_velocity'][k, leg]
        a = ctrl['feet_acceleration_reference'][k, leg] - cfg['swing_kp'] * (pref[k, leg] - pos) - cfg['swing_kd'] * (vref[k, leg] - vel)
        amount = clearance_correction(pos, vel, (pref[k, leg], vref[k, leg], a), landing,
                                      end - ct[k], cfg['dt_s'])
        if not ctrl['observed_contact'][k, leg] and amount:
            return {'opportunity_status': 'met_on_baseline', 'clock_s': float(ct[k]),
                    'offset_m': amount, 'committed_landing_z_m': float(landing)}
    return {'opportunity_status': 'did_not_trigger'}


def landing_rows(raw, result, manifest, traj, ctrl, pt, force, xyz, vz, geometry):
    events = [json.loads(l) for l in (raw / 'w55-f/contact-events.jsonl').open()]
    ct = ctrl['time']
    tt = traj['time']
    group = phase_groups(traj['phase'], traj['qpos'][:, :2])
    support = force > 1.
    rows = []
    for leg, name in enumerate(LEGS):
        lifts = [e for e in events if e['kind'] == 'liftoff_committed' and name in e['legs']]
        for num, lift in enumerate(lifts):
            start = lift['time_s']
            apex = lift['fixed_apex_s']
            end = lift['fixed_touchdown_s']
            next_lift = lifts[num+1]['time_s'] if num+1 < len(lifts) else result['simulation_s']
            limits = first_at_or_after(pt, np.array([start, apex, next_lift]))
            lo, mid, hi = np.minimum(limits, len(pt))
            tr = min(first_at_or_after(tt, start), len(tt)-1)
            row = dict(leg=name, committed_event=lift['event'], liftoff_s=start,
                       fixed_apex_s=apex, fixed_touchdown_s=end, phase_group=GROUPS[group[tr]],
                       period_s=None, phase_transition_during_period=None)
            if num:
                prev = lifts[num-1]['time_s']
                prev_i = first_at_or_after(tt, prev)
                row['period_s'] = start-prev
                row['phase_transition_during_period'] = bool(np.any(group[prev_i:tr+1] != group[tr]))
            # Emit censored swings even when no touchdown can have occurred before the episode end.
            rows.append(row)
            if mid >= hi:
                row['status'] = 'horizon_before_apex'
                continue
            if not np.any(~support[lo:mid+1, leg]):
                row['status'] = 'no_resolved_flight_before_apex'
                continue
            after = np.flatnonzero(support[mid:hi, leg])
            if not after.size:
                row['status'] = 'no_contact_after_apex'
                continue
            first = mid + int(after[0])
            if first == mid and support[mid, leg]:
                # The last takeoff-free sample gives the actual contact time, including a pre-apex impact.
                first = lo + int(np.flatnonzero(~support[lo:mid, leg])[-1]) + 1
            impact = float(pt[first])
            tk = min(first_at_or_after(tt, impact), len(tt)-1)
            row['impact_group'] = GROUPS[group[tk]]
            row['impact_substrate'] = ('material' if .8 <= xyz[first, leg, 0] <= 2.2 and
                                       abs(xyz[first, leg, 1]) <= .6 else 'rigid')
            end20 = int(min(first_at_or_after(pt, impact+.020), len(pt)))
            i = max(0, int(np.searchsorted(ct, impact + 1.e-9, side='right') - 1))
            k = min(first_at_or_after(tt, ct[i]), len(tt)-1)
            u = first_at_or_after(ct, impact)
            v = first_at_or_after(ct, next_lift)
            proven = np.flatnonzero(ctrl['confirmed_contact'][u:v, leg] & ctrl['load_bearing_contact'][u:v, leg])
            bearing = float(ct[u+int(proven[0])]) if proven.size else None
            stop = first_at_or_after(pt, bearing) if bearing is not None else hi
            off = ~support[first:stop, leg]
            zoff = xyz[first:stop, leg, 2][off]
            # Keep every individual no-force gap, including single 0.2 ms threshold recrossings.
            boundaries = np.diff(np.r_[False, off, False].astype(int))
            lengths = np.flatnonzero(boundaries == -1) - np.flatnonzero(boundaries == 1)
            row.update(status='bearing_established' if bearing is not None else 'no_bearing_before_boundary',
                       first_force_s=impact, impact_minus_committed_s=impact-end,
                       first_force_vz_m_s=float(vz[first, leg]), precontrol_s=float(ct[i]),
                       precontrol_z_error_m=float(traj['feet'][k,leg,2] - ctrl['feet_position_reference'][i, leg, 2]),
                       precontrol_vz_error_m_s=float(ctrl['foot_velocity'][i,leg,2] - ctrl['feet_velocity_reference'][i,leg,2]),
                       precontrol_mode=str(ctrl['event_mode'][i, leg]), first20ms_samples=int(end20-first),
                       first20ms_peak_N=float(force[first:end20, leg].max()),
                       first20ms_impulse_Ns=float(force[first:end20, leg].sum() * PHYS),
                       first_bearing_s=bearing, first_force_to_bearing_s=bearing-impact if bearing is not None else None,
                       observed_off_before_bearing_s=float(off.sum()*PHYS), off_intervals=int(len(lengths)),
                       longest_off_before_bearing_s=float(lengths.max(initial=0)*PHYS),
                       max_force_off_rise_above_initial_contact_m=float(max(0., (zoff-xyz[first,leg,2]).max(initial=0.))))
            # Causal first-touch latch fires on actual observed contact; an Fz crossing alone is insufficient.
            latch_ids = np.flatnonzero(geometry[mid:min(first_at_or_after(pt,end),hi), leg] &
                                       support[mid:min(first_at_or_after(pt,end),hi), leg])
            row['first_early_latch_s'] = None
            if latch_ids.size:
                latch = float(pt[mid+int(latch_ids[0])])
                receive = int(np.searchsorted(ct, latch+1.e-9, side='right'))
                if receive < v:
                    row['first_early_latch_s'] = latch
                    row['earliest_consuming_WBC_s'] = float(ct[receive])
                    row['force_still_observed_at_consumer'] = bool(ctrl['observed_contact'][receive,leg])
                    if not manifest.get('w58_gait') and ctrl['event_mode'][receive, leg] == 'swing':
                        # Numerical C1 takeover at this saved adjudicated sensor/command pair.
                        incoming = tuple(ctrl[key][receive, leg] for key in
                            ('feet_position_reference','feet_velocity_reference','feet_acceleration_reference'))
                        ti = min(first_at_or_after(tt, ct[receive]),len(tt)-1)
                        p = traj['feet'][ti,leg].copy()
                        v0 = np.zeros(3)
                        if not row['force_still_observed_at_consumer']:
                            v0[2] = -manifest['config']['contact_events']['search_speed_m_s']
                        bridge = ContactReference(ct[receive], incoming, manifest['config']['contact_events']['load_transfer_ramp_s'])
                        dp, dv, _ = bridge.update(ct[receive], 4 if row['force_still_observed_at_consumer'] else 2, p, v0)
                        row['saved_C1_p_error_m'] = float(np.linalg.norm(p+dp-incoming[0]))
                        row['saved_C1_v_error_m_s'] = float(np.linalg.norm(v0+dv-incoming[1]))
            start_w, end_w = first_at_or_after(ct, np.array([apex, min(end,impact)]))
            ids = np.arange(start_w,end_w)
            swings = ids[ctrl['event_mode'][ids,leg] == 'swing']
            row['preimpact_saved_approach'] = readback_approach(lift, leg, swings, result, traj, ctrl, manifest)
    return rows, events


def landing_metrics(rows):
    quantitative = ('impact_minus_committed_s', 'first_force_vz_m_s','precontrol_z_error_m',
        'precontrol_vz_error_m_s','first20ms_peak_N','first20ms_impulse_Ns','first_force_to_bearing_s',
        'observed_off_before_bearing_s','longest_off_before_bearing_s','max_force_off_rise_above_initial_contact_m')
    settled = sum(r['status']=='bearing_established' for r in rows)
    n = len(rows)
    result = dict(committed_leg_swings=n, status=dict(Counter(r['status'] for r in rows)),
        early_touch_latches_observed=sum(r.get('first_early_latch_s') is not None for r in rows),
        first_force_still_observed_at_consumer=sum(r.get('force_still_observed_at_consumer',False) for r in rows),
        counterfactual_approach_opportunities=sum(r.get('preimpact_saved_approach',{}).get('opportunity_status')=='met_on_baseline' for r in rows),
        primary_observed_off_s_per_committed_swing=sum(r.get('observed_off_before_bearing_s',0.) for r in rows)/n if n else None,
        bearing_fraction=settled/n if n else None, per_leg={})
    result.update({k:quant([r.get(k) for r in rows]) for k in quantitative})
    for name in LEGS:
        cycles = [r['period_s'] for r in rows if r['leg']==name and r['period_s'] is not None and not r['phase_transition_during_period']]
        result['per_leg'][name] = dict(cycles=len(cycles), period_s=quant(cycles),
            period_CV=float(np.std(cycles)/np.mean(cycles)) if cycles else None)
    return result


def score_episode(raw, result_path, output):
    raw=Path(raw); output=Path(output); result=read(result_path)
    manifest = read(raw/'w55-f/manifest.json')
    ctrl = control_arrays(raw); ct=ctrl['time']
    pt, force, xyz, vz, geometry = physical_arrays(raw, result['physical_columns'])
    with np.load(raw/'trajectory.npz') as f:
        traj = {k:f[k] for k in ('time','qpos','qvel','feet','phase','actuator_force')}
    tt = traj['time']
    group = phase_groups(traj['phase'],traj['qpos'][:,:2])
    pg = group[np.maximum(np.searchsorted(tt,pt+1.e-9,side='right')-1,0)]
    support = force > 1.
    rows, events = landing_rows(raw,result,manifest,traj,ctrl,pt,force,xyz,vz,geometry)
    groups = {}
    for i, name in enumerate(GROUPS+('motion','all')):
        if name=='motion':
            mp=pg!=3; mt=group!=3
        elif name=='all':
            mp=np.ones(len(pt),bool); mt=np.ones(len(tt),bool)
        else:
            mp=pg==i; mt=group==i
        sections = [row for row in rows if name=='all' or (name=='motion' and row['phase_group']!='hold') or row['phase_group']==name]
        groups[name] = dict(landings=landing_metrics(sections), **group_metrics(mp,mt,traj,ct,ctrl,pt,support,geometry,manifest['model_mass_kg']))
    fields=('saved_C1_p_error_m','saved_C1_v_error_m_s')
    result_set = dict(schema='s1_w58_gait_episode_v1',attempt=result['attempt'],
        case={1:'A',2:'B'}[result['case_identity']['seed']],level=result['evaluation_stage']['level'],
        gait=manifest.get('w58_gait',{}).get('gait','G0'), controller=manifest['config']['identity'],
        active_modules=manifest['config']['w57'], episode_result=str(result_path), raw=str(raw),
        stage_status=result['stage_status'],reason=result.get('reason'), observed_s=result['simulation_s'],
        physics_rows=len(pt), control_rows=len(ct), max_principal_strain=result['material'].get('max_abs_principal_strain'),
        original_slip_integral_s=result['material'].get('integral_slip_sq_s'),
        route_progress=result.get('legal_route',{}).get('progress'), groups=groups,
        contact_event_counts=dict(Counter(e['kind'] for e in events)), saved_C1={k:quant([r.get(k) for r in rows]) for k in fields},
        measurement='All physical contact and Fz>1N bitsets, all committed leg swings seen at their own clocks; unobserved post-boundary intervals stay censored. Per-event first force is from the physical force-evaluation clock; precontrol error uses reference and measurement on the same 500 Hz sample. Physical zero-force duration through a censored end is a lower bound.')
    ledger=output.with_name(output.stem+'-landings.jsonl.gz')
    output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(ledger,'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    result_set['all_committed_landings']=ledger.name
    write(output,result_set)
    return result_set


def select_gait(g0_dir, g1_dir, output):
    scores = {}
    rejected = []
    for gait, folder in [('G0', Path(g0_dir)), ('G1', Path(g1_dir))]:
        for filename in sorted(folder.glob('*.json')):
            row = read(filename)
            if row.get('schema') != 's1_w58_gait_episode_v1':
                continue
            modules = row['active_modules']
            if row['gait'] != gait or modules['material'] != 'M1' or modules['planner'] != 'P0' or modules['timing'] != 'zero':
                rejected.append(dict(path=str(filename), reason='different controller or nonzero timing'))
                continue
            key = (gait,row['case'],row['level'])
            if key in scores:
                raise ValueError(f'Duplicate gait identity/case/level in {folder}: {key}')
            scores[key] = row
    missing=[]; failed=[]; comparison={}
    for case in ('A','B'):
        for level in ('E30','E60'):
            for gait in ('G0','G1'):
                key=(gait,case,level)
                if key not in scores:
                    missing.append(dict(gait=gait,case=case,level=level))
                elif scores[key]['stage_status']!='passed':
                    failed.append(dict(gait=gait,case=case,level=level,attempt=scores[key]['attempt'],
                                       reason=scores[key]['reason']))
            if any((g,case,level) not in scores for g in ('G0','G1')):
                continue
            base,new=(scores[g,case,level] for g in ('G0','G1'))
            def measures(row):
                all_swings = row['groups']['all']['landings']
                moving = row['groups']['motion']
                return dict(off_s=all_swings['primary_observed_off_s_per_committed_swing'],
                            bearing_fraction=all_swings['bearing_fraction'],
                            low_support_fraction=moving['fewer_than_two_supports_fraction'],
                            motion_s=moving['observed_duration_s'],
                            positive_CoT=moving['positive_CoT'],
                            slip_s_per_m=row['original_slip_integral_s']/moving['base_xy_path_m']
                                         if moving['base_xy_path_m'] and row['original_slip_integral_s'] is not None else None)
            a,b = measures(base), measures(new)
            present = all(x is not None for z in (a,b) for x in z.values()) and min(a['motion_s'],b['motion_s'])>0
            gates = dict(comparable_measurements=present)
            if present:
                gates.update(off_time_lower=b['off_s'] <= a['off_s']-PHYS,
                    bearing_coverage=b['bearing_fraction'] >= a['bearing_fraction'],
                    moving_support=b['low_support_fraction'] <= a['low_support_fraction'] + PHYS/min(a['motion_s'],b['motion_s']),
                    CoT=b['positive_CoT'] <= a['positive_CoT'],
                    slip_per_distance=b['slip_s_per_m'] <= a['slip_s_per_m'])
            comparison[case+'-'+level] = dict(g0_attempt=base['attempt'],g1_attempt=new['attempt'],g0=a,g1=b,gates=gates)
    enough=not missing and not failed
    selected = enough and all(all(comparison[c+'-E60']['gates'].values()) for c in ('A','B'))
    # A missing E60 after a failed own E30 is an ineligible successor, still listed as absent.
    reason = ('passed' if selected else 'original_qualification_failed' if failed else
              'awaiting_matched_qualifications_or_gait_records' if missing else 'gait_or_cost_gate_failed')
    decision = dict(schema='s1_w58_gait_selection_v1',selected='G1' if selected else 'G0',
        reason=reason, comparison_complete=enough, missing=missing, failed=failed,
        rejected_other_identities=rejected, case_levels=comparison)
    write(output,decision)
    return decision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='cmd',required=True)
    single=commands.add_parser('episode')
    for name in ('raw','result','output'):
        single.add_argument('--'+name,type=Path,required=True)
    choice=commands.add_parser('select')
    for name in ('g0','g1','output'):
        choice.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.cmd=='episode':
        scored=score_episode(args.raw,args.result,args.output)
        print(json.dumps(dict(attempt=scored['attempt'],gait=scored['gait'],
            physical_rows=scored['physics_rows'],swings=scored['groups']['all']['landings']['committed_leg_swings'])))
    else:
        choice=select_gait(args.g0,args.g1,args.output)
        print(choice['selected'],choice['reason'])


if __name__=='__main__':
    main()
