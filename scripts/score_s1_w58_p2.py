"""Bind each P2 decision to receiver load IDs, observed support and the executed gate."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import json
from pathlib import Path

import numpy as np

from analyze_s1_w57_planner_outcomes import CrossFootLoads, LEGS, records
from qualify_s1_w57_event_planning import first_permission
from s1_w58_p2 import window_scores

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT/'experiments/terrain_memory/s1-w58-p2.json'
CFG = json.loads((ROOT/'experiments/terrain_memory/s1-w56-f-r60.json').read_text())
DT = CFG['wbc']['dt_s']


def stats(values):
    if not values:
        return dict(n=0, p50=None, p95=None, max=None)
    return dict(n=len(values), **dict(zip(('p50','p95','max'), np.percentile(values,[50,95,100]).tolist())))


class Observed:
    def __init__(self, raw):
        self.raw = raw
        path = raw/'w55-f'
        self.commit = {}
        self.touch = {}
        for event in records(path/'contact-events.jsonl'):
            if event['kind']=='liftoff_committed':
                for name, key in zip(event['legs'],event['target_event_ids']):
                    self.commit[key] = (LEGS.index(name), event['time_s'], event['fixed_apex_s'])
            if event['kind']=='actual_touchdown' and 'target_event_id' in event:
                self.touch[event['target_event_id']] = event['time_s']
        self.executed = {q['decision_id']:q for q in records(path/'handover-requests.jsonl')
                         if q['stage']=='executed'}
        self.history = CrossFootLoads(raw)
        self.shadows = defaultdict(list)
        for row in records(path/'w58-transfer-actions.jsonl'):
            self.shadows[row['decision_id']].append(row)
        if self.shadows:
            import mujoco
            model = mujoco.MjModel.from_binary_path(str(raw/'model.mjb'))
            self.force_range = model.actuator_forcerange.copy()
            with np.load(raw/'trajectory.npz') as trace:
                self.applied_target = trace['q_target']
                self.command_clock = trace['time']
            compact = ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w58/episodes'/(raw.name+'.json')
            cols = json.loads(compact.read_text())['physical_columns']
            self.pd = [cols.index(f'unclipped_pd_request_{i}') for i in range(12)]
            self.delivered = [cols.index(f'actuator_force_{i}') for i in range(12)]

    @lru_cache(maxsize=3)
    def control(self, chunk):
        keys = ('time', 'observed_contact', 'confirmed_contact', 'load_bearing_contact','event_mode',
                'measured_grf','force','body_acceleration_residual','angular_acceleration_residual','transfer_fraction')
        with np.load(self.raw/'w55-f'/f'control-{chunk+1:05d}.npz') as z:
            return {n:z[n] for n in keys}

    @lru_cache(maxsize=2)
    def material(self, chunk):
        with np.load(self.raw/f'material-{chunk:04d}.npz') as z:
            return z['foot']

    def crossing(self, leg, start, end, threshold):
        # World force-boundary p and delivered F at that same physical step.
        i, j = round(start/.0002), round(end/.0002)
        pieces = [self.material(k)[max(0,i-5000*k):min(5000,j-5000*k),[1,6+3*leg+2,32+3*leg]]
                  for k in range(i//5000,(j-1)//5000+1)]
        if not pieces:
            return None
        tr = np.vstack(pieces)
        hits = np.flatnonzero((tr[:-1,2]<threshold) & (tr[1:,2]>=threshold))
        if not len(hits):
            return None
        k = hits[0]
        a,b = tr[k:k+2]
        p = a + (threshold-a[2])/(b[2]-a[2])*(b-a)
        return dict(time_s=float(p[0]), bottom_m=float(p[1]), load_N=float(threshold))

    def query(self, row):
        q = self.executed.get(row['decision_id'])
        current = []
        for target in row['targets']:
            key = target['target_event_id']
            commit, confirmation = self.commit.get(key), self.touch.get(key)
            entry = dict(target_event_id=key, leg=target['leg'], confirmation_s=confirmation,
                         true_matching_reloads=[], touch_threshold=None, bearing_threshold=None)
            if commit and confirmation is not None:
                detached = {tuple(c['key']) for c in target['cells'] if c['private_reset']}
                entry['true_matching_reloads'] = self.history.matches(target,row['time_s'],commit,confirmation,detached)
                if target['cells'] and q:
                    for out,threshold in (('touch_threshold','touch_load_N'),('bearing_threshold','required_load_N')):
                        entry[out] = self.crossing(target['leg'],commit[2],q['actual_liftoff_s'],target[threshold])
            current.append(entry)
        observed_ready = None
        if q and all(item['confirmation_s'] is not None for item in current):
            touched = max(item['confirmation_s'] for item in current)
            legs = [item['leg'] for item in current]
            for k in range(round(touched/DT),round(q['actual_liftoff_s']/DT)+1):
                # At outgoing liftoff the named incoming support still has its pre-liftoff flags.
                if self.control(k//500)['load_bearing_contact'][k%500,legs].all():
                    observed_ready = max(touched+CFG['contact_events']['load_transfer_ramp_s'],k*DT)
                    break
        return current, observed_ready, q

    @lru_cache(maxsize=1)
    def actuator_trace(self, chunk):
        with np.load(self.raw/f'physical-{chunk:03d}.npz') as z:
            x = z['physical']
            return x[:,self.pd],x[:,self.delivered]

    def transfer_evidence(self, decision):
        samples = []
        previous = set()
        previous_time = None
        repeated = set()
        for tick in self.shadows.get(decision,[]):
            if any(tick[h]['mapped_rad'] is None for h in ('full','private_reset')):
                previous = set()
                continue
            t = tick['time_s']
            block,step = divmod(round(t/.0002),150000)
            pd,delivered = self.actuator_trace(block)
            pd,delivered = pd[step:step+10],delivered[step:step+10]
            if len(pd)!=10:
                continue
            index = round(t/DT)
            if abs(self.command_clock[index]-t)>1.e-8:
                raise ValueError(f'Physical/WBC decision clock differs at {t}')
            real = self.applied_target[index]
            readback = float(abs(real-np.array(tick['actual_mapped_rad'])).max())
            lo,hi = self.force_range.T
            alt = [np.clip(pd+60.*(np.array(tick[h]['mapped_rad'])-real),lo,hi)
                   for h in ('full','private_reset')]
            own = alt[0 if tick['active_arm']=='P2-full' else 1]
            reconstruction = abs(own-delivered).max(axis=0)
            # A physically resolved joint change exceeds both measured hold ripple
            # and same-state reconstruction error, on two consecutive WBC holds.
            spread = np.ptp(delivered,axis=0)
            effect = abs(alt[0]-alt[1]).mean(axis=0)
            affected = set(np.flatnonzero(effect > spread+reconstruction))
            if previous_time is not None and abs(t-previous_time-DT)<1e-8:
                repeated.update(affected & previous)
            previous,previous_time = affected,t
            samples.append(dict(time_s=t,peak_mean_alternate_torque_difference_Nm=float(effect.max()),
                                reconstruction_error_Nm=float(reconstruction.max()),applied_command_readback_error_rad=readback,above_hold_ripple_joints=sorted(map(int,affected))))
        return dict(transfer_command_changes=bool(repeated),repeated_joints=sorted(map(int,repeated)),samples=samples)

    def gate_evidence(self,row, q):
        if q is None or any(row[h]['winner'] is None for h in ('full','private_reset')):
            return dict(reason='unexecuted_or_infeasible',release_changes=False)
        if row['full']['winner']==row['private_reset']['winner']:
            return dict(reason='same_requested_wait',release_changes=False)
        residual = q['requested_due_s']-q['p2_due_s']
        stop = q['actual_liftoff_s']
        src = np.asarray(row['planned_pair'],int)
        # The real solved-load threshold in N is defined by the registered case mass.
        mass = row['targets'][0]['required_load_N']/(row['solved_pair_fraction']/2)/9.81
        results = {}
        for history in ('full','private_reset'):
            due = row['nominal_due_s']+row['offsets_s'][row[history]['winner']]+residual
            gate, ramp = first_permission(self.control, range(round(row['time_s']/DT)+1,round(stop/DT)+1),
                                          {'time_s':stop},src,due,mass,CFG)
            results[history] = dict(due_s=due, first_eligible_s=gate, ramp_differs_from_real=bool(ramp))
        own = {'P2-full':'full','P2-private':'private_reset'}.get(row['arm'])
        actual_reconstructed = own is not None and results[own]['first_eligible_s'] is not None and abs(
            results[own]['first_eligible_s']-stop)<DT/2
        witness = results['full']['first_eligible_s']!=results['private_reset']['first_eligible_s']
        return dict(release_changes=bool(witness and actual_reconstructed),actual_reconstructed=bool(actual_reconstructed),
                    actual_liftoff_s=stop,R_residual_held_s=residual,**self.transfer_evidence(row['decision_id']),**results)


def margins(row, uncertainty_s, common_shift=0.):
    waits = np.asarray(row['offsets_s'])
    def winner(ready):
        scores = waits**2 + np.maximum(0.,ready-row['nominal_due_s']-waits)**2
        return int(np.argmin(scores))
    return {h: all(winner(row[h]['predicted_ready_s']+common_shift+eps)==row[h]['winner']
                   for eps in (-uncertainty_s, uncertainty_s)) for h in ('full','private_reset')}


def target_attribution(row, matched_legs):
    """Would changing only the predeclared receiver's loaded depth suffice or be necessary?"""
    targets = row['targets']
    winners = {}
    for keep, change in (('full','private_reset'),('private_reset','full')):
        zs = [[(row[change] if t['leg'] in matched_legs else row[keep])[field][i]
               for i,t in enumerate(targets)] for field in ('touch_bottom_m','loaded_bottom_m')]
        winners[keep] = window_scores(targets,*zs,row['nominal_due_s'],row['offsets_s'],CFG['contact_events'])['winner']
    return any(winners[h]!=row[h]['winner'] for h in winners)


def calibration(raw, replay):
    observed = Observed(raw)
    ready_errors, numerical, effects = [], [], []
    touched_error, loaded_error = [], []
    counts = Counter()
    examples = []
    with gzip.open(replay,'rt') as stream:
        for line in stream:
            row = json.loads(line)
            counts['all_requests']+=1
            if 'full' not in row:
                continue
            counts['soft_requests']+=1
            if any(row[h]['winner'] is None or row['refined'][h]['winner'] is None for h in ('full','private_reset')):
                counts['material_rejections']+=1
                continue
            counts['nonempty_reset']+=row['released_reset_cells']>0
            counts['winner_switches']+=row['full']['winner']!=row['private_reset']['winner']
            current,ready,q = observed.query(row)
            if ready is not None:
                ready_errors.append(abs(ready-row['full']['predicted_ready_s']))
            for i,t in enumerate(current):
                for error, field, height in ((touched_error,'touch_threshold','touch_bottom_m'),
                                              (loaded_error,'bearing_threshold','loaded_bottom_m')):
                    if t[field]:
                        error.append(abs(row['full'][height][i]-t[field]['bottom_m']))
            numerical.extend(abs(row[h]['predicted_ready_s']-row['refined'][h]['predicted_ready_s'])
                             for h in ('full','private_reset'))
            effects.append(abs(row['full']['predicted_ready_s']-row['private_reset']['predicted_ready_s']))
            binds=sum(len(t['true_matching_reloads']) for t in current)
            counts['bound_requests']+=binds>0
            counts['bound_cell_events']+=binds
            examples.append(dict(decision_id=row['decision_id'],time_s=row['time_s'],
                actual_earliest_support_s=ready,full_planned_ready_s=row['full']['predicted_ready_s'],
                blind_planned_ready_s=row['private_reset']['predicted_ready_s'],true_bound_loads=binds,
                full_wait_s=row['selected_wait_s'],private_wait_s=row['offsets_s'][row['private_reset']['winner']]))
    counts.update(winner_switches=0,material_rejections=0)
    return dict(source=raw.name, role='saved development episode, real W57 P1/A clock, offline P2 predictions',
                counts=dict(counts),samples=examples,abs_full_ready_error_s=stats(ready_errors),
                abs_full_touch_bottom_error_m=stats(touched_error),abs_full_bearing_bottom_error_m=stats(loaded_error),
                abs_full_private_ready_difference_s=stats(effects),abs_step_halving_difference_s=stats(numerical))


def score_episode(raw, target_events):
    raw = Path(raw)
    defined = {r['target_id']:dict(target_id=r['target_id'], realized=r['realized'],
                                    binding_decisions=[],history_action_evidence=False) for r in target_events}
    ids = defaultdict(set)
    for r in target_events:
        for match in r['matches']:
            if r['realized'] and match['condition_met'] and match['exact_overlap_m2']>0:
                ids[match['source_id'],match['release_id'],match['target_id']].add(r['target_id'])
    path = raw/'w55-f/w58-handover.jsonl'
    if not path.is_file():
        return dict(targets=list(defined.values()),history_action_evidence=False,decision_records='absent')
    observed = Observed(raw)
    counts = Counter()
    max_effect = 0.
    calibration = json.loads(SPEC.read_text())['calibration']
    floor = (calibration['archived_p95_threshold_depth_error_m']/CFG['contact_events']['search_speed_m_s']
             + calibration['maximum_step_halving_error_s']+.0002)
    requested = []
    for row in records(path):
        counts['requests']+=1
        counts['selected_delay_requests']+=int(bool(row.get('selected_wait_s')))
        if 'full' not in row:
            continue
        counts['material_requests']+=1
        counts['nonempty_reset']+=row['released_reset_cells']>0
        if any(row[h]['winner'] is None for h in ('full','private_reset')):
            counts['unscorable_comparisons']+=1
            continue
        a,b = row['full'],row['private_reset']
        switched = a['winner']!=b['winner']
        counts['winner_switches']+=switched
        max_effect = max(max_effect, abs(a['predicted_ready_s']-b['predicted_ready_s']))
        current, ready, executed = observed.query(row)
        gate = observed.gate_evidence(row,executed)
        matched = defaultdict(list)
        for item in current:
            for pair in item['true_matching_reloads']:
                triplet = (pair['true_source_load_id'], pair['true_release_id'], pair['true_target_load_id'])
                for target_id in ids.get(triplet,()):
                    matched[target_id].append(pair)
        counts['observed_gate_switches']+=bool(gate['release_changes'])
        for key,pairs in matched.items():
            legs = {p['target_leg'] for p in pairs}
            errors = [abs(item[label]['bottom_m']-a[quantity][i]) if item[label] else None
                      for i,item in enumerate(current) if item['leg'] in legs
                      for label,quantity in (('touch_threshold','touch_bottom_m'),('bearing_threshold','loaded_bottom_m'))]
            uncertainty = max(floor,max(errors)/row['search_speed_m_s']) if errors and None not in errors else None
            # Whole-gait error can shift both candidates together. Re-evaluate both
            # rankings under the actual shared readiness and the local material error.
            common_shift = None if ready is None else ready-a['predicted_ready_s']
            robust = (uncertainty is not None and common_shift is not None
                      and abs(a['predicted_ready_s']-b['predicted_ready_s'])>uncertainty
                      and all(margins(row,uncertainty).values())
                      and all(margins(row,uncertainty,common_shift).values()))
            attribution = switched and target_attribution(row,legs)
            supported = bool(attribution and robust and (gate['release_changes'] or gate.get('transfer_command_changes')))
            defined[key]['binding_decisions'].append(dict(decision_id=row['decision_id'],time_s=row['time_s'],
                true_ids=[[p['true_source_load_id'],p['true_release_id'],p['true_target_load_id']] for p in pairs],
                target_loads=current,actual_earliest_support_s=ready,applied_arm=row['arm'],
                full_ready_s=a['predicted_ready_s'],private_ready_s=b['predicted_ready_s'],
                wait_full_s=row['offsets_s'][a['winner']],wait_private_s=row['offsets_s'][b['winner']],
                common_ready_shift_s=common_shift,uncertainty_s=uncertainty,robust_winners=bool(robust),attributable_to_target=bool(attribution),
                gate=gate,history_action_evidence=supported))
            defined[key]['history_action_evidence']|=supported
        if switched:
            requested.append(dict(decision_id=row['decision_id'],full_wait_s=row['offsets_s'][a['winner']],
                                  private_wait_s=row['offsets_s'][b['winner']],targets=list(matched),gate=gate))
    return dict(attempt=raw.name,counts=dict(counts),targets=list(defined.values()),switched_requests=requested,
                maximum_same_state_ready_difference_s=max_effect,predeclared_uncertainty_floor_s=floor,
                history_action_evidence=any(r['history_action_evidence'] for r in defined.values()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',type=Path)
    parser.add_argument('--calibrate',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=calibration(args.raw,args.calibrate) if args.calibrate else score_episode(
        args.raw,list(records(args.raw/'w58-target-events.jsonl')))
    temp=args.out.with_suffix('.pending')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temp.replace(args.out)
