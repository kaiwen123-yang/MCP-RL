"""W58 common support availability and causal handover planning."""
import json
import time

import numpy as np

from s1_w54_f_r52_backend import ControllerNumericalFailure, assemble_force_horizon
from s1_w57_planning import Forecast, footprint, original_cost
from tm02.s1_contact import StanceReferenceInfeasible

ARMS = ('B2', 'P2-full', 'P2-private', 'Fixed-wait')


def available_fraction(ground, times, grid, active_soft, contact, node_xy):
    """P1's geometric contact-availability fraction without material-state feedback."""
    duration = np.zeros(contact.shape)
    segment = np.searchsorted(times[1:], grid[:-1] + 1e-10)
    np.add.at(duration, segment, active_soft * np.diff(grid)[:, None])
    interior = np.array([[ground.interior(xy) for xy in row] for row in node_xy])
    return np.where(contact & (interior | (duration > 0)),
                    np.minimum(1., duration / np.diff(times)[:, None]), 1.)


def availability_matrix(inputs, fraction):
    return inputs['stage_B'] * np.repeat(fraction, 3, axis=1)[:, None, :]


def factor_matrices(inputs, fraction, full_B):
    """The four dynamics operators share the original force feasibility set."""
    base = inputs['stage_B']
    factor = np.repeat(fraction, 3, axis=1)[:, None, :]
    return dict(nominal=base, availability_only=availability_matrix(inputs, fraction),
                height_history_only=np.divide(full_B, factor, out=base.copy(), where=factor > 0),
                full=full_B)


def split_grid(controller, t, times, mq, mission):
    schedule = controller._forecast_events(t, float(times[-1]), mq, mission)
    cuts = {float(v) for leg in schedule for a, b, _ in leg for v in (a, b)
            if times[0] < v < times[-1] and np.min(abs(times-v)) > 1e-8}
    return np.sort(np.r_[times, list(cuts)])


def apply_availability(controller, t, mission, mq, times, foot_plan, contact, inputs, stages, states, forces, status):
    g = controller.current_material_info['material_observer'].ground
    if not any(g.interior(x) for x in foot_plan[:, :, :2].reshape(-1, 2)):
        return stages, states, forces, status
    grid = split_grid(controller, t, times, mq, mission)
    on, feet = controller._future_contact_feet(t, .5*(grid[:-1]+grid[1:]), mq, mission)
    soft = np.array([[g.interior(x[:2]) for x in row] for row in feet])
    fraction = available_fraction(g, times, grid, soft & on, inputs['contact'], foot_plan[:-1, :, :2])
    if np.all(fraction == 1.):
        return stages, states, forces, status
    updated = dict(inputs, stage_B=availability_matrix(inputs, fraction))
    qp = assemble_force_horizon(updated, controller.cfg)
    z, solve_status = controller._solve(qp['P'], qp['q'], qp['A'], qp['lower'], qp['upper'], 'w58-availability', t)
    nx = inputs['reference'].size
    f = z[nx:].reshape(forces.shape)
    reference = original_cost(updated, controller.cfg, forces)[0]
    cost = original_cost(updated, controller.cfg, f)[0]
    install = bool(cost < reference - 1e-12 * max(1., abs(cost), abs(reference)))
    with (controller.record_dir / 'w58-availability.jsonl').open('a') as out:
        out.write(json.dumps(dict(call=controller.plan_count+1, time_s=float(t),
                                  fractions=fraction.tolist(), original_score=reference,
                                  corrected_score=cost, installed=install,
                                  first_force_change_N=float(abs(f[0]-forces[0]).max())))+'\n')
    if not install:
        return stages, states, forces, status
    inputs['stage_B'] = updated['stage_B']
    stages = [dict(stage, B=inputs['stage_B'][k]) for k, stage in enumerate(stages)]
    return stages, z[:nx].reshape(states.shape), f, solve_status


def support_forecast(controller, t, mission, mq, obs, grf):
    """Predict the two committed incoming feet at the solved handover load.

    Waiting changes the acquisition window. Before that handover the three choices
    share touchdown geometry, a force scenario and the same material event grid.
    """
    cfg = controller.cfg['contact_events']
    support = [i for i in range(4) if i not in controller.next_pair]
    if not all(controller.event_mode[i]=='swing' and controller.swing[i] is not None for i in support):
        return None, [], None
    targets = []
    for leg in support:
        swing = controller.swing[leg]
        xy = swing['landing'][:2]
        targets.append(dict(leg=leg, target_event_id=controller.pending_touch[leg], xy=xy.tolist(),
                            touch_s=float(swing['end']), planned_bottom_m=float(swing['landing'][2]-controller.radius[leg]),
                            cells=[dict(key=list(map(int, key)), area_m2=float(area), **obs.history(key))
                                   for key, area in footprint(obs.ground, xy).items()]))
    if not any(row['cells'] for row in targets):
        return None, targets, None
    weight = controller.mass * 9.81
    target_force = weight * max(cfg['load_on_bodyweight_fraction'], cfg['handover_solved_bodyweight_fraction']/len(support))
    touch_check = np.array([r['touch_s'] + controller.cfg['wbc']['dt_s'] for r in targets])
    check = touch_check + cfg['load_confirm_s']
    end = max(check)
    boundaries = [t, *touch_check, *check, *(r['touch_s'] for r in targets)]
    previous = controller.last_plan
    if previous is not None:
        boundaries.extend(float(ts) for ts in previous['times'] if t < ts < end)
    grid = split_grid(controller, t, t + np.unique(np.round(np.array(boundaries)-t, 10)), mq, mission)
    mid = .5*(grid[:-1]+grid[1:])
    contact, feet = controller._future_contact_feet(t, mid, mq, mission)
    loads = np.zeros((len(mid), 4, 3))
    for j, when in enumerate(mid):
        available = list(np.flatnonzero(contact[j]))
        incoming = [i for i in support if i in available]
        carrying = [i for i in available if i not in support]
        for i in incoming:
            idx = support.index(i)
            loads[j, i, 2] = cfg['touch_on_N'] if when < touch_check[idx] else target_force
        if carrying:
            old = np.maximum(grf[carrying, 2], 0.)
            if previous is not None and previous['times'][0] <= when < previous['times'][-1]:
                k = np.searchsorted(previous['times'][1:], when)
                old = np.maximum(previous['forces'][k, carrying, 2], 0.)
            share = old/old.sum() if old.any() else np.full(len(carrying), 1./len(carrying))
            loads[j, carrying, 2] = (weight - loads[j, incoming, 2].sum())*share
    # Preserve complete transitive footprints. Spatially disjoint columns never
    # enter the same sole inversion or ownership decision.
    ground = obs.ground
    union = {tuple(cell['key']) for row in targets for cell in row['cells']}
    maps = [[set(footprint(ground, p[:2])) if use else set() for use, p in zip(on, row)]
            for on, row in zip(contact, feet)]
    while True:
        extended = union.union(*(m for row in maps for m in row if union & m))
        if extended == union:
            break
        union = extended
    related = np.array([[bool(m & union) for m in row] for row in maps])
    forecast = Forecast(ground, grid, grid, feet[:, :, :2], contact & related,
                        obs.published_bottom, controller.measured)
    forecast.backend = 'equal_columns'
    forecast.workers = int(controller.cfg['w57']['planner_material_workers'])
    for k, key in enumerate(forecast.keys):
        cell = obs.history(key)
        forecast.reset[k] &= bool(not cell['loaded'] and cell['release_event_id'] is not None)
    removed = {key for key, reset in zip(forecast.keys, forecast.reset) if reset}
    for row, a, b in zip(targets, touch_check, check):
        row['touch_row'] = int(np.argmin(abs(grid[1:]-a)))
        row['bearing_row'] = int(np.argmin(abs(grid[1:]-b)))
        row['touch_load_N'] = float(cfg['touch_on_N'])
        row['required_load_N'] = float(target_force)
        for cell in row['cells']:
            cell['private_reset'] = tuple(cell['key']) in removed
    return forecast, targets, loads

def window_scores(targets, z_touch, z_load, nominal, offsets, cfg):
    """Score lost progress and insufficient time for support acquisition, in squared seconds."""
    confirm, bearing = [], []
    speed = cfg['search_speed_m_s']
    for row, a, b in zip(targets, z_touch, z_load):
        surface = row['planned_bottom_m']
        confirm.append(row['touch_s'] + max(0., surface-a)/speed + cfg['touch_confirm_s'])
        bearing.append(max(confirm[-1], row['touch_s']+max(0., surface-b)/speed)
                       + cfg['load_confirm_s']-1./cfg['sample_Hz'])
    # Start the ramp on confirmation of all feet. Bearing is required at handover,
    # and can develop during the ramp as it does in the unchanged controller gates.
    eligible = max(max(bearing), max(confirm) + cfg['load_transfer_ramp_s'])
    rows = []
    for wait in offsets:
        short = max(0., eligible-nominal-wait)
        rows.append(dict(wait_s=wait, transfer_shortfall_s=short, progress_cost_s2=wait**2,
                         support_cost_s2=short**2, score_s2=wait**2+short**2))
    return dict(touch_bottom_m=list(map(float, z_touch)), loaded_bottom_m=list(map(float, z_load)),
                predicted_ready_s=eligible, candidates=rows,
                winner=int(np.argmin([r['score_s2'] for r in rows])))


def decide(controller, t, mission, mq, obs, grf, arm, substep=.0002):
    if arm not in ARMS:
        raise ValueError(f'Unknown W58 planning arm {arm}')
    cfg = controller.cfg['contact_events']
    offsets = [0., float(controller.cfg['wbc']['dt_s']), float(cfg['load_confirm_s'])]
    nominal = float(controller.next_liftoff)
    fc, targets, loads = support_forecast(controller, t, mission, mq, obs, grf)
    row = dict(decision_id=controller.request_count+1, time_s=float(t), arm=arm,
               nominal_due_s=nominal, offsets_s=offsets, targets=targets, private_substep_s=substep,
               search_speed_m_s=cfg['search_speed_m_s'], solved_pair_fraction=cfg['handover_solved_bodyweight_fraction'],
               planned_pair=controller.next_pair.tolist())
    if fc is None:
        row.update(reason='no_committed_material_receiver', selected=0, selected_wait_s=0.)
        return row
    if abs(obs.time-t) > 1e-9:
        raise ValueError(f'P2 consumed publication {obs.time} at {t}')
    row.update(reason='committed_bearing_and_progress', world_cells=len(fc.keys),
               released_reset_cells=int(fc.reset.sum()), protected_cells=int(fc.protected.sum()),
               material_workers=fc.workers, published_s=float(obs.time))
    for history in ('full', 'private_reset'):
        if history == 'private_reset' and not np.any(fc.reset):
            row[history] = row['full']
            continue
        try:
            result = fc.propagate(loads, substep, history)
        except StanceReferenceInfeasible as error:
            row[history] = dict(winner=None, physical_rejection=error.detail)
            continue
        heights = [[float(result['z'][r[key], r['leg']]) if r['cells'] else 0. for r in targets]
                   for key in ('touch_row', 'bearing_row')]
        row[history] = window_scores(targets, *heights, nominal, offsets, cfg)
    if arm in ('B2', 'Fixed-wait'):
        chosen = 0 if arm=='B2' else 2
    else:
        chosen = row['full' if arm=='P2-full' else 'private_reset']['winner']
    row.update(selected=chosen, selected_wait_s=None if chosen is None else offsets[chosen])
    return row


class P2ControlMixin:
    """Combine with the frozen G backend and W57 handover/WBC instrumentation."""
    def __init__(self, model, data, feet, base, out, config):
        super().__init__(model, data, feet, base, out, config)
        self.p2_arm = config['w58']['arm']
        self.p2_active = None
        self.manifest.update(w58_planner=self.p2_arm, w58_force_horizon='public contact availability',
                             w58_decision_records='w58-handover.jsonl')
        self._write_manifest()

    def select_force_plan(self, *args):
        return apply_availability(self, *args)

    def _new_handover_request(self, t, mission, mq, material_info, grf):
        if self.timing_agent is not None and not self.timing_agent.has_next_interval(t):
            return super()._new_handover_request(t, mission, mq, material_info, grf)
        start, cpu = time.perf_counter(), time.process_time()
        row = decide(self, t, mission, mq, material_info['material_observer'], grf, self.p2_arm)
        row.update(planning_wall_s=time.perf_counter()-start, planning_cpu_s=time.process_time()-cpu)
        with (self.record_dir/'w58-handover.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if row['selected'] is None:
            raise ControllerNumericalFailure(f'W58 bearing forecast outside domain at {t} for {self.p2_arm}')
        self.next_liftoff = row['nominal_due_s'] + row['selected_wait_s']
        self.p2_active = dict(p2_decision_id=row['decision_id'], original_due_s=row['nominal_due_s'],
                              p2_wait_s=row['selected_wait_s'], p2_due_s=self.next_liftoff)
        super()._new_handover_request(t, mission, mq, material_info, grf)

    def _write_request(self, record):
        if self.p2_active is not None and record.get('decision_id')==self.p2_active['p2_decision_id']:
            record = dict(record, **self.p2_active)
        return super()._write_request(record)
