"""Force-driven world-column predictions from one published observational state."""
from itertools import permutations

import numpy as np
from scipy.optimize import brentq

from .s1_contact import StanceReferenceInfeasible


def owned_columns(ground, feet, order):
    """First (lowest) sole owns each world cell. No force is counted twice."""
    seen = set()
    result = [None] * len(feet)
    for leg in order:
        ids, areas = ground.footprint(feet[leg, :2])
        selected = [(i, a) for i, a in zip(ids, areas) if i not in seen]
        result[leg] = (np.array([i for i, _ in selected], int),
                       np.array([a for _, a in selected]))
        seen.update(ids)
    return result


def inverse(ground, columns, load, dt):
    """Positive-down sole displacement and native force residual in N."""
    ids, area = columns
    if load < 0 or not np.isfinite(load):
        raise StanceReferenceInfeasible(dict(reason='invalid_normal_request', requested_force_N=float(load)))
    if not len(ids):
        if load > 0:
            raise StanceReferenceInfeasible(dict(reason='positive_load_without_owned_area', requested_force_N=float(load)))
        return 0., 0.
    e, q = ground.e[ids], ground.q[ids]
    zero = np.zeros(len(ids))
    fixed = np.ones(len(ids), bool)
    free = ground.law.step(e, q, zero, dt, ~fixed)[0][:, 0]
    # At zero force choose first contact with the tallest freely recovered column.
    lower = float((-ground.h * free).min())
    if load == 0:
        return lower, 0.

    def residual(delta):
        boundary = -delta / ground.h
        _, _, stress, _ = ground.law.step(e, q, np.full(len(ids), boundary), dt, fixed)
        pressure = np.where(boundary <= free, np.maximum(-stress[:, 0], 0), 0.)
        if hasattr(ground,'w59_reload_contact'):
            index,gain=ground.w59_reload_contact
            pressure=np.where(ids==index,gain*pressure,pressure)
        return float(np.dot(area, pressure) * 1e6 * ground.gamma - load)

    upper = ground.h * .02
    if residual(upper) < 0:
        raise StanceReferenceInfeasible(dict(reason='requested_load_exceeds_material_domain',
                                             requested_force_N=float(load), dt_s=dt))
    delta = brentq(residual, lower, upper, xtol=1e-13, rtol=1e-13)
    ee = ground.law.step(e, q, np.full(len(ids), -delta / ground.h), dt, fixed)[0]
    if np.any(np.abs(ee[-delta / ground.h <= free]) > .02):
        raise StanceReferenceInfeasible(dict(reason='principal_strain_outside_material_domain', requested_force_N=float(load)))
    return delta, residual(delta)


def normal_reference(ground, feet, leg, target, measured, dt):
    # Isolate geometry creation as well as the two constitutive evaluations.
    branch = ground.clone()
    order = np.argsort(feet[:, 2], kind='stable')
    columns = owned_columns(branch, feet, order)[leg]
    d1, r1 = inverse(branch, columns, target, dt)
    d0, r0 = inverse(branch, columns, measured, dt)
    details=({'published_reload_column':int(ground.w59_reload_contact[0]),
               'published_positive_contact_gain':float(ground.w59_reload_contact[1])}
             if hasattr(ground,'w59_reload_contact') else {})
    return d0 - d1, dict(**details,measured_N=measured, target_N=target, measured_delta_m=d0,
                         target_delta_m=d1, measured_residual_N=r0, target_residual_N=r1,
                         owned_area_m2=float(columns[1].sum()))


def candidate_step(branch, xy, contacts, force, dt, end):
    # Inactive feet have no displacement boundary. Rigid loads are external to
    # this material-only module, and are explicitly excluded from its load sum.
    feet = np.column_stack((xy, np.full(4, 1.)))
    legs = [i for i in range(4) if contacts[i] and branch.interior(xy[i])]
    inactive = sorted(set(range(4)) - set(legs))
    feet[inactive, :2] = -10.
    footprints = [set(branch.footprint(xy[i])[0]) for i in legs]
    overlap = any(a & b for j, a in enumerate(footprints) for b in footprints[:j])
    last_error = None
    for priority in permutations(legs) if overlap else [legs]:
        maps = owned_columns(branch, feet, [*priority, *inactive])
        try:
            for i in legs:
                feet[i, 2] = -inverse(branch, maps[i], float(force[i]), dt)[0]
        except StanceReferenceInfeasible as error:
            last_error = error
            continue
        owners = owned_columns(branch, feet, np.argsort(feet[:, 2], kind='stable'))
        if all(np.array_equal(owners[i][0], maps[i][0]) for i in legs):
            break
    else:
        raise StanceReferenceInfeasible(dict(reason='no_consistent_sole_ownership',
                                             detail=str(last_error), time_s=end))
    # Tangent at the consistent solution, with ownership and history frozen.
    # Keep the finite difference in the original displacement domain.
    tangents = []
    free_boundary = []
    for i in legs:
        ids, aa = owners[i]
        if not len(ids):
            tangents.append(None)
            free_boundary.append(None)
            continue
        e, q = branch.e[ids], branch.q[ids]
        fixed = np.ones(len(ids),bool)
        free = branch.law.step(e,q,np.zeros(len(ids)),dt,~fixed)[0][:,0]
        free_boundary.append(float(np.dot(-branch.h*free,aa)/aa.sum()))
        delta = -feet[i,2]
        step = 1e-9 if delta+1e-9 <= branch.h*.02 else -1e-9
        def normal_at(d):
            b = -d/branch.h
            stress = branch.law.step(e,q,np.full(len(ids),b),dt,fixed)[2][:,0]
            return float(np.dot(aa,np.where(b<=free,np.maximum(-stress,0.),0.))*1e6*branch.gamma)
        tangents.append((normal_at(delta+step)-normal_at(delta))/step)
    actual, _, _ = branch.step(feet, np.zeros((4, 3)), dt, end, log=False)
    if branch.domain_exit:
        raise StanceReferenceInfeasible(branch.domain_exit | dict(reason='candidate_strain_domain_exit'))
    residual = actual[legs, 2] - force[legs]
    if np.max(np.abs(residual), initial=0.) > 1.e-5:
        raise StanceReferenceInfeasible(dict(reason='committed_force_disagrees_with_requested_load',
                                             residual_N=residual.tolist(), time_s=end))
    return dict(time_s=end, soft_legs=legs, displacement_m=(-feet[legs, 2]).tolist(),
                force_residual_N=residual.tolist(), max_abs_strain=branch.max_strain,
                area_m2=[float(owners[i][1].sum()) for i in legs],
                normal_tangent_N_m=tangents, free_displacement_m=free_boundary)


def rollout_candidate(ground, observed_time, times, foot_xy, contact, normal_force):
    """One independent branch per candidate. All future footprint cells exist at t0."""
    if ground.prediction_steps is not None:
        raise ValueError('A candidate must start from a published observation')
    times = np.asarray(times)
    if abs(observed_time-times[0]) > 1e-9 or np.any(np.diff(times) <= 0):
        raise ValueError('Candidate clock must continue the observed clock')
    branch = ground.clone()
    for feet in foot_xy:
        for xy in feet:
            branch.footprint(xy)
    active = np.array(sorted(branch.prediction_active),int)
    keys = [tuple(np.floor(np.array(branch.xy[i])/branch.cell).astype(int)) for i in active]
    nodes = []
    for dt, end, xy, mode, force in zip(np.diff(times),times[1:],foot_xy,contact,normal_force,strict=True):
        node = candidate_step(branch, xy, mode, force, float(dt), float(end))
        node.update(cell_keys=[list(map(int,k)) for k in keys],cell_previously_observed=[k in ground.ids for k in keys],
                    e=branch.e[active].tolist(),q=branch.q[active].tolist(),
                    loaded=branch.loaded[active].tolist())
        nodes.append(node)
    return branch, nodes
