"""Actual contact-point slip on the stationary flat W59 plant, from its 5 kHz records."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from publish_s1_go2_w54 import write

LEGS = ('FL', 'FR', 'RL', 'RR')
DT = .0002


def contact_velocity_xy(material_foot, center_velocity, radius):
    """World horizontal velocity at the TM02 applied point or native sphere/plane contact."""
    soft = material_foot[:, 2:6].astype(bool)
    bottom_z = material_foot[:, 6:18].reshape(-1, 4, 3)[:, :, 2]
    v_bottom = material_foot[:, 18:30].reshape(-1, 4, 3)
    # MuJoCo places the sphere/plane contact midway between the two surfaces.
    # TM02 instead applies the resultant at the sphere bottom; the grid has no lateral velocity.
    fraction = np.where(soft, 1., 1. + bottom_z / (2. * radius))
    center_xy = center_velocity[:, :, :2]
    return center_xy + fraction[:, :, None] * (v_bottom[:, :, :2] - center_xy)


def audit_snapshot(model, data, feet, floor_id, qpos, qvel, material, reconstructed):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    position_errors, speed_errors, normal_errors = [], [], []
    for contact in data.contact:
        ids = (int(contact.geom1), int(contact.geom2))
        if floor_id not in ids:
            continue
        gid = ids[1 - ids.index(floor_id)]
        if gid not in feet:
            continue
        leg = feet.index(gid)
        if material[2 + leg]:
            continue
        pt = contact.pos.copy()
        normal = contact.frame[:3]
        spatial = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, gid, spatial, 0)
        point_v = spatial[3:] + np.cross(spatial[:3], pt - data.geom_xpos[gid])
        tangent = point_v - normal * (normal @ point_v)
        bottom = data.geom_xpos[gid].copy()
        bottom[2] -= model.geom_size[gid, 0]
        midpoint = bottom.copy()
        midpoint[2] = .5 * (bottom[2] + data.geom_xpos[floor_id, 2])
        position_errors.append(float(np.linalg.norm(midpoint - pt)))
        normal_errors.append(float(np.linalg.norm(normal[:2])))
        speed_errors.append(float(np.linalg.norm(reconstructed[leg] - tangent[:2])))
    return position_errors, speed_errors, normal_errors


def score(raw, result):
    with np.load(raw/'trajectory.npz') as z:
        time, posvel, vel = z['time'], z['qpos'], z['qvel']
        phases, attitude, tau = z['phase'], z['rpy'], z['torque']
    ctrl_dt = np.diff(time)
    path = np.linalg.norm(np.diff(posvel[:, :2], axis=0), axis=1)
    moving = np.isin(phases[:-1], [1, 3])
    work = np.maximum(0., tau[:-1] * vel[:-1, 6:]).sum(axis=1) * ctrl_dt
    model = mujoco.MjModel.from_binary_path(str(raw/'model.mjb'))
    data = mujoco.MjData(model)
    feet = [model.geom(name).id for name in LEGS]
    floor_id = model.geom('floor').id
    if model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE or model.geom_bodyid[floor_id] != 0 or any(
            model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in feet):
        raise ValueError('A stationary horizontal plane and spherical feet are required')
    mujoco.mj_forward(model, data)
    if not np.allclose(data.geom_xmat[floor_id].reshape(3, 3)[:, 2], [0., 0., 1.]):
        raise ValueError('Only the horizontal W59 plane has this contact point formula')
    radius = model.geom_size[feet, 0].copy()
    names = result['physical_columns']
    cols = {n: i for i,n in enumerate(names)}
    grf = [cols[f'grf_{leg}_z'] for leg in LEGS]
    vcols = [cols[f'foot_velocity_{leg}_{axis}_m_s'] for leg in LEGS for axis in 'xyz']
    loaded_s = np.zeros((2, 4))
    distance = np.zeros_like(loaded_s)
    center_distance = np.zeros_like(loaded_s)
    travel_distance = np.zeros_like(loaded_s)
    checks = [[], [], []]
    tick = 0
    p = None
    material_files = sorted(raw.glob('material-[0-9][0-9][0-9][0-9].npz'))
    phys_paths = iter(sorted(raw.glob('physical-*.npz')))
    for archive in material_files:
        with np.load(archive) as z:
            rows = z['foot']
        if not rows.size:
            continue
        done = 0
        while done < len(rows):
            if p is None or offset == len(p):
                with np.load(next(phys_paths)) as z:
                    p = z['physical']
                offset = 0
            count = min(len(rows) - done, len(p) - offset)
            sub, mat = p[offset:offset+count], rows[done:done+count]
            if np.max(np.abs(sub[:, cols['force_evaluation_time_s']] - mat[:, 0])) > 1.e-8:
                raise ValueError('Material and force clocks differ')
            v = sub[:, vcols].reshape(-1, 4, 3)
            xy = contact_velocity_xy(mat, v, radius)
            slip = np.linalg.norm(xy, axis=2)
            sphere = np.linalg.norm(v[:, :, :2], axis=2)
            ctrl = np.searchsorted(time, mat[:, 0] + 1.e-9, side='right') - 1
            travel = np.isin(phases[ctrl], [1, 3])
            for terrain in (0, 1):
                active = (sub[:, grf] > 1.) & (mat[:, 2:6] == terrain)
                loaded_s[terrain] += active.sum(axis=0) * DT
                distance[terrain] += (active * slip).sum(axis=0) * DT
                center_distance[terrain] += (active * sphere).sum(axis=0) * DT
                travel_distance[terrain] += (active * slip * travel[:, None]).sum(axis=0) * DT
            for j in range((-tick) % 2500, count, 2500):
                c = int(ctrl[j])
                if abs(time[c] - mat[j, 0]) <= 1.e-8:
                    for acc, obs in zip(checks, audit_snapshot(model, data, feet, floor_id, posvel[c], vel[c], mat[j], xy[j])):
                        acc.extend(obs)
            done += count
            offset += count
            tick += count
    if tick != result['physical_samples']:
        raise ValueError(f'{tick} material samples versus {result["physical_samples"]} force samples')
    d = float(path.sum())
    dm = float(path[moving].sum())
    return dict(attempt=raw.name, status=result['stage_status'], plant_s=result['simulation_s'],
        physical_samples=tick, path_m=d, moving_path_m=dm, contact_point_path_m=distance.tolist(),
        center_path_m=center_distance.tolist(), loaded_leg_s=loaded_s.tolist(),
        point_slip_per_m=float(distance.sum()/d) if d else None,
        moving_point_slip_per_m=float(travel_distance.sum()/dm) if dm else None,
        legacy_all_loaded_center_path_per_m=float(center_distance.sum()/d) if d else None,
        legacy_slip_squared_integral_s=result['material'].get('integral_slip_sq_s'),
        mechanical_positive_work_J=float(work.sum()),
        moving_positive_CoT=float(work[moving].sum()/(model.body_mass.sum()*9.81*dm)) if dm else None,
        min_body_height_m=float(posvel[:,2].min()), max_abs_roll_pitch_deg=float(np.rad2deg(np.abs(attitude[:,:2]).max())),
        direct_mujoco_rigid_contact_comparisons=len(checks[0]),
        max_position_error_m=max(checks[0], default=None),max_tangent_velocity_error_m_s=max(checks[1], default=None),
        max_horizontal_normal_component=max(checks[2], default=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--dest', required=True, type=Path)
    parser.add_argument('attempt', nargs='+')
    args = parser.parse_args()
    for attempt in args.attempt:
        result = json.loads((args.results/(attempt+'.json')).read_text())
        write(args.dest/(attempt+'.json'), score(args.raw_root/attempt, result))
        print(attempt, flush=True)
