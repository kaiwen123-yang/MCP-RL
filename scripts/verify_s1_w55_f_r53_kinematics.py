"""Compare the actual S5-r53 controller's drift with moving-point Jacobian differences."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from s1_w55_f_r53_backend import FController

ROOT = Path(__file__).resolve().parents[1]
RUN = 'tm-s1-go2-local-readiness-r12'
ATTEMPTS = {
    'r51': 'job-20261117T053419Z-w54-f-a-r51-eval-PANN_PPO_S5-s1-E30-attempt01',
    'r52': 'job-20261117T054859Z-w54-f-a-r52-eval-PANN_PPO_S5-s1-E30-attempt01',
}


def velocities(model, data, controller, qpos, qvel, eps):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_integratePos(model, data.qpos, qvel, eps)
    mujoco.mj_forward(model, data)
    linear, angular = np.zeros((2, 3, model.nv))
    centers, rolling, foot_angular, bodies = [], [], [], []
    for leg, gid in enumerate(controller.feet):
        mujoco.mj_jacGeom(model, data, linear, angular, int(gid))
        center = linear @ qvel
        omega = angular @ qvel
        centers.append(center)
        rolling.append(center + np.cross(omega, [0., 0., -controller.radius[leg]]))
        foot_angular.append(omega)
    for body in controller.body_ids:
        mujoco.mj_jacBodyCom(model, data, linear, None, int(body))
        bodies.append(linear @ qvel)
    mujoco.mj_jacSubtreeCom(model, data, linear, controller.base)
    mujoco.mj_jacBody(model, data, None, angular, controller.base)
    return dict(foot=np.array(centers), body_COM=np.array(bodies), COM=linear@qvel,
                rolling=np.array(rolling), base_angular=angular@qvel,
                foot_angular=np.array(foot_angular))


def check(source_root):
    rows = []
    eps = 1e-6
    for version, attempt in ATTEMPTS.items():
        out = source_root / 'outputs/terrain_memory/simulation' / RUN / attempt
        model = mujoco.MjModel.from_binary_path(str(out/'model.mjb'))
        data = mujoco.MjData(model)
        feet = [model.body_geomadr[model.body(leg+'_foot').id] for leg in ('FL','FR','RL','RR')]
        base = model.body('trunk').id
        data_plus, data_minus = mujoco.MjData(model), mujoco.MjData(model)
        with np.load(out/'trajectory.npz') as trajectory, TemporaryDirectory() as temporary:
            times, poses, speeds = (trajectory[k] for k in ('time','qpos','qvel'))
            data.qpos[:], data.qvel[:] = poses[0], speeds[0]
            mujoco.mj_forward(model, data)
            controller = FController(model, data, feet, base, temporary)
            for time_s in (2.06, 3.64, 4.884, 5.36):
                i = np.argmin(abs(times-time_s))
                qpos, qvel = poses[i], speeds[i]
                data.qpos[:], data.qvel[:], data.time = qpos, qvel, times[i]
                mujoco.mj_forward(model, data)
                actual = controller.model_quantities(data)
                recorded = controller.drift_rows[-1]
                plus = velocities(model, data_plus, controller, qpos, qvel, eps)
                minus = velocities(model, data_minus, controller, qpos, qvel, -eps)
                fd = {k: (v-minus[k])/(2*eps) for k,v in plus.items()}
                foot_angular = []
                for gid in feet:
                    jac = np.empty((3, model.nv))
                    mujoco.mj_jacDot(model, data, None, jac, data.geom_xpos[gid], int(model.geom_bodyid[gid]))
                    foot_angular.append(jac@qvel)
                calculated = dict(foot=actual['Jdotv'], body_COM=recorded['body_COM_corrected'],
                                  COM=actual['Jcomdotv'], rolling=actual['Jrolldotv'],
                                  base_angular=actual['Jrdotv'], foot_angular=np.array(foot_angular))
                errors = {k: float(abs(v-fd[k]).max()) for k,v in calculated.items()}
                old = {k: float(abs(recorded[k+'_raw']-fd[k]).max())
                       for k in ('foot','body_COM','COM','rolling')}
                rows.append(dict(attempt=attempt, version=version, time_s=float(times[i]),
                                 corrected_max_absolute=errors, original_max_absolute=old))
    result = dict(controller='S5-r53', physics_version=mujoco.__version__,
                  method='Centered configuration tangent; individually moving body COMs, rotating foot centers, fixed-world sphere offset. No dynamics integration or controller solve.',
                  epsilon_s=eps, states=rows, dynamic_steps=0, new_native_rollouts=0, QP_solves=0,
                  max_absolute_corrected={k:max(row['corrected_max_absolute'][k] for row in rows)
                                          for k in rows[0]['corrected_max_absolute']})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    options = parser.parse_args()
    result = check(options.source_root)
    target = ROOT / 'results/terrain_memory/simulation' / RUN / 'w55/f/r53-same-state.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['max_absolute_corrected']))
    if max(result['max_absolute_corrected'].values()) > 1e-6:
        raise ValueError('Corrected drift disagrees with the independent moving-point Jacobian difference')
