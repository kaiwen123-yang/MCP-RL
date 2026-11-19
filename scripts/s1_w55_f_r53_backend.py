"""S5-r53 moving-point kinematics for the frozen MuJoCo 3.2.3 CPU controller."""
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from s1_w54_f_r52_backend import FController as R52Controller


def moving_point_drift_323(raw, angular_velocity, point_velocity):
    """Upstream de48f417 contracted with qvel, in world coordinates."""
    correction = np.cross(angular_velocity, point_velocity)
    return raw + correction, correction


class FController(R52Controller):
    def __init__(self, model, data, feet, base, out, config=None):
        if config is None:
            root = Path(__file__).resolve().parents[1]
            config = json.loads((root / 'experiments/terrain_memory/s1-w55-f-r53.json').read_text())
        if mujoco.mj_version() != 323:
            raise ValueError('S5-r53 kinematic backport requires frozen MuJoCo 3.2.3')
        super().__init__(model, data, feet, base, out, config)
        self.record_dir = self.record_dir.rename(self.out / 'w55-f')
        self.drift_rows = []
        self.timing['model_323_point_motion'] = dict(wall_s=0., cpu_s=0., calls=0)
        self.manifest.update(schema='w55_f_records_v1', drift_chunks=[],
                             kinematics=config['revision']['drift'])
        self._write_manifest()

    def model_quantities(self, data):
        quantities = super().model_quantities(data)
        wall, cpu = time.perf_counter(), time.process_time()
        model = self.model
        foot_raw = quantities['Jdotv']
        foot_corrected, foot_term = moving_point_drift_323(
            foot_raw, quantities['foot_omega'], quantities['foot_velocity'])
        # R52 rolling uses a fixed world offset from the spherical foot center.
        # Its angular contribution is unchanged. Shift the center term once.
        rolling_raw = quantities['Jrolldotv']
        rolling_corrected = rolling_raw + foot_term
        body_raw = np.empty((len(self.body_ids), 3))
        body_omega = np.empty_like(body_raw)
        body_velocity = np.empty_like(body_raw)
        jac_point = np.empty((3, model.nv))
        jac_angular = np.empty_like(jac_point)
        jac_dot = np.empty_like(jac_point)
        for i, body in enumerate(self.body_ids):
            # mj_jacBodyCom uses this body's moving xipos, not the joint origin.
            mujoco.mj_jacBodyCom(model, data, jac_point, jac_angular, int(body))
            mujoco.mj_jacDot(model, data, jac_dot, None, data.xipos[body], int(body))
            body_raw[i] = jac_dot @ data.qvel
            body_omega[i] = jac_angular @ data.qvel
            body_velocity[i] = jac_point @ data.qvel
        body_corrected, body_term = moving_point_drift_323(body_raw, body_omega, body_velocity)
        mass_fraction = model.body_mass[self.body_ids] / self.mass
        com_corrected = mass_fraction @ body_corrected
        self.drift_rows.append(dict(
            time=float(data.time), foot_raw=foot_raw, foot_point_term=foot_term,
            foot_corrected=foot_corrected, rolling_raw=rolling_raw,
            rolling_corrected=rolling_corrected, body_COM_raw=body_raw,
            body_COM_point_term=body_term, body_COM_corrected=body_corrected,
            COM_raw=quantities['Jcomdotv'], COM_point_term=mass_fraction @ body_term,
            COM_corrected=com_corrected, base_angular=quantities['Jrdotv']))
        quantities.update(Jdotv=foot_corrected, Jrolldotv=rolling_corrected,
                          Jcomdotv=com_corrected)
        self._time('model_323_point_motion', wall, cpu)
        return quantities

    def flush(self):
        if self.drift_rows:
            wall, cpu = time.perf_counter(), time.process_time()
            path = self.record_dir / f'drift-{len(self.manifest["drift_chunks"])+1:05d}.npz'
            np.savez_compressed(path, **{key: np.asarray([row[key] for row in self.drift_rows])
                                         for key in self.drift_rows[0]})
            self.manifest['drift_chunks'].append(dict(
                path=path.name, samples=len(self.drift_rows), start_s=self.drift_rows[0]['time'],
                end_s=self.drift_rows[-1]['time'], size=path.stat().st_size))
            self.drift_rows.clear()
            self._time('record_serialization_io', wall, cpu)
        super().flush()
