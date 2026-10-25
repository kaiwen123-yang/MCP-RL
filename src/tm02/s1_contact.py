"""World-fixed finite-area columns; actual kinematics are the contact boundary."""
import copy
import numpy as np
import mujoco
from .s1_material import ReferenceColumns, LearnedColumns


class StanceReferenceInfeasible(RuntimeError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(detail['reason'])


class Ground:
    def __init__(self, terrain, material_parameters, weights=None, cell=.02):
        self.terrain = terrain
        self.kind = terrain['id']
        self.area = .001
        self.half = np.sqrt(self.area)/2
        self.cell = cell
        self.h = .05
        self.gamma = .025
        self.law = (LearnedColumns(weights) if weights else ReferenceColumns(material_parameters)) if self.kind == 'tm02_anchor' else None
        self.ids = {}
        self.xy = []
        self.e = np.empty((0, 2))
        self.q = np.empty((0, 2))
        self.stress = np.empty((0, 2))
        self.loaded = np.empty(0, bool)
        self.last_owner = np.empty(0, int)
        self.last_release = np.empty(0)
        self.ever_loaded = np.empty(0, bool)
        self.last_area = np.empty(0)
        self.last_normal = np.empty(0)
        self.last_load_event_id = np.empty(0, int)
        self.events = []
        self.event_serial = 0
        self.rate_hist = []
        self.rate_extrema = {}
        self.max_strain = 0.
        self.max_stress = 0.
        self.max_rate = 0.
        self.max_residual = 0.
        self.domain_exit = None
        self.prediction_steps = None

    def clone(self):
        other = copy.copy(self)
        for name in ('e', 'q', 'stress', 'loaded', 'last_owner', 'last_release', 'ever_loaded',
                     'last_area', 'last_normal', 'last_load_event_id'):
            setattr(other, name, getattr(self, name).copy())
        other.xy, other.ids, other.events = self.xy.copy(), self.ids.copy(), []
        other.rate_hist, other.rate_extrema = [], copy.deepcopy(self.rate_extrema)
        if self.kind=='tm02_anchor':
            other.prediction_steps = []
            other.prediction_last = np.zeros(len(self.e),int)
            other.prediction_active = set()
        return other

    def advance_prediction_query(self, ids):
        """Lazy zero-load *future* propagation from the branch's current state.

        Unqueried columns cannot affect independent-column robot predictions.
        When queried, apply exactly the same already elapsed forecast steps;
        no observed history is replayed and no current true state is replaced.
        """
        if self.prediction_steps is None or not ids:
            return
        self.prediction_active.update(ids)
        ids = np.array(ids,int)
        for step in range(int(self.prediction_last[ids].min()),len(self.prediction_steps)):
            active = ids[self.prediction_last[ids]==step]
            if len(active):
                e,q,s,_ = self.law.step(self.e[active],self.q[active],np.zeros(len(active)),
                    self.prediction_steps[step],np.zeros(len(active),bool))
                self.e[active],self.q[active],self.stress[active] = e,q,s
                self.prediction_last[active] = step+1

    def soft(self, xy):
        # A boundary-straddling equivalent flat sole belongs wholly to the hard
        # region. This explicit hard-priority edge convention avoids simultaneous
        # full-foot native and material support; interior footprints partition
        # their exact .001 m2 area among columns.
        return self.kind != 'rigid' and self.interior(xy)

    def interior(self, xy):
        return .8+self.half <= xy[0] <= 2.2-self.half and abs(xy[1]) <= .6-self.half

    def footprint(self, xy, create=True):
        if not self.interior(xy):
            return [], []
        lo, hi = xy[:2]-self.half, xy[:2]+self.half
        cells, areas = [], []
        for ix in range(int(np.floor(lo[0]/self.cell)), int(np.floor(hi[0]/self.cell))+1):
            for iy in range(int(np.floor(lo[1]/self.cell)), int(np.floor(hi[1]/self.cell))+1):
                lower = np.maximum(lo, np.array([ix, iy])*self.cell)
                upper = np.minimum(hi, (np.array([ix, iy])+1)*self.cell)
                a = float(np.prod(np.maximum(upper-lower, 0)))
                if a == 0:
                    continue
                key = (ix, iy)
                if key not in self.ids:
                    if not create:
                        continue
                    self.ids[key] = len(self.xy)
                    self.xy.append(((ix+.5)*self.cell, (iy+.5)*self.cell))
                    self.e = np.vstack((self.e, np.zeros(2)))
                    self.q = np.vstack((self.q, np.zeros(2)))
                    self.stress = np.vstack((self.stress, np.zeros(2)))
                    self.loaded = np.r_[self.loaded, False]
                    self.last_owner = np.r_[self.last_owner, -1]
                    self.last_release = np.r_[self.last_release, np.nan]
                    self.ever_loaded = np.r_[self.ever_loaded, False]
                    self.last_area = np.r_[self.last_area, 0.]
                    self.last_normal = np.r_[self.last_normal, 0.]
                    self.last_load_event_id = np.r_[self.last_load_event_id, -1]
                    if self.prediction_steps is not None:
                        self.prediction_last = np.r_[self.prediction_last,len(self.prediction_steps)]
                cells.append(self.ids[key])
                areas.append(a)
        self.advance_prediction_query(cells)
        return cells, areas

    def step(self, feet_bottom, velocity, dt, t, log=True):
        maps = [self.footprint(p[:2]) for p in feet_bottom]
        n = len(self.e)
        forces = np.zeros((4, 3))
        if n == 0 or self.kind == 'rigid':
            return forces, np.zeros(4), maps
        prev_e, prev_q = self.e.copy(), self.q.copy()
        old_loaded = self.loaded.copy()
        owner = np.full(n, -1)
        areas = np.zeros(n)
        # Each cell has one displacement boundary. If two soles overlap the
        # same cell, the lower sole owns it; force is never counted twice.
        for leg, (ids, aa) in enumerate(maps):
            for idx, a in zip(ids, aa):
                if owner[idx] < 0 or feet_bottom[leg, 2] < feet_bottom[owner[idx], 2]:
                    owner[idx], areas[idx] = leg, a
        if self.kind.startswith('kv'):
            k, d = self.terrain['k_N_per_m'], self.terrain['d_N_s_per_m']
            previous_delta = -self.e[:, 0].copy()
            # Backward Euler uses the actual imposed column displacement. This
            # includes first loading of a previously virgin spatial column;
            # foot-centre velocity alone misses that local displacement change.
            free_delta = previous_delta/(1+k*dt/d)
            preload_free_e = prev_e.copy()
            preload_free_e[:, 0] = -free_delta
            preload_free_q = preload_free_e.copy()
            self.e[:, 0] = -free_delta
            self.stress[:] = 0.
            pressure = np.zeros(n)
            for idx in np.flatnonzero(owner>=0):
                leg = owner[idx]
                delta = -feet_bottom[leg, 2]
                f = k*delta+d*(delta-previous_delta[idx])/dt
                if delta >= free_delta[idx] and f>0:
                    self.e[idx, 0] = -delta
                    pressure[idx] = f/self.area
            self.q = self.e.copy()  # KV state is its deformation, no hidden internal strain.
        else:
            active = np.array(sorted(self.prediction_active),int) if self.prediction_steps is not None else np.arange(n)
            free_e,free_q,free_s,res = self.law.step(prev_e[active],prev_q[active],np.zeros(len(active)),dt,np.zeros(len(active),bool))
            self.e[active],self.q[active],self.stress[active] = free_e,free_q,free_s
            preload_free_e, preload_free_q = self.e.copy(), self.q.copy()
            residual = np.zeros(n)
            residual[active] = res
            candidates = np.flatnonzero(owner>=0)
            if len(candidates):
                en = feet_bottom[owner[candidates], 2]/self.h
                candidates = candidates[en <= self.e[candidates, 0]]
            if len(candidates):
                en = feet_bottom[owner[candidates], 2]/self.h
                e, q, s, res = self.law.step(prev_e[candidates], prev_q[candidates], en, dt, np.ones(len(candidates), bool))
                compressive = s[:, 0]<0
                ids = candidates[compressive]
                self.e[ids], self.q[ids], self.stress[ids] = e[compressive], q[compressive], s[compressive]
                residual[candidates] = res
            pressure = np.zeros(n)
            if len(candidates):
                pressure[ids] = -self.stress[ids, 0]*1e6*self.gamma
            self.max_residual = max(self.max_residual, float(np.max(residual)))
            if self.prediction_steps is not None:
                self.prediction_steps.append(dt)
                self.prediction_last[active] = len(self.prediction_steps)
        self.loaded = (pressure>0)&(owner>=0)
        indentation = np.zeros(4)
        for idx in np.flatnonzero(self.loaded):
            leg = owner[idx]
            normal = pressure[idx]*areas[idx]
            tangential = -.5*normal*velocity[leg, :2]/np.sqrt(np.dot(velocity[leg, :2], velocity[leg, :2])+.01**2)
            forces[leg] += np.r_[tangential, normal]
            indentation[leg] = max(indentation[leg], -self.e[idx, 0]*(self.h if self.kind=='tm02_anchor' else 1.))
        if log:
            for idx in np.flatnonzero(old_loaded & ~self.loaded):
                self.last_release[idx] = t
                self.event_serial += 1
                self.events.append(dict(event_id=self.event_serial, time_s=t, patch=int(idx), type='release',
                    leg=int(self.last_owner[idx]), previous_load_event_id=int(self.last_load_event_id[idx]),
                    area_before_m2=float(self.last_area[idx]), area_m2=0., normal_before_N=float(self.last_normal[idx]),
                    normal_N=0., e_before_step=prev_e[idx].tolist(), q_before_step=prev_q[idx].tolist(),
                    preload_free_e=preload_free_e[idx].tolist(), preload_free_q=preload_free_q[idx].tolist(),
                    e=self.e[idx].tolist(), q=self.q[idx].tolist()))
            for idx in np.flatnonzero(self.loaded & (~old_loaded | (owner!=self.last_owner))):
                self.event_serial += 1
                self.events.append(dict(event_id=self.event_serial, time_s=t, patch=int(idx), type='revisit' if self.ever_loaded[idx] else 'load',
                    leg=int(owner[idx]), previous_leg=int(self.last_owner[idx]), area_m2=float(areas[idx]),
                    previous_load_event_id=int(self.last_load_event_id[idx]),
                    cross_leg=bool(self.ever_loaded[idx] and self.last_owner[idx]!=owner[idx]),
                    area_before_m2=float(self.last_area[idx]), normal_before_N=float(self.last_normal[idx]),
                    normal_N=float(pressure[idx]*areas[idx]),
                    unloaded_gap_s=float(t-self.last_release[idx]) if np.isfinite(self.last_release[idx]) else None,
                    e_before_step=prev_e[idx].tolist(), q_before_step=prev_q[idx].tolist(),
                    preload_free_e=preload_free_e[idx].tolist(), preload_free_q=preload_free_q[idx].tolist(),
                    xy_m=list(self.xy[idx]), e=self.e[idx].tolist(), q=self.q[idx].tolist()))
                self.last_load_event_id[idx] = self.event_serial
            if self.kind == 'tm02_anchor':
                self.record_material_rates(prev_e, prev_q, old_loaded, owner, areas, pressure, velocity, dt, t)
            self.ever_loaded |= self.loaded
            self.last_owner[self.loaded] = owner[self.loaded]
            self.last_area = np.where(self.loaded, areas, 0.)
            self.last_normal = pressure*areas
        self.max_strain = max(self.max_strain, float(np.max(np.abs(self.e))))
        self.max_stress = max(self.max_stress, float(np.max(np.abs(self.stress))))
        self.max_rate = max(self.max_rate, float(np.max(np.abs(self.e-prev_e))/dt))
        if self.kind == 'tm02_anchor' and self.domain_exit is None and self.max_strain>.02:
            self.domain_exit = dict(time_s=t, max_abs_principal_strain=self.max_strain)
        return forces, indentation, maps

    def record_material_rates(self, prev_e, prev_q, old_loaded, owner, areas, pressure, velocity, dt, t):
        """Separate temporal loading from spatial-column boundary changes."""
        de, qdot = self.e-prev_e, (self.q-prev_q)/dt
        continuous = old_loaded & self.loaded & (owner == self.last_owner)
        boundary = (old_loaded != self.loaded) | (old_loaded & self.loaded & (owner != self.last_owner))
        values = [('continuous_loaded_strain_rate_s_inv', np.max(np.abs(de), axis=1)/dt, continuous),
                  ('activation_release_strain_jump', np.max(np.abs(de), axis=1), boundary),
                  ('internal_qdot_s_inv', np.max(np.abs(qdot), axis=1), np.ones(len(de), bool))]
        row = [t]
        for label, magnitude, mask in values:
            candidates = np.flatnonzero(mask)
            value = float(np.max(magnitude[candidates])) if len(candidates) else 0.
            row.append(value)
            if len(candidates) and (label not in self.rate_extrema or value > self.rate_extrema[label]['magnitude']):
                idx = int(candidates[np.argmax(magnitude[candidates])])
                leg = int(owner[idx] if owner[idx]>=0 else self.last_owner[idx])
                self.rate_extrema[label] = dict(magnitude=value, time_s=t, patch=idx, leg=leg,
                    previous_leg=int(self.last_owner[idx]), loaded_before=bool(old_loaded[idx]), loaded_after=bool(self.loaded[idx]),
                    boundary_transition=('activation' if not old_loaded[idx] and self.loaded[idx] else
                        'release' if old_loaded[idx] and not self.loaded[idx] else
                        'ownership_transfer' if old_loaded[idx] and self.loaded[idx] and owner[idx]!=self.last_owner[idx] else
                        'continuous_loaded' if self.loaded[idx] else 'unloaded_recovery'),
                    area_before_m2=float(self.last_area[idx]), area_after_m2=float(areas[idx]) if self.loaded[idx] else 0.,
                    force_before_N=float(self.last_normal[idx]), force_after_N=float(pressure[idx]*areas[idx]),
                    foot_vz_over_H_s_inv=float(velocity[leg, 2]/self.h) if leg>=0 else None,
                    strain_increment=de[idx].tolist(), qdot_s_inv=qdot[idx].tolist(),
                    e_before=prev_e[idx].tolist(), e_after=self.e[idx].tolist(),
                    q_before=prev_q[idx].tolist(), q_after=self.q[idx].tolist())
        loaded_legs = np.unique(owner[self.loaded])
        peak_foot_rate = float(np.max(np.abs(velocity[loaded_legs, 2]/self.h))) if len(loaded_legs) else 0.
        row.append(peak_foot_rate)
        label = 'actual_foot_vz_over_H_s_inv'
        if len(loaded_legs) and (label not in self.rate_extrema or peak_foot_rate > self.rate_extrema[label]['magnitude']):
            leg = int(loaded_legs[np.argmax(np.abs(velocity[loaded_legs, 2]/self.h))])
            patches = np.flatnonzero(self.loaded & (owner == leg))
            normal = pressure[patches]*areas[patches]
            dominant = int(patches[np.argmax(normal)])
            self.rate_extrema[label] = dict(magnitude=peak_foot_rate, time_s=t, leg=leg,
                foot_vz_over_H_s_inv=float(velocity[leg, 2]/self.h), foot_vz_m_s=float(velocity[leg, 2]),
                thickness_m=self.h, patch=dominant, patch_selection='largest actual normal force under the peak-rate leg',
                patches=patches.tolist(), area_m2=float(areas[patches].sum()), force_N=float(normal.sum()),
                patch_areas_m2=areas[patches].tolist(), patch_normal_forces_N=normal.tolist(),
                dominant_patch_loaded_before=bool(old_loaded[dominant]),
                dominant_patch_previous_leg=int(self.last_owner[dominant]),
                e_before=prev_e[dominant].tolist(), e_after=self.e[dominant].tolist(),
                q_before=prev_q[dominant].tolist(), q_after=self.q[dominant].tolist(),
                qdot_s_inv=qdot[dominant].tolist())
        self.rate_hist.append(row)

    def force_inverse(self, xy, force_N, dt):
        """Invert the common sole displacement, without touching world state.

        The endpoint equation uses the same unilateral mixed-boundary law and
        exact overlap areas as step(). The inversion increment is the declared
        control interval; actual environmental integration keeps its 0.2 ms step.
        """
        branch = self.clone()
        ids, areas = branch.footprint(np.asarray(xy))
        if not ids or self.kind == 'rigid':
            return dict(displacement_m=0., previous_mean_displacement_m=0., force_residual_N=0.,
                        forward_force_N=0., requested_force_N=float(force_N), inverse_dt_s=dt,
                        max_abs_principal_strain=0.)
        aa = np.asarray(areas)
        ep, qp = branch.e[ids].copy(), branch.q[ids].copy()
        thickness = self.h if self.kind == 'tm02_anchor' else 1.
        previous = float(np.dot(aa, -ep[:, 0]*thickness)/aa.sum())
        if self.kind.startswith('kv'):
            k, d = self.terrain['k_N_per_m'], self.terrain['d_N_s_per_m']
            free = -ep[:, 0]/(1+k*dt/d)
            def forward(delta):
                f = np.maximum(k*delta+d*(delta+ep[:, 0])/dt, 0.)
                return float(np.dot(aa/self.area, f)), None
            lower = min(0., float(free.min()))
            upper = max(float(free.max()), (float(force_N)*self.area/aa.sum()+d/dt*float((-ep[:, 0]).max()))/(k+d/dt)) + 1e-9
        else:
            free_e, _, _, _ = self.law.step(ep, qp, np.zeros(len(ids)), dt, np.zeros(len(ids), bool))
            def forward(delta):
                ee, qq, ss, _ = self.law.step(ep, qp, np.full(len(ids), -delta/self.h), dt, np.ones(len(ids), bool))
                pressure = np.where((-delta/self.h <= free_e[:, 0]) & (ss[:, 0]<0), -ss[:, 0]*1e6*self.gamma, 0.)
                return float(np.dot(aa, pressure)), ee
            lower, upper = min(0., float(np.min(-free_e[:, 0]*self.h))), self.h*.02
        from scipy.optimize import brentq
        wanted = max(0., float(force_N))
        if wanted == 0.:
            delta = lower
        else:
            force_upper, _ = forward(upper)
            if force_upper < wanted:
                raise StanceReferenceInfeasible(dict(reason='constitutive_inverse_outside_material_domain',
                    requested_force_N=wanted, maximum_domain_force_N=force_upper, xy_m=list(xy), dt_s=dt))
            delta = brentq(lambda value: forward(value)[0]-wanted, lower, upper, xtol=1e-11, rtol=1e-12)
        force, ee = forward(delta)
        return dict(displacement_m=float(delta), previous_mean_displacement_m=previous,
                    forward_force_N=force, requested_force_N=wanted, force_residual_N=force-wanted,
                    inverse_dt_s=dt, max_abs_principal_strain=float(np.max(np.abs(ee))) if ee is not None else None)

    def displacement_for_force(self, xy, force_N, dt):
        """Inverse constitutive boundary for a candidate load, never a future observation."""
        if self.kind == 'rigid':
            return 0.
        ids, aa = self.footprint(xy)
        if not ids:
            return 0.
        weights = np.array(aa)/self.area
        if self.kind.startswith('kv'):
            k, d = self.terrain['k_N_per_m'], self.terrain['d_N_s_per_m']
            previous = float(np.dot(weights, -self.e[ids, 0]))
            return (force_N+d/dt*previous)/(k+d/dt)
        stress = np.full(len(ids), -force_N/self.area/1e6/self.gamma)
        e, _, _, _ = self.law.step(self.e[ids], self.q[ids], stress, dt, np.zeros(len(ids), bool))
        return float(-np.dot(weights, e[:, 0])*self.h)

    def observe_rigid(self, maps, forces, t):
        old = self.loaded.copy()
        owner = np.full(len(old), -1)
        areas = np.zeros(len(old))
        for leg, (ids, aa) in enumerate(maps):
            if forces[leg, 2]>0:
                owner[ids], areas[ids] = leg, aa
        self.loaded = owner>=0
        for idx in np.flatnonzero(old & ~self.loaded):
            self.last_release[idx] = t
            self.event_serial += 1
            self.events.append(dict(event_id=self.event_serial,time_s=t,patch=int(idx),type='release',leg=int(self.last_owner[idx]),
                previous_load_event_id=int(self.last_load_event_id[idx]), area_before_m2=float(self.last_area[idx]), area_m2=0.,
                normal_before_N=float(self.last_normal[idx]), normal_N=0.,
                e_before_step=[0.,0.],q_before_step=[0.,0.],preload_free_e=[0.,0.],preload_free_q=[0.,0.],e=[0.,0.],q=[0.,0.]))
        for idx in np.flatnonzero(self.loaded & (~old | (owner!=self.last_owner))):
            self.event_serial += 1
            self.events.append(dict(event_id=self.event_serial,time_s=t,patch=int(idx),type='revisit' if self.ever_loaded[idx] else 'load',
                leg=int(owner[idx]),previous_leg=int(self.last_owner[idx]),area_m2=float(areas[idx]),
                previous_load_event_id=int(self.last_load_event_id[idx]),
                cross_leg=bool(self.ever_loaded[idx] and self.last_owner[idx]!=owner[idx]),
                normal_N=float(forces[owner[idx],2]*areas[idx]/self.area), area_before_m2=float(self.last_area[idx]),
                normal_before_N=float(self.last_normal[idx]),
                unloaded_gap_s=float(t-self.last_release[idx]) if np.isfinite(self.last_release[idx]) else None,
                e_before_step=[0.,0.],q_before_step=[0.,0.],preload_free_e=[0.,0.],preload_free_q=[0.,0.],
                xy_m=list(self.xy[idx]),e=[0.,0.],q=[0.,0.]))
            self.last_load_event_id[idx] = self.event_serial
        self.ever_loaded |= self.loaded
        self.last_owner[self.loaded] = owner[self.loaded]
        self.last_area = np.where(self.loaded, areas, 0.)
        self.last_normal = np.array([forces[owner[i],2]*areas[i]/self.area if owner[i]>=0 else 0. for i in range(len(owner))])


class ContactEnvironment:
    def __init__(self, env, config, terrain, material_parameters, weights=None):
        self.env = env
        self.legs = ('FL', 'FR', 'RL', 'RR')
        self.geom = np.array([env._feet_geom_id[l] for l in self.legs])
        self.body = env.mjModel.geom_bodyid[self.geom]
        self.radius = env.mjModel.geom_size[self.geom, 0].copy()
        self.native_type = env.mjModel.geom_contype[self.geom].copy()
        self.native_affinity = env.mjModel.geom_conaffinity[self.geom].copy()
        self.ground = Ground(terrain, material_parameters)
        self.predictor = Ground(terrain, material_parameters, weights) if weights else self.ground
        self.force = np.zeros((4, 3))
        self.indent = np.zeros(4)
        self.contact = np.zeros(4, bool)
        self.material_seconds = 0.
        self.predictor_seconds = 0.
        self.material_hist = []
        self.predictor_hist = []
        self.maps = [([], []) for _ in range(4)]
        self.physics_hist = []

    def kinematics(self):
        p = self.env.feet_pos(frame='world')
        v = self.env.feet_vel(frame='world')
        bottom = np.array([p[l] for l in self.legs])
        bottom[:, 2] -= self.radius
        velocity = np.array([v[l] for l in self.legs])
        for leg in range(4):
            twist = np.zeros(6)
            mujoco.mj_objectVelocity(self.env.mjModel, self.env.mjData, mujoco.mjtObj.mjOBJ_GEOM,
                                    int(self.geom[leg]), twist, 0)
            velocity[leg] += np.cross(twist[:3], [0., 0., -self.radius[leg]])
        return bottom, velocity

    def advance(self, action, dt, t, update_predictor=True, record_material=True, record_physics=True):
        import time
        env = self.env
        # mj_step leaves Cartesian geometry at the pre-integration qpos. Refresh
        # it before imposing a displacement boundary from the new robot state.
        mujoco.mj_forward(env.mjModel, env.mjData)
        bottom, velocity = self.kinematics()
        is_soft = np.array([self.ground.soft(x[:2]) for x in bottom])
        env.mjModel.geom_contype[self.geom] = np.where(is_soft, 0, self.native_type)
        env.mjModel.geom_conaffinity[self.geom] = np.where(is_soft, 0, self.native_affinity)
        start = time.perf_counter()
        force, self.indent, self.maps = self.ground.step(bottom, velocity, dt, t)
        self.material_seconds = time.perf_counter()-start
        self.predictor_seconds = 0.
        if self.predictor is not self.ground and update_predictor:
            start = time.perf_counter()
            self.predictor.step(bottom, velocity, .002, t)
            self.predictor_seconds = time.perf_counter()-start
        env.mjData.qfrc_applied[:] = 0.
        for leg in range(4):
            if is_soft[leg]:
                mujoco.mj_applyFT(env.mjModel, env.mjData, force[leg], np.zeros(3), bottom[leg],
                                 int(self.body[leg]), env.mjData.qfrc_applied)
        _, _, terminated, _, info = env.step(action)
        native_contact, _, native_force = env.feet_contact_state(ground_reaction_forces=True)
        for leg, name in enumerate(self.legs):
            self.force[leg] = force[leg] if is_soft[leg] else native_force[name]
        self.contact = self.force[:, 2]>0
        if self.ground.kind=='rigid':
            self.ground.observe_rigid(self.maps, self.force, t)
        if record_physics:
            self.physics_hist.append(np.r_[t, bottom.ravel(), velocity.ravel(), self.force.ravel(), self.indent])
        if record_material and len(self.ground.e):
            self.material_hist.append((float(t), self.ground.e.copy(), self.ground.q.copy(),
                                       self.ground.stress.copy(), self.ground.loaded.copy()))
        if record_material and self.predictor is not self.ground and len(self.predictor.e):
            self.predictor_hist.append((float(t), self.predictor.e.copy(), self.predictor.q.copy(),
                                        self.predictor.stress.copy(), self.predictor.loaded.copy()))
        # The native ground still detects non-foot penetration even where foot
        # collisions have been disabled. A calf/body contact is a task failure.
        nonfoot = []
        for contact in env.mjData.contact:
            if 0 in env.mjModel.geom_bodyid[[contact.geom1, contact.geom2]]:
                geom = contact.geom2 if env.mjModel.geom_bodyid[contact.geom1] == 0 else contact.geom1
                if geom not in self.geom and self.ground.soft(env.mjData.geom_xpos[geom, :2]):
                    nonfoot.append(mujoco.mj_id2name(env.mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom) or str(geom))
        return terminated, info, nonfoot

    def save(self, path):
        payload = {'physics_foot_time_bottom_velocity_force_indent':np.array(self.physics_hist)}
        payload['reference_material_rate_time_continuous_jump_qdot_vz_over_H'] = np.asarray(self.ground.rate_hist).reshape(-1, 5)
        for label, history, ground in [('reference', self.material_hist, self.ground),
                                        ('predictor', self.predictor_hist, self.predictor)]:
            if history:
                payload[label+'_time'] = np.array([x[0] for x in history])
                payload[label+'_offsets'] = np.r_[0, np.cumsum([len(x[1]) for x in history])]
                for col, name in enumerate(('e', 'q', 'stress_MPa', 'loaded'), 1):
                    payload[label+'_'+name] = np.concatenate([x[col] for x in history])
                payload[label+'_patch_xy_m'] = np.array(ground.xy)
        np.savez_compressed(path, **payload)
