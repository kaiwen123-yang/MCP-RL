"""W57 causal observed-world ledger and footprint-identifiable force correction."""
import ctypes as ct
import time

import numpy as np

from s1_w60_cpu_observer import ARRAY, BYTE, INDEX, ROOT, CPUWindowObserver
from tm02.reference import IntegrationFailure


class W57MaterialObserver(CPUWindowObserver):
    def __init__(self, parameters, model_path, mode='M0'):
        super().__init__(parameters, model_path)
        if mode not in ('M0', 'M1'):
            raise ValueError(f'Unknown W57 material estimator {mode!r}')
        self.mode = mode
        self.native = ct.CDLL(str(ROOT/'work/w57-m/material.so'))
        self.native.w57_packet.argtypes = [
            ARRAY, ct.c_int, ct.c_int, ARRAY, ARRAY, ARRAY, INDEX, ct.c_double,
            ARRAY, BYTE, BYTE, ARRAY, ARRAY, BYTE, ct.c_int, ARRAY, ARRAY]
        self.force_measurements = []
        self.published_bottom = None
        self.last_packet_events = []
        self.load_time = {}
        self.release_id = {}
        self.last_correction = np.zeros((4, 8))
        self.last_prior_forces = np.zeros((10, 4))

    def completed(self, start, end, bottom, velocity, measured_force):
        super().completed(start, end, bottom, velocity)
        self.force_measurements.append(np.asarray(measured_force, dtype=float).reshape(4).copy())

    def history(self, key):
        g = self.ground
        idx = g.ids.get(tuple(key))
        if idx is None or not g.ever_loaded[idx]:
            return dict(ever_loaded=False, loaded=False, last_owner=None,
                        source_load_event_id=None, source_load_time_s=None,
                        release_event_id=None, release_time_s=None)
        return dict(ever_loaded=True, loaded=bool(g.loaded[idx]),
                    last_owner=int(g.last_owner[idx]),
                    source_load_event_id=int(g.last_load_event_id[idx]),
                    source_load_time_s=self.load_time[idx], release_event_id=self.release_id.get(idx),
                    release_time_s=float(g.last_release[idx]) if np.isfinite(g.last_release[idx]) else None)

    def _ledger(self, ids, ownership, active, ends):
        g = self.ground
        previous = np.where(g.loaded[ids], g.last_owner[ids], -1)
        owners = np.where(active, ownership, -1)
        changed = np.vstack((previous != owners[0], owners[1:] != owners[:-1]))
        for tick in np.flatnonzero(changed.any(axis=1)):
            columns = np.flatnonzero(changed[tick])
            # Release before arrivals, with world-index order inside a clock tick.
            for j in sorted(columns, key=lambda i: owners[tick, i] >= 0):
                i = int(ids[j])
                leg = int(owners[tick, j])
                g.event_serial += 1
                old_id = int(g.last_load_event_id[i])
                prior = int(g.last_owner[i])
                t = float(ends[tick])
                if leg < 0:
                    kind = 'release'
                    g.last_release[i] = t
                    self.release_id[i] = g.event_serial
                else:
                    kind = ('first_load' if prior < 0 else
                            'same_foot_reload' if prior == leg else 'cross_foot_revisit')
                    g.ever_loaded[i] = True
                    g.last_owner[i] = leg
                    g.last_load_event_id[i] = g.event_serial
                    self.load_time[i] = t
                self.last_packet_events.append(dict(event_id=g.event_serial, kind=kind,
                    world_key=tuple(round(v/g.cell-.5) for v in g.xy[i]), time_s=t,
                    owner=leg if leg>=0 else prior, previous_load_event_id=old_id if old_id>=0 else None))

    def observe(self, t, bottom, velocity):
        if len(self.packet)<10:
            return
        if len(self.packet)!=10 or abs(self.packet[-1][1]-t)>1e-9:
            raise ValueError(f'Missed a completed 500 Hz publication at {t}')
        self.mark('material_packet', t)
        begin = time.perf_counter()
        g = self.ground
        self.packet_start, self.packet_last = self.packet[0][0], self.packet[-1][0]
        ends = np.array([r[1] for r in self.packet])
        dt = np.diff(np.r_[self.time, ends])
        feet = np.stack([r[2] for r in self.packet])
        tick, leg, global_ids, areas = self.geometry(feet)
        ids = np.array(sorted(self.relaxing.union(global_ids)), dtype=int)
        n = len(ids)
        forces = np.zeros((10, 4))
        diagnostics = np.zeros((4, 8))
        self.last_packet_events = []
        self.last_areas.fill(0.)
        if n:
            local = np.searchsorted(ids, global_ids)
            normal = np.full((10,n), np.inf)
            area = np.zeros((10,n))
            owner = np.full((10,n), -1, np.int32)
            normal[tick,local] = feet[tick,leg,2]/g.h
            owner[tick,local] = leg
            area[tick,local] = areas
            np.add.at(self.last_areas, (tick,leg), areas)
            state = np.column_stack((g.e[ids],g.q[ids],g.stress[ids]))
            settled = np.array([i not in self.relaxing for i in ids], dtype='u1')
            loaded = np.array(g.loaded[ids], dtype='u1')
            trace = np.empty((10,n), dtype='u1')
            stats = np.zeros((10,4))
            bad = self.native.w57_packet(g.law.model,n,10,dt,normal,area,owner,g.gamma,
                                         state,settled,loaded,forces,stats,trace,int(self.mode=='M1'),
                                         self.force_measurements[-1],diagnostics)
            if bad>=0:
                raise IntegrationFailure(bad%n, stats[bad//n,3], 30)
            self._ledger(ids, owner, trace, ends)
            g.e[ids], g.q[ids], g.stress[ids] = state[:,:2], state[:,2:4], state[:,4:6]
            g.loaded[ids] = loaded
            g.last_area[ids] = area[-1]*loaded
            g.last_normal[ids] = np.where(loaded, -state[:,4]*1e6*g.gamma*area[-1], 0.)
            self.relaxing = set(ids[settled==0])
            peak = stats.max(axis=0)
            g.max_strain = max(g.max_strain, peak[0])
            g.max_stress = max(g.max_stress, peak[1])
            g.max_rate = max(g.max_rate, peak[2])
            g.max_residual = max(g.max_residual, peak[3])
            exited = np.flatnonzero(stats[:,0]>.02)
            if len(exited) and g.domain_exit is None:
                k = exited[0]
                g.domain_exit = dict(time_s=ends[k], max_abs_principal_strain=stats[k,0])
        self.last_forces = forces
        self.last_prior_forces = forces.copy()
        self.last_prior_forces[-1] = diagnostics[:,0]
        self.last_correction = diagnostics
        self.published_bottom = feet[-1].copy()
        self.integration_column_steps += 10*n
        self.max_packet_columns = max(self.max_packet_columns, n)
        self.windows += 1
        self.updates += 10
        self.time = float(ends[-1])
        self.packet.clear()
        self.force_measurements.clear()
        self.wall_s += time.perf_counter()-begin
        self.mark('packet_published', t)

    def timing(self):
        return super().timing() | dict(mode=self.mode,
            publication_event_count=self.ground.event_serial,
            correction='one signed 0–2 ms footprint creep/free prior; exact measured final boundary'
                       if self.mode=='M1' else 'none')
