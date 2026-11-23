"""Causal contact estimate with a continuously propagated recovery prior."""
from copy import copy
import time

import numpy as np

from s1_w57_material import W57MaterialObserver


class W58MaterialObserver(W57MaterialObserver):
    """Measure one creep direction during support; retain the kinematic prior after release."""

    def __init__(self, parameters, model_path):
        super().__init__(parameters, model_path, mode='M1')
        self.mode = 'M2'
        self.contact_ground = self.ground
        self.prior = W57MaterialObserver(parameters, model_path, mode='M0')
        self.released_mask = np.empty(0, dtype=bool)
        self.released_cell_publications = 0
        self.fusion_wall_s = 0.

    def observe(self, t, bottom, velocity):
        if len(self.packet) < 10:
            return
        # Both paths start virgin and see precisely the same ten measured boundaries.
        # Their state and ownership remain separate across publications.
        self.prior.packet[:] = self.packet
        self.prior.force_measurements[:] = self.force_measurements
        self.prior.observe(t, bottom, velocity)
        self.ground = self.contact_ground
        super().observe(t, bottom, velocity)

        started = time.perf_counter()
        supported, prior = self.contact_ground, self.prior.ground
        released = (~supported.loaded & ~prior.loaded &
                    np.isfinite(supported.last_release) & supported.ever_loaded)
        self.released_mask = released
        self.released_cell_publications += int(np.count_nonzero(released))
        if released.any():
            # Geometry IDs are allocated in causal foot order by both observers.
            # The published arrays are never fed back into either physical history.
            publication = copy(supported)
            for name in ('e', 'q', 'stress'):
                setattr(publication, name, np.where(released[:, None], getattr(prior, name),
                                                     getattr(supported, name)))
            self.ground = publication
        self.fusion_wall_s += time.perf_counter() - started

    def released_state_sensitivity(self, key):
        """Signed contact-corrected minus kinematic-prior state on an observed released cell."""
        i = self.ground.ids.get(tuple(key))
        if i is None or not self.released_mask[i]:
            return None
        return dict(free_surface_depth_m=float(-self.ground.h *
                        (self.contact_ground.e[i, 0] - self.prior.ground.e[i, 0])),
                    q11=float(self.contact_ground.q[i, 0] - self.prior.ground.q[i, 0]))

    def timing(self):
        return super().timing() | dict(
            mode='M2', correction='one signed 0–2 ms footprint direction under load, '
            'unassimilated causal prior at released jointly unloaded cells',
            supported_observer_wall_s=self.wall_s, prior_observer_wall_s=self.prior.wall_s,
            fusion_wall_s=self.fusion_wall_s,
            observation_wall_s=self.wall_s + self.prior.wall_s + self.fusion_wall_s,
            prior_column_steps=self.prior.integration_column_steps,
            released_cell_publications=self.released_cell_publications)
