"""Independent 2 ms PANN columns and private conditional recovery queries."""
import time
import numpy as np
from tm02.s1_contact import Ground


class MaterialObserver:
    def __init__(self, parameters, weights):
        self.ground = Ground({'id':'tm02_anchor'}, parameters, weights=weights)
        self.time = 0.
        self.updates = 0
        self.wall_s = 0.
        self.query_wall_s = 0.
        self.query_calls = 0

    def observe(self, t, bottom, velocity):
        if t == 0:
            return
        begin = time.perf_counter()
        self.ground.step(bottom, velocity, float(t-self.time), float(t), log=False)
        self.time = float(t)
        self.updates += 1
        self.wall_s += time.perf_counter()-begin

    def recovery(self, query_xy):
        """Copy one world-key union; integrate the same 250 x 2 ms law steps."""
        begin = time.perf_counter()
        # clone() marks a future Ground branch for lazy propagation; no Ground
        # prediction steps are used here, only its own constitutive law below.
        private = self.ground.clone()
        maps = [private.footprint(p, create=True) for p in query_xy]
        union = np.array(sorted({i for ids,_ in maps for i in ids}),int)
        values = np.zeros((4,3))
        masks = np.array([private.interior(p) for p in query_xy],bool)
        if len(union):
            e,q = private.e[union].copy(),private.q[union].copy()
            rows = {idx:j for j,idx in enumerate(union)}
            saved=[]
            for step in range(1,251):
                e,q,_,_ = private.law.step(e,q,np.zeros(len(union)),.002,np.zeros(len(union),bool))
                if step in (1,100,250):
                    saved.append(e.copy())
            for leg,(ids,area) in enumerate(maps):
                if ids:
                    for horizon,ee in enumerate(saved):
                        values[leg,horizon] = -private.h*np.dot(
                            ee[[rows[i] for i in ids],0],area)/private.area
        self.query_calls += 1
        self.query_wall_s += time.perf_counter()-begin
        return values, masks

    def states(self):
        g=self.ground
        return dict(time_s=self.time,keys=np.asarray(list(g.ids),int),
                    e=g.e.copy(),q=g.q.copy(),stress=g.stress.copy())

    def timing(self):
        return dict(observation_updates=self.updates,observation_wall_s=self.wall_s,
            query_calls=self.query_calls,query_wall_s=self.query_wall_s,
            first_compilation_s=self.ground.law.first_call_including_compile_s,
            world_cells=len(self.ground.e),sampling='Actual elapsed 2 ms; endpoint kinematic boundary. Within-window load/unload is not resolved at 0.2 ms; no true q/e correction.')
