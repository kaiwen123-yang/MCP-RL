"""Publish the W59 FP64 cross-foot recovery candidate from causal observer releases."""
from copy import copy
from pathlib import Path
import json
import numpy as np

from s1_w57_material import W57MaterialObserver
from s1_w60_cpu_observer import ROOT

PARAMS=ROOT/'experiments/terrain_memory/w59-recovery-parameters.json'


class W59MaterialObserver(W57MaterialObserver):
    """Measured M1 under support, with paired M0/M1 estimates cached at release."""
    def __init__(self,parameters,model_path):
        super().__init__(parameters,model_path,mode='M1')
        self.m1_ground=self.ground
        self.m0=W57MaterialObserver(parameters,model_path,mode='M0')
        self.coefficients=json.loads(PARAMS.read_text())['parameters']
        self.source={}
        self.changed=0

    def observe(self,t,bottom,velocity):
        if len(self.packet)<10:return
        self.m0.packet[:]=self.packet
        self.m0.force_measurements[:]=self.force_measurements
        self.m0.observe(t,bottom,velocity)
        self.ground=self.m1_ground
        super().observe(t,bottom,velocity)
        g=self.m1_ground;p=self.m0.ground
        for row in self.last_packet_events:
            key=tuple(row['world_key']);idx=g.ids[key]
            if row['kind']=='release' and not g.loaded[idx] and not p.loaded[idx]:
                self.source[key]=dict(t=float(t),owner=int(row['owner']),release_id=row['event_id'],
                    m1=np.r_[g.e[idx],g.q[idx]],m0=np.r_[p.e[idx],p.q[idx]])
            elif row['kind']!='release':
                self.source.pop(key,None)
        # The loaded M1 remains the authoritative current-contact estimator.
        # Only an explicit leg query can declare a cross-foot reload.
        self.ground=self.m1_ground

    def ground_for_leg(self,leg,xy,t):
        g=self.m1_ground
        ids,areas=g.footprint(xy,create=False)
        eligible=[]
        for i,area in zip(ids,areas):
            key=tuple(np.floor(np.asarray(g.xy[i])/g.cell).astype(int))
            r=self.source.get(key)
            if r is not None and not g.loaded[i] and r['owner']!=leg:
                eligible.append((area,-i,i,r))
        if not eligible:return g
        # Source method selects the largest observed area on tied onset ticks.
        _,_,i,r=max(eligible)
        state=self._free(r,t)
        pub=copy(g);pub.e=g.e.copy();pub.q=g.q.copy();pub.stress=g.stress.copy()
        pub.e[i]=state[:2];pub.q[i]=state[2:];pub.stress[i]=0.
        pub.w59_reload_contact=(i,self.coefficients['contact_gain'])
        self.changed+=1
        return pub

    def recovery(self,query_xy):
        values,mask=super().recovery(query_xy)
        for leg,xy in enumerate(query_xy):
            g=self.ground_for_leg(leg,xy,self.time)
            if g is self.m1_ground:continue
            ids,areas=g.footprint(xy,create=False);indices=np.asarray(ids,int)
            values[leg]=-g.h/g.area*np.dot(areas,g.law.recovery(g.e[indices],g.q[indices]))
        return values,mask

    def _free(self,record,t):
        a=self.coefficients['release_prior_blend'];rate=self.coefficients['positive_recovery_rate_s_inv']
        return (record['m1']+a*(record['m0']-record['m1']))*np.exp(-rate*max(0.,t-record['t']))

    def planning_publication(self):
        """Sparse original cell IDs + exact M1 current and selected M3 source states."""
        g=self.m1_ground;rows=[]
        for key,r in self.source.items():
            i=g.ids[key]
            if g.loaded[i]:continue
            ix,iy=key
            rows.append([(ix-40)*60+iy+30,r['owner'],r['t'],*r['m1'],*r['m0'],*g.e[i],*g.q[i]])
        return dict(rows=np.asarray(rows,dtype=np.float64).reshape(-1,15),
            t=float(self.time),coefficients=self.coefficients,law=g.law)

    def timing(self):
        return super().timing()|dict(mode='M3',m0_wall_s=self.m0.wall_s,
            release_cells_published=self.changed,parameters=self.coefficients,
            description='New release-state estimate applied only to observed, released world keys; original plant and supported M1 filter are independent.')
