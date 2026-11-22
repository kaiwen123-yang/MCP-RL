"""Numerical and decision comparison on archived r60 A/B E30 events; no plant integration."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import osqp
from scipy import sparse
from s1_w57_material import W57MaterialObserver
from s1_w57_planning import Forecast, rank
from tm02.s1_contact import StanceReferenceInfeasible

ROOT=Path(__file__).resolve().parents[1]
REL=Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
OUT=ROOT/'results'/REL/'w57/p'
PARAM=json.loads((ROOT/'experiments/tm02/r2-multihistory.json').read_text())['reference_material']['parameters']
CFG=json.loads((ROOT/'experiments/terrain_memory/s1-w56-f-r60.json').read_text())


def solve(qp):
    op=osqp.OSQP()
    op.setup(P=sparse.triu(qp['P'],format='csc'),q=qp['q'],A=qp['A'],l=qp['lower'],u=qp['upper'],
             verbose=False,eps_abs=CFG['qp']['eps_abs'],eps_rel=CFG['qp']['eps_rel'],
             max_iter=CFG['qp']['max_iter'],polishing=True,adaptive_rho=True,
             adaptive_rho_interval=50,check_termination=25,warm_starting=True)
    sol=op.solve(raise_error=False)
    if sol.info.status_val not in (1,2):
        raise RuntimeError(sol.info.status)
    return np.asarray(sol.x),dict(status=sol.info.status,primal_residual=float(sol.info.prim_res))


class Saved:
    def __init__(self,side,estimator='M0'):
        self.root=next((ROOT.parent/'MCP-RL-w56-r60'/'outputs'/REL).glob(f'*-{side}-r60-*-E30-attempt01'))
        self.raw=self.root/'w55-f'
        self.events=[json.loads(x) for x in (self.root/'material-events.jsonl').open()]
        self.loads={r['event_id']:r for r in self.events if r['type']!='release'}
        with np.load(self.root/'trajectory.npz') as tr:
            phase,time_=tr['phase'],tr['time']
            self.outbound_end=float(time_[np.flatnonzero(phase==2)[0]])
        self.estimator=estimator
        self.obs=W57MaterialObserver(PARAM,ROOT/'work/assets/B_random1000--ideal-potentials.npz',estimator)
        self.tick=0

    @lru_cache(maxsize=2)
    def material(self,i):
        with np.load(self.root/f'material-{i:04d}.npz') as z:return z['foot']

    @lru_cache(maxsize=2)
    def control(self,i):
        with np.load(self.raw/f'control-{i+1:05d}.npz') as z:
            return {k:z[k] for k in ['time','plan_call','observed_contact','measured_grf']}

    def state(self,t):
        end=int(round(t/.0002))
        for a in range(self.tick,end,10):
            rows=self.material(a//5000)[a%5000:a%5000+10]
            for row in rows:
                self.obs.completed(row[0],row[1],row[6:18].reshape(4,3),row[18:30].reshape(4,3),
                                   row[32:42:3])
            self.obs.observe(rows[-1,1],None,None)
        self.tick=end
        if abs(self.obs.time-t)>1.e-8:raise ValueError('Source/model publication clocks differ')

    def groups(self):
        families={}
        for ev in self.events:
            if ev['time_s']>self.outbound_end:break
            if ev['type']=='release':
                prior=self.loads[ev['previous_load_event_id']]
                if ev['time_s']-prior['time_s'] < .05:continue
                tag='loaded_then_release'
            elif ev['type']=='load':tag='first_loading'
            elif ev['cross_leg'] and ev.get('unloaded_gap_s') is not None:tag='cross_foot_reload'
            else:continue
            key=(tag,ev['leg'])
            if key not in families:families[key]=ev
        by_plan=defaultdict(list)
        for (tag,leg),ev in families.items():
            k=int(np.floor((ev['time_s']-.040)/.020+1.e-8))*10
            pid=int(self.control(k//500)['plan_call'][k%500])
            by_plan[pid].append(dict(family=tag,leg=leg,target_time_s=ev['time_s'],event_id=ev['event_id'],
                                     world_key=ev['world_key'] if 'world_key' in ev else self.loads[ev['previous_load_event_id']]['world_key']))
        return sorted(by_plan.items())

    def force_after(self,fc,pred,j,ts):
        # Independent held-out load: one forward step under the REAL subsequent boundary.
        # A new footprint can include cells outside the query union; report coverage and exclude
        # that leg when any non-virgin earlier state would otherwise be silently missing.
        g=self.obs.ground
        next_tick=round(ts/.0002)
        r=self.material(next_tick//5000)[next_tick%5000]
        true_force=r[32:42:3]
        branch=g.clone()
        # Forecast active world keys are advanced to this exact node. Restrict each target
        # to the known simulated world-cell union and independent future virgin keys.
        node_state=pred['states'][j]
        for points,active in zip(fc.xy,fc.active):
            for xy in points[active]:
                branch.footprint(xy)
        for k,x in zip(fc.keys,node_state):
            i=branch.ids[k];branch.e[i]=x[:2];branch.q[i]=x[2:4];branch.stress[i]=x[4:]
        cells=set(fc.keys)
        bottom=r[6:18].reshape(4,3)
        valid=[]
        for leg,p in enumerate(bottom):
            from s1_w57_planning import footprint
            keys=set(footprint(g,p[:2]))
            valid.append(bool(keys) and all(k in cells or k not in g.ids or not g.ever_loaded[g.ids[k]] for k in keys))
        # Ignore unrelated columns; truth is used only after the forecast endpoint.
        actual,_,_=branch.step(bottom,r[18:30].reshape(4,3),.0002,float(r[1]),log=False)
        return (actual[:,2]-true_force)[valid].tolist(),list(map(bool,valid))

    def evaluate(self,number,events):
        with np.load(self.raw/f'mpc-{number:06d}.npz') as z:p=dict(z)
        times=p['times']; t=float(times[0]);self.state(t)
        measured=self.control(round(t/.002)//500)['observed_contact'][round(t/.002)%500]
        fc=Forecast(self.obs.ground,times,times,p['foot_center_plan'][:-1,:,:2],p['contact'],
                    self.obs.published_bottom,measured)
        source=dict(attempt=self.root.name,mpc=number,publication_s=t,observer_windows=self.obs.windows,
                    archived_events=events,cells=len(fc.keys),protected=int(fc.protected.sum()),
                    resettable=int(fc.reset.sum()),initial_max_detached_q=float(np.abs(fc.initial[fc.reset,2:4]).max(initial=0)),
                    elapsed_query_target_s=[e['target_time_s']-t for e in events])
        if not len(fc.keys):return dict(**source,unavailable='No planned soft contact within original horizon')
        inputs={k:p[k] for k in ('x0','reference','last_force','contact','feet','force_normal_upper','stage_A','stage_B','stage_c')}
        output={};raw={}
        for h in (.025,.005,.002,.0002):
            for history in ('full','private_reset'):
                label=f'{h:g}-{history}'
                if history=='private_reset' and not fc.reset.any():
                    raw[label]=raw[f'{h:g}-full'];output[label]=dict(reused=f'{h:g}-full',reason='empty_intervention')
                    continue
                tic,core=time.perf_counter(),time.process_time()
                try:
                    result=rank(fc,inputs,p['inertia'],CFG,p['forces'],solve,h,history,True)
                except StanceReferenceInfeasible as e:
                    output[label]=dict(infeasible=e.detail); raw[label]=None;continue
                pred=result['predictions'][0]
                future_errors=[];valid=[]
                for j,time_ in enumerate(times[1:]):
                    err,ok=self.force_after(fc,pred,j,float(time_));future_errors.extend(err);valid.append(ok)
                next_z=[]
                for j,time_ in enumerate(times[1:]):
                    n=round(time_/.0002)-1;r=self.material(n//5000)[n%5000]
                    actual=r[6:18].reshape(4,3)[:,2]
                    keep=fc.active[j] & (r[32:42:3]>1)
                    next_z.extend((pred['z'][j]-actual)[keep].tolist())
                raw[label]=result
                output[label]=dict(winner=result['winner'],common_objective=result['cost'],
                    max_first_force_change_N=float(np.abs(result['corrected'][0]-p['forces'][0]).max()),
                    max_inverse_residual_N=float(pred['inverse_residual'].max()),
                    withheld_boundary_force_error_N=future_errors,withheld_valid_legs=valid,
                    planned_sole_minus_measured_when_loaded_m=next_z,
                    cpu_s=time.process_time()-core,wall_s=time.perf_counter()-tic,substeps=pred['steps'])
                print(f'P saved {self.root.name} plan {number} {label} {output[label]["wall_s"]:.3f}s',flush=True)
        def difference(a,b):
            if a is None or b is None:return None
            pa,pb=a['predictions'][0],b['predictions'][0]
            return dict(sole_m=float(np.max(np.abs(pa['z']-pb['z']))),
                        tangent_N_m=float(np.max(np.abs(pa['tangent']-pb['tangent']))),
                        e=float(np.max(np.abs(pa['states'][:,:,:2]-pb['states'][:,:,:2]))),
                        q=float(np.max(np.abs(pa['states'][:,:,2:4]-pb['states'][:,:,2:4]))),
                        stress_MPa=float(np.max(np.abs(pa['states'][:,:,4:]-pb['states'][:,:,4:]))),
                        first_force_N=float(np.max(np.abs(a['corrected'][0]-b['corrected'][0]))),
                        corrected_common_cost=float(abs(a['cost'][1]-b['cost'][1])),
                        winner_changed=bool(a['winner']!=b['winner']))
        comparison={f'{h:g}':difference(raw[f'{h:g}-full'],raw['0.0002-full']) for h in (.025,.005,.002)}
        effect={f'{h:g}':difference(raw[f'{h:g}-full'],raw[f'{h:g}-private_reset']) for h in (.025,.005,.002,.0002)}
        return dict(**source,integrations=output,integration_vs_fine=comparison,full_vs_private_reset=effect)


if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('side',choices=['a','b']);a=arg.parse_args()
    saved=Saved(a.side);OUT.mkdir(exist_ok=True,parents=True);groups=saved.groups()
    for n,family in groups:
        target=OUT/f'integrator-{a.side}-{n:06}.json'
        if target.exists():continue
        print('source',n,family,flush=True)
        row=saved.evaluate(n,family)
        temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');temporary.replace(target)
