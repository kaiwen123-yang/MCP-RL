"""Reconstruct each actual published r60 gait, then integrate on the exact forecast switch clock."""
import argparse
from functools import lru_cache
import json
from types import SimpleNamespace

import mujoco
import numpy as np

from qualify_s1_w57_planning import CFG,OUT,ROOT,Saved,solve
from s1_w56_f_r60_backend import FController
from s1_w57_planning import Forecast,rank

LEGS=('FL','FR','RL','RR')


def run(side,annotate=False):
    saved=Saved(side,'M1')
    with np.load(saved.root/'trajectory.npz') as z:
        tr={k:z[k] for k in ('qpos','qvel','command_velocity','command_omega')}
    events=[json.loads(line) for line in (saved.raw/'contact-events.jsonl').open()]
    lift=[e for e in events if e['kind']=='liftoff_committed']
    model=mujoco.MjModel.from_binary_path(str(saved.root/'model.mjb'));data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id);mujoco.mj_forward(model,data)
    label='response-masks' if annotate else 'contact-split'
    folder=ROOT/'work/w57-p'/f'{label}-{side}';folder.mkdir(exist_ok=True,parents=True)
    c=FController(model,data,[model.geom(l).id for l in LEGS],model.body('trunk').id,folder,CFG)
    @lru_cache(maxsize=3)
    def block(i):
        with np.load(saved.raw/f'control-{i+1:05d}.npz') as z:
            return {k:z[k] for k in ('time','confirmed_contact','load_bearing_contact','observed_contact',
                                      'event_mode','measured_grf','transfer_fraction','transfer_support_pair')}
    def config(tick):
        mrow=saved.material(tick//500)[(tick%500)*10]
        c.current_soft=mrow[2:6].astype(bool)
        c.current_material_info=dict(force_boundary_bottom=mrow[6:18].reshape(4,3),material_observer=saved.obs)
        data.qpos[:]=tr['qpos'][tick];data.qvel[:]=tr['qvel'][tick];data.time=tick*.002
        mujoco.mj_forward(model,data)
        c.model_data_root=data.xpos[c.base].copy()
        mission=SimpleNamespace(v=tr['command_velocity'][tick],w=tr['command_omega'][tick],phase='outbound')
        return c.model_quantities(data),mission
    outfile=OUT/f'contact-splits-{side}-M1.jsonl'
    if annotate: outfile=OUT/f'withheld-boundaries-{side}-M1.jsonl'
    with outfile.open('x') as stream:
        for n,family in saved.groups():
            with np.load(saved.raw/f'mpc-{n:06d}.npz') as z:p=dict(z)
            ts=p['times'];t=float(ts[0]);tick=round(t/.002);saved.state(t)
            previous=[e for e in lift if e['time_s']<=t+1.e-8]
            latest=previous[-1];lo=round(latest['time_s']/.002)
            c.swing=[None]*4
            for leg in range(4):
                ev=next((e for e in reversed(previous) if LEGS[leg] in e['legs']),None)
                if ev is None:continue
                k=round(ev['time_s']/.002);mq,mission=config(k)
                c._start_swing(ev['time_s'],leg,data,mq,mission,c.current_soft,
                    dict(force=block(k//500)['measured_grf'][k%500],stiffness_N_m=CFG['material_stiffness_N_m']))
            mq,mission=config(tick);row=block(tick//500);j=tick%500
            c.confirmed=row['confirmed_contact'][j];c.load_bearing=row['load_bearing_contact'][j]
            c.measured=row['observed_contact'][j];c.event_mode=row['event_mode'][j].copy()
            c.next_pair=np.setdiff1d(np.arange(4),row['transfer_support_pair'][j])
            c.next_liftoff=latest['time_s']+.25;c.last_liftoff=latest['time_s'];c.moving=True
            c.transfer_start=None
            for old in range(lo+1,tick+1):
                prev=block(old//500);i=old%500
                if np.all(prev['confirmed_contact'][i]) and np.all(prev['event_mode'][i]=='stance'):
                    c.transfer_start=max(float(prev['time'][i]),c.next_liftoff-.05);break
            ct,feet=c._future_contact_feet(t,ts,mq,mission)
            mismatch=int(np.count_nonzero(ct[:-1]!=p['contact']))
            maxxy=float(np.max(np.abs(feet[:,:,:2]-p['foot_center_plan'][:,:,:2])))
            evset=c._forecast_events(t,float(ts[-1]),mq,mission)
            cuts=sorted({v for slots in evset for a,b,_ in slots for v in (a,b)
                         if t<v<ts[-1] and np.min(np.abs(ts-v))>1.e-8})
            grid=np.sort(np.r_[ts,cuts]);mid=.5*(grid[1:]+grid[:-1])
            modes,xy=c._future_contact_feet(t,mid,mq,mission)
            fc=Forecast(saved.obs.ground,ts,grid,xy[:,:,:2],modes,saved.obs.published_bottom,
                        c.measured,feet[:-1,:,:2]);fc.workers=4;fc.backend='equal_columns'
            row=dict(attempt=saved.root.name,estimator='M1',mpc=n,t_s=t,source_events=family,
                     original_contact_node_mismatches=mismatch,max_archived_xy_difference_m=maxxy,
                     exact_planned_switches_s=cuts,intervals=len(grid)-1,world_cells=len(fc.keys),
                     detached_keys=int(fc.reset.sum()))
            if annotate:
                from s1_w57_planning import footprint
                allcells=set(fc.keys);g=saved.obs.ground;observed=[]
                for time_ in ts[1:]:
                    k=round(time_/.0002);r=saved.material(k//5000)[k%5000]
                    for leg,bottom in enumerate(r[6:18].reshape(4,3)):
                        target_keys=set(footprint(g,bottom[:2]))
                        valid=bool(target_keys) and all(key in allcells or key not in g.ids or
                          not g.ever_loaded[g.ids[key]] for key in target_keys)
                        if valid:
                            observed.append(dict(leg=leg,measured_force_N=float(r[32+3*leg]),
                                              boundary_time_s=float(r[0]),state_time_s=float(r[1])))
                row.update(withheld=observed)
            elif mismatch or maxxy>1.e-8:
                row['unavailable']='Reconstructed schedule disagrees with saved original plan'
            elif not fc.keys:
                row['unavailable']='No soft footprint in this event schedule'
            else:
                vals={}
                for h in (.025,.002,.0002):
                    for history in ('full','private_reset'):
                        if history=='private_reset' and not np.any(fc.reset):
                            vals[h,history]=vals[h,'full'];continue
                        vals[h,history]=rank(fc,p,p['inertia'],CFG,p['forces'],solve,h,history,True)
                def comp(x,y):
                    a=x['predictions'][0];b=y['predictions'][0]
                    return dict(sole_m=float(np.max(np.abs(a['z']-b['z']))),
                                tangent_N_m=float(np.max(np.abs(a['tangent']-b['tangent']))),
                                q=float(np.max(np.abs(a['states'][:,:,2:4]-b['states'][:,:,2:4]))),
                                e=float(np.max(np.abs(a['states'][:,:,:2]-b['states'][:,:,:2]))),
                                first_F_N=float(np.max(np.abs(x['corrected'][0]-y['corrected'][0]))),
                                winners=[x['winner'],y['winner']])
                row['coarse_vs_plant_step']={str(h):comp(vals[h,'full'],vals[.0002,'full']) for h in (.025,.002)}
                row['history_at_plant_step']=comp(vals[.0002,'full'],vals[.0002,'private_reset'])
                row['costs']={f'{h} {history}':v['cost'] for (h,history),v in vals.items()}
                # Known first measured boundary after each original 25ms node, not the
                # imposed candidate load. Align the private integrator endpoint explicitly.
                ix=[int(np.argmin(abs(grid[1:]-end))) for end in ts[1:]]
                row['subsequent_force_errors_N']={}
                for h in (.025,.002,.0002):
                    err=[]
                    for end,k in zip(ts[1:],ix):
                        diff,_=saved.force_after(fc,vals[h,'full']['predictions'][0],k,float(end));err.extend(diff)
                    row['subsequent_force_errors_N'][str(h)]=err
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            print(n,mismatch,maxxy,len(cuts),row.get('unavailable','evaluated'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['a','b']);p.add_argument('--associate-withheld-boundaries',action='store_true')
    a=p.parse_args();run(a.case,a.associate_withheld_boundaries)
