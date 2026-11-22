"""Screen P2 on every material-support liftoff of the original r60 A/B outbound trajectories."""
import argparse
from functools import lru_cache
import json
from types import SimpleNamespace
import time

import mujoco
import numpy as np

from qualify_s1_w57_planning import CFG, OUT, ROOT, Saved
from s1_w56_f_r60_backend import FController
from s1_w57_event_planning import compare

LEGS=('FL','FR','RL','RR')


def first_permission(raw, ticks, actual, src, due, mass, cfg):
    c=cfg['contact_events'];old_start=None;new_start=None;different_ramp=False
    support=np.setdiff1d(np.arange(4),src)
    for k in ticks:
        d=raw(k//500);i=k%500;t=float(d['time'][i]);pr=raw((k-1)//500);pi=(k-1)%500
        on_old=abs(t-actual['time_s'])<1.e-8
        if on_old:
            # Post-liftoff event_mode and counters have already been changed in this saved row.
            all_stance=True;confirmed_support=True;bearing=True
        else:
            all_stance=bool(np.all(d['event_mode'][i]=='stance'))
            confirmed_support=bool(np.all(d['confirmed_contact'][i,support]))
            bearing=bool(np.all(d['load_bearing_contact'][i,support]))
        all_ground=(all_stance and (on_old or np.all(d['confirmed_contact'][i])))
        if all_ground and new_start is None:new_start=max(t,due-.05)
        if new_start is None:continue
        z=np.clip((t-new_start)/c['load_transfer_ramp_s'],0.,1.)
        fraction=z*z*z*(10.-15.*z+6.*z*z)
        old_fraction=1. if on_old else d['transfer_fraction'][i]
        if abs(fraction-old_fraction)>1.e-10:different_ramp=True
        if t<due-1.e-9 or fraction<1.-1.e-10:continue
        prev=pr['force'][pi,:,2]
        if (all_stance and confirmed_support and bearing and
            d['measured_grf'][i,support,2].sum() >= c['handover_measured_bodyweight_fraction']*mass*9.81 and
            prev[support].sum()>=c['handover_solved_bodyweight_fraction']*mass*9.81 and
            prev[src].sum()<=c['release_solved_bodyweight_fraction']*mass*9.81 and
            np.linalg.norm(pr['body_acceleration_residual'][pi])<=c['handover_body_accel_residual_m_s2'] and
            np.linalg.norm(pr['angular_acceleration_residual'][pi])<=c['handover_angular_accel_residual_rad_s2']):
            return t,different_ramp
    return None,different_ramp


def run(side,estimator):
    saved=Saved(side,estimator);label=side if estimator=='M0' else f'{side}-{estimator}'
    folder=ROOT/'work/w57-p'/f'long-event-{label}'
    folder.mkdir(parents=True,exist_ok=True)
    with np.load(saved.root/'trajectory.npz') as z:
        tr={k:z[k] for k in ('time','qpos','qvel','command_velocity','phase')}
    raw=saved.raw
    @lru_cache(maxsize=4)
    def control(i):
        keys=('time','plan_call','measured_grf','observed_contact','confirmed_contact','load_bearing_contact',
              'event_mode','force','transfer_fraction','body_acceleration_residual','angular_acceleration_residual')
        with np.load(raw/f'control-{i+1:05d}.npz') as z:return {k:z[k] for k in keys}
    events=[json.loads(q) for q in (raw/'contact-events.jsonl').open()]
    liftoffs=[e for e in events if e['kind']=='liftoff_committed']
    m=mujoco.MjModel.from_binary_path(str(saved.root/'model.mjb'));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,m.key('home').id);mujoco.mj_forward(m,d)
    c=FController(m,d,[m.geom(l).id for l in LEGS],m.body('trunk').id,folder,CFG)
    output=OUT/f'long-event-{label}.jsonl'
    out=output.open('x')
    for ev,successor in zip(liftoffs,liftoffs[1:]):
        t=float(ev['time_s']);tick=round(t/.002)
        if t>=saved.outbound_end:break
        row=control(tick//500);j=tick%500;grf=row['measured_grf'][j]
        saved.state(t)
        pair=[LEGS.index(x) for x in ev['legs']]
        c.next_pair=np.setdiff1d(np.arange(4),pair)
        if not any(grf[i,2]>1 and saved.obs.ground.interior(saved.obs.published_bottom[i,:2]) for i in c.next_pair):
            continue
        d.qpos[:]=tr['qpos'][tick];d.qvel[:]=tr['qvel'][tick];d.time=t;mujoco.mj_forward(m,d)
        c.model_data_root=d.xpos[c.base].copy()
        c.measured=row['observed_contact'][j];c.confirmed=row['confirmed_contact'][j]
        c.load_bearing=row['load_bearing_contact'][j];c.event_mode=row['event_mode'][j].copy()
        c.transfer_start=None;c.moving=True;c.next_liftoff=t+.25
        prev=tick-1;plan=int(control(prev//500)['plan_call'][prev%500])
        with np.load(raw/f'mpc-{plan:06}.npz') as z:c.last_plan={k:z[k] for k in ('times','forces')}
        c.current_material_info=dict(material_observer=saved.obs,force_boundary_bottom=saved.obs.published_bottom)
        c.current_soft=np.array([saved.obs.ground.interior(x[:2]) for x in saved.obs.published_bottom])
        mq=c.model_quantities(d)
        mission=SimpleNamespace(v=tr['command_velocity'][tick,:3])
        for leg in pair:
            c._start_swing(t,leg,d,mq,mission,c.current_soft,
                           dict(force=grf,stiffness_N_m=CFG['material_stiffness_N_m']))
        begin,cpu=time.perf_counter(),time.process_time()
        result=compare(c,t,mission,mq,saved.obs,grf,c.next_liftoff,also_reset=True)
        end=round(successor['time_s']/.002);ts=range(tick+1,end+1)
        gate,change=first_permission(control,ts,successor,c.next_pair,result['requested_due_s'],c.mass,CFG)
        zero,_=first_permission(control,ts,successor,c.next_pair,c.next_liftoff,c.mass,CFG)
        record=dict(attempt=saved.root.name,estimator=estimator,source_event=ev['event'],publication_s=t,material_windows=saved.obs.windows,
                    source_support=c.next_pair.tolist(),most_recent_qp=plan,
                    actual_next_liftoff_s=successor['time_s'],next_liftoff_event=successor['event'],
                    nominal_gate_on_saved_path_s=zero,proposed_first_gate_on_saved_prefix_s=gate,
                    different_ramp_on_saved_prefix=change,
                    changed_handover_on_saved_prefix=(gate is None or abs(gate-successor['time_s'])>.001),
                    wall_s=time.perf_counter()-begin,cpu_s=time.process_time()-cpu,**result)
        out.write(json.dumps(record,allow_nan=False)+'\n');out.flush()
        print(side,ev['event'],t,result['reason'],len(result['common_targets']),result['selected'],
              record['wall_s'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['a','b']);p.add_argument('--estimator',choices=['M0','M1'],default='M0')
    a=p.parse_args();run(a.case,a.estimator)
