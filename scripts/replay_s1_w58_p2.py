"""W58 same-publication force factorization and P2 handover development replay."""
import argparse
from functools import lru_cache
import gzip
import json
from pathlib import Path
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from analyze_s1_w57_planner_outcomes import LEGS, records
from qualify_s1_w57_planning import CFG, ROOT, REL, solve
from replay_s1_w57_confirmed_history_actions import Actual, nominal_qp
from s1_w54_f_r52_backend import assemble_force_horizon
from s1_w56_f_r60_backend import FController
from s1_w57_planning import Forecast, original_cost, footprint
from s1_w58_p2 import available_fraction, factor_matrices, decide
from s1_w60_cpu_observer import grow_cells

OUT = ROOT/'results'/REL/'w58/p2'
DONOR = ROOT.parent/'MCP-RL-w57'
ATTEMPT = 'w57-m1-p-a-original-e30-a01'
RAW = DONOR/'outputs'/REL/ATTEMPT
WORK = ROOT/'work/w58-p2'


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/name
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def factors():
    # First actual material winner plus the first publication per independently
    # bound cross-foot load from the entire old A episode. All eleven are included.
    physical = DONOR/'results'/REL/'w57/p/physical'
    source = json.loads((physical/(ATTEMPT+'-factorized-planning.json')).read_text())
    calls = {r['call'] for r in source['observed']}
    record = {r['call']: r for r in records(RAW/'w55-f/material-candidates.jsonl') if r['call'] in calls}
    out = []
    for k in sorted(calls):
        r = record[k]
        t = r['time_s']
        tick = round(t/.002)
        with np.load(RAW/'w55-f'/r['data']) as z:
            src = dict(z)
        with np.load(RAW/'w55-f'/f'mpc-{k:06d}.npz') as z:
            plan = dict(z)
        with np.load(RAW/'w55-f'/f'control-{tick//500+1:05d}.npz') as z:
            row = {name: z[name][tick%500] for name in ('mode', 'contact_force_point', 'observed_contact')}
        saved = Actual(RAW,'M1')
        g = saved.obs.ground
        clock = round(t/.0002)-1
        bottom = saved.material(clock//5000)[clock%5000,6:18].reshape(4,3)
        keys = [tuple(map(int, key)) for key in src['world_keys']]
        grow_cells(g, keys)
        g.e[:], g.q[:], g.stress[:] = np.hsplit(src['initial'],3)
        fc = Forecast(g, plan['times'], src['segment_grid_s'], src['xy'], src['active'], bottom,
                      row['observed_contact'], plan['foot_center_plan'][:-1,:,:2])
        if fc.keys != keys:
            raise ValueError(f'Archived W57 world keys differ at call {k}')
        fc.backend = 'equal_columns'
        fc.reset = src['detached_history']
        fc.protected = src['protected']
        inputs, f0, _ = nominal_qp(plan,row,src)
        share = available_fraction(g, plan['times'], src['segment_grid_s'], fc.active,
                                   inputs['contact'], fc.node_xy)
        def Bs(force, history):
            pred = fc.propagate(force,.0002,history)
            return factor_matrices(inputs, share, fc.linearization(inputs,plan['inertia'],pred)['stage_B'])
        bs = Bs(f0,'full')
        candidates = {'nominal': f0}
        for name, matrix in bs.items():
            if name != 'nominal':
                z,_ = solve(assemble_force_horizon(dict(inputs,stage_B=matrix),CFG))
                candidates[name] = z[inputs['reference'].size:].reshape(f0.shape)
        results = {}
        for history in ('full','private_reset'):
            # Candidate force vectors are fixed across all eight scoring conditions.
            models = {name: Bs(f, history) for name,f in candidates.items()}
            results[history] = {}
            for factor in bs:
                scores = {n: original_cost(dict(inputs,stage_B=models[n][factor]), CFG, f)[0]
                          for n,f in candidates.items()}
                chosen = 'nominal'
                for n,v in scores.items():
                    if v < scores[chosen] - 1e-12*max(abs(v),abs(scores[chosen]),1.):
                        chosen = n
                results[history][factor] = dict(scores=scores, winner=chosen)
        out.append(dict(call=k, time_s=t, detached=int(fc.reset.sum()), protected=int(fc.protected.sum()),
                        first_force_vs_nominal_N={n:float(abs(f[0]-f0[0]).max()) for n,f in candidates.items()},
                        candidate_scoring=results))
    write('common-force-candidates.json',dict(attempt=ATTEMPT,selection=source['selection'],
          prescribed_step_s=.0002,native_workers=1,objective='unchanged centroidal cost',
          candidate_names=list(candidates),rows=out))


def handovers():
    saved = Actual(RAW, 'M1')
    model = mujoco.MjModel.from_binary_path(str(RAW/'model.mjb'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
    mujoco.mj_forward(model,data)
    work = WORK/'reconstructed-controller'
    work.mkdir(parents=True,exist_ok=True)
    cfg = dict(CFG, w57=dict(planner_material_workers=1))
    c = FController(model,data,[model.geom(l).id for l in LEGS],model.body('trunk').id,work,cfg)
    with np.load(RAW/'trajectory.npz') as z:
        tr = {key:z[key] for key in ('qpos','qvel','command_velocity','command_omega')}
    @lru_cache(maxsize=2)
    def control(chunk):
        names = ('plan_call','measured_grf','observed_contact','confirmed_contact','load_bearing_contact','event_mode')
        with np.load(RAW/'w55-f'/f'control-{chunk+1:05d}.npz') as z:
            return {key:z[key] for key in names}
    OUT.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    with gzip.open(OUT/'saved-requests.jsonl.gz','wt') as out:
        for q in records(RAW/'w55-f/handover-requests.jsonl'):
            if q['stage']!='request':
                continue
            t = q['sample_time_s']
            tick = round(t/.002)
            committed = [r for r in q['plan_publication'] if r['source']=='committed_swing']
            if not committed or not any(footprint(saved.obs.ground,np.array(r['xy'])) for r in committed):
                out.write(json.dumps(dict(decision_id=q['decision_id'], time_s=t,
                    reason='no_committed_material_receiver', selected=0, selected_wait_s=0.))+'\n')
                continue
            saved.state(t)
            idx = tick%500
            row = control(tick//500)
            grf = row['measured_grf'][idx]
            c.measured = row['observed_contact'][idx]
            c.confirmed = row['confirmed_contact'][idx]
            c.event_mode = row['event_mode'][idx].copy()
            c.load_bearing = row['load_bearing_contact'][idx]
            c.moving=True
            c.next_liftoff = q['nominal_due_s']
            c.next_pair = np.array(q['planned_pair'])
            c.request_count = q['decision_id']-1
            c.transfer_start = None
            c.pending_touch = [None]*4
            c.swing = [None]*4
            data.qpos[:]=tr['qpos'][tick]
            data.qvel[:]=tr['qvel'][tick]
            data.time=t
            mujoco.mj_forward(model,data)
            c.model_data_root = data.xpos[c.base].copy()
            c.current_material_info=dict(material_observer=saved.obs,force_boundary_bottom=saved.obs.published_bottom)
            c.current_soft=np.array([saved.obs.ground.interior(p[:2]) for p in saved.obs.published_bottom])
            mq = c.model_quantities(data)
            # The cap accepted at this WBC boundary is saved with the real physical command.
            mission = SimpleNamespace(v=tr['command_velocity'][tick],w=tr['command_omega'][tick])
            for item in committed:
                leg=item['leg']
                c._start_swing(t,leg,data,mq,mission,c.current_soft,
                               dict(force=grf,stiffness_N_m=cfg['material_stiffness_N_m']))
                c.pending_touch[leg]=item['event_id']
                if np.max(abs(c.swing[leg]['landing'][:2]-item['xy']))>1e-8:
                    raise ValueError(f'Committed target XY differs at {t}, leg {leg}')
            prev=tick-1
            pid=control(prev//500)['plan_call'][prev%500]
            with np.load(RAW/'w55-f'/f'mpc-{pid:06d}.npz') as z:
                c.last_plan={k:z[k] for k in ('times','forces')}
            result=decide(c,t,mission,mq,saved.obs,grf,'P2-full')
            finer=decide(c,t,mission,mq,saved.obs,grf,'P2-full',.0001)
            result['refined'] = {h:finer[h] for h in ('full','private_reset')}
            out.write(json.dumps(result, allow_nan=False)+'\n')
            out.flush()
            print(t, result['released_reset_cells'], result['full']['winner'], result['private_reset']['winner'],flush=True)
    print('handovers_elapsed_s',time.perf_counter()-start,flush=True)


if __name__=='__main__':
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument('mode',choices=['factors','handovers'])
    {'factors':factors,'handovers':handovers}[cmd.parse_args().mode]()
