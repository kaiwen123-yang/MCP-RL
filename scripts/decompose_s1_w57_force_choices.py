"""On bound physical P/A events, separate material height from within-node contact availability."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np

from analyze_s1_w57_planner_outcomes import ROOT,REL
from qualify_s1_w57_planning import CFG,PARAM,solve,Saved
from replay_s1_w57_confirmed_history_actions import Actual,nominal_qp
from replay_s1_w57_force_execution import replay_wbc
from s1_w54_f_r52_backend import assemble_force_horizon
from s1_w57_planning import Forecast,footprint,original_cost
from s1_w60_cpu_observer import grow_cells


def run(attempt):
    r=ROOT/'results'/REL/'w57';raw=ROOT/'outputs'/REL/attempt;folder=raw/'w55-f'
    receipt=json.loads((r/'attempts'/(attempt+'.json')).read_text())
    physical=json.loads((r/'p/physical'/(attempt+'.json')).read_text())
    selected=json.loads((r/'p/physical'/(attempt+'-first-selected.json')).read_text())
    calls={p['call'] for p in physical['first_matching_publication_per_true_crossfoot_target']}|{selected['call']}
    entries={e['call']:e for e in (json.loads(l) for l in (folder/'material-candidates.jsonl').open()) if e['call'] in calls}
    output=r/'p/physical'/(attempt+'-factorized-planning.json')
    if output.exists():return json.loads(output.read_text())
    results=[]
    for call in sorted(calls):
        record=entries[call];t=record['time_s'];tick=round(t/.002);start=time.perf_counter()
        with np.load(folder/record['data']) as z:src=dict(z)
        with np.load(folder/f'mpc-{call:06d}.npz') as z:plan=dict(z)
        with np.load(folder/f'control-{tick//500+1:05d}.npz') as z:row={k:z[k][tick%500] for k in z.files}
        saved=Actual(raw,receipt['material'])
        clock=round(t/.0002)-1;bottom=saved.material(clock//5000)[clock%5000,6:18].reshape(4,3).copy()
        saved.obs.time=t;saved.obs.packet_start=t-.002;saved.obs.packet_last=t-.0002;saved.obs.published_bottom=bottom
        g=saved.obs.ground;keys=[tuple(int(i) for i in k) for k in src['world_keys']]
        grow_cells(g,keys)
        g.e[:]=src['initial'][:,:2];g.q[:]=src['initial'][:,2:4];g.stress[:]=src['initial'][:,4:]
        needed={k for leg in np.flatnonzero(row['mode']==3) for k in footprint(g,bottom[leg,:2])}
        missing=sorted(needed-set(keys))
        fc=Forecast(g,plan['times'],src['segment_grid_s'],src['xy'],src['active'],bottom,row['observed_contact'],plan['foot_center_plan'][:-1,:,:2])
        if fc.keys!=keys:raise ValueError('Different physical candidate keys')
        fc.workers=4;fc.backend='equal_columns'
        input0,force0,state0=nominal_qp(plan,row,src)
        pred=fc.propagate(force0,.0002)
        lin=fc.linearization(input0,plan['inertia'],pred)
        score0,_,_=original_cost(lin,CFG,force0)
        duration=np.zeros(input0['contact'].shape)
        np.add.at(duration,fc.segment,fc.active*fc.dt[:,None])
        share=np.ones_like(duration)
        for j,dt in enumerate(np.diff(plan['times'])):
            for leg in np.flatnonzero(input0['contact'][j]):
                if duration[j,leg]>0 or g.interior(fc.node_xy[j,leg]):
                    share[j,leg]=min(1.,duration[j,leg]/dt)
        factor=np.repeat(share,3,axis=1)[:,None,:]
        B0=input0['stage_B'];Bf=lin['stage_B']
        Bh=np.divide(Bf,factor,out=B0.copy(),where=factor>0.)
        Bc=B0*factor
        work=ROOT/'work/w57-p/factorized'/attempt/f'{call:06d}'
        u0=None if missing else replay_wbc(saved,plan,state0,force0,t,row,work/'nominal')[0]
        arms={}
        for name,B in [('height_only',Bh),('availability_only',Bc),('full',Bf)]:
            z,_=solve(assemble_force_horizon(dict(input0,stage_B=B),CFG));nx=input0['reference'].size
            x=z[:nx].reshape(state0.shape);f=z[nx:].reshape(force0.shape)
            test=fc.propagate(f,.0002);evaluation=fc.linearization(input0,plan['inertia'],test)
            score,_,_=original_cost(evaluation,CFG,f)
            winner=score<score0-1e-12*max(1.,abs(score0),abs(score))
            applied=f if winner else force0
            if missing:mapped=None
            elif not winner:mapped=u0
            else:mapped=replay_wbc(saved,plan,x,f,t,row,work/name)[0]
            arms[name]=dict(original_common_cost=score0,candidate_common_cost=score,chosen=bool(winner),
                chosen_first_force_vs_nominal_N=float(np.abs(applied[0]-force0[0]).max()),
                chosen_first_Fz_N=applied[0,:,2].tolist(),
                chosen_first_mapped_vs_nominal_rad=float(np.abs(mapped-u0).max()) if mapped is not None else None,
                chosen_first_mapped_vs_actual_rad=float(np.abs(mapped-row['mapped_u']).max()) if mapped is not None else None)
        results.append(dict(call=call,published_s=t,branch=record['selected'],source='first material winner' if call==selected['call'] else 'first physically matched detached-history target',
            captured_world_cells=len(keys),missing_current_WBC_footprint_keys=missing,
            partial_support_nodes=int(np.count_nonzero((share>0)&(share<1-1e-10))),
            maximum_B_factorization_residual=float(np.max(np.abs(Bh*factor-Bf))),
            arms=arms,wall_s=time.perf_counter()-start))
    result=dict(attempt=attempt,selection='First actual material winner, plus first matched publication for every observed detached cross-foot target. Fixed case A, no trajectory selection on score.',
      interpretation='Only the candidate QP generator is factored. All alternatives use the full identical P1 constitutive+centroidal evaluator, original costs, hard constraints, unchanged M1 WBC and the original event clock. Numerical component differences are same-state intervention effects, not separate closed-loop performance. Full world-column state is read directly from the actual publication.',
      observed=results)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt');r=run(p.parse_args().attempt)
    print(r['attempt'],len(r['observed']))
