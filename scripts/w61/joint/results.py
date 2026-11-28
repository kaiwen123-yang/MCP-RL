"""Count completed real queries and score matched training and held-out comparisons."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from w61.joint.plant import CONTRACT,ROOT
from w61.joint.deliver import DATA,COMPACT


def rows():
    cfg=json.loads(CONTRACT.read_text())['study']
    by_arm={};evals={};positive=0;real=0;steps=0;simulated=0.;cpu=0.;wall=0.;matches=0
    for a in cfg['arms']:
        by_seed={}
        for s in cfg['training_seeds']:
            head=DATA/a/str(s);rounds=[]
            for node in sorted(head.glob('round-*.json')):
                u=json.loads(node.read_text());both=u['update_norm_A']>0 and u['update_norm_C']>0
                plus,minus=u['signed']
                qp,qm=[json.loads((DATA/z['result']).read_text()) for z in (plus,minus)]
                plans=[json.loads((DATA/z['result']).with_name('plans.json').read_text()) for z in (plus,minus)]
                n=min(map(len,plans))
                spread=None if not n else max(abs(p['applied']['dx']-m['applied']['dx']) for p,m in zip(*plans))
                rounds.append(dict(round=u['round'],block=u['block'],A_delta_norm=u['update_norm_A'],C_delta_norm=u['update_norm_C'],
                    support_equal=plus['support_digest']==minus['support_digest'],physical_steps_plus_minus=[plus['real_query_steps'],minus['real_query_steps']],
                    objective_plus_minus=[plus['objective'],minus['objective']],calibration_rmse_N_plus_minus=[plus['rmse_N'],minus['rmse_N']],
                    success_plus_minus=[plus['completed'],minus['completed']],reasons=[plus['reason'],minus['reason']],
                    observed_foothold_adjustment_maximum_contrast_m=spread,observed_A_tangent_contrast_N_m=qp['stiffness_N_m']-qm['stiffness_N_m']))
                positive+=int(both);matches+=int(plus['support_digest']==minus['support_digest'])
                for r in (qp,qm):real+=1;steps+=r['query_steps'];simulated+=r['physical_s'];cpu+=r['cpu_seconds'];wall+=r['wall_s']
            ev=head/'evaluation.json'
            if ev.is_file():evals[f'{a}-{s}']=json.loads(ev.read_text())
            by_seed[str(s)]=rounds
        by_arm[a]=by_seed
    trials={}
    for sta in DATA.glob('factorial/*/*/result.json'):
        u=json.loads(sta.read_text());trials[str(sta.parent.relative_to(DATA))]={k:u[k] for k in ('joint_cost','physical_cost','progress_m','completed','reason','query_steps','calibration_rmse_N')}
    comparisons={}
    if len(evals)==len(cfg['training_seeds'])*len(cfg['arms']):
        for a in cfg['arms']:
            data=[u for s in cfg['training_seeds'] for u in evals[f'{a}-{s}']['cases']]
            comparisons[a]=dict(queries=len(data),complete=sum(z['completed'] for z in data),
                mean_real_cost=float(np.mean([z['joint_cost'] for z in data])),mean_force_RMSE_N=float(np.mean([z['calibration_rmse_N'] for z in data])),
                progress_mean_m=float(np.mean([z['progress_m'] for z in data])),actual_query_steps=sum(z['query_steps'] for z in data),
                natural_terminations=dict(Counter(z['reason'].split(' at physical time')[0] for z in data)))
    return dict(runtime=(DATA/'runtime.commit').read_text().strip(),by_arm=by_arm,training_pairs=matches,training_real_queries=real,
                training_real_query_substeps=steps,training_physical_s_including_support=simulated,summed_rollout_cpu_hours=cpu/3600,
                summed_rollout_wall_s=wall,actual_simultaneous_two_head_updates=positive,unseen=evals,matched_unseen=comparisons,
                factorial=trials,originals_verified=len(list((COMPACT/'originals').glob('*.json'))),
                performance_scope='fresh forefoot-entry Go2/MuJoCo task on the world-fixed original 0.2ms reference columns; per-seed and scenario outcomes; no original route or hardware qualification')


if __name__=='__main__':
    print(json.dumps(rows(),ensure_ascii=False))
