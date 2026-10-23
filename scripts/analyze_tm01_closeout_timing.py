#!/usr/bin/env python3
"""Observed latency distributions for the three newly executed R1 Fig. 3 runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', default='r1-closeout-20261022-fig3-isolated')
a = p.parse_args()
R = ROOT/'results/terrain_memory/reproduction/pympc'/a.run
METHODS = ['acados', 'naive', 'mppi']
COLORS = ['#0072B2', '#E69F00', '#009E73']
STYLES = ['-', '--', '-.']
COMPONENTS = [
    ('optimizer_seconds', 'Synchronized optimizer', 10.),
    ('solve_seconds', 'MPC interface', 10.),
    ('control_seconds', 'Controller through executed torque', 2.),
    ('physics_seconds', 'Physics / environment step', 2.),
    ('control_and_physics_seconds', 'Controller + environment step', 2.),
    ('closed_loop_seconds', 'Online step including diagnostics', 2.),
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})


def write_csv(name, rows):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


rows, first, sources, curves = [], [], [], {}
for method in METHODS:
    folder=R/'fig3'/method
    meta=json.loads((folder/'run.json').read_text())
    episodes=list(csv.DictReader((folder/'episodes.csv').open()))
    assert len(episodes)==1 and episodes[0]['seed']=='0'
    episode=episodes[0]
    trace=ROOT/episode['trajectory']
    assert hashlib.sha256(trace.read_bytes()).hexdigest()==episode['sha256']
    with np.load(trace) as archive:
        data={key:archive[key] for key,_,_ in COMPONENTS}
        time=archive['time']; compile_mask=archive['compile_included'].astype(bool)
    mpc=np.isfinite(data['solve_seconds'])
    steady=~compile_mask
    first_tick=int(np.flatnonzero(mpc)[0])
    first.append(dict(method=method,time_s=float(time[first_tick]),
        compile_included=bool(compile_mask[first_tick]),
        **{key.replace('_seconds','_ms'):float(value[first_tick]*1000) for key,value in data.items()}))
    sources.append(dict(method=method,source_run=a.run,trajectory=episode['trajectory'],sha256=episode['sha256'],
        completed=bool(int(episode['success'])),duration_s=float(episode['duration_s']),
        run_record=str((folder/'run.json').relative_to(ROOT)),source_snapshot=meta['source_snapshot'],
        cpu=meta['cpu'],device=meta['device'],timing_context=meta['timing_context'],
        mpc_calls=int(mpc.sum()),recorded_steps=len(time),excluded_compile_steps=int(compile_mask.sum())))
    for field,label,budget in COMPONENTS:
        selections=[('mpc_calls',steady & mpc)] if field in ('optimizer_seconds','solve_seconds') else [
            ('all_steps',steady),('mpc_ticks',steady & mpc),('non_mpc_ticks',steady & ~mpc)]
        for population,mask in selections:
            values=1000*data[field][mask]
            values=values[np.isfinite(values)]
            rows.append(dict(source_run=a.run,method=method,component=field,population=population,
                count=len(values),median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),
                p99_ms=float(np.quantile(values,.99)),max_ms=float(values.max()),mean_ms=float(values.mean()),
                budget_ms=budget,over_budget_count=int((values>budget).sum()),
                over_budget_percent=float(100*np.mean(values>budget)),
                budget_meaning='100 Hz MPC update period' if budget==10 else 'Entire 500 Hz simulation-step target; components share this budget'))
            if population in ('mpc_calls','all_steps'):
                curves[method,field]=values

write_csv('latency_summary.csv',rows)
write_csv('first_mpc_calls.csv',first)
fig,axes=plt.subplots(3,2,figsize=(7.16,8.1),layout='constrained')
for axis,(field,label,budget) in zip(axes.flat,COMPONENTS):
    for method,color,style in zip(METHODS,COLORS,STYLES):
        values=np.sort(curves[method,field])
        axis.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,ls=style,lw=1.1,label=method)
    axis.axvline(budget,color='black',ls=':',lw=.9)
    axis.set(xscale='log',ylim=(0,1.01),xlabel='Observed latency (ms, log scale)',ylabel='Empirical cumulative fraction')
    axis.set_title(label+'\n'+f'{budget:g} ms budget reference',fontsize=8)
    axis.grid(alpha=.2)
axes[0,0].legend(frameon=False,fontsize=8,loc='lower right')
fig.suptitle('New Fig. 3 runs | latency distributions without first JIT call',fontsize=11)
fig.supxlabel('One 60 s episode per method; correlated time samples, not independent repetitions.\n'
    'Optimizer/interface use MPC calls; other components use all steady steps.\n'
    'The 2 ms line is a whole-step target. Simulation sampling does not establish robot real-time control.',fontsize=8)
for ext in ('png','svg'):
    fig.savefig(R/('latency_distributions.'+ext),dpi=300)
plt.close(fig)

index=json.loads((R/'index.json').read_text())
index.update(latency_sources=sources,latency_summary='latency_summary.csv',
    latency_first_calls='first_mpc_calls.csv',latency_figures=['latency_distributions.png','latency_distributions.svg'],
    latency_definition={key:desc for key,desc,_ in COMPONENTS},
    steady_state_exclusion='Only samples explicitly marked compile_included in the new raw trajectories are excluded. The first MPC call remains in first_mpc_calls.csv and all raw records.',
    constructor_and_compile_log='outputs/terrain_memory/reproduction/pympc/r1-closeout-20261022/timing.log',
    statistical_scope='Descriptive quantiles of correlated calls/steps from one 60 s episode per method on this host; no cross-hardware or real-robot deadline claim.',
    analysis_source=dict(path='scripts/analyze_tm01_closeout_timing.py',sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
(R/'index.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps(dict(first_mpc_calls=first,latency_summary=[r for r in rows if r['population'] in ('mpc_calls','all_steps')]),indent=2))
