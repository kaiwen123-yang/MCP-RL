#!/usr/bin/env python3
"""Redraw TM01 figures from actual full-episode outputs, including all failures."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("--run",default="r1-v4")
p.add_argument("--methods",nargs="+",default=["acados","naive","adaptive"])
p.add_argument("--figures-only", action="store_true")
a=p.parse_args()
R=ROOT/"results/terrain_memory/reproduction/pympc"/a.run
O=ROOT/"outputs/terrain_memory/reproduction/pympc"/a.run
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
 "axes.spines.right":False,"svg.fonttype":"none","savefig.dpi":220})
colors={"acados":"#0072B2","naive":"#E69F00","mppi":"#009E73","adaptive_fixed":"#56B4E9","adaptive":"#CC79A7"}
names={"acados":"acados","naive":"Naive, fixed gait","mppi":"MPPI, fixed gait","adaptive_fixed":"Adaptive implementation, 1.3 Hz","adaptive":"Naive, adaptive gait"}
def save(fig,name):
    fig.savefig(R/(name+".png"),bbox_inches="tight")
    fig.savefig(R/(name+".svg"),bbox_inches="tight")
    plt.close(fig)
def rows(experiment,method):
    with (R/experiment/method/"episodes.csv").open() as f:return list(csv.DictReader(f))
def trace(experiment,method,seed=0):
    return np.load(ROOT / next(x["trajectory"] for x in rows(experiment,method) if int(x["seed"])==seed))
methods=["acados","naive","mppi"]
f,ax=plt.subplots(2,1,figsize=(7.16,6.3),layout="constrained")
solve,errors=[],[]
for method in methods:
    d=trace("fig3",method)
    keep=np.isfinite(d["solve_seconds"]) & ~d["compile_included"]
    solve.append(1000*d["solve_seconds"][keep])
    errors.append(np.linalg.norm(d["velocity"]-d["command"],axis=1))
for axis,data,ylabel in zip(ax,[solve,errors],["MPC solve wall time (ms)","Linear velocity error norm (m/s)"]):
    bp=axis.boxplot(data, tick_labels=[names[m]+"\n"+json.loads((R/"fig3"/m/"run.json").read_text())["device"]+" | "+format(float(rows("fig3",m)[0]["duration_s"]),".3f")+" s" for m in methods],showmeans=True,showfliers=False,
        patch_artist=True,meanprops={"marker":"^","markerfacecolor":"black","markeredgecolor":"black","markersize":4},
        medianprops={"color":"black"})
    for box,m in zip(bp["boxes"],methods):box.set_facecolor(colors[m]);box.set_alpha(.45)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y",alpha=.2)
f.suptitle("TM01 Fig. 3 | 0.5 m/s, +/-5 N / N m")
f.supxlabel("One attempted 60 s episode per method; falls retained; samples are correlated time points.\nBoxes: IQR; line: median; triangle: mean; whiskers: 1.5 IQR. JIT shown only in raw data.",fontsize=8)
save(f,"fig3")
f,ax=plt.subplots(3,1,figsize=(7.16,7.0),sharex=True,layout="constrained")
for m in methods:
    d=trace("fig3",m)
    ax[0].plot(d["time"],d["velocity"][:,0],label=names[m],color=colors[m],lw=.7)
    ax[1].plot(d["time"],np.linalg.norm(d["velocity"]-d["command"],axis=1),color=colors[m],lw=.6)
d=trace("fig3","acados")
ax[0].axhline(.5,color="black",ls="--",lw=.8)
for i,label in enumerate(["Fx","Fy","Fz"]):ax[2].step(d["time"],d["wrench"][:,i],where="post",lw=.8,label=label)
ax[0].legend(ncol=3,fontsize=8);ax[2].legend(ncol=3)
ax[0].set_ylabel("Forward speed (m/s)");ax[1].set_ylabel("Velocity error (m/s)")
ax[2].set_ylabel("External force (N)");ax[2].set_xlabel("Simulation time (s)")
for axis in ax:axis.grid(alpha=.2)
save(f,"fig3_timeseries")
f,ax=plt.subplots(3,1,figsize=(7.16,8),layout="constrained")
for m in ["acados","adaptive"]:
    d=trace("fig4",m);label=names[m]
    ax[0].plot(d["com"][:,0],d["com"][:,1],color=colors[m],label=label,linestyle="-" if m=="acados" else "--")
    push=np.linalg.norm(d["wrench"][:,:3],axis=1)>0
    ax[0].plot(d["com"][push,0],d["com"][push,1],color=colors[m],lw=3)
    ax[1].step(d["time"],d["frequency"],where="post",color=colors[m],label=label,linestyle="-" if m=="acados" else "--")
    ax[2].plot(d["time"],d["velocity"][:,1],color=colors[m],label=label,lw=.8)
    rr=rows("fig4",m)[0]
    if rr["success"]=="0":
        ax[0].plot(d["com"][-1,0],d["com"][-1,1],"x",color=colors[m],ms=9)
        ax[2].plot(d["time"][-1],d["velocity"][-1,1],"x",color=colors[m],ms=9)
ax[0].set(xlabel="CoM x (m)",ylabel="CoM y (m)")
ax[0].legend(fontsize=8);ax[0].set_title("Thick segment: applied 40 N push; x: terminated episode",fontsize=9)
ax[1].set(ylabel="Step frequency (Hz)",ylim=(1.2,2.5),yticks=[1.3,2.0,2.4])
ax[2].axhline(.1,color="black",ls="--",lw=.8,label="Command")
ax[2].set(xlabel="Simulation time (s)",ylabel="Lateral speed (m/s)")
for axis in ax[1:]:
    axis.axvspan(2,5,color="gray",alpha=.15);axis.set_xlim(0,8)
for axis in ax:axis.grid(alpha=.2)
f.suptitle("TM01 Fig. 4 | world-y command 0.1 m/s; 40 N at 2-5 s")
save(f,"fig4")
if a.figures_only:
    raise SystemExit(0)
summary=[]
allrows=[]
for m in a.methods:
    rr=rows("table1",m);allrows+=rr
    n=len(rr);k=sum(int(x["success"]) for x in rr);phat=k/n;z=1.96
    centre=(phat+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
    summary.append(dict(method=m,episodes=n,successes=k,success_percent=100*phat,
        success_ci95_low=100*(centre-half),success_ci95_high=100*(centre+half),
        mean_episode_velocity_error_m_s=float(np.mean([float(x["velocity_error_mean_m_s"]) for x in rr])),
        mean_episode_quadratic_cost=float(np.mean([float(x["quadratic_cost_mean"]) for x in rr])),
        median_duration_s=float(np.median([float(x["duration_s"]) for x in rr])),
        mean_time_without_fall_s=float(np.mean([float(x["duration_s"]) for x in rr])),
        mean_episode_solve_ms=float(np.mean([float(x["solve_mean_ms"]) for x in rr]))))
for filename,data in [("table1.csv",summary),("episodes.csv",allrows)]:
    with (R/filename).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
f,ax=plt.subplots(1,2,figsize=(7.16,3.9),layout="constrained")
for i,s in enumerate(summary):
    rate=s["success_percent"]
    ax[0].errorbar(i,rate,yerr=[[rate-s["success_ci95_low"]],[s["success_ci95_high"]-rate]],
        fmt="o",capsize=5,color=colors[s["method"]])
    ax[0].annotate(f'{s["successes"]}/{s["episodes"]}',(i,rate),xytext=(0,10),textcoords="offset points",ha="center")
    rr=rows("table1",s["method"]);duration=np.array([float(x["duration_s"]) for x in rr])
    jitter=np.linspace(-.13,.13,len(duration))
    ax[1].scatter(i+jitter,duration,c=[colors[s["method"]]],s=12,alpha=.7)
    ax[1].plot(i,s["mean_time_without_fall_s"],"_",color="black",markersize=18,markeredgewidth=2)
ax[0].set(ylabel="Completed episodes (%)",ylim=(-5,110))
ax[1].set(ylabel="Time without a fall (s)",ylim=(0,63))
for axis in ax:
    axis.set_xticks(range(len(a.methods)),["Adaptive\nfixed 1.3 Hz" if m=="adaptive_fixed" else m for m in a.methods]);axis.grid(axis="y",alpha=.2)
f.suptitle("TM01 Table I + frequency ablation | 50 shared seeds" if "adaptive_fixed" in a.methods
           else "TM01 Table I experiment | 50 shared seeds per controller")
f.supxlabel("95% Wilson intervals. Each dot is one of 50 episodes; black bar: mean time without a fall.\n60 s is the fixed horizon. Errors on different surviving intervals are not ranked as full-task performance.",fontsize=8)
save(f,"table1")
# Show the entire declared horizon, so no post-hoc choice of a favorable cutoff is needed.
f,ax=plt.subplots(figsize=(7.16,3.8),layout="constrained")
grid=np.linspace(0,60,601)
survival=[]
for method in a.methods:
    ls={"acados":"-","naive":"--","adaptive_fixed":":","adaptive":"-."}[method]
    rr=rows("table1",method)
    duration=np.array([float(row["duration_s"]) for row in rr])
    completed=np.array([int(row["success"]) for row in rr],dtype=bool)
    alive=np.array([np.mean((duration>t) | ((duration>=t)&completed)) for t in grid])
    ax.step(grid,100*alive,where="post",color=colors[method],ls=ls,label=names[method])
    survival.extend(dict(method=method,time_s=float(t),survival_percent=float(v*100)) for t,v in zip(grid,alive))
ax.set(xlabel="Simulation time (s)",ylabel="Episodes without a fall (%)",xlim=(0,60),ylim=(0,102))
ax.grid(alpha=.2);ax.legend(fontsize=8)
ax.set_title("Complete 60 s evaluation: survival over the full horizon")
save(f,"table1_survival")
with (R/"table1_survival.csv").open("w",newline="") as output:
    writer=csv.DictWriter(output,fieldnames=list(survival[0]));writer.writeheader();writer.writerows(survival)
index=dict(run=a.run,
 included_episode_count=sum(s["episodes"] for s in summary)+5,
 table1_episode_count=sum(s["episodes"] for s in summary),videos="videos.json",
 paper_success_percent={"acados":32,"naive":37,"adaptive":80},
 summary=summary,figures=["fig3.png","fig3.svg","fig3_timeseries.png","fig4.png","fig4.svg","table1.png","table1.csv","table1_survival.png","table1_survival.csv"],
 notes=["Fixed author version method reproduction, not identical paper hardware/model.",
 "All configured random-wrench outcomes retained. Means cover each episode's observed duration, not a common full-task horizon.",
 "The paper's 0.12/0.14/0.08 Mean Cost has no fully specified scalar definition; these metrics are explicitly defined."])
(R/"index.json").write_text(json.dumps(index,indent=2)+"\n")
print(json.dumps(summary,indent=2))
