"""Offline complete-task / compute / factual-prediction tradeoff figure."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("work/go2-compact-20261019/downloaded"))
    parser.add_argument("--output", type=Path, default=Path("notes/go2-compact-2026-10-19"))
    args = parser.parse_args()
    fit = json.loads((args.input/"fit/fit-summary.json").read_text())
    execution = json.loads((args.input/"evaluation/execution.json").read_text())
    cases = {(c["method"], c["condition"]):c for c in execution["cases"]}
    methods = ["id_one_step", "compact_multistep", "kinematic_mpc", "policy_rollout_mpc"]
    names = ["One-step\nID", "Multistep\nID", "Ideal\nMPC", "Exact policy\nrollout"]
    conditions = ["nominal", "push_positive_y", "push_negative_y"]
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    markers = ["o", "^", "s"]
    plt.rcParams.update({"font.size":9, "svg.fonttype":"none", "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2,3,figsize=(11.3,7.1))
    fig.subplots_adjust(left=.075,right=.98,top=.89,bottom=.19,hspace=.48,wspace=.31)
    metrics = [("task_seconds","Task completion time (s)","a  All four methods: 3/3 success"),
               ("task_cost","Navigation cost","b  No uniform task-cost advantage"),
               ("absolute_mechanical_work","Absolute mechanical work (J)","c  Execution-resource tradeoff"),
               ("end_to_end_max_seconds","Maximum macro-step computation (s)","d  Both fitted models below 20 ms")]
    for ax,(field,label,title) in zip(axes.flat,metrics):
        for i,method in enumerate(methods):
            for j,condition in enumerate(conditions):
                ax.scatter(i+(j-1)*.13, cases[(method,condition)][field], color=colors[i],marker=markers[j],s=34)
        ax.set(xticks=range(4),xticklabels=names,ylabel=label,title=title,xlim=(-.4,3.4))
        ax.grid(axis="y",alpha=.18)
        if field == "end_to_end_max_seconds":
            ax.set_yscale("log");ax.set_ylim(.008,1.)
            ax.set_yticks([.01,.1,1]);ax.yaxis.set_minor_formatter(NullFormatter())
            ax.axhline(.2,color="#555555",linestyle=":");ax.text(.05,.23,"200 ms budget",fontsize=8)
        else:
            maximum=max(c[field] for c in cases.values());ax.set_ylim(0,maximum*1.12)
    ax=axes[1,1]
    for i,method in enumerate(methods[:2]):
        horizons=fit["arms"][method]["validation"]["autoregressive_h5"]["per_horizon"]
        ax.plot(np.arange(1,6)*.2,[h["position_rmse_m"]*100 for h in horizons],color=colors[i],marker=markers[i],
                linestyle=["-","--"][i],label=names[i].replace("\n"," "))
    ax.set(xlabel="Prediction horizon (s)",ylabel="Position RMSE (cm)",title="e  Multistep fitting reduces drift",xticks=np.arange(1,6)*.2,ylim=(0,5.5))
    ax.legend(frameon=False);ax.grid(alpha=.18)
    ax=axes[1,2]
    for i,method in enumerate(methods[:2]):
        ax.scatter(i,fit["arms"][method]["fit_wall_seconds"],color=colors[i],s=55,marker=markers[i])
    ax.set(xticks=[0,1],xticklabels=names[:2],xlim=(-.5,1.5),ylim=(0,2.1),ylabel="Model fitting wall time (s)",title="f  Same 800 updates, different cost")
    ax.grid(axis="y",alpha=.18)
    fig.suptitle("Frozen Go2: compression meets the compute budget; better prediction does not guarantee lower task cost",y=.98,fontsize=12)
    fig.text(.075,.045,"Task panels: every fixed condition is shown (circle = nominal, triangle/square = positive/negative push).\n"
             "Panel e: two held-out command-sequence episodes, overlapping windows, no confidence interval. Policy updates = 0.",fontsize=9)
    args.output.mkdir(parents=True,exist_ok=True)
    with warnings.catch_warnings(record=True) as captured:
        fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
        for obj in fig.findobj(matplotlib.text.Text):
            if obj.get_visible() and obj.get_text():
                box=obj.get_window_extent(renderer)
                if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
                    outside.append(obj.get_text())
        fig.savefig(args.output/"compression-task-tradeoff.png",dpi=180)
        fig.savefig(args.output/"compression-task-tradeoff.svg")
    checks={"case_count":len(cases),"success_count":sum(c["success"] for c in cases.values()),
            "missing_glyph_warnings":[str(w.message) for w in captured if "Glyph" in str(w.message)],
            "text_outside_canvas":outside,"error_bars":False}
    (args.output/"figure-check.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(checks)


if __name__=="__main__":main()
