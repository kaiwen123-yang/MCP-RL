"""Execute the prespecified two-by-two on the real plant and check field effects before the formal batch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from hybrid import CACHE, RESULT, ROOT
from s1_w59_learner_critic import atomic_json, file_digest

LEASE = Path(os.environ.get("W61_POOL_RUNNER","/root/autodl-tmp/MCP-RL-w61-joint/scripts/w61/lease.py"))
COLLECT = Path(__file__).with_name("collect.py")
SETTINGS = ["transfer-minus","transfer-plus","hold-minus","hold-plus"]


def admitted_call(name, count, command, ram=None, gpu=None, disk=None):
    invocation=[sys.executable,str(LEASE),"--runtime",str(ROOT),"--job",name,
        "--slots",str(count),"--cores",str(count),"--ram-gib",str(ram or 3.*count),
        "--gpu-mib",str(gpu or 650*count),"--scratch-gib",str(disk or .25*count),"--",*map(str,command)]
    return subprocess.run(invocation).returncode


def development(revision):
    base=CACHE/"development"/revision
    base.mkdir(parents=True,exist_ok=True)
    def compare(variant):
        name=f"w61-e-dev-{revision}-{variant}"
        out=base/name
        cmd=[sys.executable,"-u",COLLECT,"--out",out,"--case","A","--role","development","--method","dev","--seed","6001",
             "--variant",variant,"--duration","4.2","--quota","128"]
        code=admitted_call(name,1,cmd)
        if code or not (out/"result.json").is_file():
            return dict(variant=variant,code=code,raw=str(out),status="technical")
        r=json.loads((out/"result.json").read_text())
        rows=[json.loads(x) for x in (out/"events.jsonl").read_text().splitlines()] if (out/"events.jsonl").is_file() else []
        swings=json.loads((out/"swing.json").read_text())
        first=next((s for s in swings if s["duration_request_s"]>0),None)
        hold=next((x for x in rows if x["mode"]==1),None)
        return dict(variant=variant,raw=str(out),status=r["reason"],code=code,physical_s=r["physical_s"],requested=r["requested"],
            observed=r["observed"],first_transfer=first,first_hold_elapsed=hold["dt_s"] if hold else None,
            first_hold_requested=hold["issued_action"][3] if hold else None,control_full_ms=r["control_full_ms_p50_p95_p99"],
            rss=r["max_rss_gib"],gpu_alloc=r["cuda_peak_allocated_bytes"],wall_s=r["wall_s"],cpu_h=r["cpu_h"],
            source=r["source_commit"],identity_sha256=file_digest(out/"identity.json"),trajectory_sha256=file_digest(out/"trajectory.npz"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(compare,SETTINGS))
    usable=all(x["status"]=="horizon" and x["first_transfer"] for x in rows)
    effect={}
    if usable:
        a,b,h,k=rows
        effect=dict(xy_swing_separation_m=float(np.linalg.norm(np.subtract(a["first_transfer"]["offset"],b["first_transfer"]["offset"]))),
            apex_command_separation_m=abs(a["first_transfer"]["apex_offset_m"]-b["first_transfer"]["apex_offset_m"]),
            actual_transfer_onset_separation_s=abs(a["first_transfer"]["t"]-b["first_transfer"]["t"]),
            actual_hold_elapsed_separation_s=(k["first_hold_elapsed"] or 0)-(h["first_hold_elapsed"] or 0),
            observed_hold_elapsed=[h["first_hold_elapsed"],k["first_hold_elapsed"]])
        usable=bool(all(effect[t]>0 for t in ("xy_swing_separation_m","apex_command_separation_m","actual_transfer_onset_separation_s","actual_hold_elapsed_separation_s"))
            and h["first_hold_elapsed"] and k["first_hold_elapsed"])
    answer=dict(revision=revision,status="interface_observed" if usable else "interface_unverified",effects=effect,episodes=rows,
        used_training_events=0,query_role="development",technical=[x["variant"] for x in rows if x["status"]=="technical"])
    atomic_json(RESULT/("interface-"+revision+".json"),answer)
    return usable


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("revision")
    raise SystemExit(0 if development(p.parse_args().revision) else 1)
