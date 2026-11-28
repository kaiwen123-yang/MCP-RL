"""Compare frozen wrench-reference predictions with the simultaneous measured Go2 force channels."""
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"learning"))
from hybrid import RESULT
from orbits import nominal_response
from s1_w59_learner_critic import atomic_json, file_digest


def metrics(rows):
    value=np.asarray(rows)
    return dict(count=len(rows),mae_N=float(value.mean()),rmse_N=float(np.sqrt(np.mean(np.square(value)))),
                p95_N=float(np.percentile(value,95)))


def main():
    proof=json.loads((RESULT/"interface-v1.json").read_text())
    table=[]
    for episode in proof["episodes"]:
        raw=Path(episode["raw"])
        rows=[json.loads(x) for x in (raw/"events.jsonl").read_text().splitlines()]
        with np.load(raw/"trajectory.npz") as z:
            tr=z["control"].astype(float)
        for event in rows:
            ctx=event["response_context"]
            scale=ctx["weight"]*.25*np.sqrt(12)
            pred=nominal_response(ctx,event["mode"],event["raw"])
            initial=int(np.abs(tr[:,0]-event["start_s"]).argmin())
            for idx,dt in enumerate((.035,.06)):
                if event["dt_s"]+1.e-9<dt:
                    continue
                i=int(np.abs(tr[:,0]-(event["start_s"]+dt)).argmin())
                predicted=(scale*pred[5+16*idx:17+16*idx]).reshape(4,3)[:,2]
                for name,fields in [("solved_WBC_force",slice(87,99)),("measured_GRF",slice(75,87))]:
                    previous=tr[initial,fields].reshape(4,3)[:,2]
                    actual=tr[i,fields].reshape(4,3)[:,2]
                    table.append(dict(variant=episode["variant"],mode=event["mode"],offset=dt,name=name,
                        error=np.abs(predicted-actual).tolist(),hold_error=np.abs(previous-actual).tolist(),
                        physical_time_error_s=abs(tr[i,0]-(event["start_s"]+dt))))
    groups={}
    for name in ("solved_WBC_force","measured_GRF"):
        for mode,tag in [(None,"all"),(0,"transfer"),(1,"hold")]:
            selected=[row for row in table if row["name"]==name and (mode is None or row["mode"]==mode)]
            groups[name+"_"+tag]=dict(window_count=len(selected),
                nominal=metrics([x for r in selected for x in r["error"]]),persistence=metrics([x for r in selected for x in r["hold_error"]]))
    result=dict(source="4 forced A development episodes; near-term physical validity of the nominal local load feature. Each component is one foot. Within-rollout components are not independent learning seeds.",
        completed_development_events=sum(x["observed"] for x in proof["episodes"]),weights="normal per-leg forces in N",
        excluded="A query point after the episode/event end is unobserved and excluded.",
        fields=dict(time="columns 0/1: control start and integration end in seconds",measured_GRF="columns 75:87; measured at control start",
                    solved_WBC_force="columns 87:99; constrained WBC solution from this hold; different from the nominal load target",
                    qpos="columns 2:21 are post-integration qpos, used with column 1 for video"),
        original_receipt=file_digest(RESULT/"interface-v1.json"),time_offsets_s=[.035,.06],maximum_alignment_error_s=max(r["physical_time_error_s"] for r in table),
        groups=groups,used_training_events=0)
    atomic_json(RESULT/"development-response-validity.json",result)
    print(json.dumps(groups))


if __name__=="__main__":
    main()
