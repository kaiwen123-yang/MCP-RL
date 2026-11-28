"""Exercise the actual mixed optimizer on retained development states; never publish these actor weights."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from hybrid import RESULT, SPEC, new_heads
from orbits import nominal_response
from s1_w59_learner_critic import atomic_json
from update import OPT, update


def main():
    proof=json.loads((RESULT/"interface-v1.json").read_text())
    source=[]
    for case in proof["episodes"]:
        path=Path(case["raw"])/"events.jsonl"
        source.append([json.loads(r) for r in path.read_text().splitlines()])
    cuts=[source[0]+source[1],source[2]+source[3]]
    values={}
    directions={}
    for method in [*SPEC["methods"],"E-Fisher"]:
        actor,critic=new_heads(6001)
        ao=torch.optim.Adam(actor.parameters(),lr=OPT["learning_rate"])
        co=torch.optim.Adam(critic.parameters(),lr=OPT["learning_rate"])
        before=torch.nn.utils.parameters_to_vector(actor.parameters()).detach().clone()
        values[method]=update(actor,critic,ao,co,cuts,method,6001,0)
        directions[method]=torch.nn.utils.parameters_to_vector(actor.parameters()).detach()-before
    diff=float((directions["E-score"]-directions["E-Fisher"]).norm())
    cosine=float(torch.nn.functional.cosine_similarity(directions["E-score"][None],directions["E-Fisher"][None]))
    e=values["E-score"]
    valid=bool(diff>1.e-8 and e["max_response_singular_value"]>0 and np.isfinite(e["halving_rmse"])
        and all(np.isfinite(v["final_kl"]) and v["final_kl"]<=OPT["kl_max"]+1.e-7 for v in values.values()))
    rec=dict(status="verified" if valid else "failed",input="43 real directed A development events grouped by complete attempt; operator check; forced samples, no performance inference",
        direction_delta_l2=diff,direction_cosine=cosine,discarded_diagnostic_actor_count=4,formal_requested=0,formal_observed=0,formal_used=0,methods=values)
    atomic_json(RESULT/"development-update-v2.json",rec)
    print(json.dumps({k:(v["final_kl"],v["accepted_actor_steps"],v.get("max_response_singular_value"),v.get("chosen_multiplier")) for k,v in values.items()}))
    return valid


if __name__=="__main__":
    raise SystemExit(0 if main() else 1)
