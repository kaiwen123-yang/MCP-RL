"""Evaluate the frozen historical task critic on the independent frozen W57 R/PR/blind original E60s."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from s1_w59_learner_critic import ROOT, RELEASE, TrueTaskCritic, atomic_json, episode_arrays, file_digest, json_rows


def evaluate(source,checkpoint):
    with torch.random.fork_rng(devices=[]):model=TrueTaskCritic()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)["critic"]); model.eval()
    rel=Path("terrain_memory/simulation/tm-s1-go2-local-readiness-r12")
    episodes=[];err=[];width=[]
    for arm in ("r","pr","blind"):
        for case in ("a","b"):
            name=f"w57-m1-{arm}-{case}-original-e60-a01"
            ep=source/"results"/rel/"w57/episodes"/(name+".json")
            raw=source/"outputs"/rel/name/"timing-transitions.jsonl"
            report=json.loads(ep.read_text());rows=json_rows(raw)
            if not rows or any(r["sampling"]!="frozen_mean" for r in rows):
                raise ValueError("Expected only actual frozen W57 evaluation transitions")
            # Scoring all frozen actions uses the same task-target reconstruction as the fit.
            x,y,w,meta=episode_arrays(rows,report,score_frozen=True)
            with torch.no_grad():p=model(torch.as_tensor(np.array(x))).numpy()
            delta=np.asarray(y)-p;err.extend(delta);width.extend(w)
            episodes.append(dict(attempt=name,arm=arm,case=case.upper(),n=len(y),mae=float(np.abs(delta).mean()),
               elapsed_s=report["simulation_s"],completion_s=report["first_completion_s"],source=str(raw),
               sha256=file_digest(raw),source_commit=report["source_commit"],max_target_interval_width=max(w)))
    return dict(scope="Separate frozen original W57 E60, virgin single-roundtrip; no W59 actions or failures",
                checkpoint_sha256=file_digest(checkpoint),successful_episodes=len(episodes),real_events=len(err),
                true_physical_seconds=sum(e["elapsed_s"] for e in episodes),mae=float(np.abs(err).mean()),
                rmse=float(np.sqrt(np.mean(np.square(err)))),max_target_interval_width=max(width),episodes=episodes,
                additional_task_value_updates=0,actor_updates=0,W59_new_physics=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=ROOT.parent/"MCP-RL-w57")
    p.add_argument("--checkpoint",type=Path,default=ROOT/"work/w59/learner/historical-task-critic.pt")
    a=p.parse_args();ans=evaluate(a.source,a.checkpoint)
    atomic_json(ROOT/RELEASE/"learning"/"historical-critic-frozen-transfer.json",ans)
    print(json.dumps({k:v for k,v in ans.items() if k!="episodes"}))

