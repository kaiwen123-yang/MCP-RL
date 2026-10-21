#!/usr/bin/env python3
"""Collect disjoint, complete episodes from a fixed-seed parallel evaluation."""
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--method",default="adaptive")
a=p.parse_args()
dest=ROOT/"results/terrain_memory/reproduction/pympc"/a.run/"table1"/a.method
rows=[];sources=[]
for folder in sorted(dest.glob("batch-*")):
    with (folder/"episodes.csv").open() as f:rows.extend(csv.DictReader(f))
    sources.append(str((folder/"run.json").relative_to(ROOT)))
rows.sort(key=lambda row:int(row["seed"]))
with (dest/"episodes.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
meta=dict(experiment="table1",method=a.method,completed_episodes=len(rows),
    seeds=[int(row["seed"]) for row in rows],successes=sum(int(row["success"]) for row in rows),
    batch_run_records=sources,parallel_evaluation_processes=len(sources),
    timing_note="Solver wall times were measured during concurrent batch evaluation; use isolated Fig.3 runs for latency comparison.")
(dest/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
