"""Issue a model-free PPO allocation from measured action data and a terminal Diff blocker."""
import argparse
import json
from pathlib import Path

from s1_w59_learner_critic import ROOT, RELEASE, file_digest, atomic_json
from s1_w59_learning import read_json

def create(registry,development,blocker,critic,first_dir,out):
    reg=read_json(registry);dev=read_json(development);stop=read_json(blocker)
    # Do not spend a matched slot while another current action-conditioned model or
    # unopened admission is simply still in flight.
    if stop.get("status") not in ("scientifically_blocked","resource_blocked","blocked"):
        raise ValueError("A terminal, evidence-backed current-version Diff decision is required")
    if not stop.get("reason") or not stop.get("evidence"):
        raise ValueError("Scientific blocker lacks its actual supporting evidence")
    if len(reg.get("first_update",{}).get("5901",[]))!=8:
        raise ValueError("Independent first-parameter states are missing for seed 5901")
    if dev["registered_branches"]!=96 or dev["measured_branches"]!=96 or dev["fully_paired_directions"]!=72:
        raise ValueError("The full development action denominator is not closed")
    measured=0;floors=[]
    for st in dev["states"]:
        nominal=st["records"]
        floors.append(abs(nominal["nominal1"]["undiscounted_w38_2s"]-nominal["nominal2"]["undiscounted_w38_2s"]))
        for k,change in st["signed"].items():
            if abs(change)>max(st["nominal_replay_floor"],1e-12) and not nominal[k]["hard_failure"]:
                measured+=1
    if not measured:raise ValueError("No non-failure real action consequence above replay error")
    arms=ROOT/RELEASE/"episodes"
    for case in "ab":
        for level in ("e30","e60"):
            originals=arms.glob(f"w59-b3-{case}-original-{level}-a[0-9][0-9].json")
            if not any(read_json(p).get("stage_status")=="passed" for p in originals):
                raise ValueError(f"Missing B3 original {case}/{level}")
    ids={s["snapshot_id"] for split in reg["states"].values() for s in split}
    selected=set()
    for seed,records in reg["first_update"].items():
        if len(records)!=8:continue
        placed={s["snapshot_id"] for s in records}
        if len(placed)!=8 or placed & (ids|selected):
            raise ValueError("First-check states must remain separately frozen and physically disjoint")
        selected.update(placed)
        atomic_json(first_dir/f"seed-{seed}.json",dict(seed=int(seed),registry_sha256=file_digest(registry),
            states=[dict(r,rng_key=r["sha256"]+"/stored") for r in records]))
    receipt=dict(role="standalone_true_ppo",recipe_sha256=file_digest(ROOT/"experiments/terrain_memory/s1-w59-learner.json"),
        registry_path=str(registry.resolve()),registry_sha256=file_digest(registry),freeze_path=None,planning_substeps=0,
        ppo_eligible=True,planning_eligible=False,r3_gradient_eligible=False,online_reused_actions=False,
        critic_sha256=file_digest(critic),wm_checkpoints=[],wm_sha256=[],material="M1",gradient_admission=None,
        calibration_snapshot_ids=sorted(ids),mean_nominal_2s_return_replay_error=sum(floors)/len(floors),
        measured_development_directions=72,resolved_nonfailure=measured,
        action_source=str(development),action_source_sha256=file_digest(development),
        blocker=str(blocker),blocker_sha256=file_digest(blocker),blocked_reason=stop["reason"])
    atomic_json(out,receipt)
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for field in ("registry","development","blocker","critic","first-dir","output"):
        p.add_argument("--"+field,type=Path,required=True)
    a=p.parse_args();r=create(a.registry,a.development,a.blocker,a.critic,a.first_dir,a.output)
    print(json.dumps(r))

