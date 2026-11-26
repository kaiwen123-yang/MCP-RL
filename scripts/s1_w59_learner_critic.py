"""Task value from measured W38 credit at Go2 legal handover boundaries."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "experiments/terrain_memory/s1-w59-learner.json"
SPEC = json.loads(RECIPE_PATH.read_text())
PHASES = ("initial_stand", "outbound", "turn", "return", "terminal_stand", "complete",
          "far_hold", "home_hold", "turn_home")
RELEASE = Path("results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59")
BETA = -math.log(SPEC["discount_per_half_second"]) / 0.5
STATE_DIM = 16


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_weights(path, payload):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def file_digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            sha.update(data)
    return sha.hexdigest()


def task_observation(t, progress, phase, normalized_fz, hold_s):
    phase_vector = np.eye(len(PHASES))[PHASES.index(phase)]
    return np.asarray([t / 60., progress, *phase_vector, *normalized_fz, hold_s / 2.], np.float32)


class TrueTaskCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(STATE_DIM, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, observation):
        return self.layers(observation).squeeze(-1)


def json_rows(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def correction(rows, result):
    """Reconstruct the pre-absorption return; bound nonnegative subinterval costs."""
    last = rows[-1]
    finish = result["first_completion_s"]
    if not result["success"] or finish is None or not last["start_s"] < finish <= last["end_s"]:
        raise ValueError(f"Unqualified W57 settlement in {result['attempt']}")
    hold = result["hold"]
    if hold["start_s"] != finish or not hold["full_observation"] or hold["violation_samples"] != 0:
        raise ValueError(f"Nonstable post-completion segment in {result['attempt']}")
    terms = last["reward_terms"]
    # Post-completion leg/lap credit and old cap cost are frozen by the plant phase.
    # Legacy logs integrate loaded slip/indentation to E60. Their missing split is
    # bounded by their recorded non-positive final-event integrals.
    tail = last["end_s"] - finish
    lower = last["reward"] + tail / 60.
    upper = lower - terms["loaded_slip"] - terms["indentation"]
    return finish, lower, upper, tail


def episode_arrays(rows, result):
    end, lower, upper, post_task = correction(rows, result)
    last = rows[-1]
    ids = {row["decision_id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError(f"Repeated decision in {result['attempt']}")
    if any(abs(a["end_s"] - b["start_s"]) > 1e-7 for a, b in zip(rows, rows[1:])):
        raise ValueError(f"Non-contiguous physical credit in {result['attempt']}")
    # Invert the sum from the measured episode-end route account. This also
    # preserves any pre-first-request credit and avoids guessing its value.
    progress = float(result["legal_route"]["progress"])
    x, targets, widths = [], [], []
    for row in reversed(rows):
        if row is not last:
            decay = math.exp(-BETA * row["dt_s"])
            lower, upper = row["reward"] + decay * lower, row["reward"] + decay * upper
        progress -= row["metrics"]["signed_legal_route"]
        if row["sampling"] != "on_policy":
            continue
        obs = row["obs"]
        phase = PHASES[np.argmax(obs[50:59])]
        if phase not in ("outbound", "turn", "return", "turn_home"):
            raise ValueError("Old ledger lacks the current hold streak at a hold-state request")
        x.append(task_observation(row["start_s"], progress, phase, obs[:4], 0.))
        targets.append((lower + upper) / 2.)
        widths.append(upper - lower)
    return x, targets, widths, {
        "source_transitions": len(rows), "supervised": len(x),
        "frozen_tail_context": len(rows) - len(x), "elapsed_physics_s": result["simulation_s"],
        "post_task_observed_s": post_task, "completion_s": end,
        "fine_hold_samples": result["hold"]["observed_samples"],
    }


def saved_source(pilot, case, head):
    source = {}
    for file in sorted((pilot / head).glob(f"batch-*-{case}.jsonl")):
        for row in json_rows(file):
            source.setdefault(row["attempt"], Path(row["source_dir"]))
    return source


def collect(pilot, results):
    groups = {s: defaultdict(list) for s in ("train", "validation")}
    trials = []
    for head in ("full", "blind"):
        for case in ("A", "B"):
            source = saved_source(pilot, case, head)
            for launch in json_rows(pilot / head / f"episodes-{case}.jsonl"):
                name = launch["attempt"]
                ordinal = int(name.rsplit("-", 1)[1])
                split = "validation" if ordinal % 5 == 0 else "train"
                episode_path = results / f"{name}.json"
                path = source.get(name)
                trial = dict(attempt=name, original_head=head, case=case, split=split)
                if path is None or not episode_path.is_file():
                    trial.update(status="incomplete_no_terminal_label",
                                 reason="absent sealed event stream" if path is None else "no natural episode receipt")
                    if path is not None:
                        trial["measured_partial_rows"] = len(json_rows(path / "timing-transitions.jsonl"))
                    trials.append(trial)
                    continue
                trace_path = path / "timing-transitions.jsonl"
                rows = json_rows(trace_path)
                result = json.loads(episode_path.read_text())
                xs, ys, span, meta = episode_arrays(rows, result)
                data = groups[split]
                data["x"].extend(xs)
                data["y"].extend(ys)
                data["interval_width"].extend(span)
                data["attempt"].extend([name] * len(xs))
                data["case"].extend([case] * len(xs))
                trial.update(status="used", source=str(trace_path), episode_result=str(episode_path),
                             ledger_sha256=file_digest(trace_path), controller=result["execution_fingerprint"],
                             source_commit=result["source_commit"], **meta)
                trials.append(trial)
    return {s: {k: np.asarray(v) for k, v in g.items()} for s, g in groups.items()}, trials


def evaluate(data, critic, baseline):
    with torch.no_grad():
        pred = critic(torch.as_tensor(data["x"])).cpu().numpy()
    error = pred - data["y"]
    per_ep = []
    for name in np.unique(data["attempt"]):
        per_ep.append(float(np.abs(error[data["attempt"] == name]).mean()))
    return dict(n=len(error), episodes=len(per_ep), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.mean(error**2))), mean_error=float(error.mean()),
                constant_training_mean_mae=float(np.abs(data["y"] - baseline).mean()),
                median_episode_mae=float(np.median(per_ep)), max_target_interval_width=float(data["interval_width"].max()),
                case_mae={c: float(np.abs(error[data["case"] == c]).mean()) for c in "AB"},
                phase_mae={PHASES[int(i)]: float(np.abs(error[mask]).mean()) for i in np.unique(np.argmax(data["x"][:,2:11],axis=1))
                           if (mask := np.argmax(data["x"][:,2:11],axis=1) == i).any()})


def fit(pilot, results, raw, compact):
    torch.set_num_threads(1)
    recipe = SPEC["critic"]["standalone_fit"]
    torch.manual_seed(recipe["seed"])
    wall, cpu = time.perf_counter(), time.process_time()
    data, trials = collect(pilot, results)
    train = data["train"]
    x = torch.as_tensor(train["x"], dtype=torch.float32)
    y = torch.as_tensor(train["y"], dtype=torch.float32)
    critic = TrueTaskCritic()
    optimizer = torch.optim.Adam(critic.parameters(), lr=recipe["lr"])
    steps = 0
    for epoch in range(recipe["epochs"]):
        for ids in torch.randperm(len(x)).split(recipe["batch"]):
            pred = critic(x[ids])
            loss = (pred - y[ids]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
    raw.mkdir(parents=True, exist_ok=True)
    weights = raw / "historical-task-critic.pt"
    atomic_weights(weights, {"critic": critic.state_dict(), "dimension": STATE_DIM, "fit": recipe,
                             "weight_source": "historical W57 physical credit only", "optimizer_steps": steps})
    report = dict(schema="W59_historical_real_task_critic_v1", groups={s: evaluate(g, critic, float(y.mean())) for s,g in data.items()},
                  recipe_sha256=file_digest(RECIPE_PATH), optimizer_steps=steps, cpu_threads=torch.get_num_threads(),
                  source_identity="W57-pilot02-full/blind timing; virgin E60 per episode",
                  attempts_per_status=dict(Counter(t["status"] for t in trials)),
                  complete_source_rows=sum(t.get("source_transitions",0) for t in trials),
                  complete_physical_s=sum(t.get("elapsed_physics_s",0) for t in trials),
                  post_task_observed_s=sum(t.get("post_task_observed_s",0) for t in trials),
                  fine_hold_samples=sum(t.get("fine_hold_samples",0) for t in trials),
                  excluded_partial_rows=sum(t.get("measured_partial_rows",0) for t in trials),
                  frozen_tail_context_rows=sum(t.get("frozen_tail_context",0) for t in trials),
                  elapsed_wall_s=time.perf_counter()-wall, process_cpu_s=time.process_time()-cpu,
                  W59_new_physical_events=0, W59_actor_updates=0, imagined_steps=0, hard_failure_labelled_episodes=0,
                  checkpoint={"path":str(weights),"size":weights.stat().st_size,"sha256":file_digest(weights)},
                  scope="Reached-task W57 development. Original physical failures and W59 action gradient are uncalibrated.")
    atomic_json(compact/"historical-critic-episodes.json", trials)
    atomic_json(compact/"historical-critic.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pilot", type=Path, required=True)
    cli.add_argument("--results", type=Path, required=True)
    cli.add_argument("--raw", type=Path, default=ROOT / "work/w59/learner")
    cli.add_argument("--compact", type=Path, default=ROOT / RELEASE / "learning")
    a = cli.parse_args()
    fit(a.pilot, a.results, a.raw, a.compact)
