# Frozen Go2 task baseline

This directory runs the **complete floating-base Go2 / 12-joint / contact simulation**, with a frozen public locomotion policy. No policy/model training, external disturbance, robot SDK, hardware connection, or automatic background job is included.

The evaluated task is straight → left turn → physical doorway → stop. `task.py` defines the geometry and ordered success conditions; `planner.py` implements ordinary command-space CEM MPC and an explicitly labeled geometric follower; `simulator.py` preserves the public observation, joint ordering, action scaling and PD execution chain.

See the [Chinese result report](../../docs/go2-frozen-baseline-2026-10-19.md) and [fixed protocol](../../notes/go2-frozen-2026-10-19/PROTOCOL.md).

## Reproduce in a separate environment

From the repository root, Windows PowerShell:

```powershell
python -m venv work/go2-venv
& work/go2-venv/Scripts/python.exe -m pip install -r requirements-go2-frozen.txt
& work/go2-venv/Scripts/python.exe scripts/prepare_go2_frozen_assets.py
& work/go2-venv/Scripts/python.exe scripts/run_go2_frozen.py --controller mpc --case nominal --output outputs/go2-reproduction --raw-output work/go2-reproduction --video
```

Asset preparation clones two pinned public repositories into ignored `work/go2-public-assets/`. It reads inference weights and robot descriptions but does not execute upstream deployment scripts. Use an authenticated git/gh setup. The source policy and model have separate provenance; this is a newly constructed frozen evaluation, not the policy author's original training-environment reproduction or the user's physical robot.

Use a fresh output root for each complete evaluation. Existing run directories are never overwritten. For all declared conditions:

```powershell
foreach ($caseName in @('nominal','narrow','wide')) {
  foreach ($controllerName in @('mpc','follower')) {
    & work/go2-venv/Scripts/python.exe scripts/run_go2_frozen.py --controller $controllerName --case $caseName --output outputs/go2-six-runs --raw-output work/go2-six-runs --video
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
  }
}
```

Do not point the reproduction at the committed evidence directory. `--video` is optional and requires a working local OpenGL renderer. Simulation time pauses during MPC solving, and actual solve latency is reported separately.

## Evidence and interpretation

Each run writes `summary.json`, 10 Hz whole-task CSV, all command decisions/predictions, one-interval errors, recorded-command open-loop replay and full stage-transition states. The raw directory preserves 50 Hz joint/body/contact logs, scene XML, preparation trajectory and optional MP4.

The policy file and in-memory parameter hashes are checked before/after the run. `model_training_updates=0` also means the MPC lag parameters are **engineering initial values, not identified system parameters**. MPC changes command sequences only.

The planning footprint is a planar approximation, whereas obstacle contacts are checked on the full robot at every physics substep. Success requires the declared planar position/speed/yaw tolerances for one second and no historical obstacle contact or fall; it does not certify that every joint is motionless.

`planner_model_steps` counts CEM-scored candidate steps. Each MPC call additionally rolls out 20 feedback-initialization steps and 20 winning-trajectory steps; wall latency includes these operations. Input-squared and absolute mechanical-work integrals are proxies, not battery energy.

The three cases are individual deterministic conditions, not statistical repetitions. The follower is a useful diagnostic, not a purported state-of-the-art method. No current result establishes a need for PINN or additional actor learning.

## Fair objective, speed and model comparison (2026-10-19)

The follow-up uses separate modules and output directories. The original six runs, task geometry, robot assets, initial conditions, stage order and success thresholds remain unchanged. The policy is frozen; no policy/model fitting, PINN, hardware operation or PR merge is part of this comparison. See the [fair-comparison report](../../docs/go2-fair-comparison-2026-10-19.md), [evidence index](../../notes/go2-fair-2026-10-19/INDEX.md), and [predeclared protocol](../../notes/go2-fair-2026-10-19/PROTOCOL.md).

The 18 primary runs cover the same three cases, two speed bands and three controllers. `standard` permits straight/turn translation-command norms of 0.45/0.32 m/s; `fast` permits 0.69/0.55 m/s. These are **command ceilings**, not hard bounds on the robot's actual velocity. A shared governor implements the original command box, common turning/braking envelope and terminal recovery. The matched `follower` removes the lookahead correction's unintended tangential acceleration; `tracking` preserves the original timed-reference MPC cost; `mission` evaluates geometric tracking plus a remaining-route-time proxy. Both MPC modes retain the same 256 × 3 CEM budget and 4 s horizon. Their internal scalar costs must not be compared across objectives as task performance.

Using the environment prepared above, reproduce all 18 primary runs from the repository root:

```powershell
foreach ($caseName in @('nominal','narrow','wide')) {
  foreach ($speedName in @('standard','fast')) {
    foreach ($modeName in @('follower','tracking','mission')) {
      & work/go2-venv/Scripts/python.exe scripts/run_go2_fair.py --suite primary --mode $modeName --speed $speedName --case $caseName --output outputs/go2-fair-reproduction --raw-output work/go2-fair-reproduction
      if ($LASTEXITCODE -ne 0) { throw "Fair primary evaluation failed; preserve its output" }
    }
  }
}
```

The four additional finite-bank runs use **only nominal geometry and fast speed**: tracking/mission objective × nominal/physics predictor. Every decision builds the same six types of candidate sequences. The physics predictor evaluates them by branching the full MuJoCo state and frozen policy history; the nominal predictor uses the fixed first-order model. Feasible candidates rank first. Candidate construction is shared, but later closed-loop states and numerical candidate values can differ between runs.

```powershell
foreach ($modeName in @('tracking','mission')) {
  foreach ($predictorName in @('nominal','physics')) {
    & work/go2-venv/Scripts/python.exe scripts/run_go2_fair.py --suite bank --mode $modeName --predictor $predictorName --speed fast --case nominal --output outputs/go2-fair-reproduction --raw-output work/go2-fair-reproduction
    if ($LASTEXITCODE -ne 0) { throw "Finite-bank diagnostic failed; preserve its output" }
  }
}
```

The physics branches have privileged simulator/state-history access and substantial additional computation, recorded separately. They are **model-error diagnostics, not real-time or equal-compute controller comparisons**. Their nominal feasibility test uses a planar footprint; physics checks use full-body contact/fall flags. Nominal and physics rollouts share raw sequences and governor rules, but the governor sees their respective states, so applied sequences need not remain numerically identical. Any inferred model effect is conditional on this finite candidate bank, objective, scenario and feasibility information.

Every evaluation refuses to overwrite an existing run directory in either the public or raw output root. Use fresh roots for another attempt; retain failures and exceptions. The raw root in ignored `work/` keeps the actual 50 Hz body/joint/contact trace, warmup and scene XML. Public outputs contain the 10 Hz task trajectory, all decisions, stage transitions, prediction errors and summary/provenance. Physical candidate branches preserve states/commands at **0.2 s boundaries**, plus cumulative failure flags and budget; they are not stored as full 50 Hz branch trajectories. `--video` is optional for actual task execution.

After all 22 run summaries exist, analyze the saved results without running another simulation:

```powershell
& work/go2-venv/Scripts/python.exe scripts/analyze_go2_fair.py --runs outputs/go2-fair-reproduction --output outputs/go2-fair-reproduction-analysis
```

For the committed evidence, the analyzer's default `--runs` is `notes/go2-fair-2026-10-19/runs`; its default analysis directory is `notes/go2-fair-2026-10-19/analysis`. Unlike evaluation runs, derived analysis files may be regenerated, so select a fresh `--output` when preserving an earlier analysis. The analyzer requires all 22 declared summaries and does not hide failed evaluations.

The common evaluation reports ordered task success first, completion time for successful runs, then path error, mechanical-work proxies and computation. A failure's elapsed time is not a completion time. Each cell is one deterministic trajectory; the three cases are not statistical repetitions. Physics remains paused during planning, and every branch's additional physics steps, policy inference and solve latency are reported. No result from this comparison is by itself evidence for PINN or for changing the frozen locomotion policy.
