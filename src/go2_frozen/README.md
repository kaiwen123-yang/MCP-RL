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
