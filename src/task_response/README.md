# Task-directed response / feedback co-design candidate

Read [the method](../../docs/task-response-method-2026-10-19.md),
[conditional theory](../../docs/task-response-theory-2026-10-19.md), and
[the server protocol](../../docs/task-response-experiment-2026-10-19.md) first.

The code implements a candidate, not an established new control algorithm.
It is a synthetic planar task, not Go2/Piper deployment code. The physical
actuator state and the learned response memory are different variables.

## Commands from the repository root

```text
python scripts/task_response.py plan --config configs/task_response_pilot.json
python scripts/task_response.py check --config configs/task_response_pilot.json
```

`plan` uses arithmetic/configuration only. `check` runs random untrained
networks through a bounded complete task, computes gradients without applying
them, and checks weights are unchanged. It is not a training smoke test or a
performance comparison. Its smaller planner budget is explicitly recorded.

The explicit `train` command implements shared warmup, matched branches, fresh
A/B/C collections, PPO/model updates, calibration, logging and checkpoints.
The later authorized 2026-10-19 server pilot executed this branch for all four
arms and three seeds. The candidate did not outperform uniform joint training;
see [actual results](../../docs/server-pilot-results-2026-10-19.md).
`evaluate` requires a checkpoint and can use `--planner cem` or `--planner to`.
Those planners have different recorded work budgets; no automatic wall-time
equivalence is implied. See the protocol for later authorized server commands.

## Runtime

The existing project `.venv` contains Python 3.12, PyTorch 2.7.1+cpu and
Tianshou 0.5.1. No packages were installed or replaced for this delivery.
Use the same Tianshou version: `response_ppo.ppo_backend` calls upstream
`PPOPolicy.learn`, including its `merge_last=True` minibatch behavior.

For a later authorized, isolated Linux/CUDA environment, a proposed dependency
recipe is:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-task-response.txt
```

The PyTorch build command is documented on the [official previous-version
page](https://pytorch.org/get-started/previous-versions/). Server driver/GPU
availability and compatibility were not tested this round; check the actual
environment before installing. Do not apply the older CPU-only full pilot
freeze to a CUDA environment.

## Comparison identities

- `ordinary`: physical task PPO and supervised response identification.
- `uniform`: bidirectional response shaping/model task objective, uniform metric.
- `weighted_id`: task-weighted shaping but model identification only.
- `codesign`: normalized task metric plus model task proposal and recalibration.
- `frozen`: optional unchanged-actor diagnostic, not a certified strong baseline.

The proposed actor step is ordinary PPO on its declared shaped reward. A
standard PO method with identical objective/data can reproduce it. Command TO
optimizes commands only and must not be presented as a policy-learning rival.
Metrics, predictions and real task outcomes are reported separately.
