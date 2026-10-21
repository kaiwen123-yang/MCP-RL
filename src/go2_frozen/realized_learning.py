"""Learning objectives tied to executed feedback, not optimistic forecasts.

This module has no optimizer, simulator, policy loader, or training loop.

Implemented contracts:
* A factual predictor is fitted ONLY to recorded execution features/resources.
* Complete task credit and optional predictability costs come from those same
  recorded physical consequences. A fixed prediction is a reference, never a
  replacement for the executed task cost.
* Policy gradients enter through log probabilities of the actually sampled
  policy actions. Full-task returns, or an explicitly supplied realized suffix,
  carry downstream consequences back to earlier actions.

The default feature order is world x/y/yaw/vx/vy/yaw_rate. Conditioning identity
must include the same commands, task context, policy history/estimator state,
and execution timing used to produce BOTH prediction and execution. The caller
collects and identifies that information; a six-dimensional base state alone
is not asserted to be sufficient for the Go2 policy's closed-loop response.

The PPO/score objectives equal standard policy optimization with the same
recorded costs and information. These are structural corrections, not a claim
of originality, a realizability theorem, or evidence of useful learned feedback.
Deterministic exported Go2 policies have no behavior density in this experiment;
their data support factual fitting/metrics, not fabricated PPO updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ResponseContract:
    contract_id: str = "go2-world-response-v1"
    feature_names: tuple[str, ...] = (
        "world_x", "world_y", "world_yaw", "world_vx", "world_vy", "world_yaw_rate"
    )
    feature_scales: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    angular_indices: tuple[int, ...] = (2,)
    resource_name: str = "declared_execution_resource"
    resource_scale: float = 1.0


@dataclass(frozen=True)
class ExecutionIdentity:
    """One batch's actual feedback and conditional prediction contract.

    conditioning_id identifies the recorded command/task/history snapshots;
    execution_id identifies the physics, controller, timing and resource units.
    policy_id identifies the behavior policy that generated the recorded data.
    A target cannot be silently paired with a different task/history or policy.
    """

    policy_id: str
    execution_id: str
    conditioning_id: str
    contract_id: str = "go2-world-response-v1"


@dataclass
class ExecutedMacroBatch:
    features: Tensor                   # [M, B, D], actual macro endpoint features
    resources: Tensor                  # [M, B], actual declared macro resource
    identity: ExecutionIdentity
    task_costs: Tensor | None = None    # [M, B], true task costs including terminal events
    complete_task: bool = False         # False for a truncated/partial record


@dataclass
class PredictedMacroBatch:
    features: Tensor                   # [M, B, D], prediction under the same conditions
    resources: Tensor                  # [M, B], prediction, not a physical upper bound
    identity: ExecutionIdentity
    predictor_id: str


@dataclass(frozen=True)
class FixedResponseReference:
    features: Tensor
    resources: Tensor
    identity: ExecutionIdentity
    predictor_id: str


@dataclass
class FactualLoss:
    loss: Tensor
    feature_loss: Tensor
    resource_loss: Tensor


@dataclass
class RealizedCosts:
    task: Tensor
    predictability: Tensor
    total: Tensor


@dataclass
class PolicyLoss:
    loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


DEFAULT_CONTRACT = ResponseContract()


def _check_pair(prediction, execution, contract):
    if prediction.identity != execution.identity:
        raise ValueError("Prediction and execution must share policy/task/history/execution identity")
    if execution.identity.contract_id != contract.contract_id:
        raise ValueError("Feature/resource contract identity does not match this batch")
    if execution.features.ndim != 3 or execution.resources.shape != execution.features.shape[:2]:
        raise ValueError("Execution requires features[M,B,D] and resources[M,B]")
    if prediction.features.shape != execution.features.shape or prediction.resources.shape != execution.resources.shape:
        raise ValueError("Prediction and real execution must be aligned macro by macro")
    if execution.features.shape[-1] != len(contract.feature_names) or len(contract.feature_scales) != len(contract.feature_names):
        raise ValueError("Feature dimensions/scales must agree with the declared feature order")
    if any(scale <= 0 for scale in contract.feature_scales) or contract.resource_scale <= 0:
        raise ValueError("Feature/resource scales must be positive declared constants")


def feature_residual(predicted: Tensor, actual: Tensor,
                     contract: ResponseContract = DEFAULT_CONTRACT) -> Tensor:
    """Normalized residual with circular yaw; no wrapped yaw-rate difference."""
    residual = predicted - actual
    angular = set(contract.angular_indices)
    columns = [torch.atan2(torch.sin(residual[..., i]), torch.cos(residual[..., i]))
               if i in angular else residual[..., i] for i in range(residual.shape[-1])]
    return torch.stack(columns, dim=-1) / residual.new_tensor(contract.feature_scales)


def factual_response_loss(prediction: PredictedMacroBatch,
                          execution: ExecutedMacroBatch,
                          contract: ResponseContract = DEFAULT_CONTRACT,
                          resource_weight: float = 1.0) -> FactualLoss:
    """Fit facts; gradients reach prediction tensors, never observed labels.

    No predicted task value enters this objective. The task_costs field is not
    consumed. A model minimizing its own imagined task cost cannot improve this
    factual objective unless it also moves closer to real recorded outcomes.
    """
    _check_pair(prediction, execution, contract)
    if resource_weight < 0:
        raise ValueError("resource_weight must be nonnegative")
    residual = feature_residual(prediction.features, execution.features.detach(), contract)
    resource_error = (prediction.resources - execution.resources.detach()) / contract.resource_scale
    feature_loss, resource_loss = residual.square().mean(), resource_error.square().mean()
    return FactualLoss(feature_loss + resource_weight * resource_loss, feature_loss, resource_loss)


def freeze_response_reference(prediction: PredictedMacroBatch) -> FixedResponseReference:
    """Snapshot a reference before a policy update and keep its version fixed.

    Prediction must have been conditioned on information available before the
    corresponding execution. Calling this function after observing the outcome
    does not establish that chronological property. During PPO reuse, targets,
    task definitions, action mapping and reference inputs stay unchanged.
    """
    return FixedResponseReference(prediction.features.detach().clone(),
                                  prediction.resources.detach().clone(),
                                  prediction.identity, prediction.predictor_id)


def realized_macro_costs(execution: ExecutedMacroBatch,
                         reference: FixedResponseReference | None = None,
                         predictability_weight: float = 0.0,
                         resource_weight: float = 1.0,
                         contract: ResponseContract = DEFAULT_CONTRACT) -> RealizedCosts:
    """Build policy costs from actual task and actual response consequences.

    The optional error measures executed features against a fixed reference.
    It is not a free model objective and does not substitute predicted task
    quality for real task outcomes. These costs are detached for likelihood-
    ratio credit. The policy must receive any reference state needed to make
    this conditional reward observable, or use its recorded history; merely
    having an external task context in a log does not mean the actor saw it.
    """
    if execution.task_costs is None or execution.task_costs.shape != execution.resources.shape:
        raise ValueError("Policy costs require actual task_costs[M,B], including real terminal/event costs")
    if predictability_weight < 0 or resource_weight < 0:
        raise ValueError("Cost coefficients must be nonnegative")
    task = execution.task_costs.detach()
    penalty = torch.zeros_like(task)
    if predictability_weight:
        if reference is None:
            raise ValueError("A nonzero predictability cost requires a fixed pre-execution reference")
        _check_pair(reference, execution, contract)
        errors = feature_residual(execution.features.detach(), reference.features.detach(), contract)
        resource_error = (execution.resources.detach() - reference.resources.detach()) / contract.resource_scale
        penalty = predictability_weight * (errors.square().mean(-1) + resource_weight * resource_error.square())
    return RealizedCosts(task, penalty, task + penalty)


def realized_return_to_go(macro_costs: Tensor, complete_task: bool,
                          known_realized_suffix_cost: Tensor | None = None) -> Tensor:
    """Undiscounted costs for the declared physical finite task.

    A partial segment is accepted only with an explicitly supplied realized
    cost of its continuation. An estimated value bootstrap is a different
    estimator and is not silently treated as a measured whole-task return.
    The policy batch must not concatenate reset episodes along this time axis.
    """
    if macro_costs.ndim != 2:
        raise ValueError("macro_costs must have shape [M,B]")
    if not complete_task and known_realized_suffix_cost is None:
        raise ValueError("A truncated segment is not a complete task; supply its realized continuation cost")
    costs = macro_costs.detach()
    suffix = torch.zeros_like(costs[0]) if known_realized_suffix_cost is None else known_realized_suffix_cost.detach()
    if suffix.shape != costs.shape[1:]:
        raise ValueError("Realized suffix cost must have shape [B]")
    return costs.flip(0).cumsum(0).flip(0) + suffix


def returns_for_micro_actions(macro_costs: Tensor, macro_steps: int,
                              complete_task: bool,
                              known_realized_suffix_cost: Tensor | None = None) -> Tensor:
    """Credit every actual policy action with its macro-end and later outcomes.

    Equivalently, place each macro's actual cost at its final microstep and
    compute undiscounted returns. This preserves the full physical objective;
    it can have more variance than retaining original per-microstep costs.
    Returned shape [M*macro_steps,B] aligns with recorded micro-action densities.
    Do not use a product of macro action ratios and call it ordinary per-action
    PPO. Variable-length macro intervals need an explicit timestamp adapter.
    """
    if macro_steps < 1:
        raise ValueError("macro_steps must be positive")
    returns = realized_return_to_go(macro_costs, complete_task, known_realized_suffix_cost)
    return returns.repeat_interleave(macro_steps, dim=0)


def _cost_advantage(log_prob, returns, baseline):
    if log_prob.shape != returns.shape:
        raise ValueError("Each recorded policy-action density needs its aligned realized return")
    if baseline is None:
        return returns.detach()
    if baseline.shape != returns.shape:
        raise ValueError("The recorded action-independent baseline must match return shape")
    return (returns - baseline).detach()


def score_function_cost_loss(new_log_prob: Tensor, realized_returns: Tensor,
                             action_independent_baseline: Tensor | None = None) -> Tensor:
    """Standard on-policy cost score estimator; no model task derivative.

    new_log_prob must evaluate the actual recorded raw action under the policy
    that sampled the batch. For off-policy reuse use a justified importance
    estimator, such as the separately provided PPO surrogate. A deterministic
    action cannot be assigned a made-up Gaussian density for this purpose.
    """
    advantage = _cost_advantage(new_log_prob, realized_returns, action_independent_baseline)
    return (new_log_prob * advantage).mean()


def ppo_realized_cost_loss(new_log_prob: Tensor,
                           behavior_log_prob: Tensor | None,
                           realized_returns: Tensor,
                           action_independent_baseline: Tensor | None = None,
                           clip_epsilon: float = 0.2) -> PolicyLoss:
    """Ordinary clipped PPO expressed for costs to minimize.

    The caller evaluates new_log_prob using the recorded real policy inputs
    (native observations/history, commands, and any genuinely consumed task or
    reference inputs) and sampled raw actions. The action map is fixed during
    reuse. Minimizing max(ratio*A_cost, clipped_ratio*A_cost) is exactly the
    usual reward-PPO objective with reward=-cost. It is not a monotonic actual
    deployment guarantee and is not claimed as a novel policy optimizer.
    """
    if behavior_log_prob is None:
        raise ValueError("No behavior density: deterministic frozen Go2 records cannot be used as PPO samples")
    if behavior_log_prob.shape != new_log_prob.shape:
        raise ValueError("Behavior and candidate log probabilities must refer to the same recorded actions")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must lie between zero and one")
    advantage = _cost_advantage(new_log_prob, realized_returns, action_independent_baseline)
    log_ratio = new_log_prob - behavior_log_prob.detach()
    ratio = log_ratio.exp()
    surrogate = torch.maximum(ratio * advantage,
                              ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage)
    return PolicyLoss(surrogate.mean(), (ratio - 1.0 - log_ratio).detach().mean(),
                      ((ratio.detach() - 1.0).abs() > clip_epsilon).float().mean())


@torch.no_grad()
def factual_response_metrics(prediction: PredictedMacroBatch,
                              execution: ExecutedMacroBatch,
                              contract: ResponseContract = DEFAULT_CONTRACT) -> dict[str, Tensor]:
    """Available for deterministic frozen runs without any behavior density."""
    _check_pair(prediction, execution, contract)
    residual = feature_residual(prediction.features, execution.features, contract)
    physical_residual = residual * residual.new_tensor(contract.feature_scales)
    resource_error = prediction.resources - execution.resources
    return {"feature_rmse_native_units": physical_residual.square().mean((0, 1)).sqrt(),
            "feature_rmse_normalized": residual.square().mean((0, 1)).sqrt(),
            "resource_rmse": resource_error.square().mean().sqrt(),
            "resource_prediction_bias": resource_error.mean(),
            "actual_resource_mean": execution.resources.mean(),
            "macro_sample_count": torch.tensor(execution.resources.numel(), device=execution.resources.device)}
