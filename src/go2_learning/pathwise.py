"""Train feedback against the MPC's whole selected command sequence.

At each macro step the factual response g is corrected by the action model's
change in response when the old feedback is replaced by the current feedback:

    G_theta(x,c) = g_old(x,z,c) + F_f,theta(x,c) - F_f,old(x,c).

Both F branches start at the same state and retain state derivatives. At the
old policy the correction vanishes, so a long rollout uses the calibrated g
instead of accumulating hundreds of unanchored micro-model predictions.
This is a local model of policy change, not a newly measured response. Only
subsequent real execution can provide the labels for updating g.
"""
from __future__ import annotations

import copy
from time import perf_counter

import torch

from go2_frozen.task import Task, make_cases

from .mission import mission_cost
from .state import advance_memory, task_active, task_context
from .continuation import make_task, resolve_cases


def _frozen(module):
    result = copy.deepcopy(module).eval()
    result.requires_grad_(False)
    return result


def _wrap(angles):
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def policy_response(actor, dynamics, state, memory, command, spec, steps):
    """One held command, with feedback recomputed at every physical model step."""
    for _ in range(steps):
        alive = task_active(memory, spec)
        action = actor.mean(state, task_context(state, memory, spec), command)
        proposed = dynamics(state, action)
        state = torch.where(alive[..., None], proposed, state)
        advanced_memory = advance_memory(state, memory, spec, dynamics.dt)
        memory = torch.where(alive[..., None], advanced_memory, memory)
    return state


def changed_response(actor, old_actor, dynamics, response, state, memory,
                     command, spec):
    """Local policy-dependent response; never writes to a factual model."""
    steps = round(response.dt / dynamics.dt)
    predicted = response(state, task_context(state, memory, spec), command)
    current = policy_response(actor, dynamics, state, memory, command, spec, steps)
    # Keep the old branch's input derivative: detaching it double-counts the
    # state response at theta_old instead of cancelling the two F Jacobians.
    old = policy_response(old_actor, dynamics, state, memory, command, spec, steps)
    delta = current - old
    delta = torch.cat((delta[..., :3], _wrap(current[..., 3:6] - old[..., 3:6]),
                       delta[..., 6:]), dim=-1)
    next_state = predicted + delta
    return torch.cat((next_state[..., :3], _wrap(next_state[..., 3:6]),
                      next_state[..., 6:]), dim=-1)


def planned_task_loss(actor, old_actor, dynamics, response, value, states,
                      memories, plans, previous_commands, spec, *,
                      speed_band="fast", collision_margin=0.04):
    """Direct actor objective across all macro commands and an actual-return tail.

    The geometry cost is the same piecewise-differentiable planning proxy used
    by LearnedMPC, not a gradient of the hard physical completion time.
    The PPO term continues to use only actual time/failure returns.
    """
    x, memory = states, memories
    state_trace, memory_trace = [], []
    for k in range(plans.shape[1]):
        alive = task_active(memory, spec)
        proposed = changed_response(actor, old_actor, dynamics, response,
                                    x, memory, plans[:, k], spec)
        x = torch.where(alive[:, None], proposed, x)
        advanced_memory = advance_memory(x, memory, spec, response.dt)
        memory = torch.where(alive[:, None], advanced_memory, memory)
        state_trace.append(x)
        memory_trace.append(memory)
    states_out = torch.stack(state_trace, dim=1)
    memories_out = torch.stack(memory_trace, dim=1)
    cost = mission_cost(states_out, memories_out, plans, memories,
                        previous_commands, make_task(spec), response.dt,
                        speed_band=speed_band, collision_margin=collision_margin)
    unfinished = task_active(memory, spec)
    tail = value(x, task_context(x, memory, spec), plans[:, -1])
    cost = cost - torch.where(unfinished, tail, torch.zeros_like(tail))
    return cost.mean(), states_out, memories_out


class PlannedResponseLoss:
    """A frozen-model, whole-plan actor objective for one on-policy PPO batch."""

    def __init__(self, system, anchors, config):
        self.actor = system["actor"]
        actor_parameter = next(self.actor.parameters())
        self.device, self.dtype = actor_parameter.device, actor_parameter.dtype
        self.old_actor = _frozen(self.actor)
        self.dynamics = _frozen(system["dynamics"]).to(device=self.device, dtype=self.dtype)
        self.response = _frozen(system["response"]).to(device=self.device, dtype=self.dtype)
        self.value = _frozen(system["value"]).to(device=self.device, dtype=self.dtype)
        self.anchors = anchors
        self.case_indices = anchors["case_index"].detach().cpu()
        settings = config["pathwise"]
        self.coefficient = float(settings["coefficient"])
        self.batch_size = int(settings["anchors_per_update"])
        self.speed_band = config["speed_band"]
        self.collision_margin = float(config["planner"].get("collision_margin", 0.04))
        cases = {case.name: case for case in resolve_cases(config)}
        self.specs = [cases[name] for name in config["cases"]]
        # Cycle through all cases and throughout the actual episode, rather
        # than repeatedly learning from the first straight segment.
        generator = torch.Generator().manual_seed(int(settings["anchor_seed"]))
        self.order = torch.randperm(len(anchors["state"]), generator=generator)
        self.cursor = 0
        self.calls = self.macro_steps = self.micro_steps = 0
        self.seconds = 0.0

    def __call__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        n = min(self.batch_size, len(self.order))
        positions = (torch.arange(n) + self.cursor) % len(self.order)
        indices = self.order[positions]
        self.cursor = (self.cursor + n) % len(self.order)
        total = torch.zeros((), device=self.device, dtype=self.dtype)
        for case_index, spec in enumerate(self.specs):
            chosen = indices[self.case_indices[indices] == case_index]
            if len(chosen) == 0:
                continue
            selected = {
                key: self.anchors[key][chosen].to(device=self.device, dtype=self.dtype)
                for key in ("state", "memory", "command_sequence", "previous_command")
            }
            cost, _, _ = planned_task_loss(
                self.actor, self.old_actor, self.dynamics, self.response, self.value,
                selected["state"], selected["memory"], selected["command_sequence"],
                selected["previous_command"], spec,
                speed_band=self.speed_band, collision_margin=self.collision_margin)
            total = total + cost * len(chosen) / n
        horizon = self.anchors["command_sequence"].shape[1]
        self.calls += 1
        self.macro_steps += n * horizon
        self.micro_steps += n * horizon * round(self.response.dt / self.dynamics.dt) * 2
        # Express the auxiliary scale per planned second, independent of the
        # number of macro samples. The coefficient is fixed in the config.
        loss = self.coefficient * total / (horizon * self.response.dt)
        metrics = {
            "future_task_cost": float(total.detach()),
            "future_actor_loss": float(loss.detach()),
            "future_plan_seconds": horizon * self.response.dt,
            "future_anchor_count": n,
        }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seconds += perf_counter() - started
        return loss, metrics

    def statistics(self):
        return {
            "pathwise_calls": self.calls,
            "pathwise_macro_steps": self.macro_steps,
            "pathwise_micro_steps_both_policies": self.micro_steps,
            "pathwise_forward_seconds": self.seconds,
            "model_task_updates": 0,
            "response_rule": "factual_g_plus_current_minus_old_closed_loop_f_response",
            "task_cost": "shared planning geometry and completion proxy minus actual-return V",
        }
