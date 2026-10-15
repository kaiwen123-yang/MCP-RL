"""Response-codesign adapter using the unmodified upstream LayeredRL CEM.

This integrates original CEMPlanner/Predictor/Hierarchy/Level interfaces, not
a reproduction of SPlaTES' complete training algorithm or TD-MPC2 executor.
The upstream implementation is loaded from the recorded local checkout; it is
not copied, mocked, or reimplemented here. See ``integration-audit.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable

import torch


UPSTREAM_COMMIT = "37111d3538e18e4c6bcf1d98eef07407624c2e77"
UPSTREAM_URL = "https://github.com/nicoguertler/layeredrl"
_SOURCE_HASHES = {
    "optimizers/optimizer.py": "5958a40c319e99b0030e2e9231d0749b71725948643356492409651d2f1aa7ff",
    "utils/distributions.py": "7f94bc20cd1c8e86c4ab1fa2022bcf9b7120e0ce054c4c7251aeef3d20e2c160",
    "optimizers/cem.py": "2d70db106be77c344e6facd51a596fa0fa362a507f9053c86d602c094c2f54d4",
}


def source_provenance(method: Callable) -> dict[str, str]:
    """Return actual method identity and whole source-file hash for run logs."""
    source_file = Path(inspect.getsourcefile(method)).resolve()
    data = source_file.read_bytes()
    return {"qualname": method.__qualname__, "module": method.__module__,
            "source_file": str(source_file),
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "source_sha256_lf": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}


def _default_checkout() -> Path:
    return (Path(__file__).resolve().parents[2] / "work" /
            "four-paper-audit-2026-10-14" / "layeredrl-code")


def load_upstream_cem(checkout: str | Path | None = None):
    """Load actual upstream files under a private package namespace.

    Executing upstream package ``__init__`` imports the entire RL stack. The
    optimizer itself needs only NumPy and Torch, so normal importlib file loading
    isolates its real dependencies without stubbing Torch/Gym/Tianshou classes.
    Source hashes bind results to the exact reviewed upstream implementation.
    """
    checkout = Path(checkout) if checkout is not None else _default_checkout()
    source = checkout.resolve() / "layeredrl"
    for relative, expected in _SOURCE_HASHES.items():
        actual = hashlib.sha256((source / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"Upstream source differs from reviewed version: {relative}")
    namespace = "_response_codesign_upstream_" + hashlib.sha256(
        str(source).encode("utf-8")).hexdigest()[:12]
    if namespace + ".optimizers.cem" not in sys.modules:
        for suffix in ("", ".optimizers", ".utils"):
            module = ModuleType(namespace + suffix)
            module.__path__ = [str(source / suffix.lstrip("."))] if suffix else [str(source)]
            sys.modules[module.__name__] = module
        for relative in _SOURCE_HASHES:
            name = namespace + "." + relative.removesuffix(".py").replace("/", ".")
            spec = importlib.util.spec_from_file_location(name, source / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return sys.modules[namespace + ".optimizers.cem"].CEM


@dataclass
class PlanResult:
    action: torch.Tensor
    sequence: torch.Tensor
    cost: torch.Tensor
    margin: torch.Tensor
    feasible: torch.Tensor
    info: dict[str, Any]


class CEMResponsePlanner:
    """Receding-horizon adapter with explicit candidate feasibility checks.

    ``sequence_objective(state, commands, task)`` returns task costs and maximum
    constraint margins of shape ``[batch, samples]``. Commands have shape
    ``[batch, samples, horizon, command_dim]``; state is ``[batch, state_dim]``.
    Margins <= 0 denote a feasible candidate *under that callback's conditions*.
    Empirical residual checks are not universal physical safety certificates.

    The original CEM generates and updates proposals. A penalty steers search;
    final selection is the lowest task cost among checked feasible candidates.
    If none exists, ``feasible=False`` is returned explicitly. There is no
    implicit claim that zero or the least-violating sequence is a safe backup.

    This finite sampling algorithm is not a differentiable KKT optimum. Joint
    learning must use its declared score-function/zeroth-order estimator, or a
    separately derived differentiable planner rather than differentiating elite
    indices and calling the result a full optimizer sensitivity.
    """

    def __init__(
        self,
        sequence_objective: Callable,
        horizon: int = 4,
        command_dim: int = 1,
        command_bound: float = 1.0,
        n_samples: int = 64,
        n_iterations: int = 3,
        elite_ratio: float = 0.2,
        constraint_penalty: float = 1e4,
        device: str | torch.device = "cpu",
        checkout: str | Path | None = None,
    ):
        self.sequence_objective = sequence_objective
        self.horizon = horizon
        self.command_dim = command_dim
        self.command_bound = command_bound
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.elite_ratio = elite_ratio
        self.constraint_penalty = constraint_penalty
        self.device = torch.device(device)
        self.upstream_class = load_upstream_cem(checkout)
        self.warm_start = None
        self.n_plans = 0
        self.total_objective_calls = 0
        self.total_candidate_sequences = 0
        self.total_candidate_macro_steps = 0

    def objective_counts(self) -> dict[str, int]:
        return {"objective_calls": self.total_objective_calls,
                "candidate_sequences": self.total_candidate_sequences,
                "candidate_macro_steps": self.total_candidate_macro_steps}

    def _evaluate_sequences(self, state, commands, task):
        # Count at the actual callback boundary, including final reevaluation
        # of a full batch when at least one instance has no feasible candidate.
        sequences = commands.shape[0] * commands.shape[1]
        self.total_objective_calls += 1
        self.total_candidate_sequences += sequences
        self.total_candidate_macro_steps += sequences * commands.shape[2]
        return self.sequence_objective(state, commands, task)

    def _plan_counts(self, before: dict[str, int], batch: int) -> dict[str, int]:
        current = self.objective_counts()
        delta = {key: current[key] - before[key] for key in current}
        delta["evaluated_sequences_per_instance"] = delta["candidate_sequences"] // batch
        return delta

    def reset(self):
        self.warm_start = None

    def plan(self, state: torch.Tensor, task: Any, *, seed: int | None = None) -> PlanResult:
        counts_before = self.objective_counts()
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state[None]
        batch = state.shape[0]
        dims = self.horizon * self.command_dim
        if self.warm_start is None or self.warm_start.shape[0] != batch:
            self.warm_start = torch.zeros(batch, dims, device=self.device)
        optimizer = self.upstream_class(
            n_iterations=self.n_iterations, n_samples=self.n_samples,
            initial_sigma=torch.full((batch, dims), 0.7 * self.command_bound, device=self.device),
            elite_ratio=self.elite_ratio,
            lower_bound=torch.full((1, dims), -self.command_bound, device=self.device),
            upper_bound=torch.full((1, dims), self.command_bound, device=self.device),
            clip=True, return_mode="best", device=self.device,
        )
        best_cost = torch.full((batch,), torch.inf, device=self.device)
        best_margin = torch.full((batch,), torch.inf, device=self.device)
        best_sequence = self.warm_start.detach().clone()
        feasible_counts = torch.zeros(batch, dtype=torch.long, device=self.device)

        def evaluate(flat):
            nonlocal best_cost, best_margin, best_sequence
            commands = flat.reshape(batch, -1, self.horizon, self.command_dim)
            costs, margins = self._evaluate_sequences(state, commands, task)
            if costs.shape != flat.shape[:2] or margins.shape != costs.shape:
                raise ValueError("Objective must return cost and margin shaped [batch, samples]")
            valid = torch.isfinite(costs) & torch.isfinite(margins)
            feasible = valid & (margins <= 0)
            feasible_counts.add_(feasible.sum(dim=1))
            selected_cost, selected_index = torch.where(feasible, costs, torch.inf).min(dim=1)
            improve = selected_cost < best_cost
            row = torch.arange(batch, device=self.device)
            best_cost = torch.where(improve, selected_cost, best_cost)
            best_margin = torch.where(improve, margins[row, selected_index], best_margin)
            best_sequence = torch.where(improve[:, None], flat[row, selected_index], best_sequence)
            penalized = costs + self.constraint_penalty * torch.relu(margins)
            return torch.where(valid, penalized, torch.full_like(costs, 1e20))

        # The same warm-start and zero candidates are checked for every method.
        # Neither one is silently treated as a certified feasible fallback.
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.manual_seed(seed)
            evaluate(torch.stack((self.warm_start, torch.zeros_like(self.warm_start)), dim=1))
            optimizer.reset(self.warm_start)
            raw_sequence, raw_info = optimizer.optimize(evaluate)
            missing = ~torch.isfinite(best_cost)
            if missing.any():
                raw_cost, raw_margin = self._evaluate_sequences(
                    state, raw_sequence.reshape(batch, 1, self.horizon, self.command_dim), task)
                feasible_counts.add_((torch.isfinite(raw_cost) & torch.isfinite(raw_margin) &
                                      (raw_margin <= 0)).sum(dim=1))
                best_sequence = torch.where(missing[:, None], raw_sequence, best_sequence)
                best_cost = torch.where(missing, raw_cost[:, 0], best_cost)
                best_margin = torch.where(missing, raw_margin[:, 0], best_margin)
            sequence = best_sequence.reshape(batch, self.horizon, self.command_dim)
            feasible = torch.isfinite(best_margin) & (best_margin <= 0)
            self.warm_start = torch.cat((sequence[:, 1:], sequence[:, -1:]), dim=1).reshape(batch, dims)
        self.n_plans += 1
        return PlanResult(
            sequence[:, 0], sequence, best_cost, best_margin, feasible,
            {"upstream_class": self.upstream_class.__module__ + ".CEM",
             "upstream_commit": UPSTREAM_COMMIT, "upstream_file": sys.modules[self.upstream_class.__module__].__file__,
             **self._plan_counts(counts_before, batch),
             "feasible_candidates_per_instance": feasible_counts,
             "n_iterations": self.n_iterations, "n_samples": self.n_samples,
             "optimizer_info": raw_info, "integration_scope": "upstream-CEM-core"},
        )


def build_planner(sequence_objective: Callable, **kwargs) -> CEMResponsePlanner:
    return CEMResponsePlanner(sequence_objective, **kwargs)


def load_upstream_interfaces(checkout: str | Path | None = None) -> dict[str, type]:
    """Load original CEMPlanner, Predictor, Level and Hierarchy implementations.

    This requires real Gymnasium, Tianshou 0.5.1, TensorBoard, colorednoise and
    PyYAML. Package initializers are bypassed to avoid importing the unrelated
    complete TD-MPC2 learner. Every returned base class comes from upstream.
    """
    cem_class = load_upstream_cem(checkout)
    namespace = cem_class.__module__.split(".optimizers.cem")[0]
    source = Path(sys.modules[cem_class.__module__].__file__).parents[1]
    for suffix in ("models", "predictors", "planners", "levels", "hierarchies"):
        name = namespace + "." + suffix
        if name not in sys.modules:
            module = ModuleType(name)
            module.__path__ = [str(source / suffix)]
            sys.modules[name] = module
    # Set package exports to the actual already-loaded classes, not substitutes.
    optimizers = sys.modules[namespace + ".optimizers"]
    optimizers.CEM = cem_class
    optimizers.Optimizer = sys.modules[namespace + ".optimizers.optimizer"].Optimizer
    imports = (
        ("utils/misc.py", None),
        ("optimizers/icem.py", "ICEM"),
        ("models/model.py", "Model"),
        ("predictors/predictor.py", "Predictor"),
        ("planners/planner.py", "Planner"),
        ("planners/blackbox_planner.py", "BlackboxPlanner"),
        ("planners/cem_planner.py", "CEMPlanner"),
        ("levels/level.py", "Level"),
        ("hierarchies/hierarchy.py", "Hierarchy"),
    )
    result = {}
    for relative, export in imports:
        name = namespace + "." + relative.removesuffix(".py").replace("/", ".")
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, source / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        module = sys.modules[name]
        if export:
            value = getattr(module, export)
            setattr(sys.modules[name.rsplit(".", 1)[0]], export, value)
            result[export] = value
    return result


class SPlaTESPlannerBridge(CEMResponsePlanner):
    """Use upstream CEMPlanner/BlackboxPlanner plus a real Predictor subclass.

    The callback computes the full response rollout and task/realizability
    quantities. Its aggregate trajectory cost is placed in the first reward
    entry and remaining entries are zero. With continuation one and no terminal
    value, the unmodified upstream reward aggregation is exactly this objective.
    This retains the actual planner and predictor API, while replacing the
    original learned reward/value/MI objective explicitly.
    """

    def __init__(self, sequence_objective: Callable, **kwargs):
        self.interfaces = load_upstream_interfaces(kwargs.get("checkout"))
        super().__init__(sequence_objective, **kwargs)
        self.upstream_planner = None

    def plan(self, state: torch.Tensor, task: Any, *, seed: int | None = None) -> PlanResult:
        from gymnasium.spaces import Box
        from tianshou.data import Batch
        import numpy as np

        counts_before = self.objective_counts()
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state[None]
        batch, state_dim = state.shape
        dims = self.horizon * self.command_dim
        if self.warm_start is None or self.warm_start.shape[0] != batch:
            self.warm_start = torch.zeros(batch, dims, device=self.device)
        owner = self
        best_cost = torch.full((batch,), torch.inf, device=self.device)
        best_margin = torch.full((batch,), torch.inf, device=self.device)
        best_sequence = self.warm_start.clone()
        feasible_counts = torch.zeros(batch, dtype=torch.long, device=self.device)
        rollout_calls = 0

        def evaluate(commands):
            nonlocal best_cost, best_margin, best_sequence
            costs, margins = owner._evaluate_sequences(state, commands, task)
            valid = torch.isfinite(costs) & torch.isfinite(margins)
            feasible = valid & (margins <= 0)
            feasible_counts.add_(feasible.sum(dim=1))
            candidate_cost, index = torch.where(feasible, costs, torch.inf).min(dim=1)
            improve = candidate_cost < best_cost
            row = torch.arange(batch, device=owner.device)
            best_cost = torch.where(improve, candidate_cost, best_cost)
            best_margin = torch.where(improve, margins[row, index], best_margin)
            flat = commands.reshape(batch, commands.shape[1], dims)
            best_sequence = torch.where(improve[:, None], flat[row, index], best_sequence)
            penalty_cost = costs + owner.constraint_penalty * torch.relu(margins)
            return torch.where(valid, penalty_cost, torch.full_like(costs, 1e20))

        class ResponsePredictor(self.interfaces["Predictor"]):
            def __init__(self):
                super().__init__(model=None, val_func=None, rew_func=None,
                                 encoder=torch.nn.Identity(), latent_state_dim=state_dim,
                                 context_dim=0, device=owner.device)

            def rollout(self, obs, actions, deterministic=False):
                nonlocal rollout_calls
                rollout_calls += 1
                commands = actions.reshape(batch, -1, owner.horizon, owner.command_dim)
                total_cost = evaluate(commands).reshape(-1)
                reward = torch.zeros(actions.shape[0], 1, 1, owner.horizon, device=owner.device)
                reward[:, 0, 0, 0] = -total_cost
                # No fabricated latent trajectory: state is unused by this
                # planner configuration (terminal value/uncertainty are off).
                trajectory = Batch(rew=reward, term_prob=torch.zeros_like(reward))
                return trajectory, torch.empty(actions.shape[0], 0, device=owner.device)

            def loss(self, data):
                raise NotImplementedError("Independent response-policy trainer owns the joint loss")

            def learn(self, *args, **kwargs):
                raise NotImplementedError("Use the declared independent joint update, not SPlaTES MI")

        predictor = ResponsePredictor()
        planner = self.interfaces["CEMPlanner"](
            predictor=predictor,
            action_space=Box(-self.command_bound, self.command_bound,
                             shape=(self.command_dim,), dtype=np.float32),
            n_env_instances=batch, horizon=self.horizon,
            initial_sigma=0.7 * self.command_bound, use_icem=False,
            cem_params={"n_samples": self.n_samples, "n_iterations": self.n_iterations,
                        "elite_ratio": self.elite_ratio, "clip": True, "return_mode": "best"},
            deterministic=True, use_value_func=False, use_term_prob=False,
            device=self.device,
        )
        planner.reset(self.warm_start.reshape(batch, self.horizon, self.command_dim))
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.manual_seed(seed)
            evaluate(torch.stack((self.warm_start, torch.zeros_like(self.warm_start)), dim=1)
                     .reshape(batch, 2, self.horizon, self.command_dim))
            sequence_raw, raw_info = planner.plan(
                state, torch.ones(batch, dtype=torch.bool, device=self.device))
            missing = ~torch.isfinite(best_cost)
            if missing.any():
                raw_cost, raw_margin = self._evaluate_sequences(state, sequence_raw[:, None], task)
                feasible_counts.add_((torch.isfinite(raw_cost) & torch.isfinite(raw_margin) &
                                      (raw_margin <= 0)).sum(dim=1))
                best_sequence = torch.where(missing[:, None], sequence_raw.reshape(batch, dims), best_sequence)
                best_cost = torch.where(missing, raw_cost[:, 0], best_cost)
                best_margin = torch.where(missing, raw_margin[:, 0], best_margin)
            sequence = best_sequence.reshape(batch, self.horizon, self.command_dim)
            feasible = torch.isfinite(best_margin) & (best_margin <= 0)
            self.warm_start = torch.cat((sequence[:, 1:], sequence[:, -1:]), dim=1).reshape(batch, dims)
        self.upstream_planner = planner
        self.n_plans += 1
        info = {
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_class": type(planner).__module__ + ".CEMPlanner",
            "upstream_file": sys.modules[type(planner).__module__].__file__,
            "predictor_base_class": self.interfaces["Predictor"].__module__ + ".Predictor",
            "predictor_rollout_calls": rollout_calls,
            **self._plan_counts(counts_before, batch),
            "feasible_candidates_per_instance": feasible_counts,
            "optimizer_info": raw_info,
            "integration_scope": "upstream-CEMPlanner-and-Predictor",
            "upstream_methods": [source_provenance(planner.plan),
                                 source_provenance(planner.cost),
                                 source_provenance(planner.optimizer.optimize)],
            "predictor_base_source": source_provenance(self.interfaces["Predictor"].__init__),
        }
        return PlanResult(sequence[:, 0], sequence, best_cost, best_margin, feasible, info)


def build_splates_planner(sequence_objective: Callable, **kwargs) -> SPlaTESPlannerBridge:
    return SPlaTESPlannerBridge(sequence_objective, **kwargs)


def build_hierarchy(
    planner: CEMResponsePlanner,
    policy: Callable,
    env: Any,
    task_provider: Callable,
    command_steps: int,
    transition_callback: Callable | None = None,
    joint_update: Callable | None = None,
):
    """Create the actual upstream Hierarchy with custom original-Level subclasses.

    ``policy(observation, command)`` is the neural low-level policy. The actual
    environment remains supplied by the experiment. ``joint_update`` is called
    by the upstream ``Hierarchy.learn`` path, and the low level reports every
    transition to ``transition_callback``. This synchronously batched prototype
    does not implement mixed asynchronous resets or the original TD-MPC2 loss.
    """
    import numpy as np
    from gymnasium.spaces import Box

    upstream = load_upstream_interfaces()
    base = upstream["Level"]

    class ResponsePlanningLevel(base):
        def get_input_space(self):
            return None

        def get_action(self, mapped_env_obs, level_input, level_input_info, active_instances):
            task = task_provider(mapped_env_obs, active_instances)
            self.last_plan = planner.plan(mapped_env_obs, task)
            return self.last_plan.action, {"feasible": self.last_plan.feasible,
                                           "margin": self.last_plan.margin}

        def learn(self):
            if joint_update is not None:
                return joint_update()

    class NeuralExecutionLevel(base):
        def get_input_space(self):
            return Box(-planner.command_bound, planner.command_bound,
                       shape=(planner.command_dim,), dtype=np.float32)

        def get_action(self, mapped_env_obs, level_input, level_input_info, active_instances):
            return policy(mapped_env_obs, level_input), {}

        def process_transition(self, mapped_env_obs, level_input, action, next_mapped_env_obs,
                               terminated, truncated, active_instances):
            if transition_callback is not None:
                transition_callback(mapped_env_obs, level_input, action, next_mapped_env_obs,
                                    terminated, truncated)
            super().process_transition(mapped_env_obs, level_input, action, next_mapped_env_obs,
                                       terminated, truncated, active_instances)
            return self.n_steps_in_control[active_instances] >= command_steps

    levels = [ResponsePlanningLevel(device=planner.device), NeuralExecutionLevel(device=planner.device)]
    hierarchy = upstream["Hierarchy"](levels=levels, env=env, device=planner.device)
    hierarchy.integration_provenance = {
        "upstream_methods": [source_provenance(hierarchy.get_action),
                             source_provenance(hierarchy.process_transition),
                             source_provenance(hierarchy.learn)],
        "level_base_source": source_provenance(base.__init__),
    }
    hierarchy.reset()
    return hierarchy
