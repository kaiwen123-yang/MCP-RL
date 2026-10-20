"""Apply recorded learned actions through the unchanged Go2 physical adapter."""
from __future__ import annotations

import numpy as np

from go2_frozen.simulator import Go2Sim


class LearningGo2Sim(Go2Sim):
    """Share the original scene, contacts, PD, scaling and stand-up procedure.

    Inherited ``step`` continues to use the original deterministic public actor,
    including the inherited reset/warmup sequence. ``step_action`` applies one
    externally sampled learned action for the same 20 ms policy interval.
    Neither method changes learned parameters.
    """

    def step_action(self, command, executed_action):
        """Execute an already clipped 12-vector; raw samples belong in the log.

        This function deliberately does not infer, resample, reclip or alter the
        action supplied by TaskActor.execute. The caller records its raw action
        and behavior log probability before invoking the physical step.
        """
        command = np.asarray(command, dtype=np.float64).reshape(3)
        action = np.asarray(executed_action, dtype=np.float64).reshape(12)
        lower = np.asarray(self.policy.config["clip_actions_lower"])
        upper = np.asarray(self.policy.config["clip_actions_upper"])
        if not np.all(np.isfinite(action)) or np.any(action < lower) or np.any(action > upper):
            raise ValueError("step_action requires finite actions already clipped to the public action bounds.")
        self.command = command.copy()
        self.policy.last_action = action.copy()
        self.policy.calls += 1
        self.q_des = self.policy.default_q + self.policy.action_scale * action
        for _ in range(self.controls_per_policy):
            self._pd_step(self.q_des, self.policy.kp, self.policy.kd)
        return self.observe()
