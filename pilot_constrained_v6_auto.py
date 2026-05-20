"""
Paper 6 Pilot v6: PID-Lagrangian Constrained PPO.

Diagnostic comparator to V4. Tests whether the electronics non-convergence
is driven by asymmetric multiplier saturation (mechanism hypothesis) or
by structural testbed properties (action space, depth, etc.).

V6 differs from V4 in two ways:
  - Symmetric PID controller per multiplier (with anti-windup), so lambda
    can decrease when constraints are satisfied with margin.
  - Signed slack signal (positive when above floor, negative when below)
    so the integrator can wind down.

The penalty term during step() is identical to V4: lam * max(0, gap), so
within-episode behaviour matches V4 when constraints are violated. The
difference appears across episodes through different lambda dynamics.
"""

import os
import sys
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import FlexFlowSimEnv
from pilot_constrained_v6_pid import (
    PIDLagrangianController,
    PID_BAKERY,
    PID_ELECTRONICS,
)


class PIDLagrangianFlowEnv(gym.Wrapper):
    """
    PID-controlled dual update on Lagrangian multipliers.

    Each post-warmup step:
      - Accumulate signed per-step slack:
          util_slack = sum_i (u_i - util_floor) for i in fast_servers
          tp_slack   = tp_rate - tp_rate_floor
      - Accumulate V4-style one-sided gap (for diagnostic parity in logs).

    At episode end (in reset()):
      - Mean signed slack drives the PID controller:
          lam_util = pid_util.update(mean_util_slack)
          lam_tp   = pid_tp.update(mean_tp_slack)
    """

    def __init__(self, env, fast_server_indices,
                 util_floor=0.5, tp_rate_floor=18.0/480.0,
                 pid_config=None,
                 warmup_steps=100):
        super().__init__(env)
        if pid_config is None:
            raise ValueError("PIDLagrangianFlowEnv requires pid_config")

        self.fast = list(fast_server_indices)
        self.util_floor = float(util_floor)
        self.tp_rate_floor = float(tp_rate_floor)

        self.pid_util = PIDLagrangianController(
            Kp=pid_config["Kp_util"], Ki=pid_config["Ki_util"], Kd=pid_config["Kd_util"],
            lam_init=pid_config["lam_util_init"], lam_max=pid_config["lam_util_max"],
        )
        self.pid_tp = PIDLagrangianController(
            Kp=pid_config["Kp_tp"], Ki=pid_config["Ki_tp"], Kd=pid_config["Kd_tp"],
            lam_init=pid_config["lam_tp_init"], lam_max=pid_config["lam_tp_max"],
        )
        self.lam_util = self.pid_util.lam
        self.lam_tp = self.pid_tp.lam
        self.lam_util_max = float(pid_config["lam_util_max"])
        self.lam_tp_max = float(pid_config["lam_tp_max"])

        self.warmup = int(warmup_steps)
        self._step_count = 0

        # Signed slack (drives the PID); one-sided gap (V4-parity diagnostic)
        self._util_slack_sum = 0.0
        self._tp_slack_sum = 0.0
        self._util_gap_sum = 0.0
        self._tp_gap_sum = 0.0
        self._post_warmup_steps = 0

        # EMA smoothing of per-episode signed slack before feeding to PID.
        # Alpha 0.3 -> effective window ~5 episodes. Tests the SNR hypothesis:
        # the bare per-episode slack is dominated by noise around zero on the
        # throughput constraint, causing the controller to integrate noise.
        self._ema_alpha = 0.3
        self._util_slack_ema = 0.0
        self._tp_slack_ema = 0.0

        self.lambda_history = []
        self._episode = 0

    def reset(self, **kwargs):
        # PID update on previous episode's mean signed slack (EMA-smoothed)
        if self._post_warmup_steps > 0:
            mean_util_slack = self._util_slack_sum / self._post_warmup_steps
            mean_tp_slack = self._tp_slack_sum / self._post_warmup_steps
            mean_util_gap = self._util_gap_sum / self._post_warmup_steps
            mean_tp_gap = self._tp_gap_sum / self._post_warmup_steps

            # EMA smoothing applied before PID. Tests SNR hypothesis on tp_slack.
            self._util_slack_ema = (
                self._ema_alpha * mean_util_slack
                + (1.0 - self._ema_alpha) * self._util_slack_ema
            )
            self._tp_slack_ema = (
                self._ema_alpha * mean_tp_slack
                + (1.0 - self._ema_alpha) * self._tp_slack_ema
            )

            self.lam_util = self.pid_util.update(self._util_slack_ema)
            self.lam_tp = self.pid_tp.update(self._tp_slack_ema)

            self.lambda_history.append({
                "episode": self._episode,
                "lam_util": self.lam_util,
                "lam_tp": self.lam_tp,
                "mean_util_gap": mean_util_gap,
                "mean_tp_gap": mean_tp_gap,
                "mean_util_slack": mean_util_slack,
                "mean_tp_slack": mean_tp_slack,
                "util_slack_ema": self._util_slack_ema,
                "tp_slack_ema": self._tp_slack_ema,
            })

        self._step_count = 0
        self._util_slack_sum = 0.0
        self._tp_slack_sum = 0.0
        self._util_gap_sum = 0.0
        self._tp_gap_sum = 0.0
        self._post_warmup_steps = 0
        self._episode += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._step_count += 1

        util = info.get("utilisation", [0.0] * (max(self.fast) + 1))
        fast_utils = [float(util[i]) for i in self.fast]
        info["fast_util"] = fast_utils

        dep = info.get("total_departed", 0)
        sim_time = info.get("sim_time", self._step_count * self.env._dt)
        tp_rate = dep / max(sim_time, 1e-9)
        info["tp_rate"] = tp_rate

        # Signed per-step slack (always computed, regardless of warmup)
        util_slack = sum(u - self.util_floor for u in fast_utils)
        tp_slack = tp_rate - self.tp_rate_floor
        # One-sided gap (V4 convention)
        util_gap = sum(max(0.0, self.util_floor - u) for u in fast_utils)
        tp_gap = max(0.0, self.tp_rate_floor - tp_rate)

        penalty = 0.0
        if self._step_count > self.warmup:
            penalty = self.lam_util * util_gap + self.lam_tp * tp_gap

            self._util_slack_sum += util_slack
            self._tp_slack_sum += tp_slack
            self._util_gap_sum += util_gap
            self._tp_gap_sum += tp_gap
            self._post_warmup_steps += 1

        info["constraint_violation"] = penalty
        info["lam_util"] = self.lam_util
        info["lam_tp"] = self.lam_tp
        info["util_slack"] = util_slack
        info["tp_slack"] = tp_slack
        return obs, r - penalty, term, trunc, info


__all__ = [
    "PIDLagrangianFlowEnv",
    "PID_BAKERY",
    "PID_ELECTRONICS",
]
