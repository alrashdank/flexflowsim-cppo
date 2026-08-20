"""
Paper 6 revision: slack-corrected Lagrangian wrapper.

Motivation (code audit, 20 Aug 2026)
------------------------------------
The published V4 wrapper (pilot_constrained_v4_auto.AutoLagrangianFlowEnv)
penalises the shortfall of the CUMULATIVE throughput rate
(total_departed / sim_time) against T_min/H at every step. Because jobs need
~55-110 min to traverse the line, this signal is irreducibly positive early in
every episode for ANY policy: ShortestQueue itself accrues a mean per-step gap
of 0.001-0.006 while satisfying the episode-level constraint by 20%+. Combined
with the one-sided dual update (lr_tp=4000, no release), lambda_T saturation
within the training budget is guaranteed even for oracle policies, and on
electronics a policy achieving exactly TP = T_min NEVER satisfies the per-step
signal within the horizon. The training constraint therefore neither matches
the stated CMDP constraint (E[episode throughput] >= T_min, manuscript §3.3)
nor the validation/test criterion (episode totals).

This wrapper provides two slack modes so the published behaviour can be
ablated against the corrected one:

  slack_mode="cumrate"  Faithful re-implementation of the published V4
                        signal (control cell). Per-step hinged penalty on the
                        cumulative-rate shortfall after `warmup_steps`; dual
                        update from mean per-step hinged gaps.

  slack_mode="episode"  Corrected signal. Per-step penalty is the SIGNED
                        pro-rata decomposition of the episode Lagrangian:

                          tp:    lam_T  * (T_min/H - new_departures_t)
                          util:  lam_U  * sum_i (U_min - busy_{i,t})

                        which telescopes exactly to
                          lam_T*(T_min - TP_ep) + lam_U*H*sum_i(U_min - mean_util_i),
                        i.e. the Lagrangian of the constraint actually stated
                        in §3.3 and checked at validation/test. No warmup, no
                        ramp bias: an action's advantage depends only on the
                        departures/busy-time it causes. Dual update at episode
                        end from EPISODE-level slack:
                          hinged  (default): max(0, T_min - TP_ep)/H  -> the
                                  one-sided ratchet STOPS at satisfaction
                                  (unbiased signal => convergent, not
                                  guaranteed-divergent);
                          signed  (symmetric=True): (T_min - TP_ep)/H  -> V6-lite,
                                  lambda can decrease on over-satisfaction
                                  (still clipped to [0, lam_max]).

Multiplier scales are unit-compatible with the published V4 wrapper (per-step
rate units for tp, per-step utilisation units for util), so lam inits, learning
rates, and caps carry over unchanged and results are directly comparable.

Integration with run_experiment.py (6-line patch):
    from lagrangian_slack import SlackLagrangianFlowEnv
    # in make_env(), add a dispatch branch:
    if method in ("v4ep", "v4cum"):
        return SlackLagrangianFlowEnv(
            base, testbed_cfg["constrained_servers"],
            util_floor=u_min, tp_rate_floor=t_min / 480.0,
            slack_mode=("episode" if method == "v4ep" else "cumrate"))
Alternatively use run_ablation_2x2.py, which drives the full 2x2 with the
published validation-selection and test protocol.
"""

import numpy as np
import gymnasium as gym


class SlackLagrangianFlowEnv(gym.Wrapper):
    """Lagrangian wrapper with selectable slack signal (see module docstring)."""

    def __init__(self, env, fast_server_indices,
                 util_floor=0.5, tp_rate_floor=18.0 / 480.0,
                 lam_util_init=5.0, lam_tp_init=200.0,
                 lr_util=20.0, lr_tp=4000.0,
                 lam_util_max=500.0, lam_tp_max=20000.0,
                 warmup_steps=100,
                 slack_mode="episode",
                 symmetric=False,
                 horizon=480):
        super().__init__(env)
        if slack_mode not in ("episode", "cumrate"):
            raise ValueError(f"slack_mode must be 'episode' or 'cumrate', got {slack_mode!r}")
        self.fast = list(fast_server_indices)
        self.util_floor = float(util_floor)
        self.tp_rate_floor = float(tp_rate_floor)
        self.slack_mode = slack_mode
        self.symmetric = bool(symmetric)
        self.horizon = int(horizon)

        self.lam_util = float(lam_util_init)
        self.lam_tp = float(lam_tp_init)
        self.lr_util = float(lr_util)
        self.lr_tp = float(lr_tp)
        self.lam_util_max = float(lam_util_max)
        self.lam_tp_max = float(lam_tp_max)

        # warmup applies to the cumrate (published) signal only; the episode
        # signal must cover every step or the telescoping identity breaks.
        self.warmup = int(warmup_steps) if slack_mode == "cumrate" else 0

        # Per-episode running tallies (kept name-compatible with the V4 wrapper
        # so existing logging/analysis code keeps working).
        self._step_count = 0
        self._util_gap_sum = 0.0
        self._tp_gap_sum = 0.0
        self._violation_steps = 0
        self._episode = 0
        self._last_info = None

        self.lambda_history = []  # per-episode dicts, same keys as V4 + slack fields

    # ------------------------------------------------------------------
    # Dual update (called at episode boundary, i.e. on reset of a finished ep)
    # ------------------------------------------------------------------
    def _dual_update(self):
        if self.slack_mode == "cumrate":
            # Published V4 behaviour: mean per-step hinged gaps.
            if self._violation_steps == 0:
                return
            mean_util_gap = self._util_gap_sum / self._violation_steps
            mean_tp_gap = self._tp_gap_sum / self._violation_steps
        else:
            # Episode-level slack from the finished episode's terminal info.
            if self._last_info is None:
                return
            tp_ep = float(self._last_info.get("total_departed", 0))
            t_min = self.tp_rate_floor * self.horizon
            tp_slack = (t_min - tp_ep) / self.horizon      # per-step rate units
            util = self._last_info.get("utilisation", [])
            margins = [float(util[i]) - self.util_floor for i in self.fast]

            if self.symmetric:
                mean_tp_gap = tp_slack                      # signed
                if any(m < 0 for m in margins):
                    mean_util_gap = sum(-m for m in margins if m < 0)
                else:
                    mean_util_gap = -min(margins)           # release at the binding server's margin
            else:
                mean_tp_gap = max(0.0, tp_slack)            # hinged: stops at satisfaction
                mean_util_gap = sum(max(0.0, -m) for m in margins)

        self.lam_util = float(np.clip(
            self.lam_util + self.lr_util * mean_util_gap, 0.0, self.lam_util_max))
        self.lam_tp = float(np.clip(
            self.lam_tp + self.lr_tp * mean_tp_gap, 0.0, self.lam_tp_max))

        self.lambda_history.append({
            "episode": self._episode,
            "lam_util": self.lam_util,
            "lam_tp": self.lam_tp,
            "util_gap": float(mean_util_gap),
            "tp_gap": float(mean_tp_gap),
        })

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        if self._step_count > 0:
            self._dual_update()
        self._step_count = 0
        self._util_gap_sum = 0.0
        self._tp_gap_sum = 0.0
        self._violation_steps = 0
        self._episode += 1
        self._last_info = None
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._step_count += 1

        util = info.get("utilisation", [0.0] * (max(self.fast) + 1))
        fast_utils = [float(util[i]) for i in self.fast]
        info["fast_util"] = fast_utils

        dep = info.get("total_departed", 0)
        sim_time = info.get("sim_time", self._step_count)
        tp_rate = dep / max(sim_time, 1e-9)
        info["tp_rate"] = tp_rate

        penalty = 0.0

        if self.slack_mode == "cumrate":
            # ---- published V4 signal (control cell) ----
            util_gap = 0.0
            tp_gap = 0.0
            if self._step_count > self.warmup:
                for u in fast_utils:
                    if u < self.util_floor:
                        g = self.util_floor - u
                        util_gap += g
                        penalty += self.lam_util * g
                if tp_rate < self.tp_rate_floor:
                    tp_gap = self.tp_rate_floor - tp_rate
                    penalty += self.lam_tp * tp_gap
                self._util_gap_sum += util_gap
                self._tp_gap_sum += tp_gap
                self._violation_steps += 1
        else:
            # ---- corrected episode signal: signed pro-rata decomposition ----
            new_dep = float(info.get("new_departures", 0))
            penalty += self.lam_tp * (self.tp_rate_floor - new_dep)
            in_service = getattr(self.env.unwrapped, "_in_service", None)
            if in_service is not None:
                for i in self.fast:
                    busy = float(min(in_service[i], 1.0))
                    penalty += self.lam_util * (self.util_floor - busy)
            # tallies kept for logging parity (hinged, diagnostic only)
            self._tp_gap_sum += max(0.0, self.tp_rate_floor - tp_rate)
            self._violation_steps += 1

        self._last_info = info
        info["constraint_violation"] = penalty
        info["lam_util"] = self.lam_util
        info["lam_tp"] = self.lam_tp
        return obs, r - penalty, term, trunc, info
