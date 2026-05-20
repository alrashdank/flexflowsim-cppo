"""
Paper 6 Pilot v4: Auto-tuned Lagrangian PPO.

Lambdas update per episode based on running constraint-violation
estimates (PID-style integral term). This is the standard CMDP
machinery (Stooke et al. 2020, Tessler et al. 2019).

Algorithm:
  - Track per-episode constraint violations (util gap, throughput gap)
  - At episode end, update λ:
      λ ← clip( λ + lr_lambda * mean_violation, 0, λ_max )
  - λ rises when violations occur, falls when constraints satisfied
  - Penalty during step: λ * gap (same as fixed-λ, but λ is dynamic)

Constraints (same as v2/v3):
  - Fast-server cumulative util ≥ 0.50
  - Cumulative throughput rate ≥ 18/480 units/min
"""

import os
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import FlexFlowSimEnv


class AutoLagrangianFlowEnv(gym.Wrapper):
    """
    Auto-tuned dual ascent on Lagrangian multipliers.

    Each episode:
      - Track average per-step constraint gap (after warmup)
      - At terminal step, update λ_util and λ_tp via dual ascent:
          λ ← max(0, min(λ + lr * mean_gap, λ_max))

    This mimics the discrete dual-ascent step in Stooke et al. (2020).
    """

    def __init__(self, env, fast_server_indices,
                 util_floor=0.5, tp_rate_floor=18.0/480.0,
                 lam_util_init=5.0, lam_tp_init=200.0,
                 lr_util=20.0, lr_tp=4000.0,
                 lam_util_max=500.0, lam_tp_max=20000.0,
                 warmup_steps=100):
        super().__init__(env)
        self.fast = list(fast_server_indices)
        self.util_floor = float(util_floor)
        self.tp_rate_floor = float(tp_rate_floor)

        self.lam_util = float(lam_util_init)
        self.lam_tp = float(lam_tp_init)
        self.lr_util = float(lr_util)
        self.lr_tp = float(lr_tp)
        self.lam_util_max = float(lam_util_max)
        self.lam_tp_max = float(lam_tp_max)

        self.warmup = int(warmup_steps)
        self._step_count = 0

        # Per-episode running tallies
        self._util_gap_sum = 0.0
        self._tp_gap_sum = 0.0
        self._violation_steps = 0

        # Logging
        self.lambda_history = []  # list of (episode, lam_util, lam_tp, util_gap, tp_gap)
        self._episode = 0

    def reset(self, **kwargs):
        # Dual update on PREVIOUS episode's mean violation
        if self._violation_steps > 0:
            mean_util_gap = self._util_gap_sum / self._violation_steps
            mean_tp_gap = self._tp_gap_sum / self._violation_steps

            self.lam_util = float(np.clip(
                self.lam_util + self.lr_util * mean_util_gap,
                0.0, self.lam_util_max))
            self.lam_tp = float(np.clip(
                self.lam_tp + self.lr_tp * mean_tp_gap,
                0.0, self.lam_tp_max))

            self.lambda_history.append({
                "episode": self._episode,
                "lam_util": self.lam_util,
                "lam_tp": self.lam_tp,
                "mean_util_gap": mean_util_gap,
                "mean_tp_gap": mean_tp_gap,
            })

        self._step_count = 0
        self._util_gap_sum = 0.0
        self._tp_gap_sum = 0.0
        self._violation_steps = 0
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

        penalty = 0.0
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

        info["constraint_violation"] = penalty
        info["lam_util"] = self.lam_util
        info["lam_tp"] = self.lam_tp
        return obs, r - penalty, term, trunc, info


def evaluate_policy(model, make_env, n_eps=10, seed_base=10_000):
    rows = []
    for k in range(n_eps):
        env = make_env(seed=seed_base + k)
        obs, info = env.reset(seed=seed_base + k)
        total_r = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            if term or trunc:
                break
        dep = info["total_departed"]
        rows.append({
            "totalCost": info["total_cost"],
            "totalDeparted": dep,
            "costPerUnit": info["total_cost"] / max(dep, 1),
            "avgLeadTime": info["avg_lead_time"],
            "util": list(info["utilisation"]),
            "reward": total_r,
        })
    return rows


def summarise(label, rows, fast_idx):
    arr = lambda k: np.array([r[k] for r in rows])
    util_arr = np.array([r["util"] for r in rows])
    means = util_arr.mean(axis=0)
    print(f"\n--- {label} (n={len(rows)}) ---")
    print(f"  Throughput:        {arr('totalDeparted').mean():6.2f} ± {arr('totalDeparted').std():.2f}")
    print(f"  Cost per unit:     {arr('costPerUnit').mean():6.2f} ± {arr('costPerUnit').std():.2f}")
    print(f"  Total cost:        {arr('totalCost').mean():6.2f} ± {arr('totalCost').std():.2f}")
    print(f"  Per-server util:   {[f'{u:.3f}' for u in means]}")


CONFIG = "configs/bakery_bk50.json"
WEIGHTS = (0.8, 0.1, 0.1)
FAST_INDICES = [0, 2]
UTIL_FLOOR = 0.50
TP_FLOOR = 18.0 / 480.0
TIMESTEPS = 120_000   # Slightly longer to give λ time to converge
SEED = 42


def make_unconstrained(seed):
    return FlexFlowSimEnv(config=CONFIG, weights=WEIGHTS, seed=seed)


# Singleton wrapper to expose lambda_history at end
WRAPPER_HOLDER = {"env": None}


def make_auto(seed):
    base = FlexFlowSimEnv(config=CONFIG, weights=WEIGHTS, seed=seed)
    wrapped = AutoLagrangianFlowEnv(
        base, FAST_INDICES,
        util_floor=UTIL_FLOOR, tp_rate_floor=TP_FLOOR,
    )
    WRAPPER_HOLDER["env"] = wrapped
    return wrapped


def train_auto(timesteps, seed):
    print(f"\n=== Training Auto-Lagrangian PPO ({timesteps} steps, seed={seed}) ===")
    t0 = time.time()
    env = Monitor(make_auto(seed))
    model = PPO("MlpPolicy", env, seed=seed, verbose=0,
                learning_rate=3e-4, n_steps=2048, batch_size=64,
                gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    print(f"  Training time: {(time.time()-t0)/60:.1f} min")
    return model


if __name__ == "__main__":
    model = train_auto(TIMESTEPS, SEED)

    # Show lambda evolution
    history = WRAPPER_HOLDER["env"].lambda_history
    print(f"\n--- Lambda evolution ({len(history)} episodes tracked) ---")
    if len(history) >= 5:
        idxs = np.linspace(0, len(history) - 1, 10).astype(int)
        print(f"  {'ep':>5} {'λ_util':>10} {'λ_tp':>10} {'util_gap':>10} {'tp_gap':>10}")
        for i in idxs:
            h = history[i]
            print(f"  {h['episode']:>5} {h['lam_util']:>10.2f} {h['lam_tp']:>10.2f} "
                  f"{h['mean_util_gap']:>10.4f} {h['mean_tp_gap']:>10.5f}")

    print("\n" + "=" * 60)
    print("EVALUATION on UNCONSTRAINED env (10 episodes)")
    print("=" * 60)

    rows = evaluate_policy(model, make_unconstrained, n_eps=10)
    summarise("Auto-Lagrangian PPO", rows, FAST_INDICES)

    print("\n--- Reference: LeastUtilised ---")
    print("  Throughput:        19.60 ± 2.09")
    print("  Cost per unit:    144.37 ± 20.18")
    print("  Per-server util:  [0.874, 0.579, 0.945, 0.792]")
    print("\n--- Pilot v1 (util only, fixed λ=5) ---")
    print("  Throughput: 12.30, CPU: 231.48, util: [0.969, 0.055, 0.936, 0.037]")
    print("\n--- Pilot v2 (util+tp, fixed λ=5/200) ---")
    print("  Throughput: 13.40, CPU: 207.47, util: [0.633, 0.684, 0.409, 0.922]")
    print("\n--- Pilot v3 (util+tp, fixed λ=50/2000) ---")
    print("  Throughput:  8.70, CPU: 307.61, util: [0.970, 0.000, 0.000, 0.939]")

    print("\n=== CONSTRAINT SATISFACTION (Auto) ===")
    util_arr = np.array([r["util"] for r in rows])
    tp_arr = np.array([r["totalDeparted"] for r in rows])
    n = len(rows)
    util_v = sum(any(util_arr[k, i] < UTIL_FLOOR for i in FAST_INDICES) for k in range(n))
    tp_v = sum(tp_arr[k] < 18 for k in range(n))
    print(f"  Fast-server util ≥50%: {n - util_v}/{n} satisfied")
    print(f"  Throughput ≥18 units:  {n - tp_v}/{n} satisfied")
    print(f"  Final λ_util: {WRAPPER_HOLDER['env'].lam_util:.2f}")
    print(f"  Final λ_tp:   {WRAPPER_HOLDER['env'].lam_tp:.2f}")
