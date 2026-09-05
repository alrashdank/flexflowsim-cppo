"""R1 §5.2 / §6.5 analyses: stateless routing baselines (UniformRandom, RoundRobin)
on the test seeds, action-distribution entropy of selected checkpoints, and a
repeated-draw check of single-draw stochastic evaluation.
Outputs: results_r1/random_baselines.json, results_r1/policy_entropy.json,
         results_r1/repeated_draw_seed42.json
"""
import json, glob, numpy as np, torch
from stable_baselines3 import PPO
from env import FlexFlowSimEnv
from baselines import RandomPolicy, RoundRobinPolicy
from run_ablation_2x2 import TESTBED_CONFIG, TEST_SEEDS, evaluate_on_seeds
W = (0.8, 0.1, 0.1)

def run_rule(factory, tb, seeds):
    cfg = TESTBED_CONFIG[tb]; out = []
    for s in seeds:
        env = FlexFlowSimEnv(config=cfg["config"], weights=W, seed=int(s)); pol = factory(env, s)
        obs, _ = env.reset(); done = False
        while not done:
            a = pol.predict(obs); a = a[0] if isinstance(a, tuple) else a
            obs, _, t1, t2, info = env.step(a); done = t1 or t2
        u = info["utilisation"]; tp = info["total_departed"]; tc = info["total_cost"]
        out.append((tc / max(tp, 1e-9), tp, all(u[i] >= cfg["u_min"] for i in cfg["constrained_servers"]), tp >= cfg["tp_target"]))
    a = np.array(out, dtype=float); n = len(a); ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(n)
    return dict(cpu=a[:, 0].mean(), cpu_ci=ci(a[:, 0]), tp=a[:, 1].mean(), tp_ci=ci(a[:, 1]),
                usat=a[:, 2].mean(), tsat=a[:, 3].mean(), joint=np.mean(a[:, 2] * a[:, 3]))

def entropy(tb, cell, n_eps=10):
    cfg = TESTBED_CONFIG[tb]; rows = []
    for p in sorted(glob.glob(f"results_r1/{tb}/{cell}/seed_*/summary.json")):
        s = json.load(open(p)); m = PPO.load(p.replace("summary.json", f"checkpoints/ckpt_{s['selected_steps']}_steps"), device="cpu")
        H, top1, counts = [], [], None
        for es in TEST_SEEDS[:n_eps]:
            env = FlexFlowSimEnv(config=cfg["config"], weights=W, seed=int(es)); obs, _ = env.reset(); done = False
            nA = env.action_space.n; counts = np.zeros(nA) if counts is None else counts
            while not done:
                t, _ = m.policy.obs_to_tensor(obs)
                with torch.no_grad(): pr = m.policy.get_distribution(t).distribution.probs[0].numpy()
                H.append(-(pr * np.log(pr + 1e-12)).sum()); top1.append(pr.max())
                a = np.random.choice(nA, p=pr / pr.sum()); counts[a] += 1
                obs, _, t1, t2, _ = env.step(a); done = t1 or t2
        marg = counts / counts.sum()
        rows.append(dict(seed=s["seed"], Hnorm=float(np.mean(H) / np.log(nA)), eff_routes=float(np.exp(np.mean(H))),
                         top1=float(np.mean(top1)), marg_Hnorm=float(-(marg * np.log(marg + 1e-12)).sum() / np.log(nA)), nA=int(nA)))
    return rows

if __name__ == "__main__":
    res = {f"{tb}/{name}": run_rule(f, tb, TEST_SEEDS) for tb in ["bakery", "electronics"]
           for name, f in [("UniformRandom", lambda e, s: RandomPolicy(e, seed=int(s))), ("RoundRobin", lambda e, s: RoundRobinPolicy(e))]}
    json.dump(res, open("results_r1/random_baselines.json", "w"), indent=2)
    ent = {f"{tb}/{cell}": entropy(tb, cell) for tb in ["electronics", "bakery"] for cell in ["cost-episode", "shaped-cumrate"]}
    json.dump(ent, open("results_r1/policy_entropy.json", "w"), indent=2)
    s = json.load(open("results_r1/electronics/cost-episode/seed_42/summary.json"))
    m = PPO.load(f"results_r1/electronics/cost-episode/seed_42/checkpoints/ckpt_{s['selected_steps']}_steps", device="cpu")
    draws = []
    for d in range(5):
        torch.manual_seed(1000 + d); np.random.seed(1000 + d)
        r = evaluate_on_seeds(m, TESTBED_CONFIG["electronics"], W, TEST_SEEDS, "electronics", deterministic=False)
        draws.append((float(np.mean([x["cost_per_unit"] for x in r])), float(np.mean([min(x["util_satisfied"], x["tp_satisfied"]) for x in r]))))
    json.dump({"draws": draws}, open("results_r1/repeated_draw_seed42.json", "w"))
