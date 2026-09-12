"""Verify every quantitative claim in IJMS_manuscript.md against the archived result files."""
import json, re, glob, numpy as np, csv, pathlib
from run_ablation_2x2 import TESTBED_CONFIG, T_CRIT
M = open("/home/claude/draft/IJMS_manuscript.md").read().replace("\u2212","-")
fails=[]; checks=0
def has(s, label):
    global checks; checks+=1
    if s not in M: fails.append(f"{label}: '{s}' NOT FOUND")
def ci(xs): xs=np.asarray(xs,float); n=len(xs); return float(T_CRIT.get(n,1.96)*xs.std(ddof=1)/np.sqrt(n)) if n>1 else 0.0

OUT=pathlib.Path("results_r1/r2")
# ---------- Table 1: dispatching baselines (recompute from per-episode records) ----------
NAMES={"ShortestQueue":"ShortestQueue","LeastUtilised":"LeastUtilised","RoundRobin":"RoundRobin","UniformRandom":"UniformRandom"}
for tb,label in [("bakery","Bakery"),("electronics","Electronics")]:
    d=json.load(open(OUT/f"episodes_{tb}_base.json")); cfg=TESTBED_CONFIG[tb]; f=cfg["constrained_servers"]
    for n in NAMES:
        r=d[f"baseline/{n}"]
        tp=np.array([x["tp"] for x in r]); cpu=np.array([x["cost"]/x["tp"] for x in r])
        u=np.array([[x["util"][i] for i in f] for x in r])
        us=(u.min(1)>=cfg["u_min"]).mean(); ts=(tp>=cfg["tp_target"]).mean()
        j=((tp>=cfg["tp_target"])&(u.min(1)>=cfg["u_min"])).mean()
        z=1.959964
        row=f"| {label}, {n} | {tp.mean():.2f} ± {z*tp.std(ddof=1)/np.sqrt(50):.2f} | ${cpu.mean():.2f} ± {z*cpu.std(ddof=1)/np.sqrt(50):.2f} | {us:.0%} | {ts:.0%} | {j:.0%} |"
        has(row, f"Table 1 {label} {n}")

# ---------- Table 2: full-budget ----------
N=json.load(open(OUT/"r2_numbers.json"))
for tb,cell,lab in [("electronics","cost-episode","corrected"),("electronics","shaped-cumrate","original signal"),
                    ("bakery","cost-episode","corrected"),("bakery","shaped-cumrate","original signal")]:
    v=N[f"T3/{tb}/{cell}"]
    row=(f"| {tb} | {lab} | ${v['stoch']['cpu']:.2f} ± {v['stoch']['cpu_ci']:.2f} | {v['stoch']['joint']*100:.1f}% | "
         f"{v['val_satisfied']}/5 | {v['lam_cap']}/5 | ${v['argmax']['cpu']:.2f} ± {v['argmax']['cpu_ci']:.2f} | {v['argmax']['joint']*100:.1f}% |")
    has(row, f"Table 2 {tb} {lab}")

# ---------- Table 3: bootstrap ----------
def fmt(b,pp): 
    return (f"{b['diff']*100:+.1f} [{b['ci'][0]*100:+.1f}, {b['ci'][1]*100:+.1f}]".replace("[+0.0,","[0.0,") if pp
            else f"{b['diff']:+.2f} [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}]")
def pf(p): return "< 0.001" if p<0.001 else (f"{p:.3f}" if p<0.1 else f"{p:.2f}")
for tb in ["electronics","bakery"]:
    S=json.load(open(OUT/f"stats_{tb}.json"))
    for base in ["ShortestQueue","LeastUtilised","RoundRobin","UniformRandom"]:
        for key,pp in [("cpu",False),("joint",True)]:
            b=S["bootstrap"][f"cost-episode/stoch/{key} vs {base}"]
            has(fmt(b,pp), f"Table 3 {tb} corrected vs {base} {key}"); has(pf(b['p']), f"p {tb} {base} {key}")
    for key,pp in [("cpu",False),("joint",True)]:
        b=S["bootstrap"][f"cost-episode vs shaped-cumrate/stoch/{key}"]; has(fmt(b,pp), f"Table 3 {tb} vs original {key}")
S=json.load(open(OUT/"stats_electronics_r2.json"))
for comp in ["ShortestQueue","LeastUtilised","RoundRobin","UniformRandom","cost-episode","shaped-cumrate"]:
    for key,pp in [("cpu",False),("joint",True)]:
        b=S["bootstrap"][f"cost-episode-sym vs {comp}/{key}"]; has(fmt(b,pp), f"Table 3 symmetric vs {comp} {key}")

# ---------- Table 4: archival ----------
for run,lab in [("v4_electronics","One-sided, electronics"),("v6_electronics","PID, electronics"),
                ("v4_bakery","One-sided, bakery"),("v6_bakery","PID, bakery")]:
    r=N[f"T4/{run}"]
    for q,tag in [("q1_argmax","Q1 argmax"),("q1_stoch","Q1 stoch"),("q2_stoch","Q2 stoch")]:
        has(f"${r[q][0]:.2f} ± {r[q][1]:.2f}", f"Table 4 {lab} {tag}"); has(f"{r[q][2]*100:.0f}%", f"Table 4 {lab} {tag} sat")

# ---------- Table 5: symmetric per seed ----------
S5=[json.load(open(p)) for p in sorted(glob.glob("results_r2/electronics/cost-episode-sym/seed_*/summary.json"))]
for s in S5:
    has(f"${s['stoch_test_mean_cpu']:.2f}", f"Table 5 seed {s['seed']} stoch CPU")
    has(f"${s['test_mean_cpu']:.2f}", f"Table 5 seed {s['seed']} argmax CPU")
    has(f"{s['stoch_test_joint_sat']*100:.0f}%", f"Table 5 seed {s['seed']} sat")
has(f"${np.mean([s['stoch_test_mean_cpu'] for s in S5]):.2f} ± {ci([s['stoch_test_mean_cpu'] for s in S5]):.2f}","T5 mean stoch")
has(f"${np.mean([s['test_mean_cpu'] for s in S5]):.2f} ± {ci([s['test_mean_cpu'] for s in S5]):.2f}","T5 mean argmax")
has(f"{np.mean([s['stoch_test_joint_sat'] for s in S5])*100:.1f}%","T5 mean sat")

# ---------- Table B1: ablation ----------
for cell,lab in [("shaped-cumrate","shaped + cumulative-rate"),("shaped-episode","shaped + episode"),
                 ("cost-cumrate","cost + cumulative-rate"),("cost-episode","cost + episode")]:
    v=N[f"T2/{cell}"]
    has(f"${v['argmax']['cpu']:.2f} ± {v['argmax']['cpu_ci']:.2f}", f"B1 {lab} argmax")
    has(f"${v['stoch']['cpu']:.2f} ± {v['stoch']['cpu_ci']:.2f}", f"B1 {lab} stoch")
    has(f"{v['stoch']['joint']*100:.1f}%", f"B1 {lab} stoch sat")

# ---------- Table B3: severity ----------
for tb,S in [("electronics",json.load(open(OUT/"stats_electronics.json"))),("bakery",json.load(open(OUT/"stats_bakery.json")))]:
    sym=json.load(open(OUT/"stats_electronics_r2.json"))["severity"] if tb=="electronics" else None
    for key,row in S["severity"].items():
        T=key.split("_")[0][1:]; U=float(key.split("_U")[1])
        cells=[f"{row[n]*100:.0f}%" for n in ["ShortestQueue","LeastUtilised","RoundRobin","UniformRandom","cost-episode/stoch","shaped-cumrate/stoch"]]
        last = f"{sym[key]*100:.0f}%" if (sym and key in sym) else "–"
        line=f"| {tb} | {T} | {U:.2f} | "+" | ".join(cells)+f" | {last} |"
        if f"| {tb} | {T} | {U:.2f} |" in M and line not in M: fails.append(f"B3 row mismatch: {line}")
        if f"| {tb} | {T} | {U:.2f} |" in M: checks+=1

# ---------- inline numbers ----------
lamT=[float(list(csv.DictReader(open(p)))[-1]["lam_tp"]) for p in glob.glob("results_r1/*/cost-episode/seed_*/lambda_history.csv")]
has(f"{min(lamT):,.0f} and {max(lamT):,.0f}", "full-budget lambda range 2,525-7,858")
lamU=[float(list(csv.DictReader(open(p)))[-1]["lam_util"]) for p in glob.glob("results_r2/electronics/cost-episode-sym/seed_*/lambda_history.csv")]
symT=[np.array([float(r["lam_tp"]) for r in csv.DictReader(open(p))]) for p in glob.glob("results_r2/electronics/cost-episode-sym/seed_*/lambda_history.csv")]
has(f"{min(a.mean() for a in symT):.2f}–{max(a.mean() for a in symT):.2f}", "symmetric lambda_T mean range")
has(f"{min((a>0).mean() for a in symT)*100:.0f}–{max((a>0).mean() for a in symT)*100:.0f}%", "symmetric positive-episode fraction")
has(f"{len(symT[0]):,}", "symmetric training episodes 3,352")
# total costs
dE={**json.load(open(OUT/"episodes_electronics_rl.json")),**json.load(open(OUT/"episodes_electronics_base.json")),**json.load(open(OUT/"episodes_electronics_r2.json"))}
def meancost(pref,suf="stoch"):
    ks=[k for k in dE if k.startswith(pref) and (k.endswith(suf) or pref.startswith("baseline"))]
    return np.mean([r["cost"] for k in ks for r in dE[k]])
ALL=["cost-episode-sym/","baseline/RoundRobin","baseline/UniformRandom","cost-episode/","baseline/ShortestQueue","baseline/LeastUtilised","shaped-cumrate/"]
for pref in ALL[:5]:
    has(f"${meancost(pref):,.0f}", f"total cost {pref}")
costs=[meancost(p) for p in ALL]
def tpm(p): return np.mean([r["tp"] for k in dE if k.startswith(p) and (k.endswith("stoch") or p.startswith("baseline")) for r in dE[k]])
tps=[tpm(p) for p in ALL]
has(f"spans {(max(costs)/min(costs)-1)*100:.1f}%", "cost span")
has(f"${min(costs):,.0f}", "min cost"); has(f"${max(costs):,.0f}", "max cost")
has(f"spans {(max(tps)/min(tps)-1)*100:.1f}%", "throughput span")
has(f"{min(tps):.1f} to {max(tps):.1f} units", "throughput range")
# selected checkpoints of symmetric cell
for s in S5: has(f"{round(s['selected_steps']/1000):,}K", f"T5 ckpt seed {s['seed']}")
print(f"checks: {checks}   failures: {len(fails)}")
for f_ in fails: print("  FAIL:", f_)
