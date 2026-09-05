"""Assemble every number the R2 manuscript reports from results_r1/r2/*.json."""
import json, glob, numpy as np
from pathlib import Path
from run_ablation_2x2 import TESTBED_CONFIG, T_CRIT
OUT=Path("results_r1/r2"); T4=T_CRIT[5]; T3=T_CRIT.get(4,3.182)
def ci(xs): xs=np.asarray(xs,float); n=len(xs); return float(T_CRIT.get(n,1.96)*xs.std(ddof=1)/np.sqrt(n)) if n>1 else 0.0
def summ(recs,tb):
    cfg=TESTBED_CONFIG[tb]; f=cfg["constrained_servers"]
    cpu=np.mean([r["cost"]/r["tp"] for r in recs]); tp=np.mean([r["tp"] for r in recs])
    joint=np.mean([(r["tp"]>=cfg["tp_target"]) and min(r["util"][i] for i in f)>=cfg["u_min"] for r in recs])
    return cpu,tp,joint
N={}
print("== Table 3 (seeded re-evaluation) vs pipeline draw")
for tb in ["electronics","bakery"]:
    d=json.load(open(OUT/f"episodes_{tb}_rl.json"))
    for cell in ["cost-episode","shaped-cumrate"]:
        row={}
        for mode in ["stoch","argmax"]:
            ks=sorted(k for k in d if k.startswith(cell) and k.endswith(mode)); S=[summ(d[k],tb) for k in ks]
            row[mode]=dict(cpu=float(np.mean([s[0] for s in S])),cpu_ci=ci([s[0] for s in S]),tp=float(np.mean([s[1] for s in S])),joint=float(np.mean([s[2] for s in S])),per_seed_cpu=[s[0] for s in S],per_seed_joint=[s[2] for s in S])
        pipe=[json.load(open(p)) for p in sorted(glob.glob(f"results_r1/{tb}/{cell}/seed_*/summary.json"))]
        row["pipeline_stoch"]=dict(cpu=float(np.mean([p["stoch_test_mean_cpu"] for p in pipe])),joint=float(np.mean([min(p["stoch_test_util_sat"],p["stoch_test_tp_sat"]) for p in pipe])))
        row["pipeline_argmax"]=dict(cpu=float(np.mean([p["test_mean_cpu"] for p in pipe])),joint=float(np.mean([min(p["test_util_sat"],p["test_tp_sat"]) for p in pipe])))
        row["val_satisfied"]=sum(p["selection_status"]=="satisfied" for p in pipe); row["lam_cap"]=sum(p["lam_tp_saturated"] for p in pipe)
        N[f"T3/{tb}/{cell}"]=row
        print(f" {tb:12s} {cell:15s} stoch ${row['stoch']['cpu']:.2f}±{row['stoch']['cpu_ci']:.2f} sat {row['stoch']['joint']:.1%} TP {row['stoch']['tp']:.1f} | pipeline draw ${row['pipeline_stoch']['cpu']:.2f} sat {row['pipeline_stoch']['joint']:.1%} | argmax ${row['argmax']['cpu']:.2f}±{row['argmax']['cpu_ci']:.2f} sat {row['argmax']['joint']:.1%} (pipeline ${row['pipeline_argmax']['cpu']:.2f} sat {row['pipeline_argmax']['joint']:.1%}) val-sat {row['val_satisfied']}/5 cap {row['lam_cap']}/5")
print("== Table 4 (archival, seeded)")
A={**json.load(open(OUT/"episodes_archival_v4_electronics_v6_electronics.json")),**json.load(open(OUT/"episodes_archival_v4_bakery_v6_bakery.json"))}
for run,tb in [("v4_electronics","electronics"),("v6_electronics","electronics"),("v4_bakery","bakery"),("v6_bakery","bakery")]:
    seeds=sorted({k.split("/")[1] for k in A if k.startswith(run+"/")})
    q1a,q1s,q2s,q1aj,q1sj,q2sj=[],[],[],[],[],[]
    for sd in seeds:
        k1=[k for k in A if k.startswith(f"{run}/{sd}/q1_")][0]; k2=[k for k in A if k.startswith(f"{run}/{sd}/q2_")][0]
        a=summ(A[k1]["argmax"],tb); s1=summ(A[k1]["stoch"],tb); s2=summ(A[k2]["stoch"],tb)
        q1a.append(a[0]);q1aj.append(a[2]);q1s.append(s1[0]);q1sj.append(s1[2]);q2s.append(s2[0]);q2sj.append(s2[2])
    old=[json.load(open(p)) for p in sorted(glob.glob(f"results_r1/archival/{run}/seed_*/summary.json"))]
    N[f"T4/{run}"]=dict(n=len(seeds),q1_argmax=(float(np.mean(q1a)),ci(q1a),float(np.mean(q1aj))),q1_stoch=(float(np.mean(q1s)),ci(q1s),float(np.mean(q1sj))),q2_stoch=(float(np.mean(q2s)),ci(q2s),float(np.mean(q2sj))),
        per_seed_q1_argmax=q1a,per_seed_q1_stoch=q1s,per_seed_q1_stoch_joint=q1sj,old_q1_stoch=float(np.mean([o["q1"]["stochastic"]["cpu"] for o in old])),old_q2_stoch=float(np.mean([o["q2"]["stochastic"]["cpu"] for o in old])),old_q1_stoch_joint=float(np.mean([o["q1"]["stochastic"]["joint_sat"] for o in old])),old_q2_stoch_joint=float(np.mean([o["q2"]["stochastic"]["joint_sat"] for o in old])),q2_val_sat=sum(o["stoch_status"]=="satisfied" for o in old))
    r=N[f"T4/{run}"]; print(f" {run:15s} n={r['n']} Q1 argmax ${r['q1_argmax'][0]:.2f}±{r['q1_argmax'][1]:.2f} {r['q1_argmax'][2]:.0%} | Q1 stoch ${r['q1_stoch'][0]:.2f}±{r['q1_stoch'][1]:.2f} {r['q1_stoch'][2]:.0%} (old ${r['old_q1_stoch']:.2f} {r['old_q1_stoch_joint']:.0%}) | Q2 stoch ${r['q2_stoch'][0]:.2f}±{r['q2_stoch'][1]:.2f} {r['q2_stoch'][2]:.0%} (old ${r['old_q2_stoch']:.2f} {r['old_q2_stoch_joint']:.0%}) val-sat {r['q2_val_sat']}/{r['n']}")
    print("   per-seed Q1 argmax:",[round(x,2) for x in q1a]," Q1 stoch:",[round(x,2) for x in q1s],[f"{x:.0%}" for x in q1sj])
json.dump(N,open(OUT/"r2_numbers.json","w"),indent=1)

print("== Table 2 (pilot 400K, seeded re-evaluation)")
P={**json.load(open(OUT/"episodes_pilot_cost-cumrate_cost-episode.json")),**json.load(open(OUT/"episodes_pilot_shaped-cumrate_shaped-episode.json"))}
import csv
lam={}
for p in glob.glob("results_ablation/electronics_pilot400k/*/seed_*/summary.json"):
    s=json.load(open(p)); lam[(s["cell"],s["seed"])]=(s["lam_tp_final"],s["lam_tp_saturated"])
for cell in ["shaped-cumrate","shaped-episode","cost-cumrate","cost-episode"]:
    row={}
    for mode in ["argmax","stoch"]:
        ks=sorted(k for k in P if k.startswith(cell+"/") and k.endswith(mode)); S=[summ(P[k],"electronics") for k in ks]
        row[mode]=dict(cpu=float(np.mean([s[0] for s in S])),cpu_ci=ci([s[0] for s in S]),joint=float(np.mean([s[2] for s in S])))
    row["cap"]=sum(v[1] for (c,sd),v in lam.items() if c==cell); row["lamT"]=[v[0] for (c,sd),v in lam.items() if c==cell]
    N[f"T2/{cell}"]=row
    print(f" {cell:15s} argmax ${row['argmax']['cpu']:.2f}±{row['argmax']['cpu_ci']:.2f} {row['argmax']['joint']:.1%} | stoch ${row['stoch']['cpu']:.2f}±{row['stoch']['cpu_ci']:.2f} {row['stoch']['joint']:.1%} | cap {row['cap']}/5 lamT {sorted(round(x) for x in row['lamT'])}")
json.dump(N,open(OUT/"r2_numbers.json","w"),indent=1)
