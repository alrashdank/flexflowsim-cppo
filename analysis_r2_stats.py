"""Hierarchical paired bootstrap (seeds x episodes) and constraint-severity sweep from
results_r1/r2/episodes_*.json. Also cross-checks the per-episode means against the
numbers already in the manuscript."""
import json, numpy as np, itertools
from pathlib import Path
from run_ablation_2x2 import TESTBED_CONFIG
OUT=Path("results_r1/r2"); rng=np.random.default_rng(20260905); B=10000

def load(tb):
    d={**json.load(open(OUT/f"episodes_{tb}_rl.json")),**json.load(open(OUT/f"episodes_{tb}_base.json"))}
    cfg=TESTBED_CONFIG[tb]; fast=cfg["constrained_servers"]
    def arr(recs,tmin=cfg["tp_target"],umin=cfg["u_min"]):
        tp=np.array([r["tp"] for r in recs]); cost=np.array([r["cost"] for r in recs])
        u=np.array([[r["util"][i] for i in fast] for r in recs])
        return dict(cpu=cost/tp, tp=tp, umin=u.min(1), joint=((tp>=tmin)&(u.min(1)>=umin)).astype(float))
    return d,arr,cfg

def cell_matrix(d,arr,cell,mode,key):   # seeds x episodes
    ks=sorted(k for k in d if k.startswith(cell+"/") and k.endswith("/"+mode))
    return np.array([arr(d[k])[key] for k in ks])

def hboot(M_a, M_b):
    """Paired hierarchical bootstrap of mean(a)-mean(b). M_a: seeds x episodes (RL);
    M_b: 1 x episodes (baseline, same episode seeds) or seeds x episodes (another cell).
    Resample seeds (each policy set independently) and episodes (jointly, preserving pairing)."""
    S_a,E=M_a.shape; S_b=M_b.shape[0]; diffs=np.empty(B)
    for b in range(B):
        e=rng.integers(E,size=E); sa=rng.integers(S_a,size=S_a); sb=rng.integers(S_b,size=S_b)
        diffs[b]=M_a[np.ix_(sa,e)].mean()-M_b[np.ix_(sb,e)].mean()
    obs=M_a.mean()-M_b.mean(); lo,hi=np.percentile(diffs,[2.5,97.5])
    p=2*min((diffs<=0).mean(),(diffs>=0).mean()); return dict(diff=float(obs),ci=[float(lo),float(hi)],p=float(min(1,p)))

def main(tb):
    d,arr,cfg=load(tb); res={"means":{},"bootstrap":{},"severity":{}}
    for k in sorted(d):
        a=arr(d[k]); res["means"][k]=dict(cpu=float(a["cpu"].mean()),tp=float(a["tp"].mean()),joint=float(a["joint"].mean()))
    for cell in ["cost-episode","shaped-cumrate"]:
        for mode in ["stoch","argmax"]:
            for key in ["cpu","joint"]:
                A=cell_matrix(d,arr,cell,mode,key)
                res["means"][f"{cell}/{mode}/pooled_{key}"]=float(A.mean())
                for base in ["ShortestQueue","RoundRobin","UniformRandom","LeastUtilised"]:
                    res["bootstrap"][f"{cell}/{mode}/{key} vs {base}"]=hboot(A,arr(d[f"baseline/{base}"])[key][None,:])
    for mode in ["stoch","argmax"]:
        for key in ["cpu","joint"]:
            res["bootstrap"][f"cost-episode vs shaped-cumrate/{mode}/{key}"]=hboot(cell_matrix(d,arr,"cost-episode",mode,key),cell_matrix(d,arr,"shaped-cumrate",mode,key))
    # severity sweep: joint satisfaction at alternative thresholds, no retraining
    T=cfg["tp_target"]; tgrid=[T-2,T,T+2,T+4,T+6,T+8,T+10] if tb=="electronics" else [T-1,T,T+1,T+2,T+3,T+4]
    ugrid=[0.4,0.5,0.6,0.7,0.8]
    pols={"ShortestQueue":"baseline/ShortestQueue","LeastUtilised":"baseline/LeastUtilised","RoundRobin":"baseline/RoundRobin","UniformRandom":"baseline/UniformRandom"}
    for tmin in tgrid:
        for umin in ugrid:
            row={n:float(arr(d[k],tmin,umin)["joint"].mean()) for n,k in pols.items()}
            for cell in ["cost-episode","shaped-cumrate"]:
                ks=sorted(k for k in d if k.startswith(cell+"/") and k.endswith("/stoch"))
                row[cell+"/stoch"]=float(np.mean([arr(d[k],tmin,umin)["joint"].mean() for k in ks]))
            res["severity"][f"T{tmin}_U{umin}"]=row
    # Holm-Bonferroni over the pre-specified family: corrected cell (stochastic) vs SQ, LU, RR, Random, control; CPU and joint
    fam={k:v for k,v in res["bootstrap"].items() if k.startswith("cost-episode/stoch/") or k.startswith("cost-episode vs shaped-cumrate/stoch")}
    order=sorted(fam,key=lambda k:fam[k]["p"]); m=len(order); rejected=True
    for i,k in enumerate(order):
        thr=0.05/(m-i); rejected=rejected and fam[k]["p"]<=thr; fam[k]["holm_threshold"]=thr; fam[k]["holm_significant"]=bool(rejected)
    res["holm_family_size"]=m
    json.dump(res,open(OUT/f"stats_{tb}.json","w"),indent=1)
    print(f"== {tb}: means"); [print(f"  {k:45s} CPU {v['cpu']:7.2f} TP {v['tp']:6.2f} joint {v['joint']:.3f}") if isinstance(v,dict) else print(f"  {k:45s} {v:.3f}") for k,v in res["means"].items() if "baseline" in k or "pooled" in k]
    print("== bootstrap (diff, 95% CI, p)"); [print(f"  {k:55s} {v['diff']:+8.3f} [{v['ci'][0]:+8.3f},{v['ci'][1]:+8.3f}] p={v['p']:.4f}") for k,v in res["bootstrap"].items() if "/stoch" in k]
    print("== Holm:"); [print(f"  {k:55s} p={fam[k]['p']:.4f} thr={fam[k]['holm_threshold']:.4f} {'SIG' if fam[k]['holm_significant'] else 'ns'}") for k in order]
    print("== severity (joint sat): SQ LU RR Rand corrected control")
    for k,v in res["severity"].items(): print(f"  {k:10s} "+" ".join(f"{v[n]:.2f}" for n in ["ShortestQueue","LeastUtilised","RoundRobin","UniformRandom","cost-episode/stoch","shaped-cumrate/stoch"]))
if __name__=="__main__":
    import sys; main(sys.argv[1])
