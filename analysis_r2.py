"""R2 referee fixes: per-episode records (both modes) for the 20 full-budget runs and
the dispatching baselines on the shared test seeds; hierarchical paired bootstrap;
argmax-stability metrics; post-hoc constraint-severity sweep.
Outputs under results_r1/r2/."""
import json, glob, sys, numpy as np, torch
from pathlib import Path
from stable_baselines3 import PPO
from env import FlexFlowSimEnv
from baselines import ShortestQueuePolicy, LeastUtilisedPolicy, RandomPolicy, RoundRobinPolicy
from run_ablation_2x2 import TESTBED_CONFIG, TEST_SEEDS
W=(0.8,0.1,0.1); OUT=Path("results_r1/r2"); OUT.mkdir(exist_ok=True)

def rollout(policy_fn, tb, seed):
    torch.manual_seed(int(seed))   # A2.3 as amended: action sampling seeded by the episode seed
    cfg=TESTBED_CONFIG[tb]; env=FlexFlowSimEnv(config=cfg["config"],weights=W,seed=int(seed))
    obs,_=env.reset(); done=False
    while not done:
        obs,_,t1,t2,info=env.step(policy_fn(obs,env)); done=t1 or t2
    return dict(seed=int(seed),tp=float(info["total_departed"]),cost=float(info["total_cost"]),util=[float(u) for u in info["utilisation"]])

def episodes(tb, stage):
    """stage 'rl': all full-budget selected checkpoints, both modes. stage 'base': rules.
    stage 'r2': the Amendment R2 cell (results_r2)."""
    recs={}
    if stage in ("rl","r2"):
        root,cells=("results_r1",["cost-episode","shaped-cumrate"]) if stage=="rl" else ("results_r2",["cost-episode-sym"])
        for cell in cells:
            for p in sorted(glob.glob(f"{root}/{tb}/{cell}/seed_*/summary.json")):
                s=json.load(open(p)); m=PPO.load(p.replace("summary.json",f"checkpoints/ckpt_{s['selected_steps']}_steps"),device="cpu")
                for mode,det in [("stoch",False),("argmax",True)]:
                    fn=lambda obs,env,m=m,det=det: int(m.predict(obs,deterministic=det)[0])
                    recs[f"{cell}/seed_{s['seed']}/{mode}"]=[rollout(fn,tb,es) for es in TEST_SEEDS]
    else:
        rules={"ShortestQueue":lambda e,s:ShortestQueuePolicy(e),"LeastUtilised":lambda e,s:LeastUtilisedPolicy(e),
               "RoundRobin":lambda e,s:RoundRobinPolicy(e),"UniformRandom":lambda e,s:RandomPolicy(e,seed=int(s))}
        for name,f in rules.items():
            out=[]
            for es in TEST_SEEDS:
                pol=None
                def fn(obs,env,f=f,es=es):
                    nonlocal pol
                    if pol is None: pol=f(env,es)
                    a=pol.predict(obs); return int(a[0] if isinstance(a,tuple) else a)
                out.append(rollout(fn,tb,es))
            recs[f"baseline/{name}"]=out
    path=OUT/f"episodes_{tb}_{stage}.json"; json.dump(recs,open(path,"w")); print("wrote",path,len(recs),"policies")

def argmax_stability(tb, n_eps=5, sigmas=(0.05,0.1), root="results_r1", cells=("cost-episode","shaped-cumrate"), tag=""):
    cfg=TESTBED_CONFIG[tb]; res={}
    # reference state set: states visited under UniformRandom on the first n_eps test seeds
    states=[]
    for es in TEST_SEEDS[:n_eps]:
        env=FlexFlowSimEnv(config=cfg["config"],weights=W,seed=int(es)); rng=np.random.default_rng(int(es)); obs,_=env.reset(); done=False
        while not done:
            states.append(obs.copy()); obs,_,t1,t2,_=env.step(int(rng.integers(env.action_space.n))); done=t1 or t2
    S=torch.as_tensor(np.array(states),dtype=torch.float32)
    for cell in cells:
        argmaxes=[]; margins=[]; flips={s:[] for s in sigmas}
        for p in sorted(glob.glob(f"{root}/{tb}/{cell}/seed_*/summary.json")):
            s=json.load(open(p)); m=PPO.load(p.replace("summary.json",f"checkpoints/ckpt_{s['selected_steps']}_steps"),device="cpu")
            with torch.no_grad():
                logits=m.policy.get_distribution(S).distribution.logits; pr=torch.softmax(logits,-1)
            top2=torch.topk(pr,2,dim=-1).values; margins.append(float((top2[:,0]-top2[:,1]).mean()))
            am=logits.argmax(-1); argmaxes.append(am)
            g=torch.Generator().manual_seed(0)
            for sg in sigmas:
                noisy=logits+sg*torch.randn(logits.shape,generator=g); flips[sg].append(float((noisy.argmax(-1)!=am).float().mean()))
        A=torch.stack(argmaxes); pair=[float((A[i]==A[j]).float().mean()) for i in range(len(A)) for j in range(i+1,len(A))]
        res[cell]=dict(n_states=len(states),mean_top1_top2_margin=float(np.mean(margins)),margin_per_seed=margins,
                       cross_seed_argmax_agreement=float(np.mean(pair)),chance_agreement=1/ int(cfg and FlexFlowSimEnv(config=cfg["config"],weights=W,seed=0).action_space.n),
                       flip_rate={str(s):float(np.mean(v)) for s,v in flips.items()})
        print(cell,json.dumps(res[cell],indent=None)[:300])
    json.dump(res,open(OUT/f"argmax_stability_{tb}{tag}.json","w"),indent=2)

def pilot(cells):
    tb="electronics"; recs={}
    for cell in cells.split(","):
        for p in sorted(glob.glob(f"results_ablation/electronics_pilot400k/{cell}/seed_*/summary.json")):
            s=json.load(open(p)); m=PPO.load(p.replace("summary.json",f"checkpoints/ckpt_{s['selected_steps']}_steps"),device="cpu")
            for mode,det in [("stoch",False),("argmax",True)]:
                recs[f"{cell}/seed_{s['seed']}/{mode}"]=[rollout(lambda o,e,m=m,d=det:int(m.predict(o,deterministic=d)[0]),tb,es) for es in TEST_SEEDS]
            print(cell,s["seed"],flush=True)
    path=OUT/f"episodes_pilot_{cells.replace(',','_')}.json"; json.dump(recs,open(path,"w")); print("wrote",path)

def archival(which):
    from archival_reeval import RUNS
    recs={}
    for run in which.split(","):
        src,tb,budget=RUNS[run]
        for p in sorted(glob.glob(f"results_r1/archival/{run}/seed_*/summary.json")):
            s=json.load(open(p)); seed=s["seed"]
            for q,st in [("q1",s["q1_published_ckpt"]),("q2",s["q2_stoch_ckpt"])]:
                key=f"{run}/seed_{seed}/{q}_{st}"
                if q=="q2" and st==s["q1_published_ckpt"]: recs[key]=recs[f"{run}/seed_{seed}/q1_{st}"]; continue
                ck=Path(src)/f"seed_{seed}"/"checkpoints"/f"ckpt_{st}_steps"
                if not ck.with_suffix(".zip").exists(): ck=Path(src)/f"seed_{seed}"/"final"
                m=PPO.load(str(ck),device="cpu")
                recs[key]={mode:[rollout(lambda o,e,m=m,d=det:int(m.predict(o,deterministic=d)[0]),tb,es) for es in TEST_SEEDS] for mode,det in [("stoch",False),("argmax",True)]}
                print(key,{mode:round(float(np.mean([r["cost"]/r["tp"] for r in v])),2) for mode,v in recs[key].items()},flush=True)
    path=OUT/f"episodes_archival_{which.replace(',','_')}.json"; json.dump(recs,open(path,"w")); print("wrote",path)

if __name__=="__main__":
    what=sys.argv[1]; tb=sys.argv[2]
    if what in("rl","base","r2"): episodes(tb,what)
    elif what=="stab": argmax_stability(tb)
    elif what=="stab_r2": argmax_stability(tb,root="results_r2",cells=("cost-episode-sym",),tag="_r2")
    elif what=="arch": archival(tb)
    elif what=="pilot": pilot(tb)
