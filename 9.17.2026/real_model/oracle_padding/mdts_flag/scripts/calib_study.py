#!/usr/bin/env python3
"""Calibration study over all 48 layers: how much the hot-expert pattern moves across workloads, and what a fixed
hot/cold layout (hot set = top-64 by calibration mean count, capacities = largest count seen in calibration) costs on
held-out prompts. Each workload's first half of prompts is its calibration part, the second half its evaluation part.
Usage: calib_study.py <T> <out dir> name=dir [name=dir ...]"""
import sys, os, glob, json, itertools, numpy as np
T, out = int(sys.argv[1]), sys.argv[2]; os.makedirs(out, exist_ok=True)
W = {}
for a in sys.argv[3:]:
    n, d = a.split('=', 1); fs = sorted(glob.glob(f'{d}/T{T}_*.npz'), key=lambda f: (len(f), f))
    W[n] = np.array([[np.bincount(r, minlength=128) for r in np.load(f)['routing_idx'].reshape(48, -1)] for f in fs])   # [P, 48, 128]
names = list(W); L = 48; rng = np.random.default_rng(0)
half = {n: len(W[n]) // 2 for n in names}
cal = {n: W[n][:half[n]] for n in names}; ev = {n: W[n][half[n]:] for n in names}
def top(mean, k): return set(np.argsort(-mean)[:k])
def jac(a, b): return len(a & b) / len(a | b)
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
lines = []; P = lambda *x: (print(*x), lines.append(' '.join(str(y) for y in x)))
P(f"T={T}; workloads: " + ', '.join(f"{n} {len(W[n])}" for n in names))
# ---- A. similarity per layer
simJ64 = np.zeros((len(names), len(names), L)); simJ16 = np.zeros_like(simJ64); simR = np.zeros_like(simJ64); noise64 = {}; noise16 = {}
for l in range(L):
    means = {n: W[n][:, l].mean(0) for n in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            simJ64[i, j, l] = jac(top(means[a], 64), top(means[b], 64)); simJ16[i, j, l] = jac(top(means[a], 16), top(means[b], 16)); simR[i, j, l] = spearman(means[a], means[b])
    for n in names:
        h1, h2 = W[n][:half[n], l].mean(0), W[n][half[n]:, l].mean(0)
        noise64.setdefault(n, []).append(jac(top(h1, 64), top(h2, 64))); noise16.setdefault(n, []).append(jac(top(h1, 16), top(h2, 16)))
P("\nA. Agreement on which experts are hot (median over the 48 layers). Diagonal = two halves of the same workload (sampling noise floor).")
for title, M, nz in (("top-64 Jaccard", simJ64, noise64), ("top-16 Jaccard", simJ16, noise16)):
    P(f"  {title}: " + ' '.join(f"{n[:9]:>9s}" for n in names))
    for i, a in enumerate(names):
        P(f"  {a[:14]:>14s} " + ' '.join(f"{(np.median(nz[a]) if i == j else np.median(M[i, j])):9.2f}" for j in range(len(names))))
P("  Spearman rank correlation of per-expert load (median over layers): " + ', '.join(f"{a}-{b} {np.median(simR[i, j]):.2f}" for (i, a), (j, b) in itertools.combinations(enumerate(names), 2) if (a, b) in {(names[0], x) for x in names} or True)[:2000])
off = np.array([simJ64[i, j] for i, j in itertools.combinations(range(len(names)), 2)])            # [pairs, L]
P(f"  cross-workload top-64 Jaccard per layer: median {np.median(off):.2f}; by layer, first/last: " + ' '.join(f"{np.median(off[:, l]):.2f}" for l in (0, 1, 2, 12, 24, 36, 44, 47)) + "  (layers 0,1,2,12,24,36,44,47)")
# ---- B. shared core
core_sz, core_share, union_sz = [], {n: [] for n in names}, []
for l in range(L):
    sets = [top(W[n][:, l].mean(0), 64) for n in names]; core = set.intersection(*sets); union_sz.append(len(set.union(*sets))); core_sz.append(len(core))
    for n in names: core_share[n].append(W[n][:, l][:, sorted(core)].sum() / W[n][:, l].sum() if core else 0.0)
P(f"\nB. Experts in the top-64 of every workload: median {int(np.median(core_sz))} per layer (range {min(core_sz)}-{max(core_sz)}); union of all top-64 sets: median {int(np.median(union_sz))} of 128")
P("   share of each workload's assignments carried by that shared core (median over layers): " + ', '.join(f"{n} {100*np.median(core_share[n]):.0f}%" for n in names))
# ---- C. fixed layout from calibration, evaluated on held-out prompts
def fixed_eval(calib, evals):
    """calib: [P, L, 128]; evals: dict name -> [P, L, 128]. Per layer: hot = top-64 by calib mean; capacities = calib max per stage."""
    res = {}
    hot = np.zeros((L, 128), bool); capH = np.zeros(L, int); capC = np.zeros(L, int)
    for l in range(L):
        idx = np.argsort(-calib[:, l].mean(0))[:64]; hot[l, idx] = True; capH[l] = calib[:, l][:, hot[l]].max(); capC[l] = calib[:, l][:, ~hot[l]].max()
    for n, e in evals.items():
        needH = np.array([[p[l][hot[l]].max() for l in range(L)] for p in e]); needC = np.array([[p[l][~hot[l]].max() for l in range(L)] for p in e])   # [P, L]
        dropped = np.array([[np.maximum(p[l][hot[l]] - capH[l], 0).sum() + np.maximum(p[l][~hot[l]] - capC[l], 0).sum() for l in range(L)] for p in e])
        res[n] = dict(needH=needH, needC=needC, capH=capH, capC=capC, drop_prompts=float((dropped.sum(1) > 0).mean()), drop_layers=float((dropped > 0).mean()), drop_share=float(dropped.sum() / e.sum()))
    return res
def dyn_need(e):   # per-card dynamic split: largest count, and 17th-largest per card (native blocks of 32)
    return np.array([[max(np.sort(p[l][32*k:32*k+32])[::-1][16] for k in range(4)) for l in range(L)] for p in e])
P("\nC. Fixed layout (hot set and capacities from calibration prompts), evaluated on each workload's held-out half")
P("   calibration        eval       cap hot/cold (median layer)  needed cold: median layer, worst layer (max over prompts)  prompts with any drop  layers x prompts dropping  assignments dropped")
scen = [('same workload', lambda n: cal[n]), ('all workloads', lambda n: np.concatenate([cal[x] for x in names])), ('all but this one', lambda n: np.concatenate([cal[x] for x in names if x != n]))]
tabC = {}
for sname, cf in scen:
    for n in names:
        r = fixed_eval(cf(n), {n: ev[n]})[n]; tabC[(sname, n)] = r
        nc = r['needC']
        P(f"   {sname:16s} {n:10s} {int(np.median(r['capH'])):4d}/{int(np.median(r['capC'])):<4d}                     {int(np.median(nc.max(0))):4d}, {nc.max():4d}                                    {100*r['drop_prompts']:5.0f}%              {100*r['drop_layers']:6.1f}%                {100*r['drop_share']:.3f}%")
P("   per-card dynamic split for reference (needed cold capacity = 17th-largest count on a card): " + ', '.join(f"{n} median layer {int(np.median(dyn_need(ev[n]).max(0)))}, worst {dyn_need(ev[n]).max()}" for n in names))
P("   the hard bound: a stage never needs more than T = " + str(T) + " rows")
# ---- D. calibration size: pooled calibration halves of all workloads, random subsets
pool = np.concatenate([cal[x] for x in names]); evall = {n: ev[n] for n in names}
P("\nD. Calibration size: random subsets of the pooled calibration halves (all workloads), 20 draws each; evaluated on all held-out halves")
P("   n prompts   prompts with any drop (median of draws)   assignments dropped   cap cold (median layer)")
for n_ in [x for x in (8, 16, 32, 64, 128) if x < len(pool)] + [len(pool)]:
    dp, ds, cc = [], [], []
    for d in range(20 if n_ < len(pool) else 1):
        sub = pool[rng.choice(len(pool), n_, replace=False)]; r = fixed_eval(sub, evall)
        allp = np.concatenate([[r[n]['drop_prompts']] * len(ev[n]) for n in names]); dp.append(allp.mean()); ds.append(np.mean([r[n]['drop_share'] for n in names])); cc.append(np.median(r[names[0]]['capC']))
    P(f"   {n_:9d}   {100*np.median(dp):8.0f}%                                 {100*np.median(ds):.3f}%              {np.median(cc):.0f}")
# ---- E. layer sensitivity
lay = np.median(off, 0); order = np.argsort(lay)
P("\nE. Layers whose hot set moves most across workloads (lowest median cross-workload top-64 Jaccard): " + ', '.join(f"L{l} {lay[l]:.2f}" for l in order[:6]) + "; least: " + ', '.join(f"L{l} {lay[l]:.2f}" for l in order[-6:]))
open(f'{out}/calib_study_T{T}.txt', 'w').write('\n'.join(lines) + '\n')
np.savez(f'{out}/calib_study_T{T}.npz', names=np.array(names), simJ64=simJ64, simJ16=simJ16, simR=simR, core_sz=np.array(core_sz))
