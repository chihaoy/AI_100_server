#!/usr/bin/env python3
"""Co-activation-aware expert placement: can a 32-per-card assignment put a token's 8 experts on fewer cards?
Per layer, A[e, f] = calibration tokens that chose both e and f; a balanced 4-way partition is searched by greedy pairwise
swaps (Kernighan-Lin style) that maximize within-card co-activation, from the native placement and from random starts;
evaluated on held-out prompts (first half of each workload's T=128 prompts calibrates, second half evaluates). The three
chat-template tokens at positions 0-2 (identical in every prompt) are excluded. Also reports the per-card cold-capacity
need (17th-largest count per card, per 128-token prompt) under each placement.
Variant of coact_placement.py: the load-constrained search and leave-one-workload-out.
Usage: coact_placement2.py <mdts_flag dir>"""
import sys, os, glob, numpy as np
R = sys.argv[1]; RESTARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 4; rng = np.random.default_rng(0)
cal, ev, ev_prompts, names = [], [], [], []
for d in sorted(glob.glob(f'{R}/realcase/routing*')):
    fs = sorted(glob.glob(d + '/T128_*.npz'), key=lambda f: (len(f), f)); h = len(fs) // 2
    names.append(os.path.basename(d).replace('routing_', '').replace('routing', 'gsm8k'))
    for k, f in enumerate(fs):
        r = np.load(f)['routing_idx'][:, 3:, :].astype(np.int64)          # [48, 125, 8], template tokens dropped
        (cal if k < h else ev).append(r)
        if k >= h: ev_prompts.append(r)
cal = np.concatenate(cal, 1); ev = np.concatenate(ev, 1)                  # [48, N, 8]
def onehot(r):  # [N, 8] -> [N, 128]
    X = np.zeros((len(r), 128), np.float64); np.put_along_axis(X, r, 1.0, axis=1); return X
def cold_need_of(cnts, g):   # cnts [P, 128] per-prompt counts -> max over prompts and cards of the 17th-largest count
    return max(int(np.sort(cnts[:, g == c], axis=1)[:, -17].max()) for c in range(4))
def partition(A, g0, iters=4000, cnts=None, bound=None):
    g = g0.copy(); G = np.eye(4)[g]; W = A @ G                            # W[e, c] = co-activation of e with card c
    idx = np.arange(128); banned = np.zeros((128, 128), bool)
    for _ in range(iters):
        own = W[idx, g]; cross = W[:, g]                                   # cross[e, f] = W[e, g[f]]
        gain = cross - own[:, None] + cross.T - own[None, :] - 2 * A
        gain[(g[:, None] == g[None, :]) | banned] = -np.inf
        e, f = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[e, f] <= 1e-9: break
        if bound is not None:
            h = g.copy(); h[e], h[f] = g[f], g[e]
            if cold_need_of(cnts, h) > bound: banned[e, f] = banned[f, e] = True; continue
        ge, gf = g[e], g[f]; W[:, ge] += A[:, f] - A[:, e]; W[:, gf] += A[:, e] - A[:, f]; g[e], g[f] = gf, ge; banned[:] = False
    return g
def objective(A, g): return sum(A[np.ix_(g == c, g == c)].sum() for c in range(4)) / 2
def metrics(r, g):   # r [N, 8] experts, g expert -> card
    c = g[r]; k = np.stack([(c == j).sum(1) for j in range(4)], 1)
    return (k > 0).sum(1), k.max(1)
native = np.arange(128) // 32
# ---- constrained placement (cold need on calibration prompts <= native's) and leave-one-workload-out
cal_prompts = []
for d in sorted(glob.glob(f'{R}/realcase/routing*')):
    fs = sorted(glob.glob(d + '/T128_*.npz'), key=lambda f: (len(f), f)); h = len(fs) // 2
    cal_prompts += [np.load(f)['routing_idx'].astype(np.int64) for f in fs[:h]]
print("\n=== load-constrained: calibration cold need may not exceed the native placement's")
resc = []; coldc = []
for L in range(48):
    X = onehot(cal[L]); A = X.T @ X; np.fill_diagonal(A, 0)
    cnts = np.array([np.bincount(p[L].ravel(), minlength=128) for p in cal_prompts]); bound = cold_need_of(cnts, native)
    g = partition(A, native, cnts=cnts, bound=bound); resc.append(metrics(ev[L], g))
    coldc += [max(np.sort(np.bincount(p[L].ravel(), minlength=128)[g == c])[::-1][16] for c in range(4)) for p in ev_prompts]
nc = np.concatenate([x[0] for x in resc]); mx = np.concatenate([x[1] for x in resc])
print(f"  constrained: cards per token mean {nc.mean():.2f}; spans 1/2/3/4 = " + " / ".join(f"{(nc == k).mean():.2%}" for k in (1, 2, 3, 4)) + f"; busiest card {mx.mean():.2f} of 8; off-card {8 - mx.mean():.2f}; held-out cold need max {max(coldc)}")
print("\n=== leave-one-workload-out (calibrate on all prompts of 7 workloads, evaluate the 8th), unconstrained")
allw = []
for d in sorted(glob.glob(f'{R}/realcase/routing*')):
    fs = sorted(glob.glob(d + '/T128_*.npz'), key=lambda f: (len(f), f))
    allw.append(np.concatenate([np.load(f)['routing_idx'][:, 3:, :].astype(np.int64) for f in fs], 1))
out = []
for i, w in enumerate(names):
    c_nat, c_co = [], []
    for L in range(0, 48, 3):                                                # every third layer, for time
        trn = np.concatenate([allw[j][L] for j in range(len(allw)) if j != i]); X = onehot(trn); A = X.T @ X; np.fill_diagonal(A, 0)
        g = partition(A, native); c_nat.append(metrics(allw[i][L], native)[0].mean()); c_co.append(metrics(allw[i][L], g)[0].mean())
    out.append(f"{w} {np.mean(c_nat):.2f}->{np.mean(c_co):.2f}")
print("  cards per token, native -> co-activation placement from the other 7: " + ", ".join(out))
