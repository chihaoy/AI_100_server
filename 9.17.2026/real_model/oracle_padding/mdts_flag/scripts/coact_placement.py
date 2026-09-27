#!/usr/bin/env python3
"""Co-activation-aware expert placement: can a 32-per-card assignment put a token's 8 experts on fewer cards?
Per layer, A[e, f] = calibration tokens that chose both e and f; a balanced 4-way partition is searched by greedy pairwise
swaps (Kernighan-Lin style) that maximize within-card co-activation, from the native placement and from random starts;
evaluated on held-out prompts (first half of each workload's T=128 prompts calibrates, second half evaluates). The three
chat-template tokens at positions 0-2 (identical in every prompt) are excluded. Also reports the per-card cold-capacity
need (17th-largest count per card, per 128-token prompt) under each placement.
Usage: coact_placement.py <mdts_flag dir> [restarts]"""
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
def partition(A, g0, iters=4000):
    g = g0.copy(); G = np.eye(4)[g]; W = A @ G                            # W[e, c] = co-activation of e with card c
    idx = np.arange(128)
    for _ in range(iters):
        own = W[idx, g]; cross = W[:, g]                                   # cross[e, f] = W[e, g[f]]
        gain = cross - own[:, None] + cross.T - own[None, :] - 2 * A
        gain[g[:, None] == g[None, :]] = -np.inf
        e, f = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[e, f] <= 1e-9: break
        ge, gf = g[e], g[f]; W[:, ge] += A[:, f] - A[:, e]; W[:, gf] += A[:, e] - A[:, f]; g[e], g[f] = gf, ge
    return g
def objective(A, g): return sum(A[np.ix_(g == c, g == c)].sum() for c in range(4)) / 2
def metrics(r, g):   # r [N, 8] experts, g expert -> card
    c = g[r]; k = np.stack([(c == j).sum(1) for j in range(4)], 1)
    return (k > 0).sum(1), k.max(1)
native = np.arange(128) // 32
res = {'native': [], 'coact': []}; place = np.zeros((48, 128), np.int64); cold_need = {'native': [], 'coact': []}
for L in range(48):
    X = onehot(cal[L]); A = X.T @ X; np.fill_diagonal(A, 0)
    best = partition(A, native); bobj = objective(A, best)
    for _ in range(RESTARTS):
        g0 = rng.permutation(np.repeat(np.arange(4), 32)); g = partition(A, g0); o = objective(A, g)
        if o > bobj: best, bobj = g, o
    assert (np.bincount(best, minlength=4) == 32).all(); place[L] = best
    for key, g in (('native', native), ('coact', best)):
        nc, mx = metrics(ev[L], g); res[key].append((nc, mx))
        for p in ev_prompts:
            cnt = np.bincount(p[L].ravel(), minlength=128)
            cold_need[key].append(max(np.sort(cnt[g == c])[::-1][16] for c in range(4)))
N = ev.shape[1]
print(f"calibration tokens {cal.shape[1]}, held-out tokens {N} (8 workloads, T=128 prompts, template tokens excluded), restarts {RESTARTS}")
for key in ('native', 'coact'):
    nc = np.concatenate([x[0] for x in res[key]]); mx = np.concatenate([x[1] for x in res[key]])
    print(f"  {key:6s}: cards per token mean {nc.mean():.2f}; spans 1/2/3/4 cards = " + " / ".join(f"{(nc == k).mean():.2%}" for k in (1, 2, 3, 4)) +
          f"; experts on the token's busiest card mean {mx.mean():.2f} of 8; assignments off that card {8 - mx.mean():.2f}; cold need (17th count/card) max {max(cold_need[key])}")
gain_l = [np.concatenate([res['native'][L][0]]).mean() - res['coact'][L][0].mean() for L in range(48)]
print("  layers with the largest drop in cards per token: " + ", ".join(f"L{L} {res['native'][L][0].mean():.2f}->{res['coact'][L][0].mean():.2f}" for L in np.argsort(gain_l)[::-1][:6]))
np.save(f'{R}/realcase/coact_placement_T128.npy', place)
