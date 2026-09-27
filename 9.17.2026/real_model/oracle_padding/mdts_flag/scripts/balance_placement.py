#!/usr/bin/env python3
"""Placements that balance instead of localize (arXiv 2510.05497, insights 4-5), judged by what they cost in this design:
the per-card cold-capacity need (17th-largest count per card) of the runtime sort and the padded rows of the run-time lane
split (scripts/split_budget.py rule, 4 extra lanes per card). Placements per layer from the T=128 calibration halves:
  native   expert e on card e // 32
  remap    the paper's remap: experts by decreasing calibration load, each to the least-loaded card with < 32 experts
  spread   swap search that MINIMIZES within-card co-activation (the opposite of coact_placement.py)
  coact    coact_placement.py's localizing placement, for reference
Evaluated out of sample on the T=256 and T=512 chunks; the spread placement is saved to realcase/spread_placement_T128.npy.
Usage: balance_placement.py <mdts_flag dir>"""
import sys, os, glob, numpy as np
R = sys.argv[1]; rng = np.random.default_rng(0)
cal = []
for d in sorted(glob.glob(f'{R}/realcase/routing*')):
    fs = sorted(glob.glob(d + '/T128_*.npz'), key=lambda f: (len(f), f))
    cal += [np.load(f)['routing_idx'][:, 3:, :].astype(np.int64) for f in fs[:len(fs) // 2]]
cal = np.concatenate(cal, 1)                                               # [48, N, 8]
ch = {256: [], 512: []}
for d in sorted(glob.glob(f'{R}/realcase/routing*')):
    for f in sorted(glob.glob(d + '/T*_*.npz')):
        T = int(os.path.basename(f)[1:].split('_')[0]); r = np.load(f)['routing_idx'].astype(np.int64)
        if T == 512: ch[512].append(r); ch[256] += [r[:, :256], r[:, 256:]]
        elif T == 256: ch[256].append(r)
def onehot(r): X = np.zeros((len(r), 128)); np.put_along_axis(X, r, 1.0, axis=1); return X
def swap_search(A, g0, sign, iters=4000):     # sign +1 maximizes within-card co-activation, -1 minimizes it
    g = g0.copy(); W = A @ np.eye(4)[g]; idx = np.arange(128)
    for _ in range(iters):
        own = W[idx, g]; cross = W[:, g]; gain = sign * (cross - own[:, None] + cross.T - own[None, :] - 2 * A)
        gain[g[:, None] == g[None, :]] = -np.inf
        e, f = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[e, f] <= 1e-9: break
        ge, gf = g[e], g[f]; W[:, ge] += A[:, f] - A[:, e]; W[:, gf] += A[:, e] - A[:, f]; g[e], g[f] = gf, ge
    return g
def remap(load):
    g = np.zeros(128, np.int64); tot = np.zeros(4); n = np.zeros(4, int)
    for e in np.argsort(-load, kind='stable'):
        c = min((c for c in range(4) if n[c] < 32), key=lambda c: tot[c]); g[e] = c; tot[c] += load[e]; n[c] += 1
    return g
native = np.arange(128) // 32; coact = np.load(f'{R}/realcase/coact_placement_T128.npy')
P = {'native': np.tile(native, (48, 1)), 'remap': np.zeros((48, 128), np.int64), 'spread': np.zeros((48, 128), np.int64), 'coact': coact}
for L in range(48):
    X = onehot(cal[L]); A = X.T @ X; np.fill_diagonal(A, 0)
    P['remap'][L] = remap(X.sum(0)); P['spread'][L] = swap_search(A, native, -1)
np.save(f'{R}/realcase/spread_placement_T128.npy', P['spread'])   # read by tier_budget.py
def place(n, pools):        # the split_budget.py prefix rule
    placed = np.zeros_like(n, bool)
    for lanes, cap in pools:
        need = np.where(placed | (n == 0), 0, -(-n // cap)); cum = np.cumsum(need, axis=1)
        first_bad = np.where((need > 0) & (cum > lanes), np.arange(32)[None], 32).min(1)
        placed |= (need > 0) & (cum <= lanes) & (np.arange(32)[None] < first_bad[:, None])
    return ~((n > 0) & ~placed).any(1)
def headroom(n, pools, T):
    if not place(n, pools).all(): return 0.0
    lo, hi = 1.0, 3.0
    for _ in range(22):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if place(np.minimum(np.ceil(n * mid), T).astype(np.int64), pools).all() else (lo, mid)
    return lo
for T, rs in ch.items():
    print(f"=== T={T} ({len(rs)} chunks, not used for the placements)")
    for name, pl in P.items():
        rows = []; ncards = []
        for r in rs:
            for L in range(48):
                cnt = np.bincount(r[L].ravel(), minlength=128)
                rows.append(np.concatenate([-np.sort(-cnt[pl[L] == c]) for c in range(4)]).reshape(4, 32))
                ncards.append(np.mean([(np.bincount(x, minlength=4) > 0).sum() for x in pl[L][r[L]]]))
        n = np.concatenate(rows); cold = n[:, 16].max(); hot_sum = n[:, :16].sum(1)
        best = None
        for C_h in range(T // 16, T + 1, T // 32):
            for C_c in range(T // 16, T // 2 + 1, T // 32):
                for C_x in (T // 16, T // 8, T // 4):
                    pools = [(16, C_h), (16, C_c), (4, C_x)]; rr = 16 * (C_h + C_c) + 4 * C_x
                    if best is not None and rr >= best[0]: continue
                    if headroom(n, pools, T) >= 1.28: best = (rr, C_h, C_c, C_x)
        cur = 16 * T + 16 * (T // 8)
        print(f"  {name:6s}: cards per token {np.mean(ncards):.2f}; runtime-sort cold need max {cold} (capacity today {T // 8}); "
              f"busiest card / mean card load: median {np.median([x.sum(1).max() / x.sum(1).mean() for x in rows]):.2f}, max {np.max([x.sum(1).max() / x.sum(1).mean() for x in rows]):.2f}; "
              f"split with 4 extra lanes: {best[1]}/{best[2]}/{best[3]} -> {best[0] / cur:.0%} of today's rows")
