#!/usr/bin/env python3
"""Run-time lane split, feasibility on calibration routing. Per card and layer, experts sorted by count (descending) are
placed by a prefix rule that a graph can compute with cumulative sums: hot pool of 16 lanes x C_h (expert i takes
ceil(n_i / C_h) lanes while the running total stays <= 16), then the cold pool of 16 lanes x C_c (same rule on the rest),
then an extra pool of k lanes x C_x (k in 1, 2, 4, 8 keeps the core mapping even, E19). Feasible if every active expert is
placed. Designs are ranked by padded rows among those with at least the current design's headroom (uniform scaling of the
counts, capped at T, over the calibration chunks); the held-out prompts of E28/E29 are checked out of sample.
Usage: split_budget.py <mdts_flag dir>"""
import sys, os, glob, numpy as np
R = sys.argv[1]; RC = f'{R}/realcase'; ch = {256: [], 512: []}
cnt = lambda x: np.array([np.bincount(y.ravel(), minlength=128) for y in x])
for d in sorted(glob.glob(RC + '/routing*')):
    for f in sorted(glob.glob(d + '/T*_*.npz')):
        T = int(os.path.basename(f)[1:].split('_')[0]); r = np.load(f)['routing_idx']
        if T == 512: ch[512].append(cnt(r)); ch[256] += [cnt(r[:, :256]), cnt(r[:, 256:])]
        elif T == 256: ch[256].append(cnt(r))
def place(n, pools):
    """n: [M, 32] counts sorted descending per row; pools: [(lanes, cap)]; returns feasible mask [M] and lanes used per pool."""
    left = n.copy(); ok = np.ones(len(n), bool); used = []
    placed = np.zeros_like(n, bool)
    for lanes, cap in pools:
        need = np.where(placed | (n == 0), 0, -(-n // cap))                  # lanes each unplaced expert needs in this pool
        # prefix of the unplaced experts (they are a suffix of the sorted list, so keep order)
        cum = np.cumsum(need, axis=1); take = (need > 0) & (cum <= lanes)
        # prefix rule: stop at the first expert that does not fit
        first_bad = np.where((need > 0) & (cum > lanes), np.arange(32)[None], 32).min(1)
        take &= np.arange(32)[None] < first_bad[:, None]
        placed |= take; used.append(np.where(take, need, 0).sum(1))
    ok = ~((n > 0) & ~placed).any(1)
    return ok, used
def headroom(n, pools, T):
    """largest s such that min(ceil(s * counts), T) still places every expert (a count can never exceed T)"""
    if not place(n, pools)[0].all(): return 0.0
    lo, hi = 1.0, 3.0
    for _ in range(25):
        mid = (lo + hi) / 2; ok, _ = place(np.minimum(np.ceil(n * mid), T).astype(np.int64), pools)
        lo, hi = (mid, hi) if ok.all() else (lo, mid)
    return lo
for T in (256, 512):
    X = np.array(ch[T]).reshape(-1, 48, 4, 32); n = -np.sort(-X.reshape(-1, 32), axis=1)                # all (chunk, layer, card) rows
    H = np.fromfile(f'{R}/full_model/ref{T}/counts_i32.bin', np.int32).reshape(48, 4, 32); nh = -np.sort(-H.reshape(-1, 32), axis=1)
    s_cur = (T // 8) / n[:, 16].max(); cur_rows = 16 * T + 16 * (T // 8)                              # current: one lane per expert
    print(f"T={T} ({len(ch[T])} chunks): current 16x{T} + 16x{T // 8} = {cur_rows} rows/card, headroom x{s_cur:.2f}; "
          f"active experts per card mean {(n > 0).sum(1).mean():.1f}, max {(n > 0).sum(1).max()}")
    for k in (0, 4, 8):
        best = None
        for C_h in range(T // 16, T + 1, T // 32):
            for C_c in range(T // 16, T // 2 + 1, T // 32):
                for C_x in ((0,) if k == 0 else (T // 16, T // 8, T // 4)):
                    pools = [(16, C_h), (16, C_c)] + ([(k, C_x)] if k else []); rows = 16 * C_h + 16 * C_c + k * C_x
                    if best is not None and rows >= best[0]: continue
                    s = headroom(n, pools, T)
                    if s >= s_cur: best = (rows, C_h, C_c, C_x, s, headroom(nh, pools, T))
        if best is None: print(f"  split, extra lanes/card {k}: nothing with the current headroom"); continue
        rows, C_h, C_c, C_x, s, sh = best
        print(f"  split, extra lanes/card {k}: C_hot {C_h}, C_cold {C_c}" + (f", C_extra {C_x}" if k else "") +
              f" -> {rows} rows/card = {rows / cur_rows:.0%} of current, {rows / (32 * T):.0%} of naive; headroom x{s:.2f}, held-out prompt x{sh:.2f}")
