#!/usr/bin/env python3
"""Tiered capacities by per-card rank, the no-split alternative to the run-time lane split (split_budget.py). The per-card
runtime sort orders each card's 32 experts by count; tier i takes the next L_i ranks (L_i in 1, 2, 4, 8, 16 lanes per
card, so every stage maps one lane per core or divides evenly, E19) at capacity C_i = headroom x the largest count of the
tier's top rank over the calibration chunks (capped at T, rounded up to 16). Every expert keeps one lane. Reports the
cheapest tier structures with 2-5 stages against today's 16 x T + 16 x T/8, under the native and the spread placement
(balance_placement.py), and checks the held-out prompts. Usage: tier_budget.py <mdts_flag dir>"""
import sys, os, glob, itertools, numpy as np
R = sys.argv[1]; HEAD = 1.28
ch = {256: [], 512: []}
for d in sorted(glob.glob(f'{R}/realcase/routing*')):
    for f in sorted(glob.glob(d + '/T*_*.npz')):
        T = int(os.path.basename(f)[1:].split('_')[0]); r = np.load(f)['routing_idx'].astype(np.int64)
        if T == 512: ch[512].append(r); ch[256] += [r[:, :256], r[:, 256:]]
        elif T == 256: ch[256].append(r)
def sorted_per_card(rs, place):   # -> [n, 32] per (chunk, layer, card) counts sorted descending
    out = []
    for r in rs:
        for L in range(48):
            cnt = np.bincount(r[L].ravel(), minlength=128)
            for c in range(4): out.append(-np.sort(-cnt[place[L] == c]))
    return np.array(out)
def structures(max_tiers):
    for k in range(2, max_tiers + 1):
        for lanes in itertools.product((1, 2, 4, 8, 16), repeat=k):
            if sum(lanes) == 32: yield lanes
def rup(x, T): return int(min(T, 16 * np.ceil(x / 16)))
native = np.tile(np.arange(128) // 32, (48, 1))
places = {'native': native}
sp = f'{R}/realcase/spread_placement_T128.npy'
if os.path.exists(sp): places['spread'] = np.load(sp)
for T, rs in ch.items():
    cur = 16 * T + 16 * (T // 8)
    held = np.fromfile(f'{R}/full_model/ref{T}/counts_i32.bin', np.int32).reshape(48, 128)
    print(f"=== T={T} ({len(rs)} calibration chunks); today 16x{T} + 16x{T // 8} = {cur} rows per card")
    for pname, pl in places.items():
        n = sorted_per_card(rs, pl); mx = n.max(0)                      # largest count at each rank
        nh = np.array([-np.sort(-held[L][pl[L] == c]) for L in range(48) for c in range(4)])
        print(f"  {pname}: largest count at ranks 1,2,3,4,5,8,9,12,13,16,17,24,25,32 = {[int(mx[i - 1]) for i in (1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 24, 25, 32)]}")
        best = {}
        for lanes in structures(5):
            caps, rows, start = [], 0, 0
            for Lk in lanes:
                cap = rup(HEAD * mx[start], T) if start > 0 else T; caps.append(max(cap, 16)); rows += Lk * caps[-1]; start += Lk
            k = len(lanes)
            if k not in best or rows < best[k][0]: best[k] = (rows, lanes, caps)
        for k, (rows, lanes, caps) in sorted(best.items()):
            ok = all((nh[:, sum(lanes[:i]):sum(lanes[:i + 1])] <= caps[i]).all() for i in range(len(lanes)))
            print(f"    {k} stages: lanes/card {lanes}, capacities {caps} -> {rows} rows = {rows / cur:.0%} of today; held-out prompt fits: {ok}")
