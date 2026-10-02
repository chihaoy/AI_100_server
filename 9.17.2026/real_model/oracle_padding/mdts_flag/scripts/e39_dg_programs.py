#!/usr/bin/env python3
"""Program list for the direct-gather E39 session at T=128 or T=256: RankTier, the rank sort at capacity T and the static
naive T/T at their best two-layer tile sizes (E38a), every block size with direct gather at the default tile size (the tile
hardly matters for blocks: within 3% at T=512) for the dropless budget and, where their layer-0/1 budgets differ, the
calibrated and trace-specialized budgets, and the 64-row dropless program without direct gather as the reference.
Usage: e39_dg_programs.py <mdts_flag dir> <T>"""
import sys, numpy as np
R, T = sys.argv[1], int(sys.argv[2])
best = {128: dict(E='512', C='512', A='def'), 256: dict(E='1024', C='512', A='512')}[T]
progs = [f'e38E:{best["E"]}', f'e38C:{best["C"]}', f'e38A:{best["A"]}']
for B in (16, 32, 64, 128):
    seen = []
    for kind, lab in (('dropless', ''), ('calibrated', 'cal'), ('oracle', 'orc')):
        b = tuple(np.load(f'{R}/full_model/e39/budget_T{T}_B{B}_{kind}.npy')[:2])
        if b not in seen: progs.append(f'blk{B}{lab}dg:def'); seen.append(b)
progs.append('blk64:def')
print(' '.join(progs))
