#!/usr/bin/env python3
"""Program list for the second E39 session of a chunk length: RankTier's anchor plus the calibrated and trace-specialized
block budgets, each at the fastest tile size of the dropless program with the same block size in the first session.
Budget variants whose layer-0/1 budgets equal the dropless budget (the two-layer cut would be the same program) are skipped.
Direct-gather variants of RankTier and of the 64-row block program are added when their two-layer cuts exist.
Usage: e39_next.py <mdts_flag dir> <T> <first session log>"""
import sys, re, numpy as np
R, T, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]
means = dict(re.findall(r'(\S+) ([\d.]+) ms', open(log).read().split('session means (both passes): ')[1].split('\n')[0]))
progs = ['e38E:' + {128: '512', 256: '1024', 512: '1024'}[T]]
for B in (16, 32, 64, 128):
    tiles = {ts: float(means[f'blk{B}_{ts}']) for ts in ('def', '512', '1024') if f'blk{B}_{ts}' in means}
    if not tiles: continue
    best = min(tiles, key=tiles.get); d = np.load(f'{R}/full_model/e39/budget_T{T}_B{B}_dropless.npy')[:2]; seen = [tuple(d)]
    for kind, lab in (('calibrated', 'cal'), ('oracle', 'orc')):
        b = tuple(np.load(f'{R}/full_model/e39/budget_T{T}_B{B}_{kind}.npy')[:2])
        if b not in seen: progs.append(f'blk{B}{lab}:{best}'); seen.append(b)
import os
for lab in ('e38Edg', 'blk64dg'):   # direct-gather variants (build_full_tiered.py --direct-gather), at the anchor's and blk64's best tiles
    if os.path.exists(f'{R}/full_model/trunc2_{T}_{lab}/model.onnx'):
        if lab == 'e38Edg': progs.append(f'{lab}:' + progs[0].split(':')[1])
        else:
            t64 = {ts: float(means[f'blk64_{ts}']) for ts in ('def', '512', '1024') if f'blk64_{ts}' in means}
            if t64: progs.append(f'{lab}:' + min(t64, key=t64.get))
print(' '.join(progs))
