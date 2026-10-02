#!/usr/bin/env python3
"""Block budgets for the fixed-size block baseline (experiment 2, E39). A card packs its local assignments into blocks of B
rows, one expert per block: expert e needs ceil(n_e / B) blocks. Budgets per card:
  dropless    min(ceil(8T/B) + 31, 32 * ceil(T/B)) (any routing: up to 8T local assignments), rounded up to 16
  calibrated  1.28 x the largest active-block count of any card, layer and calibration chunk (RankTier's headroom rule),
              rounded up to 16 and capped at the dropless budget, one budget for all layers
  oracle      per layer, the largest active-block count over the cards for the timed prompt (device routing counts of
              the E38 static program, mapped from its card-major lane order back to experts), rounded up to 16
              (trace-specialized, not deployable); the first version read the FP32 reference counts at T=128 and 256 (no
              device run existed yet) and took the T=512 lanes for experts, which gives the same budgets in layers 0-1
Calibration chunks as in tier_budget.py: every realcase/routing*/T<T>_*.npz (T=256 also from the halves of the 512 chunks).
Writes full_model/e39/budget_T<T>_B<B>_<kind>.npy ([48] per-layer budgets) and prints rows per card against RankTier.
Usage: e39_block_budget.py <mdts_flag dir>"""
import sys, os, glob, json, numpy as np
R = sys.argv[1]; O = f'{R}/full_model/e39'; os.makedirs(O, exist_ok=True); HEAD = 1.28
rup = lambda x: int(16 * np.ceil(x / 16))
TIERS = {128: [(8, 128), (8, 48), (16, 16)], 256: [(8, 256), (8, 64), (16, 32)], 512: [(8, 512), (8, 128), (16, 64)]}
chunks = {128: [], 256: [], 512: []}
for d in sorted(glob.glob(f'{R}/realcase/routing*')):
    for f in sorted(glob.glob(d + '/T*_*.npz')):
        T = int(os.path.basename(f)[1:].split('_')[0]); r = np.load(f)['routing_idx'].astype(np.int64)
        if T in chunks: chunks[T].append(r)
        if T == 512: chunks[256] += [r[:, :256], r[:, 256:]]
def counts(r): return np.array([np.bincount(r[L].ravel(), minlength=128) for L in range(48)])      # [48, 128]
ORDER = [32 * c + j for c in range(4) for j in range(16)] + [32 * c + 16 + j for c in range(4) for j in range(16)]   # program A's lanes
def device_counts(T):   # expert-order counts of the timed prompt from the static program's routing counts (card-major lanes)
    fs = sorted(glob.glob(f'{R}/full_model/e38/run_T{T}_S*_A_*/result.json')); assert fs, f'no E38 static-program run at T={T}'
    lane = np.array(json.load(open(fs[0]))['routing_counts']).reshape(48, 128); c = np.zeros_like(lane); c[:, ORDER] = lane
    return c
for T in (128, 256, 512):
    C = np.array([counts(r) for r in chunks[T]])                 # [n, 48, 128]
    dev = device_counts(T); tier_rows = sum(l * c for l, c in TIERS[T])
    print(f'\n=== T={T}: {len(C)} calibration chunks; RankTier {"+".join(f"{l}x{c}" for l, c in TIERS[T])} = {tier_rows} rows per card; '
          f'naive T/T {32 * T}; assignments per card in the timed prompt: mean {dev.reshape(48, 4, 32).sum(-1).mean():.0f}, max {dev.reshape(48, 4, 32).sum(-1).max()}')
    print('| B | dropless N | rows | calibrated N | rows | oracle N, layers 0-1 (max over 48) | rows, layers 0-1 | active blocks per card in the prompt, layer 0 |')
    print('|---:|---:|---:|---:|---:|---|---:|---|')
    for B in (16, 32, 64, 128):
        drop = rup(min(-(-8 * T // B) + 31, 32 * (T // B)))
        act = -(-C // B)                                         # blocks per expert
        per_card = act.reshape(len(C), 48, 4, 32).sum(-1)        # [n, 48, 4]
        cal = min(drop, rup(HEAD * per_card.max()))
        dact = (-(-dev // B)).reshape(48, 4, 32).sum(-1)         # [48, 4]
        orc = np.array([rup(v) for v in dact.max(1)])
        for kind, v in (('dropless', [drop] * 48), ('calibrated', [cal] * 48), ('oracle', orc)):
            np.save(f'{O}/budget_T{T}_B{B}_{kind}.npy', np.array(v, np.int64))
        over = int((dact > cal).sum())
        print(f'| {B} | {drop} | {drop * B} | {cal} | {cal * B} | {orc[0]}, {orc[1]} ({orc.max()}) | {orc[0] * B}, {orc[1] * B} | {dact[0].tolist()} |' + (f' prompt exceeds calibrated budget in {over} (layer, card)' if over else ''))
