#!/usr/bin/env python3
"""Per-card active-expert balance of expert placements, from the device-captured prompt-41 routing counts (48 layers).
Run from oracle_padding/. Placements: native order, deployed load-sorted hot/cold, hot-contiguous + cold round-robin,
snake and LPT on per-expert token counts, and the per-layer lower bound ceil(active/4)."""
import json, numpy as np
plan = json.load(open('sorted128_fp16_plan.json'))['layers']
counts = np.fromfile('sorted128_fp16_run/counts_i32.bin', np.int32).reshape(48, 128)   # sorted bank-position order
res = {k: [] for k in ('native', 'sorted_contig', 'sorted_hot_coldsnake', 'snake_by_count', 'lpt_by_count', 'ideal', 'total_active')}
for L in range(48):
    order = np.array(plan[str(L)]['order']); nat = np.empty(128, np.int32); nat[order] = counts[L]; active = nat > 0
    res['total_active'].append(int(active.sum())); res['ideal'].append(int(np.ceil(active.sum() / 4)))
    pos_of = np.empty(128, int); pos_of[order] = np.arange(128)
    cards = {'native': (np.arange(128) % 64) // 16, 'sorted_contig': (pos_of % 64) // 16}
    hcs = cards['sorted_contig'].copy(); cold = pos_of >= 64; hcs[cold] = (pos_of[cold] - 64) % 4; cards['sorted_hot_coldsnake'] = hcs
    rank = np.argsort(-nat, kind='stable'); snake = np.empty(128, int); snake[rank] = np.tile(np.r_[0, 1, 2, 3, 3, 2, 1, 0], 16); cards['snake_by_count'] = snake
    lpt = np.empty(128, int); load = np.zeros(4); cap = np.zeros(4, int)
    for e in rank:
        c = int(np.argmin(np.where(cap < 32, load, np.inf))); lpt[e] = c; load[c] += float(nat[e] > 0); cap[c] += 1
    cards['lpt_by_count'] = lpt
    for k, card_of in cards.items(): res[k].append(int(np.bincount(card_of[active], minlength=4).max()))
for k in ('native', 'sorted_contig', 'sorted_hot_coldsnake', 'snake_by_count', 'lpt_by_count', 'ideal'):
    a = np.array(res[k]); print(f'{k:>22s} mean max-card active {a.mean():6.2f}  sum {a.sum():5d}  est GEMM ms/layer at 0.1 ms/expert {a.mean()*0.1:.2f}')
print('active experts per layer: mean %.1f min %d max %d' % (np.mean(res['total_active']), min(res['total_active']), max(res['total_active'])))
