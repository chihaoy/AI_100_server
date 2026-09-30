#!/usr/bin/env python3
"""Ladder table from the E36 runs: each step against the step before it (same session), cumulative against naive T/T and
production. Batch B steps are chained through naive + tileadd, which runs in both batches. Usage: e36_summary.py <e36 dir>"""
import sys, os, json, numpy as np
O = sys.argv[1]
def med(n):
    p = f'{O}/run_{n}/result.json'
    return json.load(open(p))['timing']['median_ms'] if os.path.exists(p) else float('nan')
names = ['1 production', '2 + flag, head-parallel attention', '3 + stack rewrites', '4 + token-owned combine (naive T/T)', '5 + elementwise final sum',
         '6 + runtime sort hot/cold', '7 + 3 tiers']
print('| Step | ' + ' | '.join(f'T={T}' for T in (128, 256, 512)) + ' |'); print('|---|' + '---:|' * 3)
cols = {}
for T in (128, 256, 512):
    a_naive = np.nanmean([med(f'T{T}_4_naive'), med(f'T{T}_4_naive_again')])
    a_tile = med(f'T{T}_5_tileadd'); b_tile = np.nanmean([med(f'T{T}_5_tileadd_B'), med(f'T{T}_5_tileadd_B_again')])
    scale = a_tile / b_tile                        # express batch B in batch A's terms through the shared program
    v = [med(f'T{T}_1_production'), med(f'T{T}_2_flag_headpar'), med(f'T{T}_3_rewrites'), a_naive, a_tile,
         med(f'T{T}_6_runtime_sort') * scale, med(f'T{T}_7_tiers') * scale]
    cols[T] = v
for i, nm in enumerate(names):
    cells = []
    for T in (128, 256, 512):
        v = cols[T]; s = f'{v[i]:.1f} ms'
        if i > 0: s += f' ({(v[i] / v[i - 1] - 1) * 100:+.1f}%)'
        cells.append(s)
    print(f'| {nm} | ' + ' | '.join(cells) + ' |')
print('| **Steps 3-5, reductions** | ' + ' | '.join(f'{cols[T][1] / cols[T][4]:.2f}×' for T in (128, 256, 512)) + ' |')
print('| **Steps 6-7, padding** | ' + ' | '.join(f'{cols[T][4] / cols[T][6]:.2f}×' for T in (128, 256, 512)) + ' |')
print('| **Final vs naive T/T** | ' + ' | '.join(f'{cols[T][3] / cols[T][6]:.2f}×' for T in (128, 256, 512)) + ' |')
print('| **Final vs production** | ' + ' | '.join(f'{cols[T][0] / cols[T][6]:.2f}×' for T in (128, 256, 512)) + ' |')
json.dump({str(T): dict(zip(names, cols[T])) for T in cols}, open(f'{O}/e36_ladder.json', 'w'), indent=1, default=float)
