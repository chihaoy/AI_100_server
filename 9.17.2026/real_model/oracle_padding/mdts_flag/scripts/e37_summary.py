#!/usr/bin/env python3
"""2x2 ablation table from the E37b runs: reduction series x padding series on the full model, per chunk length.
Session 1 holds N, R, P, D (each timed twice, the two medians averaged); session 2 holds N, P2, P, and P2 is expressed in
session-1 terms through the geometric mean of the N and P ratios between the sessions. Usage: e37_summary.py <e37 dir>"""
import sys, os, json, numpy as np
O = sys.argv[1]
def med(n):
    p = f'{O}/run_{n}/result.json'
    return json.load(open(p))['timing']['median_ms'] if os.path.exists(p) else float('nan')
def pair(a): return np.nanmean([med(a), med(a + '_again')])
rows = {}
for T in (128, 256, 512):
    N, R, P, D = (pair(f'T{T}_{x}') for x in 'NRPD')
    n2, p2, pp = pair(f'T{T}_S2_N'), pair(f'T{T}_S2_P2'), pair(f'T{T}_S2_P')
    P2 = p2 * np.sqrt((N / n2) * (P / pp))
    rows[T] = dict(N=N, R=R, P=P, D=D, P2=P2, e36naive=med(f'T{T}_E36naive'))
fmt = lambda v: '–' if np.isnan(v) else f'{v:.1f} ms'
x = lambda v: '–' if np.isnan(v) else f'{v:.2f}×'
print('| Cell | Reduction | Padding | ' + ' | '.join(f'T={T}' for T in rows) + ' |'); print('|---|---|---|' + '---:|' * len(rows))
for key, red, pad in (('N', 'naive (export combine)', 'naive T/T'), ('R', 'ours', 'naive T/T'), ('P2', 'naive (export combine)', 'rank sort, 2 tiers'),
                      ('P', 'naive (export combine)', 'rank sort + 3 tiers'), ('D', 'ours', 'rank sort + 3 tiers (RankTier)')):
    print(f'| {key} | {red} | {pad} | ' + ' | '.join(fmt(r[key]) for r in rows.values()) + ' |')
print()
print('| Speedup | ' + ' | '.join(f'T={T}' for T in rows) + ' |'); print('|---|' + '---:|' * len(rows))
for name, f in (('Reduction series alone, N/R', lambda r: r['N'] / r['R']),
                ('Padding series alone, N/P', lambda r: r['N'] / r['P']),
                ('Rank sort alone, N/P2', lambda r: r['N'] / r['P2']),
                ('Both, N/D', lambda r: r['N'] / r['D']),
                ('Padding on top of our reduction, R/D', lambda r: r['R'] / r['D']),
                ('Reduction on top of our padding, P/D', lambda r: r['P'] / r['D']),
                ('Interaction, (N/D) / ((N/R)(N/P))', lambda r: (r['N'] / r['D']) / ((r['N'] / r['R']) * (r['N'] / r['P'])))):
    print(f'| {name} | ' + ' | '.join(x(f(r)) for r in rows.values()) + ' |')
print('\nE36 naive T/T program re-timed in session 1 (cross-session anchor): ' + ', '.join(f'T={T} {fmt(r["e36naive"])}' for T, r in rows.items()))
json.dump({str(T): r for T, r in rows.items()}, open(f'{O}/e37_ablation.json', 'w'), indent=1, default=float)
