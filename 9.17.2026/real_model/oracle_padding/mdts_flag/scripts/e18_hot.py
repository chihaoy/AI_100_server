#!/usr/bin/env python3
"""Hot/cold stage engine time per core from a profile's work.csv. Usage: e18_hot.py <analysis dir> <label>"""
import sys, csv, collections, statistics as st
d, label = sys.argv[1], sys.argv[2]; rows = list(csv.DictReader(open(f'{d}/work.csv')))
HOT = {'MatMul', 'MatMul_1', 'MatMul_2'}; COLD = {'MatMul_3', 'MatMul_4', 'MatMul_5'}
def short(n): return n.split('/mlp/')[-1] if '/mlp/' in n else n
def base(n): return short(n).split('_rc')[0]
agg = collections.defaultdict(lambda: collections.defaultdict(float)); span = collections.defaultdict(lambda: [1e18, -1e18])
t0 = min(float(r['start']) for r in rows)
for r in rows:
    b = base(r['node']); stg = 'hot' if b in HOT else ('cold' if b in COLD else None)
    if not stg or int(r['card']) < 0: continue
    k = {'aicconvolutiond32': 'hmx', 'blockdequantize_mxfp6': 'dequant', 'aiccopytovtcm': 'wdma'}.get(r['kind'])
    if not k: continue
    agg[(stg, int(r['card']), int(r['core']))][k] += float(r['dur'])
    if k == 'hmx': s = span[(stg, int(r['card']))]; s[0] = min(s[0], float(r['start'])); s[1] = max(s[1], float(r['end']))
print(label)
for stg in ('hot', 'cold'):
    for k in ('hmx', 'dequant', 'wdma'):
        v = [a[k] for key, a in agg.items() if key[0] == stg]
        if v: print(f"  {stg:4s} {k:7s} us/core min {min(v):6.0f} med {st.median(v):6.0f} max {max(v):6.0f}")
    print(f"  {stg:4s} HMX span per card, ms: " + ', '.join(f"{c}: {(s[0]-t0)/1e3:.2f}-{(s[1]-t0)/1e3:.2f}" for (g, c), s in sorted(span.items()) if g == stg))
