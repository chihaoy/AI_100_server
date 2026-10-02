#!/usr/bin/env python3
"""Tile predication in the expert stages of one layer (E43), from a detail_analyze.py work table and the routing counts of
the same prompt. The export ends every expert lane with Where(row < n_e, y, 0); the compiler cuts the lane's rows into tiles
(the size follows -size-split-granularity and the cores per lane) and gives each tile a runtime predicate (elementcmplte +
aicreducebooltoscalar) that gates the tile's activation gather, GEMMs and SiLU; the zero rows come from aicselectsplatrange.
Per stage and card this reports: lanes and empty lanes; predicates per lane; the rows the SiLU actually processed (sigmoid
bytes / (768 x 2)) against tile * ceil(n / tile) for every candidate tile (the tile that matches every card is printed);
the lanes whose weights were read (weight bytes / 3.52 MiB per expert: bank-gather reads, or a static stage's DDR-to-VTCM
reads less spills) against the non-empty lanes; HMX busy.
Usage: tile_predication.py <analysis sample dir> <result.json with routing_counts> <tiers, e.g. 16x512,16x512> [layer=1]
  The routing counts must be in the program's own lane order (tier after tier, card after card), as every E38 program reports
  them; program A's stages are its two 16-lane tiers."""
import sys, csv, re, json, collections, numpy as np
d, res, spec = sys.argv[1], sys.argv[2], sys.argv[3]; L = int(sys.argv[4]) if len(sys.argv) > 4 else 1
P = f'/model/layers.{L}/mlp/'; EXPERT_MIB = 3.515625
tiers = [tuple(int(v) for v in t.split('x')) for t in spec.split(',')]
counts = np.array(json.load(open(res))['routing_counts']).reshape(-1, 128)[L]
def stage(n):
    t = n[len(P):]
    m = re.match(r'tr(\d+)(?:/|_gather_)', t)
    if m: return int(m.group(1))
    if re.match(r'(MatMul|MatMul_1|MatMul_2|act_fn|Where_3|CtxGather3D)(/|$)', t): return 0
    if re.match(r'(MatMul_3|MatMul_4|MatMul_5|act_fn_1|to_stage1_masked|CtxGather3D_2)(/|$)', t): return 1
    return None
S = collections.defaultdict(lambda: dict(pred=0, sig=0, gather=0.0, inb=0.0, spill=0.0, hmx=0.0))
for r in csv.DictReader(open(f'{d}/work.csv')):
    n = r['node']
    if not n.startswith(P): continue
    st = stage(n)
    if st is None: continue
    x = S[(st, int(r['card']))]; k = r['kind'].strip(); b = int(r['bytes'] or 0)
    if k == 'aicreducebooltoscalar': x['pred'] += 1
    if k == 'sigmoid' and n.endswith('Sigmoid'): x['sig'] += b
    if r['engine'] == 'HMX' and re.search(r'MatMul(_\d)?$', n): x['hmx'] += float(r['dur'])
    if 'DDR' in r['memory'] and 'DMA' in r['engine']:
        if '_gather_' in n: x['gather'] += b / 2**20
        elif re.search(r'MatMul(_\d)?$', n): x['spill' if 'fromvtcm' in k else 'inb'] += b / 2**20
off = 0
print(f'== {d}, layer {L}, tiers {spec}')
for t, (lanes, C) in enumerate(tiers):
    blk = counts[off:off + 4 * lanes].reshape(4, lanes); off += 4 * lanes
    cand = [C // k for k in (1, 2, 4, 8, 16) if C % k == 0 and C // k >= 16]
    rows = {c: S[(t, c)]['sig'] / (768 * 2) for c in range(4)}
    match = [tile for tile in cand if all(abs(rows[c] - sum(tile * -(-int(n) // tile) for n in blk[c])) < 0.5 for c in range(4))]
    wl = {c: ((S[(t, c)]['gather'] if S[(t, c)]['gather'] > 0 else S[(t, c)]['inb'] - S[(t, c)]['spill']) / EXPERT_MIB) for c in range(4)}
    for c in range(4):
        n = blk[c]; e = int((n == 0).sum())
        print(f'   tier {t} ({lanes}x{C}) card {c}: {lanes} lanes, {e} empty, max {int(n.max())} tokens; predicates per lane {S[(t, c)]["pred"] / lanes:.2f}; '
              f'rows computed {rows[c]:.0f} of {lanes * C} provisioned ({int(n.sum())} tokens); weights read for {wl[c]:.1f} lanes ({lanes - e} non-empty); '
              f'HMX busy {S[(t, c)]["hmx"] / 1e3:.2f} ms')
    print(f'   tier {t}: rows computed == tile * ceil(n / tile) on every card for tile = {match if match else "none of " + str(cand)}')
