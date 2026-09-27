#!/usr/bin/env python3
"""Per-layer critical-path timestamps (ms from the layer's first attention op) from a detail_analyze work.csv:
attention end, router end, per-card sort end, first expert-weight stream, first expert GEMM, MoE end.
Usage: layer_timeline.py <analysis dir> <label>"""
import sys, csv, re, collections
d, label = sys.argv[1], sys.argv[2]; rows = list(csv.DictReader(open(f'{d}/work.csv')))
W = collections.defaultdict(lambda: collections.defaultdict(lambda: [1e18, -1e18]))
EXP = re.compile(r'^MatMul(_[1-5])?(_rc\d+)?$')
for r in rows:
    m = re.match(r'/model/layers\.(\d+)/(self_attn|input_layernorm|mlp|post_attention_layernorm)/(.*)', r['node'])
    if not m or int(r['card']) < 0: continue
    L, blk, rest = int(m.group(1)), m.group(2), m.group(3); s, e = float(r['start']), float(r['end'])
    keys = []
    if blk in ('self_attn', 'input_layernorm'): keys.append('attention')
    if blk == 'mlp' and rest.startswith('gate/'): keys.append('router')
    if blk == 'mlp' and rest.startswith('dc_topk'): keys.append('sort')
    if blk == 'mlp' and ((EXP.match(rest) and r['kind'] in ('aiccopytovtcm', 'aicgather')) or rest.startswith('dc_gather')): keys.append('weights')
    if blk == 'mlp' and EXP.match(rest) and r['kind'] == 'aicconvolutiond32': keys.append('gemm')
    if blk in ('mlp', 'post_attention_layernorm'): keys.append('moe')
    for k in keys: w = W[L][k]; w[0] = min(w[0], s); w[1] = max(w[1], e)
print(label)
for L in sorted(W):
    w = W[L]; t0 = w['attention'][0]; f = lambda k, i: f"{(w[k][i]-t0)/1e3:6.3f}" if w[k][0] < 1e17 else '   -  '
    print(f"  layer {L}: attention end {f('attention',1)} | router end {f('router',1)} | sort end {f('sort',1)} | first weight stream {f('weights',0)} | first expert GEMM {f('gemm',0)} | MoE end {f('moe',1)}  (ms from attention start)")
