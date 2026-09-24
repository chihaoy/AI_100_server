#!/usr/bin/env python3
"""Per-layer attention / MoE spans (ms) from a profile's work.csv. Usage: spans.py <analysis dir> <label>"""
import sys, csv, re, collections
d, label = sys.argv[1], sys.argv[2]; rows = list(csv.DictReader(open(f'{d}/work.csv')))
t0 = min(float(r['start']) for r in rows)
span = collections.defaultdict(lambda: [1e18, -1e18])
for r in rows:
    m = re.match(r'/model/layers\.(\d+)/(self_attn|mlp|input_layernorm|post_attention_layernorm)/', r['node'])
    if not m: continue
    k = (int(m.group(1)), 'attn' if m.group(2) in ('self_attn', 'input_layernorm') else 'mlp'); s = span[k]
    s[0] = min(s[0], float(r['start'])); s[1] = max(s[1], float(r['end']))
print(label)
for L in sorted({k[0] for k in span}):
    a, m = span[(L, 'attn')], span[(L, 'mlp')]
    print(f"  layer {L}: attention {(a[0]-t0)/1e3:6.2f}-{(a[1]-t0)/1e3:6.2f} ms (span {(a[1]-a[0])/1e3:5.2f})   moe {(m[0]-t0)/1e3:6.2f}-{(m[1]-t0)/1e3:6.2f} ms (span {(m[1]-m[0])/1e3:5.2f})")
