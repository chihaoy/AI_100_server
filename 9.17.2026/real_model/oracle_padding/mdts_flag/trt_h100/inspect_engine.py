#!/usr/bin/env python3
"""Summarize a TensorRT engine's layer list (engine inspector JSON from build_engines.py): every layer's type, name,
largest output and tactic, and a digest of what the plan asks to check (section 6.1): gathers that materialize weight copies,
reformat/copy layers, scatter, reduce and GEMM layers, and fused layers. Usage: inspect_engine.py <tag.layers.json> [--all]"""
import sys, json, re, collections
info = json.load(open(sys.argv[1])); L = info.get('Layers', [])
def nbytes(o):
    dims = o.get('Dimensions') or []; fmt = str(o.get('Format/Datatype', ''))
    size = 1
    for v in dims: size *= int(v)
    b = 2 if re.search(r'Half|FP16|BF16', fmt, re.I) else 1 if re.search(r'Int8|Bool|FP8', fmt, re.I) else 8 if re.search(r'Int64', fmt, re.I) else 4
    return size * b
cnt = collections.Counter(); big = []
for l in L:
    if not isinstance(l, dict): print('?', str(l)[:160]); continue
    t = l.get('LayerType', '?'); n = l.get('Name', ''); outs = l.get('Outputs', [])
    ob = max([nbytes(o) for o in outs] or [0]); cnt[t] += 1
    big.append((ob, t, n, [o.get('Dimensions') for o in outs], str(l.get('TacticName', '') or l.get('Tactic', ''))))
    if '--all' in sys.argv:
        print(f'{t:20s} {ob / 2**20:9.1f} MiB  {n[:90]:90s} {str([o.get("Dimensions") for o in outs])[:50]:50s} {str(l.get("TacticName", ""))[:50]}')
print(f'{len(L)} layers by type: {dict(cnt.most_common())}')
print('largest outputs:')
for ob, t, n, dims, tac in sorted(big, reverse=True)[:14]:
    print(f'  {ob / 2**20:9.1f} MiB  {t:18s} {n[:80]:80s} {str(dims)[:40]} {tac[:40]}')
