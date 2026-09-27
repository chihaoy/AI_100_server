#!/usr/bin/env python3
"""Per-card view of a 2-layer full-model profile (detail_analyze work.csv): hot-stage window and GEMM median per card, the final
cross-card sum (Einsum tiles or elementwise adds) per card, and where marker ops ran (which trace card holds logical slice 0).
Usage: e31_cards.py <analysis dir> <label>"""
import sys, csv, re, statistics as st, collections
d, label = sys.argv[1], sys.argv[2]; rows = list(csv.DictReader(open(f'{d}/work.csv')))
print(f"=== {label}")
mk = collections.defaultdict(set)
for r in rows:
    for m in ('/model/embed_tokens/', '/lm_head/', '/model/norm/'):
        if r['node'].startswith(m) and int(r['card']) >= 0: mk[m].add(int(r['card']))
print("  marker ops on trace cards: " + ", ".join(f"{m.strip('/').split('/')[-1]} {sorted(v)}" for m, v in mk.items()))
for L in (0, 1):
    P = f'/model/layers.{L}/mlp/'; out = []
    for c in range(4):
        hot = [r for r in rows if r['node'].startswith(P) and int(r['card']) == c and re.search(r'/mlp/MatMul(_[12])?$', r['node'])]
        g = [float(r['dur']) for r in hot if r['engine'].upper().startswith('HMX')]
        t0 = min(float(r['start_ms']) for r in hot); t1 = max(float(r['end_ms']) for r in hot)
        fin = [r for r in rows if r['node'].startswith(P) and int(r['card']) == c and re.search(r'/mlp/(final_tile|fa_|ft\d+_)', r['node'])]
        fs = f"final sum {min(float(r['start_ms']) for r in fin):.3f}-{max(float(r['end_ms']) for r in fin):.3f} on {len({r['core'] for r in fin})} cores" if fin else "final sum: none"
        out.append(f"card {c}: hot {t1 - t0:.3f} ms (GEMM median {st.median(g):.1f} us); {fs}")
    cold_end = max(float(r['end_ms']) for r in rows if r['node'].startswith(P) and re.search(r'/mlp/MatMul_[345]$', r['node']))
    fin_all = [r for r in rows if r['node'].startswith(P) and re.search(r'/mlp/(final_tile|fa_|ft\d+_)', r['node'])]
    print(f"  layer {L}: tail last cold GEMM -> final sum end {max(float(r['end_ms']) for r in fin_all) - cold_end:.3f} ms")
    for o in out: print("    " + o)
p2p = list(csv.DictReader(open(f'{d}/p2p_summary.csv')))
fam = collections.defaultdict(float)
for r in p2p:
    n = r['node']; k = 'MoE partial sums' if re.search(r'/mlp/(to_partials|fa_|ft\d+_|final)', n) else 'o_proj' if n.endswith('o_proj') else 'norm exchanges' if 'layernorm' in n else 'other'
    fam[k] += float(r['MiB'])
print("  P2P MiB: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(fam.items(), key=lambda kv: -kv[1])))
