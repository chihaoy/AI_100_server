#!/usr/bin/env python3
"""Experiment 2 (E39) table: fixed-size block scheduling against RankTier on two-layer cuts, per chunk length, from the
run_e39.sh session logs. Each program is the mean of its two passes; ratios use RankTier (e38E) of the same session.
Block budgets per card for layers 0-1 from full_model/e39/budget_T<T>_B<B>_<kind>.npy. Usage: e39_summary.py <mdts_flag dir>"""
import sys, re, glob, json, os, numpy as np
R = sys.argv[1]; O = f'{R}/full_model/e39'
TIERS = {128: [(8, 128), (8, 48), (16, 16)], 256: [(8, 256), (8, 64), (16, 32)], 512: [(8, 512), (8, 128), (16, 64)]}
KIND = {'': 'dropless', 'cal': 'calibrated', 'orc': 'oracle'}
res = {}
for T in (128, 256, 512):
    rows = []
    for f in sorted(glob.glob(f'{O}/e39_T{T}_P*.txt')):
        t = open(f).read()
        if 'session means' not in t: continue
        S = os.path.basename(f)[len(f'e39_T{T}_'):-4]
        means = {k: float(v) for k, v in re.findall(r'(\S+) ([\d.]+) ms', t.split('session means (both passes): ')[1].split('\n')[0])}
        ident = dict(re.findall(r'logits (\S+) vs \S+: bit-identical (\w+)', t))
        anchor = next((v for k, v in means.items() if k.startswith('e38E_')), None)
        for k, v in means.items():
            lab, tile = k.rsplit('_', 1); m = re.fullmatch(r'blk(\d+)(cal|orc)?(dg|iv)?', lab)
            if m:
                B, kind, fix = int(m.group(1)), m.group(2) or '', m.group(3)
                bud = np.load(f'{O}/budget_T{T}_B{B}_{KIND[kind]}.npy')[:2]
                desc, nrows = f'blocks of {B}, {KIND[kind]}{ {"dg": ", direct gather", "iv": ", invalid-index padding", None: ""}[fix] }, {"/".join(map(str, sorted(set(bud.tolist()))))} per card', int(bud.max()) * B
            else:
                desc = {'e38E': 'RankTier', 'e38C': 'rank sort, capacity T', 'e38A': 'static naive T/T', 'e38Edg': 'RankTier, direct gather'}.get(lab, lab)
                nrows = sum(l * c for l, c in TIERS[T]) if lab.startswith('e38E') else 32 * T
            rows.append(dict(session=S, program=k, desc=desc, tile=tile, rows=nrows, ms=v, vs_ranktier=v / anchor if anchor else None, bit_identical=ident.get(k, 'ref')))
    if not rows: continue
    res[T] = rows
    print(f'\n### T={T} (two-layer cuts; RankTier {"+".join(f"{l}x{c}" for l, c in TIERS[T])} = {sum(l * c for l, c in TIERS[T])} rows per card)\n')
    print('| Session | Program | Tile | Rows per card | Two layers | vs RankTier | Logits = first program |'); print('|---|---|---|---:|---:|---:|---|')
    for r in rows: print(f"| {r['session']} | {r['desc']} | {r['tile']} | {r['rows']} | {r['ms']:.2f} ms | {r['vs_ranktier']:.2f}x | {r['bit_identical']} |")
json.dump({str(k): v for k, v in res.items()}, open(f'{O}/e39_table.json', 'w'), indent=1)
