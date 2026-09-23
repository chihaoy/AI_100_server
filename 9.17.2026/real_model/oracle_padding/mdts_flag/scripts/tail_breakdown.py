#!/usr/bin/env python3
"""Phase and combine-tail breakdown of a detail_analyze output dir. Usage: tail_breakdown.py <analysis dir> <T> <label>"""
import csv, sys, collections, statistics as st
adir, T, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
W = [r for r in csv.DictReader(open(f'{adir}/work.csv'))]
for r in W: r['card'] = int(r['card']); r['core'] = int(r['core']); r['s'] = float(r['start_ms']); r['e'] = float(r['end_ms']); r['d'] = float(r['dur']); r['n'] = r['node'].split('/mlp/')[-1]
HOT, COLD = {'MatMul', 'MatMul_1', 'MatMul_2'}, {'MatMul_3', 'MatMul_4', 'MatMul_5'}
def group(r):
    n, k = r['n'], r['kind']
    if n in HOT or n in COLD:
        if k == 'aiccopytovtcm': return 'weights DMA'
        if k == 'aicconvolutiond32': return 'GEMM (HMX)'
        if k == 'blockdequantize_mxfp6': return 'dequantize (HVX)'
        return 'stage other'
    if n.startswith('tc_g_'): return 'combine: gather'
    if n.startswith(('tc_gm_', 'tc_mk_', 'tc_idx_', 'tc_mask', 'tc_slot')): return 'combine: mask/index'
    if n.startswith(('tc_add_', 'tc_rs_')): return 'combine: add stages'
    if n.startswith('tc_reduce_tile_'): return 'combine: lane reduce'
    if n.startswith('final_tile_') or n == 'Einsum_4': return 'combine: card reduce'
    if n.startswith('reduce_tile_') or n == 'Einsum_3' or n.startswith('reduce_slice_'): return 'old: local tiles'
    if n == 'ConstantOfShape_1': return 'old: zero splat'
    if n.startswith('CtxScatter3D') and 'Int' not in n: return 'old: scatter'
    if n == 'CtxGather3D_5' or n == 'Add_2' or n == 'Where_7': return 'old: read-modify-write'
    if n.startswith('CtxGather3D'): return 'token gathers (x)'
    if 'aicopstats' in r['node'] or 'seminc' in n or k in ('aicendcyclestats',): return 'bookkeeping'
    return 'routing/other'
for r in W: r['g'] = group(r)
dev_end = max(r['e'] for r in W if r['g'] != 'bookkeeping'); print(f"=== {label}: device (last non-bookkeeping op) {dev_end:.3f} ms; {dev_end/T*1e3:.1f} us/token")
rows = []
for card in range(4):
    C = [r for r in W if r['card'] == card and r['g'] != 'bookkeeping']
    hw = [r for r in C if r['n'] in HOT and r['kind'] in ('aiccopytovtcm', 'aicconvolutiond32')]; cw = [r for r in C if r['n'] in COLD and r['kind'] in ('aiccopytovtcm', 'aicconvolutiond32')]
    h0, h1 = min(r['s'] for r in hw), max(r['e'] for r in hw); c0, c1 = min(r['s'] for r in cw), max(r['e'] for r in cw); end = max(r['e'] for r in C)
    nh = len({r['core'] for r in hw if r['kind'] == 'aicconvolutiond32'}); nc = len({r['core'] for r in cw if r['kind'] == 'aicconvolutiond32'})
    rows.append((card, h0, h1 - h0, nh, c0 - h1, c1 - c0, nc, end - c1, end))
print('   card | prologue | hot stage (experts, us/exp) | gap | cold stage (experts, us/exp) | tail after cold | end')
for card, h0, hd, nh, gap, cd, nc, tail, end in rows:
    print(f'   {card}    | {h0:.2f}     | {hd:.2f} ms ({nh}, {hd/max(nh,1)*1e3:.0f})          | {gap:.2f} | {cd:.2f} ms ({nc}, {cd/max(nc,1)*1e3:.0f})         | {tail:.2f} ms          | {end:.2f}')
c0_tail_start = rows[0][1] + rows[0][2] + rows[0][4] + rows[0][5]
print(f"   tail composition (ops starting after each card's cold stage end): wall window, and per-engine busy time (max core / median core, ms)")
comp = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float))); win = {}
for card, h0, hd, nh, gap, cd, nc, tail, end in rows:
    tstart = h0 + hd + gap + cd
    for r in W:
        if r['card'] != card or r['g'] == 'bookkeeping' or r['s'] < tstart - 1e-6: continue
        comp[r['g']][r['engine']][(card, r['core'])] += r['d'] / 1e3
        w = win.get(r['g'], (1e9, 0)); win[r['g']] = (min(w[0], r['s']), max(w[1], r['e']))
for gname in sorted(comp, key=lambda k: -max(max(v.values()) for v in comp[k].values())):
    parts = []
    for eng, d in sorted(comp[gname].items(), key=lambda kv: -max(kv[1].values())):
        vals = list(d.values()); parts.append(f'{eng} {max(vals):.2f}/{st.median(vals):.2f}')
    print(f"   {gname:24s} [{win[gname][0]:.2f}-{win[gname][1]:.2f}]  " + '; '.join(parts))
# whole-layer group totals (all phases), max-core busy per engine
tot = collections.defaultdict(lambda: collections.defaultdict(float))
for r in W:
    if r['g'] != 'bookkeeping': tot[(r['g'], r['engine'])][(r['card'], r['core'])] += r['d'] / 1e3
print('   whole layer, max-core busy ms by group/engine:', '; '.join(f'{g}/{e} {max(v.values()):.2f}' for (g, e), v in sorted(tot.items(), key=lambda kv: -max(kv[1].values()))[:10]))
