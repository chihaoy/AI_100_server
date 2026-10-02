#!/usr/bin/env python3
"""Per-stage MoE breakdown of one layer from a detail_analyze.py work table (instrumented two-layer profiles, E38/E39).
Stages: the export's stage 0 (MatMul, MatMul_1, MatMul_2) and stage 1 (MatMul_3..5), or tier / block clones tr<t>/; a stage's
weight bytes are the DDR reads of its bank gathers (tr<t>_gather_*, runtime-indexed weights; a stage with
gathers reads no weights through its MatMul nodes) or of its own MatMul nodes (static weights: DDR-to-VTCM reads less the
VTCM-to-DDR spills of the same nodes, which are read back once). Per stage and card: GEMM window (first to last HMX op), HMX
busy summed over cores, MXFP6 dequantize busy, weight MiB read from DDR, activation spills MiB written to DDR. Families: sort (per-card TopK and orders), block metadata (bk_*), native routing chain (nat*),
token-owned combine (to_*), final sum (final_/ft/fa_). Usage: stage_profile.py <analysis sample dir> <label> [layer, default 1]"""
import sys, csv, re, json, collections
d, label = sys.argv[1], sys.argv[2]; L = int(sys.argv[3]) if len(sys.argv) > 3 else 1
P = f'/model/layers.{L}/mlp/'
rows = [r for r in csv.DictReader(open(f'{d}/work.csv')) if r['node'].startswith(P)]
def stage(n):
    t = n[len(P):]
    m = re.match(r'tr(\d+)(?:/|_gather_)', t)
    if m: return f'tr{m.group(1)}'
    if re.match(r'(MatMul|MatMul_1|MatMul_2)(/|$)', t): return 'st0'
    if re.match(r'(MatMul_3|MatMul_4|MatMul_5)(/|$)', t): return 'st1'
    return None
def family(n):
    t = n[len(P):]
    for k, pat in (('sort', r'(dc_topk|dc_cnt|dc_loc|dc_gloc|tr_order|dc_sorted)'), ('block metadata', r'bk_'), ('routing chain', r'(nat|dc_counts|dc_routed|ScatterElements|TopK|Softmax|gate)'),
                   ('combine', r'(to_|dn_|CtxScatter3D(_\d+)?(/|$)|CtxGather3D_2|Einsum_[234]|ConstantOfShape_1|tc_|ta_)'), ('final sum', r'(final_|ft\d|fa_)')):
        if re.match(pat, t): return k
    return 'other'
S = collections.defaultdict(lambda: collections.defaultdict(lambda: dict(g0=1e9, g1=-1e9, hmx=0.0, deq=0.0, gmib=0.0, inmib=0.0, spill=0.0, wmib=0.0)))
Fm = collections.defaultdict(lambda: dict(s=1e9, e=-1e9, busy=0.0)); win = [1e9, -1e9]; p2p = 0; card_end = collections.defaultdict(float); gemm_end = collections.defaultdict(float)
for r in rows:
    s, e, du, card = float(r['start_ms']), float(r['end_ms']), float(r['dur']), int(r['card'])
    win[0] = min(win[0], s); win[1] = max(win[1], e); card_end[card] = max(card_end[card], e)
    if r['engine'] == 'P2P': p2p += int(r['bytes'] or 0)
    st = stage(r['node'])
    if st and r['engine'] == 'HMX': gemm_end[card] = max(gemm_end[card], e)
    if st:
        x = S[st][card]
        if r['engine'] == 'HMX': x['g0'] = min(x['g0'], s); x['g1'] = max(x['g1'], e); x['hmx'] += du
        if r['kind'] == 'blockdequantize_mxfp6': x['deq'] += du
        if 'DDR' in r['memory'] and 'DMA' in r['engine']:
            b = int(r['bytes'] or 0) / 2**20
            if '_gather_' in r['node']: x['gmib'] += b
            elif re.search(r'MatMul(_\d)?$', r['node']): x['spill' if 'fromvtcm' in r['kind'] else 'inmib'] += b
    else:
        f = Fm[family(r['node'])]; f['s'] = min(f['s'], s); f['e'] = max(f['e'], e); f['busy'] += du
if 'tr0' in S and 'st0' in S and all(v['g1'] < 0 for v in S['tr0'].values()):   # tier 0 kept the export's stage-0 nodes; its gathers are tr0_gather_*
    for c, v in S.pop('tr0').items(): S['st0'][c]['gmib'] += v['gmib']
for st in S:
    for v in S[st].values(): v['wmib'] = v['gmib'] if v['gmib'] > 0 else v['inmib'] - v['spill']
out = dict(label=label, layer=L, moe_window_ms=[win[0], win[1]], stages={}, families={})
tail = {c: card_end[c] - gemm_end[c] for c in gemm_end}; out['tail_after_last_gemm_ms'] = tail; out['p2p_mib'] = p2p / 2**20
print(f'== {label}, layer {L}: MoE ops {win[0]:.3f}-{win[1]:.3f} ms ({win[1] - win[0]:.3f} ms); after the last expert GEMM per card {[round(tail[c], 3) for c in sorted(tail)]} ms; P2P within the MoE {p2p / 2**20:.1f} MiB')
order = sorted(S, key=lambda k: min(v['g0'] for v in S[k].values()))
for st in order:
    cs = S[st]; g = [(cs[c]['g1'] - cs[c]['g0']) for c in sorted(cs) if cs[c]['g1'] > 0]
    out['stages'][st] = {c: v for c, v in cs.items()}
    wins = ', '.join('%.3f-%.3f' % (cs[c]['g0'], cs[c]['g1']) for c in sorted(cs) if cs[c]['g1'] > 0)
    print(f'   {st:4s}: GEMM window per card {wins} (length {min(g) if g else 0:.3f}-{max(g) if g else 0:.3f} ms); '
          f'HMX busy per card {[round(cs[c]["hmx"] / 1e3, 2) for c in sorted(cs)]} ms; dequantize {[round(cs[c]["deq"] / 1e3, 2) for c in sorted(cs)]} ms; weights from DDR {[round(cs[c]["wmib"], 1) for c in sorted(cs)]} MiB; spills to DDR {[round(cs[c]["spill"], 1) for c in sorted(cs)]} MiB')
for k, f in sorted(Fm.items(), key=lambda kv: kv[1]['s']):
    out['families'][k] = f
    print(f'   {k:15s}: {f["s"]:.3f}-{f["e"]:.3f} ms, busy {f["busy"] / 1e3:.2f} ms (all cores)')
json.dump(out, open(f'{d}/stage_profile_L{L}.json', 'w'), indent=1)
