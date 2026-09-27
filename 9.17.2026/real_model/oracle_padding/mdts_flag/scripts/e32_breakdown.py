#!/usr/bin/env python3
"""End-to-end breakdown of a full-model profile (detail_analyze output): (1) core-busy time by op family, (2) per-layer
phase timestamps on the critical card and their medians over the layers, (3) cross-card bytes by family, (4) per-SoC hot-stage spread.
Usage: e32_breakdown.py <analysis dir> <label>"""
import sys, csv, re, json, collections, statistics as st
d, label = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(f'{d}/work.csv'))); summ = json.load(open(f'{d}/summary.json'))
LAY = re.compile(r'^/model/layers\.(\d+)/')
def family(n):
    if n.startswith('/model/embed_tokens'): return 'embedding'
    if n.startswith(('/lm_head', '/Flatten', '/model/norm')): return 'final norm + lm_head'
    if '/input_layernorm/' in n or '/post_attention_layernorm/' in n: return 'rmsnorm (+ exchange)'
    if '/self_attn/' in n:
        if re.search(r'q_proj|k_proj|v_proj', n): return 'attn: q/k/v proj'
        if re.search(r'o_proj|ReduceSum', n): return 'attn: o_proj + group sum'
        if re.search(r'CtxScatter|CtxGather|Where_1|kv', n): return 'attn: KV cache'
        if re.search(r'MatMul_qk|MatMul_pv|Softmax|scores|Add_mask|Div|Mul_scale|Where', n): return 'attn: scores/softmax/pv'
        return 'attn: norms/rope/other'
    if '/mlp/' in n:
        t = n.split('/mlp/', 1)[1]
        if re.match(r'MatMul(_[12])?$', t): return 'moe: hot stage'
        if re.match(r'MatMul_[345]$', t): return 'moe: cold stage'
        if t.startswith('dc_gather'): return 'moe: weight gather'
        if t.startswith(('nat/', 'dc_', 'nat_')): return 'moe: routing chain + sort'
        if t.startswith(('to_', 'final_', 'fa_', 'ft')): return 'moe: combine'
        if re.match(r'(gate|Softmax|TopK|ReduceSum|Div|ScatterElements|Cast)', t): return 'moe: router'
        if re.match(r'(Mul|Sigmoid|Silu|act_fn)', t): return 'moe: activation'
        return 'moe: token gather/index'
    return 'other'
busy = collections.defaultdict(float); unk = collections.Counter()
for r in rows:
    f = family(r['node']); busy[f] += float(r['dur']) / 1e3
    if f in ('other', 'moe: token gather/index'): unk[r['node'].split('/mlp/')[-1][:40]] += float(r['dur']) / 1e3
dev = summ['device_ms']; ncore = len({(r['card'], r['core']) for r in rows if int(r['card']) >= 0})
print(f"=== {label}: device {dev:.1f} ms, {ncore} cores; total busy {sum(busy.values()):.0f} core-ms = {sum(busy.values()) / (ncore * dev):.0%} of core-time")
for f, v in sorted(busy.items(), key=lambda kv: -kv[1]): print(f"  {f:28s} {v:9.1f} core-ms  {v / sum(busy.values()):6.1%}")
print("  largest items in 'moe: token gather/index' and 'other':", ", ".join(f"{k} {v:.0f}" for k, v in unk.most_common(8)))
# ---- per-layer phases from explicit markers (as layer_timeline.py), ms; max over cards unless noted
EXP = re.compile(r'^MatMul(_[1-5])?$')
W = collections.defaultdict(lambda: collections.defaultdict(lambda: [1e18, -1e18]))
byL = collections.defaultdict(list)
for r in rows:
    m = re.match(r'/model/layers\.(\d+)/(self_attn|input_layernorm|mlp|post_attention_layernorm)/(.*)', r['node'])
    if not m or int(r['card']) < 0: continue
    L, blk, rest = int(m.group(1)), m.group(2), m.group(3); s_, e_ = float(r['start_ms']), float(r['end_ms']); byL[L].append(r)
    keys = []
    if blk == 'input_layernorm': keys.append('attn')
    if blk == 'self_attn' and not re.search(r'ReduceSum_o|hp/osum', rest): keys.append('attn')
    if blk == 'self_attn' and re.search(r'ReduceSum_o|hp/osum', rest): keys.append('osum')
    if blk == 'mlp' and rest.startswith('gate/'): keys.append('router')
    if blk == 'mlp' and rest.startswith('dc_topk'): keys.append('sort')
    if blk == 'mlp' and rest.startswith('dc_rows_s0'): keys.append('table')
    if blk == 'mlp' and ((EXP.match(rest) and r['kind'] in ('aiccopytovtcm', 'aicgather')) or rest.startswith('dc_gather')): keys.append('weights')
    if blk == 'mlp' and EXP.match(rest) and r['kind'] == 'aicconvolutiond32': keys.append('hot' if re.match(r'MatMul(_[12])?$', rest) else 'cold')
    if blk == 'mlp' and rest.startswith(('final_', 'fa_', 'ft')): keys.append('final')
    for k in keys: w = W[L][k]; w[0] = min(w[0], s_); w[1] = max(w[1], e_)
layers = sorted(W); ph = []
for L in layers:
    w = W[L]; g = lambda k, i: w[k][i] if w[k][0] < 1e17 else float('nan')
    ph.append(dict(L=L, start=g('attn', 0), attn=g('attn', 1) - g('attn', 0), osum=g('osum', 1) - g('attn', 1),
                   to_router=g('router', 1) - g('osum', 1), sort=g('sort', 1) - g('router', 1), table=g('table', 1) - g('router', 1),
                   to_weights=g('weights', 0) - g('router', 1), to_gemm=g('hot', 0) - g('router', 1),
                   hot=g('hot', 1) - g('hot', 0), hot_to_cold=g('cold', 0) - g('hot', 1), cold=g('cold', 1) - g('cold', 0),
                   combine=g('final', 1) - g('cold', 1), end=g('final', 1)))
for i in range(len(ph) - 1):
    ph[i]['period'] = ph[i + 1]['start'] - ph[i]['start']; ph[i]['to_next'] = ph[i + 1]['start'] - ph[i]['end']
keys = ['period', 'attn', 'osum', 'to_router', 'sort', 'to_weights', 'to_gemm', 'hot', 'hot_to_cold', 'cold', 'combine', 'to_next']
names = dict(period='layer period (input norm start to next)', attn='attention core: input norm .. o_proj',
             osum='o_proj group sum (ReduceSum_o) after the core', to_router='group sum done -> router done (residual, norm, gate)',
             sort='router done -> per-card sort done', table='router done -> hot token table done',
             to_weights='router done -> first expert weight stream', to_gemm='router done -> first hot GEMM',
             hot='hot stage: first .. last hot GEMM (slowest SoC)', hot_to_cold='hot end -> first cold GEMM (index build)',
             cold='cold stage', combine='cold end -> final sum done', to_next='final sum done -> next input norm start')
mid = [p for p in ph if 1 <= p['L'] <= 46] if len(ph) > 3 else ph
print("  per-layer sequence, median over layers 1-46 (min, max), ms; negative = overlap:")
for k in keys:
    v = [p[k] for p in mid if k in p and p[k] == p[k]]
    if v: print(f"    {names[k]:58s} {st.median(v):7.3f}  ({min(v):.3f}, {max(v):.3f})")
print(f"  layer 0 starts at {ph[0]['start']:.3f} ms, last final sum ends at {ph[-1]['end']:.3f} ms, device ends at {dev:.3f} ms")
# ---- per-SoC hot stage over all layers
def span(sel):
    s_ = [float(r['start_ms']) for r in sel]; e_ = [float(r['end_ms']) for r in sel]
    return (min(s_), max(e_)) if s_ else (float('nan'), float('nan'))
hs = collections.defaultdict(list)
for L in layers:
    for c in range(4):
        sel = [r for r in byL[L] if int(r['card']) == c and family(r['node']) == 'moe: hot stage' and r['kind'] == 'aicconvolutiond32']
        if sel: a, b = span(sel); hs[c].append(b - a)
print("  hot stage per SoC, median over layers (ms): " + ", ".join(f"SoC {c} {st.median(v):.3f}" for c, v in sorted(hs.items())) +
      f"; slowest SoC per layer: {dict(collections.Counter(max(range(4), key=lambda c: hs[c][i]) for i in range(len(layers))))}")
# ---- cross-card bytes
p2p = list(csv.DictReader(open(f'{d}/p2p_summary.csv'))); pb = collections.defaultdict(float)
for r in p2p: pb[family(r['node'])] += float(r['MiB'])
print(f"  cross-card: {sum(pb.values()):.0f} MiB total; " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(pb.items(), key=lambda kv: -kv[1])[:6]))
json.dump(dict(label=label, device_ms=dev, busy_core_ms=busy, phases=ph, hot_per_soc=hs, p2p_MiB=pb), open(f'{d}/e32_breakdown.json', 'w'), indent=1, default=float)
