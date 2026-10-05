#!/usr/bin/env python3
"""Per-layer breakdown of an instrumented full-model RankTier profile (E44), from a detail_analyze.py work table and the
routing counts of the same prompt (any full-model run of the same program; counts in its lane order: tier 0 = 4 x 8 lanes,
tier 1 = 4 x 8, tier 2 = 4 x 16). For every layer and SoC: attention core (input norm .. o_proj), the o_proj group sum,
post-attention norm and router, the per-SoC sort, each tier in execution order as [first weight fetch .. last GEMM] with the
gap before it, its DRAM weight fetch, busiest-core dequantize and GEMM time, GEMM cores and rows computed, the combine and the
final sum. Prints the whole-model busy time by family, a table per layer (slowest SoC per phase), medians over layers 1-46,
and the per-SoC spread; writes e44_layers.json next to the work table.
Attention-side ops that end after the SoC's router starts (late buffer copies) are left out of the attention markers.
Usage: e44_layers.py <analysis sample dir> <result.json with routing_counts> [tiers, default 8x512,8x128,16x64]"""
import sys, csv, re, json, collections, statistics as st
d, res = sys.argv[1], sys.argv[2]; spec = sys.argv[3] if len(sys.argv) > 3 else '8x512,8x128,16x64'
TIERS = [tuple(int(v) for v in s.split('x')) for s in spec.split(',')]
counts = json.load(open(res))['routing_counts']; summ = json.load(open(f'{d}/summary.json'))
LAY = re.compile(r'^/model/layers\.(\d+)/(input_layernorm|self_attn|post_attention_layernorm|mlp)/(.*)$')

def tier_of(t):                      # tier index of an expert-stage node of the MoE, else None
    m = re.match(r'tr(\d+)(/|_gather_|_rows)', t)
    if m: return int(m.group(1))
    if re.match(r'(MatMul|MatMul_1|MatMul_2|act_fn|Where_3|CtxGather3D|Mul_4|Mul_5)(/|$)', t): return 0
    return None

def family(blk, t, kind, engine):
    if blk == 'input_layernorm' or blk == 'post_attention_layernorm': return 'rmsnorm (+ exchange)'
    if blk == 'self_attn':
        if re.search(r'ReduceSum_o|hp/osum', t): return 'attn: o_proj group sum'
        if re.search(r'o_proj', t): return 'attn: o_proj'
        if re.search(r'q_proj|k_proj|v_proj', t): return 'attn: q/k/v proj'
        if re.search(r'CtxScatter|CtxGather', t): return 'attn: KV cache'
        return 'attn: scores/softmax/pv/rope'
    ti = tier_of(t)
    if ti is not None:
        if kind == 'aicgather' and '_gather_' in t: return 'moe: weight fetch (DRAM)'
        if kind == 'blockdequantize_mxfp6': return 'moe: dequantize'
        if kind == 'aicmulticastvtcm' and re.search(r'MatMul(_\d)?$', t): return 'moe: weight multicast'
        if engine == 'HMX': return f'moe: tier {ti} GEMMs'
        return 'moe: tier activation/gather/mask'
    if t.startswith('gate/'): return 'moe: router'
    if t.startswith(('to_',)): return 'moe: token-owned combine'
    if t.startswith(('final_', 'fa_', 'ft')): return 'moe: final sum'
    if t.startswith(('nat', 'dc_', 'tr_order', 'ScatterElements', 'TopK', 'Softmax', 'ReduceSum', 'Div', 'Cast')): return 'moe: routing chain + sort'
    return 'moe: other'

rows = list(csv.DictReader(open(f'{d}/work.csv')))
ENDS = collections.defaultdict(list)   # attention-side ends; ops the compiler schedules after the router starts (late copies) are dropped below
busy = collections.Counter(); ncore = set()
# per (L, card): phase windows and tier stats
W = collections.defaultdict(lambda: collections.defaultdict(lambda: [1e18, -1e18]))
TS = collections.defaultdict(lambda: dict(start=1e18, end=-1e18, f0=1e18, f1=-1e18, fmib=0.0, q=collections.Counter(),
                                          g=collections.Counter(), rows=collections.Counter()))
for r in rows:
    c = int(r['card'])
    if c < 0: continue
    ncore.add((c, r['core']))
    m = LAY.match(r['node'])
    if not m:
        n = r['node']; f = 'embedding' if n.startswith('/model/embed_tokens') else ('final norm + lm_head' if n.startswith(('/lm_head', '/model/norm', '/Flatten')) else 'other')
        busy[f] += float(r['dur']) / 1e3; continue
    L, blk, t = int(m.group(1)), m.group(2), m.group(3); k = r['kind'].strip(); s, e = float(r['start_ms']), float(r['end_ms'])
    busy[family(blk, t, k, r['engine'])] += float(r['dur']) / 1e3
    w = W[(L, c)]
    def put(key): w[key][0] = min(w[key][0], s); w[key][1] = max(w[key][1], e)
    put('layer')
    if blk == 'input_layernorm': put('attn')
    if blk == 'self_attn' and re.search(r'o_proj', t) and not re.search(r'ReduceSum_o|hp/osum', t): ENDS[(L, c, 'oproj')].append(e)
    if blk == 'self_attn' and re.search(r'ReduceSum_o|hp/osum', t): ENDS[(L, c, 'osum')].append(e)
    if blk == 'mlp' and t.startswith('gate/'): put('router')
    if blk == 'mlp' and t.startswith('dc_topk'): put('sort')
    if blk == 'mlp' and t.startswith(('final_', 'fa_', 'ft')): put('final')
    if blk == 'mlp':
        ti = tier_of(t)
        if ti is not None:
            x = TS[(L, c, ti)]; core = int(r['core'])
            is_fetch = k == 'aicgather' and '_gather_' in t
            if is_fetch: x['f0'] = min(x['f0'], s); x['f1'] = max(x['f1'], e); x['fmib'] += int(r['bytes'] or 0) / 2**20
            if k == 'blockdequantize_mxfp6': x['q'][core] += float(r['dur'])
            if r['engine'] == 'HMX': x['g'][core] += float(r['dur']); x['end'] = max(x['end'], e)
            if is_fetch or k == 'blockdequantize_mxfp6' or r['engine'] == 'HMX': x['start'] = min(x['start'], s)
            if k == 'sigmoid' and t.endswith('Sigmoid'): x['rows'][core] += int(r['bytes'] or 0) / 1536

layers = sorted({L for L, _ in W}); cards = sorted({c for _, c in W})
for (L, c, key), ends in ENDS.items():   # the router cannot start before the head-group sum is done, so later ops are off the critical path
    r0 = W[(L, c)]['router'][0]; keep = [e for e in ends if e <= r0] or ends
    W[(L, c)][key] = [min(keep), max(keep)]
dev = summ.get('device_ms', float('nan')); tot = sum(busy.values())
print(f"=== E44: device {dev:.1f} ms on {len(ncore)} cores; busy {tot:.0f} core-ms = {tot / (len(ncore) * dev):.0%} of core-time; {len(layers)} layers")
for f, v in sorted(busy.items(), key=lambda kv: -kv[1]): print(f"  {f:34s} {v:8.1f} core-ms {v / tot:6.1%}")

def lane_counts(L, c, ti):
    off = sum(4 * l for l, _ in TIERS[:ti]); l = TIERS[ti][0]
    return counts[L][off + c * l: off + (c + 1) * l]
out = []
for L in layers:
    per = {}
    for c in cards:
        w = W[(L, c)]; g = lambda k, i: w[k][i] if w[k][0] < 1e17 else float('nan')
        tiers = []
        for ti in range(len(TIERS)):
            x = TS[(L, c, ti)]
            if x['end'] < 0: continue
            lc = lane_counts(L, c, ti); L_, C_ = TIERS[ti]
            tiers.append(dict(tier=ti, spec=f'{L_}x{C_}', start=x['start'], end=x['end'], span=x['end'] - x['start'],
                              fetch=x['f1'] - x['f0'] if x['f0'] < 1e17 else 0.0, fetch_mib=x['fmib'],
                              deq_busiest=max(x['q'].values()) / 1e3 if x['q'] else 0.0, gemm_busiest=max(x['g'].values()) / 1e3 if x['g'] else 0.0,
                              gemm_cores=len(x['g']), rows_busiest=max(x['rows'].values()) if x['rows'] else 0.0, rows_total=sum(x['rows'].values()),
                              tokens_max=max(lc), tokens=sum(lc), empty=sum(1 for v in lc if v == 0)))
        tiers.sort(key=lambda v: v['start'])
        for i, tv in enumerate(tiers): tv['gap_before'] = tv['start'] - (tiers[i - 1]['end'] if i else g('sort', 1))
        per[c] = dict(attn0=g('attn', 0), oproj1=g('oproj', 1), osum1=g('osum', 1), router1=g('router', 1), sort1=g('sort', 1),
                      tiers=tiers, tier_end=max(t['end'] for t in tiers) if tiers else float('nan'), final1=g('final', 1))
    M = lambda k, f=max: f(per[c][k] for c in cards)
    o = dict(L=L, per_soc=per, A0=M('attn0', min), A1=M('oproj1'), O1=M('osum1'), R1=M('router1'), S1=M('sort1'), TE=M('tier_end'), F1=M('final1'))
    o.update(attn=o['A1'] - o['A0'], osum=o['O1'] - o['A1'], norm_router=o['R1'] - o['O1'], sort=o['S1'] - o['R1'],
             tiers_total=o['TE'] - o['S1'], tail=o['F1'] - o['TE'])
    out.append(o)
for i in range(len(out) - 1):
    out[i]['period'] = out[i + 1]['A0'] - out[i]['A0']; out[i]['to_next'] = out[i + 1]['A0'] - out[i]['F1']

# ---- per-layer table on the layer's critical timeline (phases add up to the period)
PH = [('attn', 'attention core: input norm .. o_proj'), ('osum', 'o_proj sum over the 4 head groups (cross-SoC)'),
      ('norm_router', 'residual, post-attention norm, router'), ('sort', 'router done -> per-SoC sort done'),
      ('tiers_total', 'expert tiers: sort done -> last tier done'), ('tail', 'combine tail: last tier -> final sum done'),
      ('to_next', 'final sum done -> next layer starts')]
def tiermax(o, ti, key): return max((t[key] for c in cards for t in o['per_soc'][c]['tiers'] if t['tier'] == ti), default=float('nan'))
print("\n  per layer, ms (critical timeline: each phase ends when its last SoC finishes; tier spans = max over SoCs, execution order)")
print("   L  period  attn  osum  norm+rtr sort tiers  tail  next | " + "  ".join(f"t{ti}:{L_}x{C_} span/gemm" for ti, (L_, C_) in enumerate(TIERS)) + " | tok1 tok9 tok17")
for o in out:
    tok = [tiermax(o, ti, 'tokens_max') for ti in range(len(TIERS))]
    ts = "  ".join(f"{tiermax(o, ti, 'span'):5.2f}/{tiermax(o, ti, 'gemm_busiest'):4.2f}" + " " * 7 for ti in range(len(TIERS)))
    print(f"  {o['L']:2d}  {o.get('period', float('nan')):6.2f} {o['attn']:5.2f} {o['osum']:5.2f} {o['norm_router']:6.2f}  {o['sort']:5.2f} {o['tiers_total']:5.2f} {o['tail']:5.2f} {o.get('to_next', float('nan')):5.2f} | {ts}| " + " ".join(f"{int(v):4d}" for v in tok))

# ---- medians over layers 1..46
mid = [o for o in out if 1 <= o['L'] <= len(out) - 2] or out
def med(vals): v = [x for x in vals if x == x]; return (st.median(v), min(v), max(v)) if v else (float('nan'),) * 3
print("\n  medians over layers 1-46 (min, max), ms:")
a, b, cc = med([o.get('period', float('nan')) for o in mid]); print(f"    {'layer period':52s} {a:6.3f} ({b:.3f}, {cc:.3f})")
for key, name in PH:
    a, b, cc = med([o.get(key, float('nan')) for o in mid]); print(f"    {name:52s} {a:6.3f} ({b:.3f}, {cc:.3f})")
for ti, (L_, C_) in enumerate(TIERS):
    for key, name in (('span', 'span'), ('gap_before', 'gap before'), ('fetch', 'DRAM weight fetch window'), ('deq_busiest', 'busiest-core dequantize'), ('gemm_busiest', 'busiest-core GEMMs')):
        a, b, cc = med([tiermax(o, ti, key) for o in mid]); print(f"    tier {ti} {L_}x{C_:<4d} {name:40s} {a:6.3f} ({b:.3f}, {cc:.3f})")
# ---- tier rule check over all layers: busiest core rows vs tile rule, second-core activation
over = collections.Counter(); act2 = collections.Counter()
for o in out:
    for c in cards:
        for t in o['per_soc'][c]['tiers']:
            L_, C_ = TIERS[t['tier']]; per_core = C_ * L_ // 16 if L_ < 16 else C_
            if t['tokens_max'] > per_core: over[t['tier']] += 1; act2[t['tier']] += t['gemm_cores'] > (L_ - t['empty'])
print("\n  (layer, SoC) pairs whose busiest lane holds more tokens than one core's rows: " + ', '.join(f"tier {k}: {v} ({act2[k]} of them with extra GEMM cores)" for k, v in sorted(over.items())) if over else "\n  no lane exceeded one core's rows")
# ---- per-SoC: which SoC finishes each layer last
last = collections.Counter(max(cards, key=lambda c: o['per_soc'][c]['final1']) for o in out)
print("  SoC that finishes the layer last: " + ', '.join(f"SoC {c} in {n} layers" for c, n in sorted(last.items())))
json.dump(dict(device_ms=dev, busy_core_ms=busy, layers=out, tiers=TIERS), open(f'{d}/e44_layers.json', 'w'), indent=1, default=float)
