#!/usr/bin/env python3
import json, statistics as st, os
from pathlib import Path
S = Path('/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad')
B = Path('/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding')
def pooled(rows, key, variant, field='samples'):
    xs = [s for r in rows if r.get('label', r.get('capacity')) == key and r['variant'] == variant for s in r[field]]
    return st.median(xs) if xs else float('nan')
def prof(d):
    g = lambda k: (d.get(k) if d.get(k) is not None else float('nan'))
    cc = d.get('cold_cores_per_card'); cc = ' '.join(str(v) for v in cc.values()) if isinstance(cc, dict) else ''
    return f"dev {g('device_ms'):6.3f} | hot {g('hot_span_ms'):.3f} gap {g('between_groups_ms'):.3f} cold {g('cold_span_ms'):.3f} unpack {g('last_unpack_ms'):6.3f} combine {g('combine_ms'):.3f} | cold cores {d.get('cold_cores', 0):2d} [{cc}] | red max-core {g('local_reduction_max_core_ms'):.3f}"
print('==================== E1: cold-capacity sweep (layer 2 replay, hot C128, real routing) ====================')
if (S/'e1/timing.json').exists():
    rows = json.load(open(S/'e1/timing.json'))
    profiles = json.load(open(S/'e1/profiles/profiles.json')) if (S/'e1/profiles/profiles.json').exists() else {}
    print(f"{'cold cap':>8s} | {'host median ms: base':>21s} {'flag':>8s} {'flag/base':>9s} | flag profile (representative capture)")
    for cap in (128, 64, 32, 16, 8, 4, 2):
        b, f = pooled(rows, cap, 'base'), pooled(rows, cap, 'flag'); p = profiles.get(f'c{cap}_flag', {}).get('chosen')
        print(f"{cap:>8d} | {b:21.3f} {f:8.3f} {f/b:9.3f} | {prof(p) if p else '-'}")
    print('  baseline profiles (from cold_capacity_control analysis, earlier session):')
    for cap in (128, 64, 32, 16, 8, 4, 2):
        try: d = json.load(open(B/f'cold_capacity_control/c{cap}/analysis/summary.json')); print(f"{cap:>8d} | {prof(d)}")
        except Exception as e: print(cap, 'n/a', e)
print('\n==================== E2: hot capacity with cold 2 ====================')
if (S/'e2/timing.json').exists():
    rows = json.load(open(S/'e2/timing.json')); profiles = json.load(open(S/'e1/profiles/profiles.json')) if (S/'e1/profiles/profiles.json').exists() else {}
    for lab in ('hot128_cold2', 'hot96_cold2', 'hot92_cold2'):
        b, f = pooled(rows, lab, 'base'), pooled(rows, lab, 'flag')
        pf = profiles.get({'hot128_cold2': 'c2_flag'}.get(lab, lab + '_flag'), {}).get('chosen'); pb = profiles.get(lab + '_base', {}).get('chosen')
        print(f"{lab:>14s} | base {b:7.3f} flag {f:7.3f} ratio {f/b:.3f}\n{'':>16s} flag: {prof(pf) if pf else '-'}\n{'':>16s} base: {prof(pb) if pb else '-'}")
print('\n==================== E3: all-active placement/capacity plans (synthetic routing, 128 experts active) ====================')
if (S/'e3/results.json').exists():
    res = json.load(open(S/'e3/results.json'))
    W = ['uniform', 'mild_a', 'skew_a', 'skew_b', 'skew_c']
    print(f"{'plan':>18s} " + ' '.join(f'{w:>19s}' for w in W) + '   (base -> flag ms)')
    for plan, e in res.items():
        cells = []
        for w in W:
            if w in e['base']: cells.append(f"{st.median(e['base'][w]):6.3f}->{st.median(e['flag'][w]):6.3f}{'' if e['bit_exact'].get(w) else '*'}")
            else: cells.append(' ' * 14)
        print(f"{plan:>18s} " + ' '.join(f'{c:>19s}' for c in cells))
    sel = {'one static plan': {w: 'aggregate_c32_32' for w in W},
           'fixed layout + capacity oracle': {'uniform': 'aggregate_c8_8', 'mild_a': 'aggregate_c32_4', 'skew_a': 'aggregate_c32_4', 'skew_b': 'aggregate_c32_32', 'skew_c': 'aggregate_c32_32'},
           'fixed shape + placement oracle': {'uniform': 'sorted_c_c32_32', 'mild_a': 'sorted_c_c32_32', 'skew_a': 'sorted_a_c32_32', 'skew_b': 'sorted_b_c32_32', 'skew_c': 'aggregate_c32_32'},
           'joint oracle (frozen)': {'uniform': 'stripe_c8_8', 'mild_a': 'sorted_a_c32_4', 'skew_a': 'sorted_a_c32_2', 'skew_b': 'sorted_b_c32_32', 'skew_c': 'aggregate_c32_32'},
           'no regrouping, 32/32': {w: 'identity_c32_32' for w in W}}
    print('\n  policy means over the five workloads (frozen selections from RESULTS.md):')
    for name, m in sel.items():
        for var in ('base', 'flag'):
            vals = [st.median(res[m[w]][var][w]) for w in W if m[w] in res and w in res[m[w]][var]]
            if len(vals) == len(W): print(f"    {name:>32s} {var:4s}: mean {st.mean(vals):.4f} ms  [{' '.join(f'{v:.3f}' for v in vals)}]")
            else: print(f"    {name:>32s} {var:4s}: incomplete ({len(vals)}/5)")
    for var in ('base', 'flag'):
        best = [min((st.median(e[var][w]), p) for p, e in res.items() if w in e[var]) for w in W]
        print(f"    {'best tested plan per workload':>32s} {var:4s}: mean {st.mean(v for v, _ in best):.4f} ms  [{' '.join(f'{v:.3f}({p})' for v, p in best)}]")
print('\n==================== E4: expert regrouping with real routing (native order vs sorted hot/cold) ====================')
if (S/'e4/timing.json').exists():
    rows = json.load(open(S/'e4/timing.json'))
    for lab in ('native_128_128', 'native_92_34'):
        print(f"{lab:>16s} | base {pooled(rows, lab, 'base'):7.3f} flag {pooled(rows, lab, 'flag'):7.3f}")
    if (S/'e4/profiles/profiles.json').exists():
        for k, v in json.load(open(S/'e4/profiles/profiles.json')).items():
            if v.get('chosen'): print(f"{k:>22s} | {prof(v['chosen'])}")
