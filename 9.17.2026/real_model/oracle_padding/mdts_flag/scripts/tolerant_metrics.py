"""Same phase metrics as moe_qwen3_cold_capacity.analyze_trace, without rejecting overlapping phases; adds per-card stage spans."""
import sys, re
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools')
from moe_qwen3_profile import summarize_trace
STEM = '/model/layers.2/mlp/'
NAMES = {'hot': {STEM + n for n in ['MatMul', 'MatMul_1', 'MatMul_2']}, 'cold': {STEM + n for n in ['MatMul_3', 'MatMul_4', 'MatMul_5']}}
def analyze(path):
    rows, execution_ms = summarize_trace(path)
    result = dict(device_ms=execution_ms); bounds = {}
    for stage in NAMES:
        sel = [r for r in rows if r['node'] in NAMES[stage] and r['engine'] == 'HMX' and r['kind'] == 'aicconvolutiond32']
        if not sel: result.update({f'{stage}_span_ms': None}); continue
        lo, hi = min(r['first_us'] for r in sel), max(r['last_us'] for r in sel); bounds[stage] = (lo, hi)
        busy, counts = defaultdict(float), defaultdict(int)
        for r in sel: busy[(r['card'], r['core'])] += r['busy_us']; counts[(r['card'], r['core'])] += r['events']
        percard = {c: (min(r['first_us'] for r in sel if r['card'] == c), max(r['last_us'] for r in sel if r['card'] == c)) for c in sorted({r['card'] for r in sel})}
        result.update({f'{stage}_span_ms': (hi - lo) / 1000, f'{stage}_hmx_busy_max_core_ms': max(busy.values()) / 1000,
                       f'{stage}_hmx_busy_median_core_ms': float(np.median(list(busy.values()))) / 1000, f'{stage}_hmx_events_total': sum(counts.values()),
                       f'{stage}_cores': len(busy), f'{stage}_cores_per_card': {c: sum(1 for k in busy if k[0] == c) for c in range(4)},
                       f'{stage}_card_spans_ms': {c: [round((a - min(b[0] for b in percard.values())) / 1000, 3), round((b_ - min(b[0] for b in percard.values())) / 1000, 3)] for c, (a, b_) in percard.items()}})
    red = [r for r in rows if r['node'] == STEM + 'Einsum_3' and r['engine'] == 'HVX' and r['kind'] == 'aicbatchedreduceadd']
    if not red: red = [r for r in rows if 'reduce_tile' in r['node'] and r['engine'] == 'HVX' and r['kind'] == 'aicbatchedreduceadd']
    comb = [r for r in rows if r['node'] == STEM + 'Einsum_4' and r['engine'] in ['HMX', 'HVX'] and r['kind'] != 'aicendcyclestats']
    if 'hot' in bounds and 'cold' in bounds:
        result['between_groups_ms'] = (bounds['cold'][0] - bounds['hot'][1]) / 1000
    if red and comb and 'cold' in bounds:
        result['last_unpack_ms'] = (min(r['first_us'] for r in red) - bounds['cold'][1]) / 1000
        result['combine_ms'] = (max(r['last_us'] for r in comb) - min(r['first_us'] for r in red)) / 1000
        result['reduction_max_core_ms'] = max(defaultdict(float, {(r['card'], r['core']): 0}).__class__(float, {}).get(0, 0) for r in red[:1]) if False else None
        busy_red = defaultdict(float)
        for r in red: busy_red[(r['card'], r['core'])] += r['busy_us']
        result['local_reduction_max_core_ms'] = max(busy_red.values()) / 1000
    # first HMX (router-free replay start) offsets for reference
    if 'hot' in bounds: result['hot_start_ms_from_first_event'] = None
    return result
