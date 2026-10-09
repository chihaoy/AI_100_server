#!/usr/bin/env python3
"""Per-phase breakdown of the MoE combine in one layer of instrumented two-layer profiles (detail_analyze.py work tables), to
compare combine formulations: index build, gather, mask multiply, stage add, the reduction to per-card partials, the final
cross-card sum. Per phase: operations, busy time summed over cores, bytes, cross-core multicast bytes inside the phase, the
wall window on each card, and how many cores worked. Also the exposed tail (last expert GEMM to the end of the final sum, per
card) and, for per-token-tile reductions, whether the tiles overlap or run one after another.
Usage: combine_breakdown.py <label=work.csv> ... [--layer 1]"""
import sys, csv, re, collections
args = [a for a in sys.argv[1:] if not a.startswith('--')]; L = 1
if '--layer' in sys.argv: L = int(sys.argv[sys.argv.index('--layer') + 1])
P = f'/model/layers.{L}/mlp/'
PHASES = [('index build', r'(to_(idx|mf|m\d|on|in|card|lane|local|row|slot|stage|pos|topk|vals|tab|offs|lanes|caps|tier|ge|order|routes|shape|data|mask)|'
           r'tc_(slotsafe|mask|maskf|stage1|idx_|mk_|data_|slot_)|tk_(idx|mf|row|lanerow|data|idxr|mask))'),
          ('gather', r'(to_g_|tk_g_|tc_g_)'), ('mask multiply', r'(to_gm|tk_gm|tc_gm)'), ('stage add', r'(to_sum\b|to_sum\d|tk_sum\b|tc_add)'),
          ('reduce to partials', r'(to_sum4|to_partials|tk_sum3|tk_reduce|tc_rs|tc_reduce)'), ('final sum', r'(final_|ft\d|fa_)')]
def phase(t):
    for name, pat in PHASES:
        if re.match(pat, t): return name
    return None
for spec in args:
    label, path = spec.split('=', 1)
    rows = list(csv.DictReader(open(path)))
    gemm_end = collections.defaultdict(float)
    for r in rows:
        if r['node'].startswith(P) and int(r['card']) >= 0 and r['engine'] == 'HMX' and re.search(r'MatMul(_\d)?(/|$)', r['node'].replace(P, '')):
            gemm_end[int(r['card'])] = max(gemm_end[int(r['card'])], float(r['end_ms']))
    agg = collections.defaultdict(lambda: dict(n=0, busy=0.0, bytes=0, mc=0, win={}, cores=set()))
    tiles = collections.defaultdict(lambda: [1e9, 0.0])
    fin_end = collections.defaultdict(float)
    for r in rows:
        if not r['node'].startswith(P) or int(r['card']) < 0: continue   # card -1: the host queue
        t = r['node'].replace(P, ''); ph = phase(t)
        if ph is None: continue
        a = agg[ph]; c = int(r['card']); s, e = float(r['start_ms']), float(r['end_ms'])
        a['n'] += 1; a['busy'] += float(r['dur']) / 1000; a['bytes'] += int(float(r['bytes'] or 0))
        if 'multicast' in r['kind']: a['mc'] += int(float(r['bytes'] or 0))
        w = a['win'].setdefault(c, [1e9, 0.0]); w[0] = min(w[0], s); w[1] = max(w[1], e); a['cores'].add((c, int(r['core'])))
        m = re.match(r'(tc_reduce_tile_|tc_reduce_c\d_t|final_tile_)(\d+)', t)
        if m and ph == 'reduce to partials': tw = tiles[(c, int(m.group(2)))]; tw[0] = min(tw[0], s); tw[1] = max(tw[1], e)
        if ph == 'final sum': fin_end[c] = max(fin_end[c], e)
    print(f'== {label}: layer {L}')
    print(f'   {"phase":20s} {"ops":>6s} {"busy core-ms":>13s} {"MiB":>8s} {"multicast MiB":>14s} {"cores":>6s}  window per card (ms)')
    for name, _ in PHASES:
        if name not in agg: continue
        a = agg[name]; wins = ', '.join(f'{a["win"][c][1] - a["win"][c][0]:.2f}' for c in sorted(a['win']))
        print(f'   {name:20s} {a["n"]:6d} {a["busy"]:13.2f} {a["bytes"] / 2**20:8.1f} {a["mc"] / 2**20:14.1f} {len(a["cores"]):6d}  {wins}')
    tail = [fin_end[c] - gemm_end[c] for c in sorted(fin_end)]
    print('   exposed tail, last expert GEMM -> final sum done, per card: ' + ', '.join(f'{x:.2f}' for x in tail) + ' ms')
    if tiles:
        for c in sorted({k[0] for k in tiles}):
            ts = sorted((v for k, v in tiles.items() if k[0] == c))
            overlap = sum(max(0.0, ts[i][1] - ts[i + 1][0]) for i in range(len(ts) - 1))
            span = ts[-1][1] - ts[0][0]; work = sum(e - s for s, e in ts)
            print(f'   card {c}: {len(ts)} reduction tiles, span {span:.2f} ms, sum of tile windows {work:.2f} ms, overlap between consecutive tiles {overlap:.2f} ms')
    try:
        cp = list(csv.DictReader(open(path.replace('work.csv', 'critical_path.csv'))))
        tail_ph = collections.defaultdict(float); start = max(gemm_end.values()) if gemm_end else 0
        for r in cp:
            if not r['node'].startswith(P) or float(r['end_ms']) <= start: continue
            ph = phase(r['node'].replace(P, '')) or 'other'
            tail_ph[ph] += (float(r['end_ms']) - max(float(r['start_ms']), start))
        if tail_ph: print('   critical path after the last expert GEMM, by phase (ms): ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(tail_ph.items(), key=lambda x: -x[1])))
    except FileNotFoundError: pass
