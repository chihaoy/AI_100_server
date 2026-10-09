#!/usr/bin/env python3
"""Summarize the TensorRT/H100 results (plan sections 5 and 10): per chunk length and launch mode, the latency of every arm
with the range of its batch medians, the capacity speedups S_dense = G00/G10 and S_token = G01/G11, the aggregation speedups
R_uniform = G00/G01 and R_tiered = G10/G11, the interaction S_token/S_dense, the cost of ranking and weight selection
(G01/S01, G00/S00) and the native reference. All ratios compare arms timed in the same session (same run_trt.py call).
Usage: summarize.py <results dir> [layer=L20]"""
import sys, os, csv, json, glob
R = sys.argv[1]; L = sys.argv[2] if len(sys.argv) > 2 else 'L20'
for d in sorted((p for p in glob.glob(f'{R}/{L}_T*') if p.rsplit('_T', 1)[1].isdigit()), key=lambda p: int(p.rsplit('_T', 1)[1])):
    T = int(d.rsplit('_T', 1)[1]); f = f'{d}/latency.csv'
    if not os.path.exists(f): continue
    rows = list(csv.DictReader(open(f)))
    nat = {}
    if os.path.exists(f'{d}/native_latency.csv'):
        for r in csv.DictReader(open(f'{d}/native_latency.csv')): nat[r['launch_mode']] = r
    for m in sorted({r['launch_mode'] for r in rows}):
        lat = {r['variant']: float(r['latency_us']) for r in rows if r['launch_mode'] == m}
        rng = {r['variant']: (float(r['batch_median_min_us']), float(r['batch_median_max_us'])) for r in rows if r['launch_mode'] == m}
        print(f'\n## T={T}, {m}')
        print('| arm | latency (us) | batch medians |'); print('|---|---:|---|')
        for v in ('S00', 'S01', 'G00', 'G01', 'U00', 'U01', 'G10', 'G11'):
            if v in lat: print(f'| {v} | {lat[v]:.1f} | {rng[v][0]:.1f}-{rng[v][1]:.1f} |')
        if m in nat: print(f"| native ({nat[m]['backend']}) | {float(nat[m]['latency_us']):.1f} | {float(nat[m]['batch_median_min_us']):.1f}-{float(nat[m]['batch_median_max_us']):.1f} |")
        g = lambda a, b: lat[a] / lat[b] if a in lat and b in lat else float('nan')
        sd, st = g('G00', 'G10'), g('G01', 'G11')
        print(f'S_dense {sd:.3f}, S_token {st:.3f}, interaction S_token/S_dense {st / sd:.3f}; R_uniform {g("G00", "G01"):.3f}, '
              f'R_tiered {g("G10", "G11"):.3f}; ranking+gathered weights G01/S01 {g("G01", "S01"):.3f}, G00/S00 {g("G00", "S00"):.3f}'
              + (f'; G11 vs native {lat["G11"] / float(nat[m]["latency_us"]):.3f}' if m in nat and 'G11' in lat else ''))
        if 'U01' in lat:
            print(f'stage organization G00/U00 {g("G00", "U00"):.3f}, G01/U01 {g("G01", "U01"):.3f}; capacities within the stages '
                  f'U00/G10 {g("U00", "G10"):.3f}, U01/G11 {g("U01", "G11"):.3f}')
