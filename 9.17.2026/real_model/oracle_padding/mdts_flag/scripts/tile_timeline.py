#!/usr/bin/env python3
"""Per-tile chain timing of the token-centric combine: gather -> lane reduce -> P2P -> card reduce. Usage: tile_timeline.py <analysis dir> [tiles]"""
import csv, sys, collections, re
adir = sys.argv[1]; show = int(sys.argv[2]) if len(sys.argv) > 2 else 6
W = [r for r in csv.DictReader(open(f'{adir}/work.csv'))]
for r in W: r['card'] = int(r['card']); r['core'] = int(r['core']); r['s'] = float(r['start_ms']); r['e'] = float(r['end_ms']); r['n'] = r['node'].split('/mlp/')[-1]
def tile_of(n):
    m = re.search(r'_t(\d+)$', n) or re.search(r'(?:tc_reduce_tile_|final_tile_)(\d+)$', n); return int(m.group(1)) if m else None
mem = collections.Counter((r['n'].split('_s')[0] if r['n'].startswith('tc_g') else re.sub(r'\d+$', 'N', r['n']), r['engine'], r['kind'], r['memory']) for r in W if r['n'].startswith(('tc_', 'final_tile')))
print('memory placement of the combine ops (node pattern, engine, kind, memory: events):')
for k, v in sorted(mem.items(), key=lambda kv: -kv[1])[:14]: print('   ', k, v)
print(f'per tile (first {show}): lane-reduce window per card [start-end] and core | P2P rows | card-reduce window and card/core')
for i in range(show):
    lr = {c: [r for r in W if r['card'] == c and r['n'] == f'tc_reduce_tile_{i}' and r['engine'] == 'HVX'] for c in range(4)}
    p2p = [r for r in W if r['engine'] == 'P2P' and (r['n'] == f'tc_reduce_tile_{i}' or r['n'] == f'final_tile_{i}')]
    fr = [r for r in W if r['n'] == f'final_tile_{i}' and r['engine'] == 'HVX']
    g0 = [r for r in W if r['n'] == f'tc_g_s0_t{i}' and r['card'] == 0]
    lrs = '; '.join(f"c{c}[{min(x['s'] for x in v):.2f}-{max(x['e'] for x in v):.2f}]@{sorted({x['core'] for x in v})}" if v else f'c{c}: -' for c, v in lr.items())
    p2ps = f"{len(p2p)} rows [{min(x['s'] for x in p2p):.2f}-{max(x['e'] for x in p2p):.2f}]" if p2p else 'none'
    frs = f"[{min(x['s'] for x in fr):.2f}-{max(x['e'] for x in fr):.2f}] card/core {sorted({(x['card'], x['core']) for x in fr})}" if fr else '-'
    gs = f"[{min(x['s'] for x in g0):.2f}-{max(x['e'] for x in g0):.2f}]" if g0 else '-'
    print(f'   tile {i:2d}: gather c0 {gs} | lane reduce {lrs} | P2P {p2ps} | card reduce {frs}')
# where do the card reduces run?
fr_all = [r for r in W if r['n'].startswith('final_tile_') and r['engine'] == 'HVX']; print('card-reduce cards:', collections.Counter(r['card'] for r in fr_all), 'cores on card0:', sorted({r['core'] for r in fr_all if r['card'] == 0}))
