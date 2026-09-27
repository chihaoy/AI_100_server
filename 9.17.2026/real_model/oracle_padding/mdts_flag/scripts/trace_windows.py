#!/usr/bin/env python3
"""Per-card execution windows from merged opstats traces: first/last op per card and per node family, all samples.
Usage: trace_windows.py <trace dir> [node regex groups...]"""
import sys, re, json, glob, collections
tdir = sys.argv[1]; files = sorted(glob.glob(f'{tdir}/*merged*.trace.json'))
CORE = re.compile(r'_(?:slice|Partition)(\d+)_Core_(\d+)(?:_(.*))?$')
for f in files[:3]:
    raw = json.load(open(f))['traceEvents']
    threads = {(e['pid'], e['tid']): e['args']['name'] for e in raw if e['ph'] == 'M' and e['name'] == 'thread_name'}
    procs = {e['pid']: e['args']['name'] for e in raw if e['ph'] == 'M' and e['name'] == 'process_name'}
    win = collections.defaultdict(lambda: [1e18, -1e18]); fam = collections.defaultdict(lambda: [1e18, -1e18, 0.0]); p2p = [1e18, -1e18, 0]
    for e in raw:
        if e['ph'] != 'X': continue
        th = threads.get((e['pid'], e['tid']), ''); m = CORE.search(th); s, d = float(e['ts']), float(e.get('dur', 0))
        a = e.get('args', {}); op = a.get('opName', '') or e.get('name', '')
        if m is None:
            if 'oc' in a: p2p[0] = min(p2p[0], s); p2p[1] = max(p2p[1], s + d); p2p[2] += 1
            continue
        card, eng = int(m.group(1)), m.group(3) or 'execution'
        if eng == 'execution': w = win[card]; w[0] = min(w[0], s); w[1] = max(w[1], s + d); continue
        if e['name'].startswith('sync ') or e['name'].startswith('barrier '): continue
        mm = re.search(r'/toy/(c\d|sum)/([a-z_]+)', op) or re.search(r'(/model/[^ ]+)', op)
        key = (card, mm.group(1) + '/' + mm.group(2)) if mm and mm.lastindex == 2 else (card, 'other')
        fz = fam[key]; fz[0] = min(fz[0], s); fz[1] = max(fz[1], s + d); fz[2] += d
    t0 = min(w[0] for w in win.values())
    print(f"{f.split('/')[-1][:70]}: processes {sorted(set(procs.values()))[:4]}")
    print("   card execution windows (ms): " + ', '.join(f"{c}: {(w[0]-t0)/1e3:.3f}-{(w[1]-t0)/1e3:.3f}" for c, w in sorted(win.items())))
    if p2p[2]: print(f"   P2P events {p2p[2]}: {(p2p[0]-t0)/1e3:.3f}-{(p2p[1]-t0)/1e3:.3f} ms")
    for (c, k), (s, e_, b) in sorted(fam.items(), key=lambda kv: (kv[0][0], kv[1][0])):
        if b > 5: print(f"     card {c} {k:18s} {(s-t0)/1e3:7.3f}-{(e_-t0)/1e3:7.3f} ms  busy {b/1e3:7.3f} ms")
