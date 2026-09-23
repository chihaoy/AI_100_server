#!/usr/bin/env python3
"""Quick device-trace summary: P2P bytes per node, GEMM spans per stage, device duration. Usage: trace_quick.py <merged trace json> [label]"""
import json, re, sys
from collections import defaultdict
STEM = '/model/layers.2/mlp/'
CORE = re.compile(r'_slice(\d+)_Core_(\d+)(?:_(.*))?$')
path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else path
raw = json.load(open(path))['traceEvents']
threads = {(e['pid'], e['tid']): e['args']['name'] for e in raw if e['ph'] == 'M' and e['name'] == 'thread_name'}
execution, p2p, hmx, waits = [], [], [], []
seen = set()
for e in raw:
    if e['ph'] != 'X': continue
    a = e.get('args', {}); th = threads.get((e['pid'], e['tid']), ''); m = CORE.search(th)
    if 'oc' in a:
        key = (e['pid'], e['tid'], int(a['oc']))
        if key in seen: continue
        seen.add(key)
    s, d = float(e['ts']), float(e.get('dur', 0))
    if m and (m[3] or 'execution') == 'execution':
        execution.append((s, s + d)); continue
    port = a.get('PortDescription', '')
    if port and '->' in port:
        p2p.append((a.get('opName', ''), int(a.get('opOutputSize', 0)), s, s + d, port)); continue
    if m and m[3] == 'HMX' and 'oc' in a and a.get('opKind', '').strip() == 'aicconvolutiond32':
        hmx.append((a.get('opName', ''), int(m[1]), int(m[2]), s, s + d, d))
origin = min(x[0] for x in execution); finish = max(x[1] for x in execution)
def short(n):
    n = re.sub(r'_slice\d+.*$', '', n); n = re.sub(r'/__\d+$', '', n); n = re.sub(r'/n\d+$', '', n); return n.replace(STEM, '')
print(f'== {label}: device {(finish-origin)/1e3:.3f} ms')
byn = defaultdict(lambda: [0, 0, 1e18, 0])
for n, b, s, e, port in p2p:
    r = byn[short(n)]; r[0] += 1; r[1] += b; r[2] = min(r[2], s); r[3] = max(r[3], e)
tot = 0
for n, (c, b, s, e) in sorted(byn.items(), key=lambda kv: -kv[1][1]):
    tot += b; print(f'   P2P {n:28s} {c:3d} sends {b/2**20:8.3f} MiB  window {(s-origin)/1e3:.3f}-{(e-origin)/1e3:.3f} ms')
print(f'   P2P total {tot/2**20:.3f} MiB')
stage = defaultdict(list)
for n, card, core, s, e, d in hmx:
    key = short(n); key = re.sub(r'(_n\d+|_l\d+|_T)$', '', key)
    stage[key].append((s, e, d, card, core))
for k in sorted(stage):
    v = stage[k]; s0 = min(x[0] for x in v); e1 = max(x[1] for x in v)
    percore = defaultdict(float)
    for s, e, d, card, core in v: percore[(card, core)] += d
    med = sorted(percore.values())[len(percore)//2]
    print(f'   HMX {k:12s} span {(s0-origin)/1e3:.3f}-{(e1-origin)/1e3:.3f} ms ({(e1-s0)/1e3:.3f})  events {len(v)} on {len(percore)} cores, median busy/core {med:.1f} us')
# per-core gap from this core's last UP GEMM to its first DOWN GEMM, per stage
import statistics as st
for label2, up, down in (('hot', 'MatMul_1', 'MatMul_2'), ('cold', 'MatMul_4', 'MatMul_5')):
    ends, starts = defaultdict(float), defaultdict(lambda: 1e18)
    for n, card, core, s, e, d in hmx:
        k = re.sub(r'(_n\d+|_l\d+|_T)$', '', short(n))
        if k == up: ends[(card, core)] = max(ends[(card, core)], e)
        if k == down: starts[(card, core)] = min(starts[(card, core)], s)
    gaps = [(starts[c] - ends[c]) / 1e3 for c in ends if c in starts]
    if gaps:
        print(f'   {label2} stage: per-core gap last-UP-end -> first-DOWN-start over {len(gaps)} cores: min {min(gaps):.3f}  median {st.median(gaps):.3f}  max {max(gaps):.3f} ms; down first-start spread across cores {(max(starts.values())-min(starts.values()))/1e3:.3f} ms')
