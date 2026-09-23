#!/usr/bin/env python3
"""Per-card node timeline and a critical-path walk that skips bookkeeping ops. Usage: detail_timeline.py <trace> <meta dir> <analysis dir> [card]"""
import sys, re, json, csv
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools'); from moe_qwen3_profile import canon
trace, meta, adir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); card_sel = int(sys.argv[4]) if len(sys.argv) > 4 else 0
STEM = '/model/layers.2/mlp/'; CORE = re.compile(r'_slice(\d+)_Core_(\d+)(?:_(.*))?$'); BOOK = {'aicendcyclestats', 'aicoutputsemaphoreinc', 'aicinputsemaphoreinc'}
short = lambda n: canon(n).replace(STEM, '')
raw = json.load(open(trace))['traceEvents']; threads = {(e['pid'], e['tid']): e['args']['name'] for e in raw if e['ph'] == 'M' and e['name'] == 'thread_name'}
work, execution = {}, []
for e in raw:
    if e['ph'] != 'X': continue
    a = e.get('args', {}); m = CORE.search(threads.get((e['pid'], e['tid']), '')); card, core, eng = (int(m[1]), int(m[2]), m[3] or 'execution') if m else (-1, -1, 'P2P')
    s, d = float(e['ts']), float(e.get('dur', 0))
    if eng == 'execution': execution.append((s, s + d)); continue
    if 'oc' not in a or e['name'].startswith(('barrier ', 'sync ')): continue
    work.setdefault((e['pid'], e['tid'], int(a['oc'])), dict(card=card, core=core, engine=eng, kind=a.get('opKind', '').strip(), node=short(a.get('opName', '')), start=s, end=s + d, dur=d, port=a.get('PortDescription', ''), bytes=int(a.get('opOutputSize', 0))))
origin = min(a for a, _ in execution); ms = lambda t: (t - origin) / 1e3
pending, incoming = {}, defaultdict(list)
for e in raw:
    if e['ph'] == 's': pending[e['id']] = e
    elif e['ph'] == 'f' and e['id'] in pending:
        src = pending.pop(e['id']); a = (src['pid'], src['tid'], int(src['args']['oc'])); b = (e['pid'], e['tid'], int(e['args']['oc']))
        if a in work and b in work: incoming[b].append((a, src.get('cat') == 'true'))
# critical path from the last real op
real = [(k, r) for k, r in work.items() if r['card'] >= 0 and r['kind'] not in BOOK]
k, cur = max(real, key=lambda kr: kr[1]['end']); chain = []; seen = set()
for _ in range(60):
    chain.append(cur); seen.add(k); preds = [(a, work[a], t) for a, t in incoming.get(k, []) if a not in seen and work[a]['kind'] not in BOOK]
    if not preds: break
    true = [p for p in preds if p[2]] or preds; k, cur, _ = max(true, key=lambda p: p[1]['end'])
print(f'critical path (backwards from the last real op; device end {ms(max(b for _, b in execution)):.3f} ms):')
for r in chain[:30]: print(f"   card{r['card']} core{r['core']:2d} {r['engine']:12s} {r['node']:26s} {r['kind']:24s} {r['port']:6s} {ms(r['start']):.3f}-{ms(r['end']):.3f} ({r['dur']:.0f} us)")
# per-card node timeline (window of each node on the selected card, with max-core busy), sorted by start
by = defaultdict(list)
for r in work.values():
    if r['card'] == card_sel: by[(r['node'], r['engine'], r['kind'])].append(r)
rows = []
for (n, eng, kind), rs in by.items():
    busy = defaultdict(float)
    for r in rs: busy[r['core']] += r['dur']
    rows.append(dict(node=n, engine=eng, kind=kind, start=ms(min(r['start'] for r in rs)), end=ms(max(r['end'] for r in rs)), cores=len(busy), max_core_us=max(busy.values()), events=len(rs), MiB=sum(r['bytes'] for r in rs) / 2**20))
rows.sort(key=lambda r: r['start'])
print(f'\ncard {card_sel} node windows (start-end ms, cores, max-core busy us), skipping tiny bookkeeping:')
for r in rows:
    if r['kind'] in BOOK or (r['end'] - r['start'] < 0.02 and r['max_core_us'] < 5): continue
    print(f"   {r['start']:6.3f}-{r['end']:6.3f} {r['node']:26s} {r['engine']:12s} {r['kind']:24s} cores {r['cores']:2d} max-core {r['max_core_us']:7.1f} us ev {r['events']:4d} {r['MiB']:6.1f} MiB")
with open(adir / f'card{card_sel}_timeline.csv', 'w', newline='') as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
