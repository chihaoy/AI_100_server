#!/usr/bin/env python3
"""Top op kinds by summed duration per engine in an opstats trace; per-card max-core totals for selected kinds.
Usage: mx_kinds.py <trace.json> [kind_regex]"""
import json,re,sys,collections,statistics as st
trace=sys.argv[1]; pat=re.compile(sys.argv[2]) if len(sys.argv)>2 else None
CORE=re.compile(r'_slice(\d+)_Core_(\d+)(?:_(.*))?$')
raw=json.load(open(trace))['traceEvents']; threads={(e['pid'],e['tid']):e['args']['name'] for e in raw if e['ph']=='M' and e['name']=='thread_name'}
tot=collections.defaultdict(float); percore=collections.defaultdict(float); n=collections.Counter()
for e in raw:
    if e['ph']!='X': continue
    m=CORE.search(threads.get((e['pid'],e['tid']),''))
    if not m or not m[3] or e['name'].startswith(('sync','barrier')): continue
    k=e.get('args',{}).get('opKind','').strip(); eng=m[3]; d=float(e.get('dur',0))
    tot[(eng,k)]+=d; n[(eng,k)]+=1; percore[(int(m[1]),int(m[2]),eng,k)]+=d
print("engine/kind: total ms over all cores, events, max single-core ms")
for (eng,k),v in sorted(tot.items(), key=lambda kv:-kv[1])[:18]:
    mx=max(val for (c,co,e2,k2),val in percore.items() if e2==eng and k2==k)
    print(f"   {eng:14s} {k:28s} {v/1e3:9.2f} ms  n={n[(eng,k)]:6d}  max-core {mx/1e3:6.2f} ms")
if pat:
    print(f"per-card max-core ms for kinds matching /{pat.pattern}/:")
    for card in range(4):
        vals=collections.defaultdict(float)
        for (c,co,eng,k),val in percore.items():
            if c==card and pat.search(k): vals[co]+=val
        if vals: print(f"   card{card}: max {max(vals.values())/1e3:.2f} med {st.median(vals.values())/1e3:.2f} min {min(vals.values())/1e3:.2f} ms over {len(vals)} cores")
