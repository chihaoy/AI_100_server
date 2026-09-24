#!/usr/bin/env python3
"""Where do the ops of a truncated full-model profile run? Groups work.csv rows by op family and reports cards/cores,
busy time and bytes; lists P2P senders. Usage: placement.py <analysis dir> [layer-prefix e.g. /model/layers.0/]"""
import sys, csv, re, collections
d = sys.argv[1]; pref = sys.argv[2] if len(sys.argv) > 2 else ''
rows = list(csv.DictReader(open(f'{d}/work.csv')))
def fam(node):
    n = node
    if '/self_attn/hp/' in n: return 'attn-hp/' + n.split('/hp/')[-1]
    if '/self_attn/' in n: return 'attn/' + re.sub(r'_\d+$', '', n.split('/self_attn/')[-1])
    if '/mlp/' in n: return 'mlp/' + re.sub(r'_\d+$', '', n.split('/mlp/')[-1])
    return 'other/' + n.split('/')[-1]
agg = collections.defaultdict(lambda: dict(cards=set(), cores=set(), dur=0.0, bytes=0, kinds=collections.Counter(), n=0))
for r in rows:
    if pref and not r['node'].startswith(pref): continue
    a = agg[fam(r['node'])]; a['cards'].add(int(r['card'])); a['cores'].add((int(r['card']), int(r['core']))); a['dur'] += float(r['dur']); a['bytes'] += int(float(r['bytes'] or 0)); a['kinds'][r['kind']] += 1; a['n'] += 1
print(f"{'family':40s} {'cards':10s} {'cores':>5s} {'busy ms':>8s} {'MiB':>7s}  kinds")
for k, a in sorted(agg.items(), key=lambda kv: -kv[1]['dur']):
    if a['dur'] < 50 and 'hp' not in k: continue
    print(f"{k[:40]:40s} {str(sorted(a['cards'])):10s} {len(a['cores']):5d} {a['dur']/1e3:8.2f} {a['bytes']/2**20:7.1f}  {dict(a['kinds'].most_common(3))}")
p2p = [r for r in rows if 'multicast' in r['kind'] and r['port']]
by = collections.defaultdict(lambda: [0, 0.0]); 
for r in p2p: by[fam(r['node'])][0] += int(float(r['bytes'] or 0)); by[fam(r['node'])][1] += float(r['dur'])
print('\nP2P/multicast bytes by family (MiB, busy ms):')
for k, (b, t) in sorted(by.items(), key=lambda kv: -kv[1][0])[:12]: print(f"  {k[:50]:50s} {b/2**20:8.2f} {t/1e3:8.2f}")
