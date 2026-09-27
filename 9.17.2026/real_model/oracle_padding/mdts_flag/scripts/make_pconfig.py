#!/usr/bin/env python3
"""Build a one-partition-per-card MDP config from a compiler dump: IR nodes whose name contains a card tag go to that
card's partition, everything else to partition 0; node order is the compiler's dump order.
Usage: make_pconfig.py <model.onnx> <dump.json> <out.json> <tag regex with one group = card index> [<last-partition regex>]"""
import sys, re, json, onnx
model, dump, out, tag = sys.argv[1], sys.argv[2], sys.argv[3], re.compile(sys.argv[4]); last = re.compile(sys.argv[5]) if len(sys.argv) > 5 else None
m = onnx.load(model, load_external_data=False); order = {n.name: i for i, n in enumerate(m.graph.node)}
d = json.load(open(dump)); names = [n for p in d['partitions'] for n in p.get('nodeList', [])]
def base(n):
    b = n
    while b not in order and '/' in b: b = b.rsplit('/', 1)[0]
    return b
unknown = [n for n in names if base(n) not in order]
pos = {n: (order.get(base(n), 10**9), k) for k, n in enumerate(names)}
parts = [[] for _ in range(4)]
for n in names:                                  # the compiler's dump order is topological; keep it
    mt = tag.search(n); parts[3 if (last and last.search(n)) else (int(mt.group(1)) if mt else 0)].append(n)
cfg = {'connections': [{'devices': [0, 1, 2, 3], 'type': 'p2p'}],
       'partitions': [{'name': f'Partition{c}', 'nodeList': parts[c], 'devices': [{'deviceId': c, 'numCores': 16}]} for c in range(4)]}
json.dump(cfg, open(out, 'w'), indent=1)
print(f'{out}: {len(names)} IR nodes ({len(unknown)} without an ONNX parent: {unknown[:4]}); per partition {[len(p) for p in parts]}')
