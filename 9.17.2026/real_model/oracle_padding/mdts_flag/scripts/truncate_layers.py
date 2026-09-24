#!/usr/bin/env python3
"""Truncate a full-model graph to its first N decoder layers: KV retained state kept for those layers, the final norm and
lm_head rewired to layer N-1's residual output, routing_counts dropped, dead nodes and unused initializers removed.
Also writes a custom_io.yaml restricted to the surviving KV IO. Usage: truncate_layers.py <src dir> <out dir> <N> <custom_io.yaml>"""
import sys, os, re, onnx
src, out, N, io_yaml = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; os.makedirs(out, exist_ok=True)
m = onnx.load(f'{src}/model.onnx', load_external_data=False); g = m.graph
lay = re.compile(r'^/model/layers\.(\d+)/')
last = max(int(lay.match(n.name).group(1)) for n in g.node if lay.match(n.name))
old, new = f'/model/layers.{last}/Add_1_output_0', f'/model/layers.{N-1}/Add_1_output_0'
hits = 0
for n in g.node:
    for k, x in enumerate(n.input):
        if x == old: n.input[k] = new; hits += 1
assert hits >= 1, 'final norm input not found'
keep = [n for n in g.node if not (lay.match(n.name) and int(lay.match(n.name).group(1)) >= N)]
del g.node[:]; g.node.extend(keep)
def idx(name): return int(re.search(r'\.(\d+)(_RetainedState)?$', name).group(1))
for i in [i for i in g.input if i.name.startswith('past_') and idx(i.name) >= N]: g.input.remove(i)
for o in [o for o in g.output if o.name.endswith('_RetainedState') and idx(o.name) >= N]: g.output.remove(o)
for o in [o for o in g.output if o.name == 'routing_counts']: g.output.remove(o)
needed = {o.name for o in g.output}; keep = []
for n in reversed(list(g.node)):
    if any(o in needed for o in n.output): keep.append(n); needed.update(n.input)
keep.reverse(); del g.node[:]; g.node.extend(keep)
used = {x for n in g.node for x in n.input}
for t in [t for t in g.initializer if t.name not in used]: g.initializer.remove(t)
for link in ('weights', 'weights_fp16', 'weights_native_fp16', 'regrouped'):
    p = f'{src}/{link}'
    if os.path.islink(p) and not os.path.exists(f'{out}/{link}'): os.symlink(os.path.realpath(p), f'{out}/{link}')
onnx.save(m, f'{out}/model.onnx'); _c = os.getcwd(); os.chdir(out); onnx.checker.check_model('model.onnx'); os.chdir(_c)
# custom IO restricted to surviving names
names = {i.name for i in g.input} | {o.name for o in g.output}
blocks = open(io_yaml).read().split('\n\n'); kept = [b for b in blocks if b.strip() and re.search(r'IOName:\s*(\S+)', b).group(1) in names]
open(f'{out}/custom_io.yaml', 'w').write('\n\n'.join(kept) + '\n')
print(f'{out}: {N} layers, inputs {len(g.input)}, outputs {len(g.output)}, nodes {len(g.node)}, initializers {len(g.initializer)}, custom IO entries {len(kept)}')
