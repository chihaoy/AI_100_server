#!/usr/bin/env python3
"""Row-chunked hot stage: the hot gate/up/SiLU/down chain runs k times on consecutive row chunks of the [64, C, 2048]
gathered activation (Slice on axis 1), outputs concatenated back. No routing or weight change. Usage: make_rowchunk.py <src dir> <out dir> <k>"""
import sys, os, onnx, numpy as np
from onnx import helper as h, numpy_helper as nh
src, out, K = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(out, exist_ok=True)
m = onnx.load(f'{src}/model.onnx', load_external_data=False); g = m.graph; STEM = '/model/layers.2/mlp/'
bn = {n.name: n for n in g.node}; C = int(nh.to_array(next(t for t in g.initializer if t.name == 'probe_stage0_end'))[0]); assert C % K == 0, (C, K)
CHAIN = ['MatMul', 'MatMul_1', 'act_fn/Sigmoid', 'act_fn/Mul', 'Mul_4', 'MatMul_2']
chain = [bn[STEM + c] for c in CHAIN]; X = bn[STEM + 'MatMul'].input[0]; final = bn[STEM + 'MatMul_2'].output[0]
internal = {o for n in chain for o in n.output}
new = []; outs = []
def i64(name, v): g.initializer.append(nh.from_array(np.array(v, np.int64), name)); return name
ax = i64(STEM + 'rc_axis1', [1])
for k in range(K):
    xs = f'{STEM}rc_x_{k}'; new.append(h.make_node('Slice', [X, i64(f'{STEM}rc_start_{k}', [k * C // K]), i64(f'{STEM}rc_end_{k}', [(k + 1) * C // K]), ax], [xs], name=xs))
    ren = {X: xs}
    for n in chain:
        c = onnx.NodeProto(); c.CopyFrom(n); c.name = f'{n.name}_rc{k}'
        for i, x in enumerate(c.input):
            if x in ren: c.input[i] = ren[x]
        for i, o in enumerate(c.output): ren[o] = f'{o}_rc{k}'; c.output[i] = ren[o]
        new.append(c)
    outs.append(ren[final])
new.append(h.make_node('Concat', outs, [final], axis=1, name=STEM + 'rc_concat'))
nodes = []
for n in g.node:
    if n.name == STEM + 'MatMul': nodes.extend(new)
    if n in chain: continue
    nodes.append(n)
del g.node[:]; g.node.extend(nodes)
for link in ('weights.bin', 'input_f16.bin', 'expected_counts_i32.bin', 'info.json'):
    if os.path.exists(f'{src}/{link}') and not os.path.exists(f'{out}/{link}'): os.symlink(os.path.realpath(f'{src}/{link}'), f'{out}/{link}')
_c = os.getcwd(); os.chdir(out); onnx.checker.check_model(m); onnx.save(m, 'model.onnx'); os.chdir(_c)
print(f'{out}: hot chain replicated {K}x on {C // K}-row chunks; nodes {len(g.node)}')
