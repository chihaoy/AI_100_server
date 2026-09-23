#!/usr/bin/env python3
"""Prefill-only variant of a full-model graph: KV-cache inputs replaced by zero initializers, *_RetainedState outputs dropped,
so the compiled program exposes only input_ids/position_ids/logits(/routing_counts). Usage: kvfree_graph.py <src dir> <out dir>"""
import sys, os, onnx, numpy as np
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
m = onnx.load(f'{src}/model.onnx', load_external_data=False); g = m.graph
kv_in = [i for i in g.input if i.name.startswith('past_')]; kv_out = [o for o in g.output if o.name.endswith('_RetainedState')]
new_nodes = []
for i in kv_in:   # zero cache as a plain fp32 initializer: -convert-to-fp16 treats it like any weight
    dims = [d.dim_value if d.dim_value else {'batch_size': 1, 'ctx_len': 256}[d.dim_param] for d in i.type.tensor_type.shape.dim]
    g.initializer.append(nh.from_array(np.zeros(dims, np.float16), i.name))   # fp16: the compiler leaves fp32 constants unconverted
for i in kv_in: g.input.remove(i)
for o in kv_out: g.output.remove(o)
nodes = new_nodes + list(g.node); del g.node[:]; g.node.extend(nodes)   # constants first, before any consumer
# prune nodes that only fed the dropped outputs
needed = {o.name for o in g.output}; keep = []
for n in reversed(list(g.node)):          # reverse topological walk: keep a node only if one of its outputs is needed
    if any(o in needed for o in n.output): keep.append(n); needed.update(n.input)
keep.reverse(); del g.node[:]; g.node.extend(keep)
for link in ('weights', 'weights_fp16', 'weights_native_fp16', 'regrouped'):
    p = f'{src}/{link}'
    if os.path.islink(p) and not os.path.exists(f'{out}/{link}'): os.symlink(os.path.realpath(p), f'{out}/{link}')
onnx.save(m, f'{out}/model.onnx'); _c = os.getcwd(); os.chdir(out); onnx.checker.check_model('model.onnx'); os.chdir(_c)
print(f'{out}: removed {len(kv_in)} KV inputs and {len(kv_out)} retained outputs; inputs now {[i.name for i in g.input]}, outputs {[o.name for o in g.output]}, nodes {len(g.node)}')
