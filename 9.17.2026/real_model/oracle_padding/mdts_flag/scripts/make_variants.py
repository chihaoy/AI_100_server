#!/usr/bin/env python3
"""Rewrite only the two down-projection MatMuls of the current-best layer-2 replay graph (compile-only probes)."""
import os, sys, shutil
import numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
SRC = '/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/routing_retune/c2_tree/model.onnx'
OUT = sys.argv[1]
STEM = '/model/layers.2/mlp/'
DOWN = [('MatMul_2', 'Mul_4_output_0'), ('MatMul_5', 'Mul_9_output_0')]
base = onnx.load(SRC, load_external_data=False)
init = {t.name: t for t in base.graph.initializer}
def ext(t):
    d = {e.key: e.value for e in t.external_data}
    return os.path.join(os.path.dirname(SRC), d['location']), int(d['offset']), int(d['length'])
def read_w(t):
    loc, off, ln = ext(t)
    with open(loc, 'rb') as f:
        f.seek(off); buf = f.read(ln)
    return np.frombuffer(buf, np.float16).reshape(list(t.dims))
def node_index(g, name):
    for i, n in enumerate(g.node):
        if n.name == STEM + name: return i
    raise KeyError(name)
def ext_init(name, arr, fh, path_rel, offsets):
    off = fh.tell(); fh.write(arr.tobytes()); ln = fh.tell() - off
    t = onnx.TensorProto()
    t.name = name; t.data_type = TP.FLOAT16; t.dims.extend(arr.shape)
    t.data_location = TP.EXTERNAL
    for k, v in (('location', path_rel), ('offset', str(off)), ('length', str(ln))):
        e = t.external_data.add(); e.key = k; e.value = v
    return t
def save(m, name):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    link = os.path.join(d, 'regrouped')
    if not os.path.exists(link): os.symlink(os.path.dirname(SRC) + '/regrouped', link)
    onnx.save(m, os.path.join(d, 'model.onnx'))
    onnx.checker.check_model(os.path.join(d, 'model.onnx'))
    print('wrote', d, 'nodes', len(m.graph.node))

# V4: down as Einsum, same weights
m = onnx.load(SRC, load_external_data=False)
for nm, _ in DOWN:
    n = m.graph.node[node_index(m.graph, nm)]
    n.op_type = 'Einsum'; del n.attribute[:]
    n.attribute.extend([h.make_attribute('equation', 'bck,bkn->bcn')])
pass  # v4 already written

# V3: down split along N into 4 x [64,768,512] MatMuls + Concat(axis=2)
m = onnx.load(SRC, load_external_data=False)
d = os.path.join(OUT, 'v3_splitn'); os.makedirs(d, exist_ok=True)
with open(os.path.join(d, 'down_split.bin'), 'wb') as fh:
    for nm, hin in DOWN:
        i = node_index(m.graph, nm); n = m.graph.node[i]; wname = n.input[1]; hin = n.input[0]
        W = read_w(init[wname]); assert W.shape == (64, 768, 2048)
        new_nodes, outs = [], []
        for k in range(4):
            t = ext_init(f'{wname}_n{k}', np.ascontiguousarray(W[:, :, k*512:(k+1)*512]), fh, 'down_split.bin', None)
            m.graph.initializer.append(t)
            o = f'{STEM}{nm}_n{k}_out'
            new_nodes.append(h.make_node('MatMul', [hin, t.name], [o], name=f'{STEM}{nm}_n{k}')); outs.append(o)
        new_nodes.append(h.make_node('Concat', outs, [n.output[0]], name=f'{STEM}{nm}_concat', axis=2))
        m.graph.node.remove(n)
        for j, nn in enumerate(new_nodes): m.graph.node.insert(i + j, nn)
        m.graph.initializer.remove([t for t in m.graph.initializer if t.name == wname][0])
save(m, 'v3_splitn')

# V5: down split along the lane (batch) axis into 4 x [16,C,768]@[16,768,2048] + Concat(axis=0)
m = onnx.load(SRC, load_external_data=False)
d = os.path.join(OUT, 'v5_splitlane'); os.makedirs(d, exist_ok=True)
with open(os.path.join(d, 'down_lane.bin'), 'wb') as fh:
    for nm, hin in DOWN:
        i = node_index(m.graph, nm); n = m.graph.node[i]; wname = n.input[1]; hin = n.input[0]
        W = read_w([t for t in m.graph.initializer if t.name == wname][0])
        starts = nh.from_array(np.array([0,16,32,48], np.int64), f'{STEM}{nm}_lane_starts')
        ends = nh.from_array(np.array([16,32,48,64], np.int64), f'{STEM}{nm}_lane_ends')
        axis0 = nh.from_array(np.array([0], np.int64), f'{STEM}{nm}_axis0')
        m.graph.initializer.extend([starts, ends, axis0])
        new_nodes, outs = [], []
        for k in range(4):
            t = ext_init(f'{wname}_l{k}', np.ascontiguousarray(W[k*16:(k+1)*16]), fh, 'down_lane.bin', None)
            m.graph.initializer.append(t)
            s0 = nh.from_array(np.array([k*16], np.int64), f'{STEM}{nm}_s{k}'); e0 = nh.from_array(np.array([(k+1)*16], np.int64), f'{STEM}{nm}_e{k}')
            m.graph.initializer.extend([s0, e0])
            hs = f'{STEM}{nm}_h_l{k}'; o = f'{STEM}{nm}_l{k}_out'
            new_nodes.append(h.make_node('Slice', [hin, s0.name, e0.name, axis0.name], [hs], name=f'{STEM}{nm}_hslice{k}'))
            new_nodes.append(h.make_node('MatMul', [hs, t.name], [o], name=f'{STEM}{nm}_l{k}')); outs.append(o)
        new_nodes.append(h.make_node('Concat', outs, [n.output[0]], name=f'{STEM}{nm}_concat', axis=0))
        m.graph.node.remove(n)
        for j, nn in enumerate(new_nodes): m.graph.node.insert(i + j, nn)
        m.graph.initializer.remove([t for t in m.graph.initializer if t.name == wname][0])
save(m, 'v5_splitlane')

# V8: transposed formulation  out = Transpose( W_d^T[64,2048,768] @ Transpose(h) )  -- weight has gate/up's shape
m = onnx.load(SRC, load_external_data=False)
d = os.path.join(OUT, 'v8_transposed'); os.makedirs(d, exist_ok=True)
with open(os.path.join(d, 'down_T.bin'), 'wb') as fh:
    for nm, hin in DOWN:
        i = node_index(m.graph, nm); n = m.graph.node[i]; wname = n.input[1]; hin = n.input[0]
        W = read_w([t for t in m.graph.initializer if t.name == wname][0])
        t = ext_init(f'{wname}_T', np.ascontiguousarray(W.transpose(0, 2, 1)), fh, 'down_T.bin', None)  # [64,2048,768]
        m.graph.initializer.append(t)
        hT, oT = f'{STEM}{nm}_hT', f'{STEM}{nm}_outT'
        new_nodes = [h.make_node('Transpose', [hin], [hT], name=f'{STEM}{nm}_transpose_h', perm=[0, 2, 1]),
                     h.make_node('MatMul', [t.name, hT], [oT], name=f'{STEM}{nm}'),
                     h.make_node('Transpose', [oT], [n.output[0]], name=f'{STEM}{nm}_transpose_out', perm=[0, 2, 1])]
        m.graph.node.remove(n)
        for j, nn in enumerate(new_nodes): m.graph.node.insert(i + j, nn)
        m.graph.initializer.remove([t for t in m.graph.initializer if t.name == wname][0])
save(m, 'v8_transposed')
