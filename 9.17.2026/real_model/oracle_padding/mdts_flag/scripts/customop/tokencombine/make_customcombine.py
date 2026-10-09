#!/usr/bin/env python3
"""E49 step 2: replace the per-card gather and sum of the token-owned combine by the TokenCombine custom op, every layer.
Input: a graph with build_full_dyncard.py's naive token-owned combine (two stages, e.g. full512_naive_tileadd_hpos or its
two-layer cut). Per layer and stage s, CtxGather3D(to_data_s [4, 16C, 2048], to_idx_s [4, T*8]) -> Mul(mask) -> Add over
stages -> Reshape [4, T, 8, 2048] -> ReduceSum over k becomes TokenCombine(to_data_s, Where(mask, to_idx_s, -1), k=8) per
stage and one Add, which writes to_partials; the final cross-card sum that reads to_partials is unchanged.
Usage: make_customcombine.py <src dir> <out dir>   (compile with -register-custom-op=<this dir>/custom_op_config.yaml)"""
import os, re, sys, json, onnx, numpy as np
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
m = onnx.load(f'{src}/model.onnx', load_external_data=False); g = m.graph
names = {n.name: n for n in g.node}
layers = sorted({int(x) for x in re.findall(r'/model/layers\.(\d+)/mlp/to_partials', ' '.join(names))})
assert layers, 'no token-owned combine found'
g.initializer.extend([nh.from_array(np.array(-1, np.int32), 'tcc_minus1')])
new_nodes, drop = {}, set()
for L in layers:
    STEM, P = f'/model/layers.{L}/mlp/', f'L{L}_'
    red = names[STEM + 'to_partials']; assert red.op_type == 'ReduceSum'
    T8 = None
    adds = []
    for s in (0, 1):
        gth = names[STEM + f'to_g_s{s}']; assert gth.op_type == 'CtxGather3D' and list(gth.input) == [P + f'to_data_s{s}', P + f'to_idx_s{s}']
        msk = names[STEM + f'to_mask_s{s}']; assert msk.op_type == 'Reshape' and msk.input[0] == P + f'to_mf4_s{s}'
        shape_m = nh.to_array(next(t for t in g.initializer if t.name == msk.input[1])); T8 = int(shape_m[1])
        sh = f'tcc_shape_idx_{T8}'
        if not any(t.name == sh for t in g.initializer): g.initializer.append(nh.from_array(np.array([4, T8], np.int64), sh))
        new_nodes.setdefault(red.name, []).extend([
            h.make_node('Reshape', [P + f'to_mf4_s{s}', sh], [P + f'tcc_mf_s{s}'], name=STEM + f'tcc_mf_s{s}'),
            h.make_node('Cast', [P + f'tcc_mf_s{s}'], [P + f'tcc_valid_s{s}'], to=TP.BOOL, name=STEM + f'tcc_valid_s{s}'),
            h.make_node('Where', [P + f'tcc_valid_s{s}', P + f'to_idx_s{s}', 'tcc_minus1'], [P + f'tcc_idx_s{s}'], name=STEM + f'tcc_idx_s{s}'),
            h.make_node('TokenCombine', [P + f'to_data_s{s}', P + f'tcc_idx_s{s}'], [P + f'tcc_part_s{s}'], domain='QAic', k=8, name=STEM + f'tcc_combine_s{s}')])
        adds.append(P + f'tcc_part_s{s}')
        drop |= {gth.name, STEM + f'to_gm_s{s}'}
    new_nodes[red.name].append(h.make_node('Add', adds, list(red.output), name=STEM + 'tcc_partials'))
    drop |= {STEM + 'to_sum', STEM + 'to_sum4', red.name}
nodes = []
for n in g.node:
    if n.name in new_nodes: nodes.extend(new_nodes[n.name])
    if n.name not in drop: nodes.append(n)
del g.node[:]; g.node.extend(nodes)
needed = {o.name for o in g.output}; keep = []
for n in reversed(list(g.node)):
    if any(o in needed for o in n.output): keep.append(n); needed.update(n.input)
keep.reverse(); del g.node[:]; g.node.extend(keep)
used = {x for n in g.node for x in n.input}
for t in [t for t in g.initializer if t.name not in used]: g.initializer.remove(t)
if not any(o.domain == 'QAic' for o in m.opset_import): m.opset_import.append(h.make_opsetid('QAic', 1))
for link in ('weights', 'weights_fp16', 'weights_native_fp16', 'regrouped', 'weights_hp', 'weights_cardmajor_fp16'):
    p = f'{src}/{link}'
    if os.path.exists(p) and not os.path.exists(f'{out}/{link}'): os.symlink(os.path.realpath(p), f'{out}/{link}')
if os.path.exists(f'{src}/custom_io.yaml') and not os.path.exists(f'{out}/custom_io.yaml'): os.symlink(os.path.realpath(f'{src}/custom_io.yaml'), f'{out}/custom_io.yaml')
onnx.save(m, f'{out}/model.onnx')
json.dump(dict(src=src, layers=layers, op='TokenCombine'), open(f'{out}/info.json', 'w'))
print(f'{out}: TokenCombine in {len(layers)} layers ({2 * len(layers)} ops), nodes {len(g.node)}')
