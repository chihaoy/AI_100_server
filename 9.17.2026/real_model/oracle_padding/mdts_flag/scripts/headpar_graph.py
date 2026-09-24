#!/usr/bin/env python3
"""Head-parallel attention rewrite for the QEfficient Qwen3-MoE graph.

Every attention block is re-expressed with the KV-head group (4 groups x 8 query heads) as a leading batch axis, so that under
-mdts-mos=1 (no weight splitting across cards) the compiler can place one KV head per card, exactly the layout the default
head-sliced compile produces, and the KV-cache CtxScatter/CtxGather of each head stay on one card (retained-state pairing).
  q: Expand(x)[4,T,2048] @ Wq_g[4,2048,1024] -> [4,T,8,128] -> q_norm -> [4,8,T,128] -> rotary
  k: Expand(x) @ Wk_g[4,2048,128] -> [4,T,1,128] -> k_norm -> [4,1,T,128] -> rotary -> [1,4,T,128] -> CtxScatter (unchanged node)
  v: Expand(x) @ Wv_g[4,2048,128] -> [1,4,T,128] -> CtxScatter_1 (unchanged node)
  attention per group on the gathered cache, o: [4,T,1024] @ Wo_g[4,1024,2048] -> ReduceSum over groups -> [1,T,2048]
Wq/Wk/Wv are re-laid out as new fp16 external files (<out>/weights_hp/), Wo is a pure reshape of the existing file.
Usage: headpar_graph.py <src dir> <out dir>"""
import sys, os, re, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
src, out = sys.argv[1], sys.argv[2]; os.makedirs(f'{out}/weights_hp', exist_ok=True)
m = onnx.load(f'{src}/model.onnx', load_external_data=False); g = m.graph
init = {t.name: t for t in g.initializer}
def ext_path(t): return f"{src}/{ {e.key: e.value for e in t.external_data}['location'] }"
def load_w(t): return np.fromfile(ext_path(t), np.float16).reshape(list(t.dims))
def add_ext(name, arr):
    path = f'{out}/weights_hp/{name}'; arr.astype(np.float16).tofile(path)
    t = TP(); t.name = name; t.data_type = TP.FLOAT16; t.dims.extend(arr.shape); t.data_location = TP.EXTERNAL
    for k, v in (('location', f'weights_hp/{name}'), ('offset', '0'), ('length', str(arr.nbytes))): e = t.external_data.add(); e.key = k; e.value = v
    g.initializer.append(t); return name
consts = {}
def const(name, arr):
    if name not in consts: g.initializer.append(nh.from_array(arr, name)); consts[name] = True
    return name
I64 = lambda n, v: const(n, np.array(v, np.int64))
layers = sorted({int(mm.group(1)) for n in g.node for mm in [re.match(r'^/model/layers\.(\d+)/self_attn/', n.name)] if mm})
by_name = {n.name: n for n in g.node}
new_nodes_pre, new_nodes_post, drop = {}, {}, set()
G, HQ, D = 4, 8, 128
for L in layers:
    P = f'/model/layers.{L}/self_attn/'; N = lambda s: P + 'hp/' + s
    x = by_name[P + 'q_proj/MatMul'].input[0]
    wq, wk, wv, wo = (init[by_name[P + f'{p}_proj/MatMul'].input[1]] for p in ('q', 'k', 'v', 'o'))
    Wq = add_ext(f'L{L}_wq', load_w(wq).reshape(2048, G, HQ * D).transpose(1, 0, 2))      # [4,2048,1024]
    Wk = add_ext(f'L{L}_wk', load_w(wk).reshape(2048, G, D).transpose(1, 0, 2))           # [4,2048,128]
    Wv = add_ext(f'L{L}_wv', load_w(wv).reshape(2048, G, D).transpose(1, 0, 2))
    del wo.dims[:]; wo.dims.extend([G, HQ * D, 2048])                                       # [4,1024,2048], same bytes
    qn = by_name[P + 'q_norm/CustomRMSNorm']; kn = by_name[P + 'k_norm/CustomRMSNorm']
    cos, sin = by_name[P + 'Mul'].input[1], by_name[P + 'Mul_1'].input[1]
    scale = by_name[P + 'Mul_8'].input[1]; mask = by_name[P + 'Where_4'].input[0]
    pre = []
    def mk(op, ins, outs, name, **kw): pre.append(h.make_node(op, ins, outs, name=N(name), **kw)); return outs[0]
    xe = mk('Expand', [x, I64('hp_shape_4_1_1', [G, 1, 1])], [N('xe')], 'Expand_x')
    def rotary(t, tag):   # t: [G,H,T,128] ; cos/sin [1,1,T,128]
        c = mk('Mul', [t, cos], [N(f'{tag}_cos')], f'Mul_cos_{tag}')
        a = mk('Slice', [t, I64('hp_zero', [0]), I64('hp_half', [D // 2]), I64('hp_axm1', [-1])], [N(f'{tag}_x1')], f'Slice_x1_{tag}')
        b = mk('Slice', [t, I64('hp_half', [D // 2]), I64('hp_D', [D]), I64('hp_axm1', [-1])], [N(f'{tag}_x2')], f'Slice_x2_{tag}')
        nb = mk('Neg', [b], [N(f'{tag}_negx2')], f'Neg_{tag}')
        rot = mk('Concat', [nb, a], [N(f'{tag}_rot')], f'Concat_rot_{tag}', axis=-1)
        s = mk('Mul', [rot, sin], [N(f'{tag}_sin')], f'Mul_sin_{tag}')
        return mk('Add', [c, s], [N(f'{tag}_rope')], f'Add_rope_{tag}')
    q = mk('MatMul', [xe, Wq], [N('q')], 'q_proj')                                          # [4,T,1024]
    q = mk('Reshape', [q, I64('hp_shape_q4', [G, -1, HQ, D])], [N('q4')], 'Reshape_q')       # [4,T,8,128]
    q = mk(qn.op_type, [q, qn.input[1]], [N('qn')], 'q_norm', domain=qn.domain, **{a.name: h.get_attribute_value(a) for a in qn.attribute})
    q = mk('Transpose', [q], [N('qt')], 'Transpose_q', perm=[0, 2, 1, 3])                    # [4,8,T,128]
    q = rotary(q, 'q')
    k = mk('MatMul', [xe, Wk], [N('k')], 'k_proj')                                          # [4,T,128]
    k = mk('Reshape', [k, I64('hp_shape_k4', [G, -1, 1, D])], [N('k4')], 'Reshape_k')        # [4,T,1,128]
    k = mk(kn.op_type, [k, kn.input[1]], [N('kn')], 'k_norm', domain=kn.domain, **{a.name: h.get_attribute_value(a) for a in kn.attribute})
    k = mk('Transpose', [k], [N('kt')], 'Transpose_k', perm=[0, 2, 1, 3])                    # [4,1,T,128]
    k = rotary(k, 'k')
    k_upd = mk('Reshape', [k, I64('hp_shape_1_4_T_D', [1, G, -1, D])], [N('k_upd')], 'Reshape_kupd')   # [1,4,T,128]
    v = mk('MatMul', [xe, Wv], [N('v')], 'v_proj')                                          # [4,T,128]
    v_upd = mk('Reshape', [v, I64('hp_shape_1_4_T_D', [1, G, -1, D])], [N('v_upd')], 'Reshape_vupd')
    by_name[P + 'CtxScatter'].input[2] = k_upd; by_name[P + 'CtxScatter_1'].input[2] = v_upd
    new_nodes_pre[P + 'CtxScatter'] = pre
    post = []
    def mk2(op, ins, outs, name, **kw): post.append(h.make_node(op, ins, outs, name=N(name), **kw)); return outs[0]
    kc = by_name[P + 'CtxGather'].output[0]; vc = by_name[P + 'Where_1'].output[0]           # [1,4,ctx,128]
    kg = mk2('Reshape', [kc, I64('hp_shape_4_1_ctx_D', [G, 1, -1, D])], [N('kg')], 'Reshape_kc')
    kg = mk2('Expand', [kg, I64('hp_shape_4_8_1_1', [G, HQ, 1, 1])], [N('kge')], 'Expand_k')  # [4,8,ctx,128]
    kT = mk2('Transpose', [kg], [N('kT')], 'Transpose_kT', perm=[0, 1, 3, 2])                 # [4,8,128,ctx]
    sc = mk2('MatMul', [q, kT], [N('scores')], 'MatMul_qk')                                  # [4,8,T,ctx]
    sc = mk2('Mul', [sc, scale], [N('scaled')], 'Mul_scale')
    fill = mk2('ConstantOfShape', [mk2('Shape', [sc], [N('sc_shape')], 'Shape_sc')], [N('neg_inf')], 'ConstantOfShape_mask', value=nh.from_array(np.array([-np.inf], np.float16)))
    sc = mk2('Where', [mask, fill, sc], [N('masked')], 'Where_mask')
    pr = mk2('Softmax', [sc], [N('probs')], 'Softmax', axis=-1)
    pr = mk2('Cast', [mk2('Cast', [pr], [N('probs32')], 'Cast_probs32', to=TP.FLOAT)], [N('probs16')], 'Cast_probs16', to=TP.FLOAT16)
    vg = mk2('Reshape', [vc, I64('hp_shape_4_1_ctx_D', [G, 1, -1, D])], [N('vg')], 'Reshape_vc')
    vg = mk2('Expand', [vg, I64('hp_shape_4_8_1_1', [G, HQ, 1, 1])], [N('vge')], 'Expand_v')  # [4,8,ctx,128]
    o = mk2('MatMul', [pr, vg], [N('ctx')], 'MatMul_pv')                                     # [4,8,T,128]
    o = mk2('Transpose', [o], [N('ctxt')], 'Transpose_ctx', perm=[0, 2, 1, 3])                # [4,T,8,128]
    o = mk2('Reshape', [o, I64('hp_shape_4_T_1024', [G, -1, HQ * D])], [N('ctx2')], 'Reshape_ctx')
    o = mk2('MatMul', [o, wo.name], [N('o_part')], 'o_proj')                                 # [4,T,2048]
    old_o = by_name[P + 'o_proj/MatMul']
    post.append(h.make_node('ReduceSum', [o, I64('hp_axes0', [0])], [old_o.output[0]], name=N('ReduceSum_o'), keepdims=1))   # [1,T,2048]
    new_nodes_post[old_o.name] = post; drop.add(old_o.name)
nodes = []
for n in g.node:
    if n.name in new_nodes_pre: nodes.extend(new_nodes_pre[n.name])
    if n.name in drop: nodes.extend(new_nodes_post[n.name]); continue
    nodes.append(n)
del g.node[:]; g.node.extend(nodes)
needed = {o.name for o in g.output}; keep = []
for n in reversed(list(g.node)):
    if any(o in needed for o in n.output): keep.append(n); needed.update(n.input)
keep.reverse(); del g.node[:]; g.node.extend(keep)
used = {x for n in g.node for x in n.input}
for t in [t for t in g.initializer if t.name not in used]: g.initializer.remove(t)
del g.value_info[:]
for link in ('weights', 'weights_fp16', 'weights_native_fp16', 'regrouped'):
    p = f'{src}/{link}'
    if os.path.islink(p) and not os.path.exists(f'{out}/{link}'): os.symlink(os.path.realpath(p), f'{out}/{link}')
if os.path.exists(f'{src}/custom_io.yaml') and not os.path.exists(f'{out}/custom_io.yaml'): os.symlink(os.path.realpath(f'{src}/custom_io.yaml'), f'{out}/custom_io.yaml')
onnx.save(m, f'{out}/model.onnx'); _c = os.getcwd(); os.chdir(out); onnx.checker.check_model('model.onnx'); os.chdir(_c)
print(f'{out}: {len(layers)} layers rewritten, nodes {len(g.node)}, initializers {len(g.initializer)}, new weight files {3*len(layers)}')
