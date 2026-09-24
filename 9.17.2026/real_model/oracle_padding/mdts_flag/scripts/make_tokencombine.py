#!/usr/bin/env python3
"""Combine redesign on a layer-2 replay graph (make_cap/make_chunk output dir).
Usage: make_tokencombine.py <src dir> <out dir> tiledfinal|tokencentric|tokenowned|to_fp16|to_idx|to_rs
 to_fp16: tokenowned + fp16 cast of the per-card partials; to_idx: + broadcast index build; to_rs: + reduce-scatter cross-card combine
 tiledfinal  : Einsum_4 ('dth->th' over [4,T,2048]) -> 16 token tiles + Concat
 tokencentric: drop the dense [64,T,2048] accumulator (zero splat, scatters, read-modify-write, 16 reduce tiles); instead,
               per card and per token tile, CtxGather3D the weighted expert outputs by the per-(lane,token) slot index
               (sentinel INT32_MAX gathers zero rows), add the two stages per lane, ReduceSum over the card's 16 lanes,
               concat tiles and cards into [4,T,2048], then the tiled final combine."""
import sys, os, shutil, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
src, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
m = onnx.load(f'{src}/model.onnx', load_external_data=False); g = m.graph; STEM = '/model/layers.2/mlp/'; DOM = 'com.qualcomm.cloud'
T = g.input[0].type.tensor_type.shape.dim[0].dim_value; assert T % 16 == 0; W = T // 16
bn = {n.name: n for n in g.node}
def const(name, arr): t = nh.from_array(np.asarray(arr), name); g.initializer.append(t); return name
i64 = lambda name, vals: const(name, np.array(vals, np.int64))
new_nodes = []
# ---------------- tiled final combine: replace Einsum_4 by 16 token tiles
e4 = bn[STEM + 'Einsum_4']; assert e4.op_type == 'Einsum' and h.get_attribute_value(e4.attribute[0]) == b'dth->th'
partials = e4.input[0]   # [4, T, 2048]
ax1 = i64('tc_final_axis', [1]); outs = []
for i in range(16):
    sl, rd = f'tc_final_slice_{i}', f'tc_final_reduced_{i}'
    new_nodes += [h.make_node('Slice', [partials, i64(f'tc_final_start_{i}', [i * W]), i64(f'tc_final_end_{i}', [(i + 1) * W]), ax1], [sl], name=STEM + f'final_slice_{i}'),
                  h.make_node('Einsum', [sl], [rd], equation='dth->th', name=STEM + f'final_tile_{i}')]
    outs.append(rd)
final_concat = h.make_node('Concat', outs, list(e4.output), axis=0, name=e4.name)
nodes = []
for n in g.node: nodes.extend(new_nodes + [final_concat] if n.name == e4.name else [n])
del g.node[:]; g.node.extend(nodes); new_nodes = []; bn = {n.name: n for n in g.node}
# ---------------- token-centric combine
if mode == 'tokencentric':
    slot = {0: STEM + 'Where_1_output_0', 1: STEM + 'Where_5_output_0'}          # [64, T] int32, INT32_MAX where not routed
    data = {0: STEM + 'Where_3_output_0'}                                          # [64, C0, 2048] weighted, invalid slots zeroed
    # stage 1 weighted outputs masked alone (Where_7 masks Add_2 = stage0 rows + stage1; we want stage1 only)
    w7 = bn[STEM + 'Where_7']; assert w7.input[1] == STEM + 'Add_2_output_0'
    zeros_src = {n.output[0]: n for n in g.node}[w7.input[2]]; assert zeros_src.op_type == 'ConstantOfShape'
    zero_value = [a for a in zeros_src.attribute if a.name == 'value']
    new_nodes += [h.make_node('Shape', [STEM + 'Mul_10_output_0'], ['tc_stage1_shape'], name=STEM + 'tc_stage1_shape'),
                  h.make_node('ConstantOfShape', ['tc_stage1_shape'], ['tc_stage1_zeros'], name=STEM + 'tc_stage1_zeros', value=zero_value[0].t) if zero_value else h.make_node('ConstantOfShape', ['tc_stage1_shape'], ['tc_stage1_zeros'], name=STEM + 'tc_stage1_zeros'),
                  h.make_node('Where', [w7.input[0], STEM + 'Mul_10_output_0', 'tc_stage1_zeros'], ['tc_stage1_masked'], name=STEM + 'tc_stage1_masked')]; data[1] = 'tc_stage1_masked'
    ax2 = i64('tc_axis2', [2]); zero_i32 = const('tc_zero_i32', np.array(0, np.int32)); shape4 = i64('tc_shape4', [4, 16, W, 2048])
    mask = {0: STEM + 'Greater_output_0', 1: STEM + 'Greater_1_output_0'}   # bool [64, T]: token routed to this lane
    for s in (0, 1):   # safe index (0 where not routed) and an fp16 mask; the device gather does not zero-fill out-of-range rows
        new_nodes += [h.make_node('Where', [mask[s], slot[s], zero_i32], [f'tc_slotsafe_s{s}'], name=STEM + f'tc_slotsafe_s{s}'),
                      h.make_node('Cast', [mask[s]], [f'tc_mask_s{s}'], to=TP.FLOAT16, name=STEM + f'tc_mask_s{s}'),
                      h.make_node('Unsqueeze', [f'tc_mask_s{s}', ax2], [f'tc_maskf_s{s}'], name=STEM + f'tc_maskf_s{s}')]
        slot[s] = f'tc_slotsafe_s{s}'
    # keep the 64-lane axis as the batch/partition axis throughout (as the MatMuls and the old accumulator tiles do); per token tile:
    # gather [64, W, 2048] per stage, mask, add stages, reshape [4, 16, W, 2048], reduce the 16 lanes per card, then the 4 cards
    final_outs = []
    for i in range(16):
        ts, te = i64(f'tc_tok_start_{i}', [i * W]), i64(f'tc_tok_end_{i}', [(i + 1) * W]); gathered = []
        for s in (0, 1):
            idx, gth, mk, gm = f'tc_idx_s{s}_t{i}', f'tc_g_s{s}_t{i}', f'tc_mk_s{s}_t{i}', f'tc_gm_s{s}_t{i}'
            new_nodes += [h.make_node('Slice', [slot[s], ts, te, ax1], [idx], name=STEM + idx),
                          h.make_node('CtxGather3D', [data[s], idx], [gth], name=STEM + gth, domain=DOM),       # [64, W, 2048]
                          h.make_node('Slice', [f'tc_maskf_s{s}', ts, te, ax1], [mk], name=STEM + mk),
                          h.make_node('Mul', [gth, mk], [gm], name=STEM + gm)]
            gathered.append(gm)
        add, rs, lanes, cards = f'tc_add_t{i}', f'tc_rs_t{i}', f'tc_lanes_t{i}', f'tc_final_reduced_{i}'
        new_nodes += [h.make_node('Add', gathered, [add], name=STEM + add),
                      h.make_node('Reshape', [add, shape4], [rs], name=STEM + rs),
                      h.make_node('Einsum', [rs], [lanes], equation='dpth->dth', name=STEM + f'tc_reduce_tile_{i}'),   # [4, W, 2048]
                      h.make_node('Einsum', [lanes], [cards], equation='dth->th', name=STEM + f'final_tile_{i}')]      # [W, 2048]
        final_outs.append(cards)
    # replace the tiled-final subgraph built above: its Concat keeps the name/output of Einsum_4
    keep_nodes = [n for n in g.node if not (n.name.startswith(STEM + 'final_slice_') or n.name.startswith(STEM + 'final_tile_'))]
    fc = next(n for n in keep_nodes if n.name == STEM + 'Einsum_4'); del fc.input[:]; fc.input.extend(final_outs)
    pos = next(i for i, n in enumerate(keep_nodes) if n.name == STEM + 'Einsum_4'); keep_nodes[pos:pos] = new_nodes
    del g.node[:]; g.node.extend(keep_nodes); new_nodes = []

# ---------------- token-owned combine: per card, gather each token's eight (lane, slot) rows, sum them locally
TO = mode in ('tokenowned', 'to_fp16', 'to_idx', 'to_rs'); cast16 = mode in ('to_fp16', 'to_idx', 'to_rs'); trim_idx = mode in ('to_idx', 'to_rs'); rs = mode == 'to_rs'
if TO:
    inits = {t.name: t for t in g.initializer}
    C = {0: int(nh.to_array(inits['probe_stage0_stop'])), 1: int(nh.to_array(inits['probe_stage1_stop']))}
    routes = (bn.get(STEM + 'probe_input_routes') or bn['probe_input_routes']).output[0]                       # [T, 128] routing weights, position order p = stage*64 + lane
    w7 = bn[STEM + 'Where_7']; zsrc = {n.output[0]: n for n in g.node}[w7.input[2]]; assert zsrc.op_type == 'ConstantOfShape'; zval = [a_ for a_ in zsrc.attribute if a_.name == 'value']
    data = {0: STEM + 'Where_3_output_0', 1: 'to_stage1_masked'}
    new_nodes += [h.make_node('Shape', [STEM + 'Mul_10_output_0'], ['to_stage1_shape'], name=STEM + 'to_stage1_shape'),
                  h.make_node('ConstantOfShape', ['to_stage1_shape'], ['to_stage1_zeros'], name=STEM + 'to_stage1_zeros', **({'value': zval[0].t} if zval else {})),
                  h.make_node('Where', [w7.input[0], STEM + 'Mul_10_output_0', 'to_stage1_zeros'], ['to_stage1_masked'], name=STEM + 'to_stage1_masked')]
    i32 = lambda name, v: const(name, np.array(v, np.int32))
    k8 = i64('to_k8', [8]); c64, c16, zero = i32('to_c64', 64), i32('to_c16', 16), i32('to_zero', 0)
    new_nodes += [h.make_node('TopK', [routes, k8], ['to_vals', 'to_idx64'], axis=1, largest=1, sorted=1, name=STEM + 'to_topk'),
                  h.make_node('Cast', ['to_idx64'], ['to_pos'], to=TP.INT32, name=STEM + 'to_pos'),
                  h.make_node('Div', ['to_pos', 'to_c64'], ['to_stage'], name=STEM + 'to_stage'), h.make_node('Mul', ['to_stage', 'to_c64'], ['to_stage64'], name=STEM + 'to_stage64'),
                  h.make_node('Sub', ['to_pos', 'to_stage64'], ['to_lane'], name=STEM + 'to_lane'),
                  h.make_node('Div', ['to_lane', 'to_c16'], ['to_card'], name=STEM + 'to_card'), h.make_node('Mul', ['to_card', 'to_c16'], ['to_card16'], name=STEM + 'to_card16'),
                  h.make_node('Sub', ['to_lane', 'to_card16'], ['to_local'], name=STEM + 'to_local')]
    slot_src = {0: STEM + 'Where_1_output_0', 1: STEM + 'Where_5_output_0'}; ax0 = i64('to_axis0', [0]); rs_axes = i64('to_reduce_axes', [2]); shape_g = i64('to_shape_g', [4, T * 8]); shape_m = i64('to_shape_m', [4, T * 8, 1]); shape4 = i64('to_shape4', [4, T, 8, 2048])
    gathered = []
    for s_ in (0, 1):
        cs = i32(f'to_C{s_}', C[s_]); st = i32(f'to_s{s_}', s_); shape_d = i64(f'to_shape_d{s_}', [4, 16 * C[s_], 2048])
        new_nodes += [h.make_node('Transpose', [slot_src[s_]], [f'to_slotT_s{s_}'], perm=[1, 0], name=STEM + f'to_slotT_s{s_}'),
                      h.make_node('GatherElements', [f'to_slotT_s{s_}', 'to_lane'], [f'to_slot_s{s_}'], axis=1, name=STEM + f'to_slot_s{s_}'),       # [T, 8]
                      h.make_node('Mul', ['to_local', cs], [f'to_localrow_s{s_}'], name=STEM + f'to_localrow_s{s_}'),
                      h.make_node('Add', [f'to_localrow_s{s_}', f'to_slot_s{s_}'], [f'to_row_s{s_}'], name=STEM + f'to_row_s{s_}'),
                      h.make_node('Equal', ['to_stage', st], [f'to_instage_s{s_}'], name=STEM + f'to_instage_s{s_}'),
                      h.make_node('Reshape', [data[s_], shape_d], [f'to_data_s{s_}'], name=STEM + f'to_data_s{s_}')]
        idx_parts, mask_parts = [], []
        if trim_idx:
            if s_ == 0:
                range4 = const('to_range4', np.arange(4, dtype=np.int32).reshape(4, 1, 1))
                new_nodes += [h.make_node('Unsqueeze', ['to_card', ax0], ['to_card_u'], name=STEM + 'to_card_u'), h.make_node('Equal', ['to_card_u', 'to_range4'], ['to_oncard4'], name=STEM + 'to_oncard4')]   # [4, T, 8]
            new_nodes += [h.make_node('Unsqueeze', [f'to_instage_s{s_}', ax0], [f'to_instage_u_s{s_}'], name=STEM + f'to_instage_u_s{s_}'),
                          h.make_node('And', ['to_oncard4', f'to_instage_u_s{s_}'], [f'to_m4_s{s_}'], name=STEM + f'to_m4_s{s_}'),
                          h.make_node('Unsqueeze', [f'to_row_s{s_}', ax0], [f'to_row_u_s{s_}'], name=STEM + f'to_row_u_s{s_}'),
                          h.make_node('Where', [f'to_m4_s{s_}', f'to_row_u_s{s_}', 'to_zero'], [f'to_idx4_s{s_}'], name=STEM + f'to_idx4_s{s_}'),
                          h.make_node('Cast', [f'to_m4_s{s_}'], [f'to_mf4_s{s_}'], to=TP.FLOAT16, name=STEM + f'to_mf4_s{s_}')]
        for c in (range(4) if not trim_idx else []):
            cc = i32(f'to_card{c}', c) if s_ == 0 else f'to_card{c}'
            new_nodes += [h.make_node('Equal', ['to_card', f'to_card{c}'], [f'to_oncard{c}_s{s_}'], name=STEM + f'to_oncard{c}_s{s_}'),
                          h.make_node('And', [f'to_instage_s{s_}', f'to_oncard{c}_s{s_}'], [f'to_m{c}_s{s_}'], name=STEM + f'to_m{c}_s{s_}'),
                          h.make_node('Where', [f'to_m{c}_s{s_}', f'to_row_s{s_}', 'to_zero'], [f'to_idx{c}_s{s_}'], name=STEM + f'to_idx{c}_s{s_}'),
                          h.make_node('Cast', [f'to_m{c}_s{s_}'], [f'to_mf{c}_s{s_}'], to=TP.FLOAT16, name=STEM + f'to_mf{c}_s{s_}'),
                          h.make_node('Unsqueeze', [f'to_idx{c}_s{s_}', ax0], [f'to_idx{c}_s{s_}_u'], name=STEM + f'to_idx{c}_s{s_}_u'),
                          h.make_node('Unsqueeze', [f'to_mf{c}_s{s_}', ax0], [f'to_mf{c}_s{s_}_u'], name=STEM + f'to_mf{c}_s{s_}_u')]
            idx_parts.append(f'to_idx{c}_s{s_}_u'); mask_parts.append(f'to_mf{c}_s{s_}_u')
        if not trim_idx: new_nodes += [h.make_node('Concat', idx_parts, [f'to_idx4_s{s_}'], axis=0, name=STEM + f'to_idx4_s{s_}'), h.make_node('Concat', mask_parts, [f'to_mf4_s{s_}'], axis=0, name=STEM + f'to_mf4_s{s_}')]
        new_nodes += [h.make_node('Reshape', [f'to_idx4_s{s_}', shape_g], [f'to_idx_s{s_}'], name=STEM + f'to_idx_s{s_}'),
                      h.make_node('Reshape', [f'to_mf4_s{s_}', shape_m], [f'to_mask_s{s_}'], name=STEM + f'to_mask_s{s_}'),
                      h.make_node('CtxGather3D', [f'to_data_s{s_}', f'to_idx_s{s_}'], [f'to_g_s{s_}'], name=STEM + f'to_g_s{s_}', domain=DOM),        # [4, T*8, 2048]
                      h.make_node('Mul', [f'to_g_s{s_}', f'to_mask_s{s_}'], [f'to_gm_s{s_}'], name=STEM + f'to_gm_s{s_}')]
        gathered.append(f'to_gm_s{s_}')
    red_out = 'to_partials_raw' if cast16 else 'to_partials'
    new_nodes += [h.make_node('Add', gathered, ['to_sum'], name=STEM + 'to_sum'), h.make_node('Reshape', ['to_sum', shape4], ['to_sum4'], name=STEM + 'to_sum4'),
                  h.make_node('ReduceSum', ['to_sum4', rs_axes], [red_out], keepdims=0, name=STEM + 'to_partials')]                              # [4, T, 2048]
    if cast16: new_nodes.append(h.make_node('Cast', ['to_partials_raw'], ['to_partials'], to=TP.FLOAT16, name=STEM + 'to_partials16'))
    if rs:   # reduce-scatter: quarters of tokens become the card-partitioned batch axis; each card reduces its quarter over the 4 cards
        Q = T // 4; e4name = STEM + 'Einsum_4'; e4out = STEM + 'Einsum_4_output_0'
        new_nodes += [h.make_node('Reshape', ['to_partials', i64('rs_shape_q', [4, 4, Q, 2048])], ['rs_cq'], name=STEM + 'rs_reshape_q'),
                      h.make_node('Transpose', ['rs_cq'], ['rs_qc'], perm=[1, 0, 2, 3], name=STEM + 'rs_alltoall'),                              # [4 quarters, 4 cards, Q, 2048]
                      h.make_node('Einsum', ['rs_qc'], ['rs_red'], equation='qcth->qth', name=STEM + 'rs_reduce'),                              # per quarter, sum over cards
                      h.make_node('Reshape', ['rs_red', i64('rs_shape_y', [T, 2048])], [e4out], name=e4name)]
        nodes = [n for n in g.node if not (n.name.startswith(STEM + 'final_slice_') or n.name.startswith(STEM + 'final_tile_') or n.name == e4name)]
        pos = next(i for i, n in enumerate(nodes) if e4out in n.input); nodes[pos:pos] = new_nodes
    else:
        for n in g.node:
            if n.name.startswith(STEM + 'final_slice_'): n.input[0] = 'to_partials'
        pos = next(i for i, n in enumerate(g.node) if n.name == STEM + 'final_slice_0'); nodes = list(g.node); nodes[pos:pos] = new_nodes
    del g.node[:]; g.node.extend(nodes); new_nodes = []
needed = {o.name for o in g.output}; keep = []
for n in reversed(list(g.node)):
    if any(o in needed for o in n.output): keep.append(n); needed.update(n.input)
removed = len(g.node) - len(keep); keep.reverse(); del g.node[:]; g.node.extend(keep)
used = needed; inits = [t for t in g.initializer if t.name in used]; del g.initializer[:]; g.initializer.extend(inits)
del g.value_info[:]
for f in ('weights.bin', 'input_f16.bin', 'expected_counts_i32.bin', 'info.json'):
    if os.path.exists(f'{src}/{f}') and not os.path.exists(f'{out}/{f}'): os.symlink(os.path.realpath(f'{src}/{f}'), f'{out}/{f}')
onnx.save(m, f'{out}/model.onnx'); _c = os.getcwd(); os.chdir(out); onnx.checker.check_model('model.onnx'); os.chdir(_c)
names = {n.name for n in g.node}; gone = [k for k in ('ConstantOfShape_1', 'CtxScatter3D', 'CtxGather3D_5', 'Add_2', 'CtxScatter3D_1', 'Reshape_6', 'reduce_tile_0', 'Einsum_3') if STEM + k not in names]
if TO: print('   ops:', sorted({n.op_type for n in g.node if n.name.startswith(STEM + 'to_')}))
print(f'{out}: mode={mode} T={T} nodes={len(g.node)} pruned={removed} removed={gone}')
