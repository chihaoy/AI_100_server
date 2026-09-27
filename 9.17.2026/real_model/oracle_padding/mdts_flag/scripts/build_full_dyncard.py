#!/usr/bin/env python3
"""Full-model port of the per-card runtime sort (E22, make_dyncard4) and the token-owned combine (E15), all 48 layers.
Input: a build_full_stack.py output with --rewrites 1 --combine dense (native order, capacity 128/128, T=128).
  --mode naive  : token-owned combine only (static native lanes, capacity 128/128)
  --mode dyncard: per card, the 32 native experts are sorted by token count at run time (16 hot, 16 cold lanes); mask /
                  prefix-scan / slot chain once over 128 experts in native order; one [128, ...] bank constant per matrix
                  gathered in card-major lane order; cold capacity --C_cold; then the token-owned combine
Usage: build_full_dyncard.py <src dir> <out dir> --mode naive|dyncard [--C_cold 16] [--T 128]"""
import sys, os, re, json, heapq, argparse, collections, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--mode', choices=['naive', 'dyncard', 'sortfirst'], required=True)
ap.add_argument('--C_cold', type=int, default=16); ap.add_argument('--T', type=int, default=128)
ap.add_argument('--final', choices=['einsum', 'addtree', 'tileadd'], default='einsum', help='form of the final cross-card sum: 16 Einsum tiles (collective template), '
                'elementwise (p0+p1)+(p2+p3) on the whole [T, 2048], or the same per token tile (github_pack/FINAL_COMBINE_16CORE.md)')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
T, DOM, C_HOT, C_COLD = a.T, 'com.qualcomm.cloud', a.T, (a.C_cold if a.mode in ('dyncard', 'sortfirst') else a.T)
m = onnx.load(f'{a.src}/model.onnx', load_external_data=False); g = m.graph
nodes = list(g.node); init = {t.name: t for t in g.initializer}; new_inits = []
key = {id(n): 10 * i for i, n in enumerate(nodes)}
def add_init(t):
    if t.name not in init: init[t.name] = t; new_inits.append(t)
    return t.name
i64 = lambda nm, v: add_init(nh.from_array(np.array(v, np.int64), nm)); i32 = lambda nm, v: add_init(nh.from_array(np.array(v, np.int32), nm))
removed_banks = set()
cval = {n.output[0]: nh.to_array(h.get_attribute_value(n.attribute[0])) for n in nodes if n.op_type == 'Constant'}
def cvalue(x):
    if x in cval: return int(np.asarray(cval[x]).reshape(-1)[0])
    t = init.get(x); return int(nh.to_array(t).reshape(-1)[0]) if t is not None and t.data_location != 1 else None
def layer_view(STEM):
    ln = [n for n in nodes if n.name.startswith(STEM)]; bn = {n.name: n for n in ln}
    prod = {o: n for n in ln for o in n.output}; cons = collections.defaultdict(list)
    for n in ln:
        for x in n.input: cons[x].append(n)
    return ln, bn, prod, cons
def dyncard(L, sortfirst=False):
    """E22 per-card runtime sort on layer L; returns (new nodes, dropped node names, position-order routes tensor).
    sortfirst: sort, then permute the routing columns into lane order and let the original two stage chains run on them"""
    STEM, P = f'/model/layers.{L}/mlp/', f'L{L}_'
    ln, bn, prod, cons = layer_view(STEM)
    # capacities: hot stays T, cold -> C_COLD (per-layer constants, as build_full_stack --caps)
    for stage, cap in ((1, C_COLD),):
        sl, rg = bn[STEM + 'Slice_1'], bn[STEM + 'Range_3']; assert sl.op_type == 'Slice' and rg.op_type == 'Range'
        sl.input[2] = i64(P + 'cold_end', [cap]); rg.input[1] = i32(P + 'cold_stop', cap)
    # banks: one [128, ...] constant per matrix from the two adjacent native stage banks
    mm = {k: bn[STEM + k] for k in ('MatMul', 'MatMul_1', 'MatMul_2', 'MatMul_3', 'MatMul_4', 'MatMul_5')}
    banks = {}
    for mat, (s0, s1) in (('gate', ('MatMul', 'MatMul_3')), ('up', ('MatMul_1', 'MatMul_4')), ('down', ('MatMul_2', 'MatMul_5'))):
        t0, t1 = init[mm[s0].input[1]], init[mm[s1].input[1]]
        e0, e1 = ({x.key: x.value for x in t.external_data} for t in (t0, t1))
        assert e0['location'] == e1['location'] and int(e1['offset']) == int(e0['offset']) + int(e0['length']), (L, mat, e0, e1)
        tb = TP(); tb.name = P + f'bank_{mat}'; tb.data_type = TP.FLOAT16; tb.dims.extend([128] + list(t0.dims)[1:]); tb.data_location = TP.EXTERNAL
        for k, v in (('location', e0['location']), ('offset', e0['offset']), ('length', str(2 * int(e0['length'])))): x = tb.external_data.add(); x.key = k; x.value = v
        add_init(tb); banks[s0] = (tb.name, 0); banks[s1] = (tb.name, 1); removed_banks.update([t0.name, t1.name])
    # routes: the scatter that builds the dense [T, 128] routing weights now writes routes_native
    sc = bn[STEM + 'ScatterElements']; routes_native = P + 'routes_native'; old_routes = sc.output[0]; sc.output[0] = routes_native
    c_ = lambda nm: P + nm
    if sortfirst:
        new = [h.make_node('Greater', [routes_native, add_init(nh.from_array(np.array(0, np.float16), 'dc_zero16'))], [c_('dc_routed')], name=STEM + 'dc_routed'),
               h.make_node('Cast', [c_('dc_routed')], [c_('dc_routed16')], name=STEM + 'dc_routed16', to=TP.FLOAT16),
               h.make_node('ReduceSum', [c_('dc_routed16'), i64('dc_ax0', [0])], [c_('dc_counts')], name=STEM + 'dc_counts', keepdims=0)]
        hot_g, cold_g = [], []
        for k in range(4):
            new += [h.make_node('Slice', [c_('dc_counts'), i64(f'dc_lo{k}', [32 * k]), i64(f'dc_hi{k}', [32 * k + 32]), 'dc_ax0'], [c_(f'dc_cnt{k}')], name=STEM + f'dc_cnt{k}'),
                    h.make_node('TopK', [c_(f'dc_cnt{k}'), i64('dc_k32', [32])], [c_(f'dc_sorted{k}'), c_(f'dc_loc64_{k}')], name=STEM + f'dc_topk{k}', axis=0, largest=1, sorted=1),
                    h.make_node('Cast', [c_(f'dc_loc64_{k}')], [c_(f'dc_loc{k}')], name=STEM + f'dc_loc{k}', to=TP.INT32),
                    h.make_node('Slice', [c_(f'dc_loc{k}'), i64('dc_0', [0]), i64('dc_16', [16]), 'dc_ax0'], [c_(f'dc_hot{k}')], name=STEM + f'dc_hot{k}'),
                    h.make_node('Slice', [c_(f'dc_loc{k}'), 'dc_16', i64('dc_32', [32]), 'dc_ax0'], [c_(f'dc_cold{k}')], name=STEM + f'dc_cold{k}'),
                    h.make_node('Add', [c_(f'dc_hot{k}'), i32(f'dc_base{k}', 32 * k)], [c_(f'dc_ghot{k}')], name=STEM + f'dc_ghot{k}'),
                    h.make_node('Add', [c_(f'dc_cold{k}'), f'dc_base{k}'], [c_(f'dc_gcold{k}')], name=STEM + f'dc_gcold{k}')]
            hot_g.append(c_(f'dc_ghot{k}')); cold_g.append(c_(f'dc_gcold{k}'))
        new += [h.make_node('Concat', hot_g, [c_('dc_order0')], name=STEM + 'dc_order0', axis=0), h.make_node('Concat', cold_g, [c_('dc_order1')], name=STEM + 'dc_order1', axis=0),
                h.make_node('Concat', [c_('dc_order0'), c_('dc_order1')], [c_('dc_order')], name=STEM + 'dc_order', axis=0),
                h.make_node('Gather', [routes_native, c_('dc_order')], [old_routes], name=STEM + 'dc_routes_pos', axis=1)]   # original chains now see lane order
        for s0, mmn in mm.items():
            bname, stage = banks[s0]
            new.append(h.make_node('Gather', [bname, c_(f'dc_order{stage}')], [c_(f'W_{s0}')], name=STEM + f'dc_gather_{s0}', axis=0)); mmn.input[1] = c_(f'W_{s0}')
        return new, set(), old_routes
    # native-order chain: clone stage 0's chain (Gather_6 -> slot table, counts, slot index) over all 128 experts
    targets0 = {STEM + 'CtxScatter3DInt_output_0', STEM + 'Einsum_1_output_0', STEM + 'Where_1_output_0'}
    targets1 = {STEM + 'CtxScatter3DInt_1_output_0', STEM + 'Einsum_2_output_0', STEM + 'Where_5_output_0'}
    def ancestors(ts):
        seen = set(); st = list(ts)
        while st:
            t = st.pop(); n = prod.get(t)
            if n is None or n.name in seen: continue
            seen.add(n.name); st.extend(n.input)
        return seen
    def descendants(t0):
        seen = set(); st = [t0]
        while st:
            t = st.pop()
            for n in cons.get(t, []):
                if n.name not in seen: seen.add(n.name); st.extend(n.output)
        return seen
    rg = {cvalue(n.input[1]): n for n in cons[STEM + 'Transpose_1_output_0'] if n.op_type == 'Gather'}      # stage routes (layer 0 numbers them 6/12, others 6/13)
    wg = {cvalue(n.input[1]): n for n in cons[STEM + 'Unsqueeze_7_output_0'] if n.op_type == 'Gather'}      # stage routing weights (7/13 or 7/14)
    assert set(rg) == {0, 1} and set(wg) == {0, 1}, (L, rg, wg)
    chain0 = ancestors(targets0) & descendants(rg[0].output[0]); chain1 = ancestors(targets1) & descendants(rg[1].output[0])
    assert STEM + 'CtxScatter3DInt' in chain0 and STEM + 'Where_1' in chain0 and STEM + 'Einsum_1' in chain0, (L, sorted(chain0)[:6])
    ren = {rg[0].output[0]: P + 'nat_routesT'}; new = []
    for n in [n for n in ln if n.name in chain0]:
        c = onnx.NodeProto(); c.CopyFrom(n); c.name = STEM + 'nat/' + n.name.replace(STEM, '')
        for i, x in enumerate(c.input):
            if x in ren: c.input[i] = ren[x]
            elif x in init and list(init[x].dims)[:1] == [64] and 'retune_zero' in x:
                z = nh.to_array(init[x]); c.input[i] = add_init(nh.from_array(np.zeros((128, z.shape[1]), z.dtype), f'nat128_zero_{z.shape[1]}'))
        for i, o in enumerate(c.output): ren[o] = P + 'nat_' + o.replace(STEM, ''); c.output[i] = ren[o]
        new.append(c)
    nat = {k: ren[STEM + k] for k in ('CtxScatter3DInt_output_0', 'Einsum_1_output_0', 'Where_1_output_0', 'Greater_output_0')}
    c_ = lambda nm: P + nm
    new += [h.make_node('Transpose', [routes_native], [c_('nat_routesT')], name=STEM + 'nat_transpose', perm=[1, 0]),
            h.make_node('Unsqueeze', [c_('nat_routesT'), i64('dc_axm1', [-1])], [c_('nat_wT')], name=STEM + 'nat_unsq'),
            h.make_node('Greater', [routes_native, add_init(nh.from_array(np.array(0, np.float16), 'dc_zero16'))], [c_('dc_routed')], name=STEM + 'dc_routed'),
            h.make_node('Cast', [c_('dc_routed')], [c_('dc_routed16')], name=STEM + 'dc_routed16', to=TP.FLOAT16),
            h.make_node('ReduceSum', [c_('dc_routed16'), i64('dc_ax0', [0])], [c_('dc_counts')], name=STEM + 'dc_counts', keepdims=0)]
    hot_g, cold_g = [], []
    for k in range(4):
        new += [h.make_node('Slice', [c_('dc_counts'), i64(f'dc_lo{k}', [32 * k]), i64(f'dc_hi{k}', [32 * k + 32]), 'dc_ax0'], [c_(f'dc_cnt{k}')], name=STEM + f'dc_cnt{k}'),
                h.make_node('TopK', [c_(f'dc_cnt{k}'), i64('dc_k32', [32])], [c_(f'dc_sorted{k}'), c_(f'dc_loc64_{k}')], name=STEM + f'dc_topk{k}', axis=0, largest=1, sorted=1),
                h.make_node('Cast', [c_(f'dc_loc64_{k}')], [c_(f'dc_loc{k}')], name=STEM + f'dc_loc{k}', to=TP.INT32),
                h.make_node('Slice', [c_(f'dc_loc{k}'), i64('dc_0', [0]), i64('dc_16', [16]), 'dc_ax0'], [c_(f'dc_hot{k}')], name=STEM + f'dc_hot{k}'),
                h.make_node('Slice', [c_(f'dc_loc{k}'), 'dc_16', i64('dc_32', [32]), 'dc_ax0'], [c_(f'dc_cold{k}')], name=STEM + f'dc_cold{k}'),
                h.make_node('Add', [c_(f'dc_hot{k}'), i32(f'dc_base{k}', 32 * k)], [c_(f'dc_ghot{k}')], name=STEM + f'dc_ghot{k}'),
                h.make_node('Add', [c_(f'dc_cold{k}'), f'dc_base{k}'], [c_(f'dc_gcold{k}')], name=STEM + f'dc_gcold{k}')]
        hot_g.append(c_(f'dc_ghot{k}')); cold_g.append(c_(f'dc_gcold{k}'))
    routes_pos = c_('routes_pos')
    new += [h.make_node('Concat', hot_g, [c_('dc_order0')], name=STEM + 'dc_order0', axis=0), h.make_node('Concat', cold_g, [c_('dc_order1')], name=STEM + 'dc_order1', axis=0),
            h.make_node('Concat', [c_('dc_order0'), c_('dc_order1')], [c_('dc_order')], name=STEM + 'dc_order', axis=0),
            h.make_node('Gather', [routes_native, c_('dc_order')], [routes_pos], name=STEM + 'dc_routes_pos', axis=1)]
    for s, names in ((0, (STEM + 'CtxScatter3DInt_output_0', STEM + 'Einsum_1_output_0', STEM + 'Where_1_output_0', wg[0].output[0], STEM + 'Greater_output_0')),
                     (1, (STEM + 'CtxScatter3DInt_1_output_0', STEM + 'Einsum_2_output_0', STEM + 'Where_5_output_0', wg[1].output[0], STEM + 'Greater_1_output_0'))):
        for srcname, dst in zip((nat['CtxScatter3DInt_output_0'], nat['Einsum_1_output_0'], nat['Where_1_output_0'], c_('nat_wT'), nat['Greater_output_0']), names):
            new.append(h.make_node('Gather', [srcname, c_(f'dc_order{s}')], [dst], name=STEM + f'dc_rows_s{s}_' + dst.replace(STEM, ''), axis=0))
    for s0, mmn in mm.items():
        bname, stage = banks[s0]; wname = mmn.input[1]
        new.append(h.make_node('Gather', [bname, c_(f'dc_order{stage}')], [c_(f'W_{s0}')], name=STEM + f'dc_gather_{s0}', axis=0)); mmn.input[1] = c_(f'W_{s0}')
    drop = chain0 | chain1 | {rg[0].name, rg[1].name, wg[0].name, wg[1].name} | {STEM + n for n in ('Reshape_1', 'Transpose_1', 'Unsqueeze_7', 'Transpose')}
    # anything else that read the old dense routes (other than the dropped Transpose) reads the position-order routes
    for n in ln:
        if n.name not in drop:
            for i, x in enumerate(n.input):
                if x == old_routes: n.input[i] = routes_pos
    return new, drop, routes_pos
def tokenowned(L, routes):
    """E15 token-owned combine on layer L with routes in position order (p = stage*64 + card*16 + local)."""
    STEM, P = f'/model/layers.{L}/mlp/', f'L{L}_'; c_ = lambda nm: P + nm
    ln, bn, prod, cons = layer_view(STEM); W = T // 16; new = []
    e4 = bn[STEM + 'Einsum_4']; assert e4.op_type == 'Einsum' and h.get_attribute_value(e4.attribute[0]) == b'dth->th'
    def addtree(src, tag, out):   # elementwise (p0 + p1) + (p2 + p3) over the card axis of src [4, rows, 2048]
        ps = []
        for c in range(4):
            new.extend([h.make_node('Slice', [src, i64(f'fa_c{c}', [c]), i64(f'fa_c{c + 1}', [c + 1]), i64('fa_axis0', [0])], [c_(f'{tag}_s{c}')], name=STEM + f'{tag}_slice_{c}'),
                        h.make_node('Squeeze', [c_(f'{tag}_s{c}'), 'fa_axis0'], [c_(f'{tag}_p{c}')], name=STEM + f'{tag}_squeeze_{c}')])
            ps.append(c_(f'{tag}_p{c}'))
        new.extend([h.make_node('Add', [ps[0], ps[1]], [c_(f'{tag}_a01')], name=STEM + f'{tag}_add01'), h.make_node('Add', [ps[2], ps[3]], [c_(f'{tag}_a23')], name=STEM + f'{tag}_add23')])
        return h.make_node('Add', [c_(f'{tag}_a01'), c_(f'{tag}_a23')], [out], name=STEM + f'{tag}_add')
    if a.final == 'addtree':
        final_concat = addtree(c_('to_partials'), 'fa', e4.output[0])
    else:
        outs = []
        for i in range(16):
            new.append(h.make_node('Slice', [c_('to_partials'), i64(f'tc_final_start_{i}_{T}', [i * W]), i64(f'tc_final_end_{i}_{T}', [(i + 1) * W]), i64('tc_final_axis', [1])], [c_(f'final_slice_{i}')], name=STEM + f'final_slice_{i}'))
            if a.final == 'einsum': new.append(h.make_node('Einsum', [c_(f'final_slice_{i}')], [c_(f'final_red_{i}')], equation='dth->th', name=STEM + f'final_tile_{i}'))
            else: new.append(addtree(c_(f'final_slice_{i}'), f'ft{i}', c_(f'final_red_{i}')))
            outs.append(c_(f'final_red_{i}'))
        final_concat = h.make_node('Concat', outs, list(e4.output), axis=0, name=STEM + 'final_concat')
    C = {0: C_HOT, 1: C_COLD}
    w7 = bn[STEM + 'Where_7']; zsrc = prod[w7.input[2]]; assert zsrc.op_type == 'ConstantOfShape'; zval = [x for x in zsrc.attribute if x.name == 'value']
    data = {0: STEM + 'Where_3_output_0', 1: c_('to_stage1_masked')}
    new += [h.make_node('Shape', [STEM + 'Mul_10_output_0'], [c_('to_stage1_shape')], name=STEM + 'to_stage1_shape'),
            h.make_node('ConstantOfShape', [c_('to_stage1_shape')], [c_('to_stage1_zeros')], name=STEM + 'to_stage1_zeros', **({'value': zval[0].t} if zval else {})),
            h.make_node('Where', [w7.input[0], STEM + 'Mul_10_output_0', c_('to_stage1_zeros')], [c_('to_stage1_masked')], name=STEM + 'to_stage1_masked'),
            h.make_node('TopK', [routes, i64('to_k8', [8])], [c_('to_vals'), c_('to_idx64')], axis=1, largest=1, sorted=1, name=STEM + 'to_topk'),
            h.make_node('Cast', [c_('to_idx64')], [c_('to_pos')], to=TP.INT32, name=STEM + 'to_pos'),
            h.make_node('Div', [c_('to_pos'), i32('to_c64', 64)], [c_('to_stage')], name=STEM + 'to_stage'), h.make_node('Mul', [c_('to_stage'), 'to_c64'], [c_('to_stage64')], name=STEM + 'to_stage64'),
            h.make_node('Sub', [c_('to_pos'), c_('to_stage64')], [c_('to_lane')], name=STEM + 'to_lane'),
            h.make_node('Div', [c_('to_lane'), i32('to_c16', 16)], [c_('to_card')], name=STEM + 'to_card'), h.make_node('Mul', [c_('to_card'), 'to_c16'], [c_('to_card16')], name=STEM + 'to_card16'),
            h.make_node('Sub', [c_('to_lane'), c_('to_card16')], [c_('to_local')], name=STEM + 'to_local')]
    slot_src = {0: STEM + 'Where_1_output_0', 1: STEM + 'Where_5_output_0'}; gathered = []
    for s_ in (0, 1):
        new += [h.make_node('Transpose', [slot_src[s_]], [c_(f'to_slotT_s{s_}')], perm=[1, 0], name=STEM + f'to_slotT_s{s_}'),
                h.make_node('GatherElements', [c_(f'to_slotT_s{s_}'), c_('to_lane')], [c_(f'to_slot_s{s_}')], axis=1, name=STEM + f'to_slot_s{s_}'),
                h.make_node('Mul', [c_('to_local'), i32(f'to_C_{C[s_]}', C[s_])], [c_(f'to_localrow_s{s_}')], name=STEM + f'to_localrow_s{s_}'),
                h.make_node('Add', [c_(f'to_localrow_s{s_}'), c_(f'to_slot_s{s_}')], [c_(f'to_row_s{s_}')], name=STEM + f'to_row_s{s_}'),
                h.make_node('Equal', [c_('to_stage'), i32(f'to_s{s_}', s_)], [c_(f'to_instage_s{s_}')], name=STEM + f'to_instage_s{s_}'),
                h.make_node('Reshape', [data[s_], i64(f'to_shape_d_{C[s_]}', [4, 16 * C[s_], 2048])], [c_(f'to_data_s{s_}')], name=STEM + f'to_data_s{s_}')]
        idx_parts, mask_parts = [], []
        for k in range(4):
            new += [h.make_node('Equal', [c_('to_card'), i32(f'to_card{k}', k)], [c_(f'to_oncard{k}_s{s_}')], name=STEM + f'to_oncard{k}_s{s_}'),
                    h.make_node('And', [c_(f'to_instage_s{s_}'), c_(f'to_oncard{k}_s{s_}')], [c_(f'to_m{k}_s{s_}')], name=STEM + f'to_m{k}_s{s_}'),
                    h.make_node('Where', [c_(f'to_m{k}_s{s_}'), c_(f'to_row_s{s_}'), i32('to_zero', 0)], [c_(f'to_idx{k}_s{s_}')], name=STEM + f'to_idx{k}_s{s_}'),
                    h.make_node('Cast', [c_(f'to_m{k}_s{s_}')], [c_(f'to_mf{k}_s{s_}')], to=TP.FLOAT16, name=STEM + f'to_mf{k}_s{s_}'),
                    h.make_node('Unsqueeze', [c_(f'to_idx{k}_s{s_}'), i64('to_axis0', [0])], [c_(f'to_idx{k}_s{s_}_u')], name=STEM + f'to_idx{k}_s{s_}_u'),
                    h.make_node('Unsqueeze', [c_(f'to_mf{k}_s{s_}'), 'to_axis0'], [c_(f'to_mf{k}_s{s_}_u')], name=STEM + f'to_mf{k}_s{s_}_u')]
            idx_parts.append(c_(f'to_idx{k}_s{s_}_u')); mask_parts.append(c_(f'to_mf{k}_s{s_}_u'))
        new += [h.make_node('Concat', idx_parts, [c_(f'to_idx4_s{s_}')], axis=0, name=STEM + f'to_idx4_s{s_}'), h.make_node('Concat', mask_parts, [c_(f'to_mf4_s{s_}')], axis=0, name=STEM + f'to_mf4_s{s_}'),
                h.make_node('Reshape', [c_(f'to_idx4_s{s_}'), i64(f'to_shape_g_{T}', [4, T * 8])], [c_(f'to_idx_s{s_}')], name=STEM + f'to_idx_s{s_}'),
                h.make_node('Reshape', [c_(f'to_mf4_s{s_}'), i64(f'to_shape_m_{T}', [4, T * 8, 1])], [c_(f'to_mask_s{s_}')], name=STEM + f'to_mask_s{s_}'),
                h.make_node('CtxGather3D', [c_(f'to_data_s{s_}'), c_(f'to_idx_s{s_}')], [c_(f'to_g_s{s_}')], name=STEM + f'to_g_s{s_}', domain=DOM),
                h.make_node('Mul', [c_(f'to_g_s{s_}'), c_(f'to_mask_s{s_}')], [c_(f'to_gm_s{s_}')], name=STEM + f'to_gm_s{s_}')]
        gathered.append(c_(f'to_gm_s{s_}'))
    new += [h.make_node('Add', gathered, [c_('to_sum')], name=STEM + 'to_sum'), h.make_node('Reshape', [c_('to_sum'), i64(f'to_shape4_{T}', [4, T, 8, 2048])], [c_('to_sum4')], name=STEM + 'to_sum4'),
            h.make_node('ReduceSum', [c_('to_sum4'), i64('to_reduce_axes', [2])], [c_('to_partials')], keepdims=0, name=STEM + 'to_partials')]
    return new + [final_concat], {e4.name}
# ---------------- apply to every layer
added = []
for L in range(48):
    STEM = f'/model/layers.{L}/mlp/'
    anchor = next(n for n in nodes if n.name == STEM + 'Transpose'); base = key[id(anchor)]
    drop = set(); routes = STEM + 'ScatterElements_output_0'
    if a.mode in ('dyncard', 'sortfirst'):
        new, d, routes = dyncard(L, sortfirst=(a.mode == 'sortfirst')); drop |= d
        for j, n in enumerate(new): key[id(n)] = base + j / (len(new) + 1)
        added += new
    new, d = tokenowned(L, routes); drop |= d
    e4pos = key[id(next(n for n in nodes if n.name == STEM + 'Einsum_4'))]
    for j, n in enumerate(new): key[id(n)] = e4pos + j / (len(new) + 1)
    added += new
    nodes = [n for n in nodes if n.name not in drop]
    if L % 8 == 0: print(f'layer {L} done; nodes {len(nodes) + len(added)}', flush=True)
nodes = nodes + added
# ---------------- prune dead nodes, stable topological order (heap on the original position)
needed = {o.name for o in g.output}; prod = {o: n for n in nodes for o in n.output}; live = set(); st = list(needed)
while st:
    t = st.pop(); n = prod.get(t)
    if n is None or id(n) in live: continue
    live.add(id(n)); st.extend(n.input)
nodes = [n for n in nodes if id(n) in live]
avail = {i.name for i in g.input} | set(init); prod = {o: n for n in nodes for o in n.output}
indeg = {id(n): sum(1 for x in n.input if x and x in prod) for n in nodes}; cons = collections.defaultdict(list)
for n in nodes:
    for x in n.input:
        if x and x in prod: cons[x].append(n)
heap = [(key[id(n)], i, n) for i, n in enumerate(nodes) if indeg[id(n)] == 0]; heapq.heapify(heap); order = []; cnt = len(nodes)
while heap:
    _, _, n = heapq.heappop(heap); order.append(n)
    for o in n.output:
        for c in cons.get(o, []):
            indeg[id(c)] -= 1
            if indeg[id(c)] == 0: cnt += 1; heapq.heappush(heap, (key[id(c)], cnt, c))
assert len(order) == len(nodes), f'topological sort failed: {len(order)} of {len(nodes)}'
del g.node[:]; g.node.extend(order)
used = {x for n in order for x in n.input}
keep = [t for t in list(g.initializer) + new_inits if t.name in used]; seen = set(); keep = [t for t in keep if not (t.name in seen or seen.add(t.name))]
del g.initializer[:]; g.initializer.extend(keep); del g.value_info[:]
for link in ('weights', 'weights_fp16', 'weights_native_fp16', 'regrouped'):
    p = f'{a.src}/{link}'
    if os.path.islink(p) and not os.path.exists(f'{a.out}/{link}'): os.symlink(os.path.realpath(p), f'{a.out}/{link}')
onnx.save(m, f'{a.out}/model.onnx'); _c = os.getcwd(); os.chdir(a.out); onnx.checker.check_model('model.onnx'); os.chdir(_c)
json.dump(dict(src=a.src, mode=a.mode, T=T, C_hot=C_HOT, C_cold=C_COLD, final=a.final), open(f'{a.out}/info.json', 'w'))
print(f'{a.out}: mode {a.mode}, final {a.final}, capacities {C_HOT}/{C_COLD}; nodes {len(order)}; initializers {len(keep)}; banks replaced {len(removed_banks)}')
