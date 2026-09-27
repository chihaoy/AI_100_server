#!/usr/bin/env python3
"""Tiered capacities for the per-card runtime sort (E34), all 48 layers of the full model.
The per-card sort of build_full_dyncard.py --mode dyncard ranks each card's 32 experts by token count at run time. Instead
of 16 hot lanes at capacity T and 16 cold lanes, tier i takes the next L_i ranks of every card (L_i lanes per card) at
capacity C_i; every expert keeps exactly one lane. Tier 0 is the original stage-0 subgraph fed by the first ranks; every
further tier is a clone of that subgraph with its own row order, bank gathers and capacity (Slice end and Range stop); the
unused stage-1 subgraph is pruned. The token-owned combine is generalized to any number of tiers, and the per-layer
routing-counts output lists the tiers' lanes in order (still 128 per layer). Lanes per card per tier should be 1, 2, 4, 8
or 16 (even core mapping, E19) and sum to 32. Capacities come from scripts/tier_budget.py.
Input: a build_full_stack.py output with --rewrites 1 --combine dense for the same --T.
Usage: build_full_tiered.py <src dir> <out dir> --T 512 --tiers 8x512,8x128,16x64 [--final einsum|addtree|tileadd]
       [--placement <[48,128] expert-to-card npy>]"""
import sys, os, re, json, heapq, argparse, collections, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--T', type=int, required=True)
ap.add_argument('--tiers', required=True, help='comma list of <lanes per card>x<capacity>, e.g. 8x512,8x128,16x64')
ap.add_argument('--final', choices=['einsum', 'addtree', 'tileadd'], default='einsum')
ap.add_argument('--placement', default=None, help='[48,128] expert -> card assignment (32 per card); default native e // 32')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
T, DOM = a.T, 'com.qualcomm.cloud'
TIERS = [tuple(int(v) for v in s.split('x')) for s in a.tiers.split(',')]
assert sum(l for l, _ in TIERS) == 32 and all(l in (1, 2, 4, 8, 16) for l, _ in TIERS) and all(0 < c <= T for _, c in TIERS), TIERS
NT = len(TIERS); START = [sum(l for l, _ in TIERS[:i]) for i in range(NT)]; OFF = [4 * s for s in START]
PLACE = np.load(a.placement).astype(np.int64) if a.placement else np.tile(np.arange(128) // 32, (48, 1))
assert all((np.bincount(PLACE[L], minlength=4) == 32).all() for L in range(48))
m = onnx.load(f'{a.src}/model.onnx', load_external_data=False); g = m.graph
nodes = list(g.node); init = {t.name: t for t in g.initializer}; new_inits = []
key = {id(n): 10 * i for i, n in enumerate(nodes)}
def add_init(t):
    if t.name not in init: init[t.name] = t; new_inits.append(t)
    return t.name
i64 = lambda nm, v: add_init(nh.from_array(np.array(v, np.int64), nm)); i32 = lambda nm, v: add_init(nh.from_array(np.array(v, np.int32), nm))
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
def ancestors(ts, prod):
    seen = set(); st = list(ts)
    while st:
        t = st.pop(); n = prod.get(t)
        if n is None or n.name in seen: continue
        seen.add(n.name); st.extend(n.input)
    return seen
def descendants(ts, cons):
    seen = set(); st = list(ts)
    while st:
        t = st.pop()
        for n in cons.get(t, []):
            if n.name not in seen: seen.add(n.name); st.extend(n.output)
    return seen
count_tensors = {}
def resize_lane_consts(n, lanes):
    """Stage constants folded at export for 64 lanes (layer 0 has a [64, 1] zeros input to Max): repeat a row-uniform one for `lanes`."""
    for i, x in enumerate(n.input):
        v = cval.get(x)
        if v is None and x in init and init[x].data_location != TP.EXTERNAL and len(init[x].dims) and init[x].dims[0] == 64: v = nh.to_array(init[x])
        if v is None: continue
        v = np.asarray(v)
        if v.ndim >= 1 and v.shape[0] == 64 and lanes != 64:
            assert (v == v[:1]).all(), (n.name, x, 'lane constant is not row-uniform')
            n.input[i] = add_init(nh.from_array(np.repeat(v[:1], lanes, axis=0), f'tr_l{lanes}_' + re.sub(r'[^A-Za-z0-9_]', '_', x)))
def tiered(L):
    """Per-card runtime sort with tiered capacities on layer L; returns (new nodes, dropped names, tiers' data/slot/count tensors)."""
    STEM, P = f'/model/layers.{L}/mlp/', f'L{L}_'; c_ = lambda nm: P + nm
    ln, bn, prod, cons = layer_view(STEM)
    mm = {k: bn[STEM + k] for k in ('MatMul', 'MatMul_1', 'MatMul_2', 'MatMul_3', 'MatMul_4', 'MatMul_5')}
    banks = {}
    for mat, (s0, s1) in (('gate', ('MatMul', 'MatMul_3')), ('up', ('MatMul_1', 'MatMul_4')), ('down', ('MatMul_2', 'MatMul_5'))):
        t0, t1 = init[mm[s0].input[1]], init[mm[s1].input[1]]
        e0, e1 = ({x.key: x.value for x in t.external_data} for t in (t0, t1))
        assert e0['location'] == e1['location'] and int(e1['offset']) == int(e0['offset']) + int(e0['length']), (L, mat, e0, e1)
        tb = TP(); tb.name = P + f'bank_{mat}'; tb.data_type = TP.FLOAT16; tb.dims.extend([128] + list(t0.dims)[1:]); tb.data_location = TP.EXTERNAL
        for k, v in (('location', e0['location']), ('offset', e0['offset']), ('length', str(2 * int(e0['length'])))): x = tb.external_data.add(); x.key = k; x.value = v
        add_init(tb); banks[s0] = tb.name
    sc = bn[STEM + 'ScatterElements']; routes_native = P + 'routes_native'; old_routes = sc.output[0]; sc.output[0] = routes_native
    # native-order chain (as build_full_dyncard.py): clone stage 0's chain over all 128 experts
    targets0 = {STEM + 'CtxScatter3DInt_output_0', STEM + 'Einsum_1_output_0', STEM + 'Where_1_output_0'}
    targets1 = {STEM + 'CtxScatter3DInt_1_output_0', STEM + 'Einsum_2_output_0', STEM + 'Where_5_output_0'}
    rg = {cvalue(n.input[1]): n for n in cons[STEM + 'Transpose_1_output_0'] if n.op_type == 'Gather'}
    wg = {cvalue(n.input[1]): n for n in cons[STEM + 'Unsqueeze_7_output_0'] if n.op_type == 'Gather'}
    assert set(rg) == {0, 1} and set(wg) == {0, 1}, (L, rg, wg)
    chain0 = ancestors(targets0, prod) & descendants([rg[0].output[0]], cons); chain1 = ancestors(targets1, prod) & descendants([rg[1].output[0]], cons)
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
    new += [h.make_node('Transpose', [routes_native], [c_('nat_routesT')], name=STEM + 'nat_transpose', perm=[1, 0]),
            h.make_node('Unsqueeze', [c_('nat_routesT'), i64('dc_axm1', [-1])], [c_('nat_wT')], name=STEM + 'nat_unsq'),
            h.make_node('Greater', [routes_native, add_init(nh.from_array(np.array(0, np.float16), 'dc_zero16'))], [c_('dc_routed')], name=STEM + 'dc_routed'),
            h.make_node('Cast', [c_('dc_routed')], [c_('dc_routed16')], name=STEM + 'dc_routed16', to=TP.FLOAT16),
            h.make_node('ReduceSum', [c_('dc_routed16'), i64('dc_ax0', [0])], [c_('dc_counts')], name=STEM + 'dc_counts', keepdims=0)]
    # per-card sort over the card's 32 experts (placement order), then tier slices of the ranks
    perm = np.argsort(PLACE[L], kind='stable').astype(np.int32)          # experts grouped by card
    native = (perm == np.arange(128)).all()
    counts_in = c_('dc_counts')
    if not native:
        new.append(h.make_node('Gather', [c_('dc_counts'), i32(P + 'tr_perm', perm)], [c_('tr_counts_perm')], name=STEM + 'tr_counts_perm', axis=0)); counts_in = c_('tr_counts_perm')
    tier_parts = collections.defaultdict(list)
    for k in range(4):
        new += [h.make_node('Slice', [counts_in, i64(f'dc_lo{k}', [32 * k]), i64(f'dc_hi{k}', [32 * k + 32]), i64('dc_ax0', [0])], [c_(f'dc_cnt{k}')], name=STEM + f'dc_cnt{k}'),
                h.make_node('TopK', [c_(f'dc_cnt{k}'), i64('dc_k32', [32])], [c_(f'dc_sorted{k}'), c_(f'dc_loc64_{k}')], name=STEM + f'dc_topk{k}', axis=0, largest=1, sorted=1),
                h.make_node('Cast', [c_(f'dc_loc64_{k}')], [c_(f'dc_loc{k}')], name=STEM + f'dc_loc{k}', to=TP.INT32),
                h.make_node('Add', [c_(f'dc_loc{k}'), i32(f'dc_base{k}', 32 * k)], [c_(f'dc_gloc{k}')], name=STEM + f'dc_gloc{k}')]
        glob_k = c_(f'dc_gloc{k}')
        if not native:
            new.append(h.make_node('Gather', [P + 'tr_perm', c_(f'dc_gloc{k}')], [c_(f'tr_gid{k}')], name=STEM + f'tr_gid{k}', axis=0)); glob_k = c_(f'tr_gid{k}')
        for t, (lanes, _) in enumerate(TIERS):
            new.append(h.make_node('Slice', [glob_k, i64(f'tr_s{START[t]}', [START[t]]), i64(f'tr_s{START[t] + lanes}', [START[t] + lanes]), i64('dc_ax0', [0])],
                                   [c_(f'tr_order{t}_c{k}')], name=STEM + f'tr_order{t}_c{k}'))
            tier_parts[t].append(c_(f'tr_order{t}_c{k}'))
    orders = []
    for t in range(NT):
        new.append(h.make_node('Concat', tier_parts[t], [c_(f'tr_order{t}')], name=STEM + f'tr_order{t}', axis=0)); orders.append(c_(f'tr_order{t}'))
    # tier 0: the original stage-0 subgraph fed by rows of the native chain in tier-0 order
    stage_in = {STEM + 'CtxScatter3DInt_output_0': nat['CtxScatter3DInt_output_0'], STEM + 'Einsum_1_output_0': nat['Einsum_1_output_0'],
                STEM + 'Where_1_output_0': nat['Where_1_output_0'], wg[0].output[0]: c_('nat_wT'), STEM + 'Greater_output_0': nat['Greater_output_0']}
    out0 = STEM + 'Where_3_output_0'
    body = (ancestors([out0], prod) & descendants(list(stage_in) + [mm[k].input[1] for k in ('MatMul', 'MatMul_1', 'MatMul_2')], cons)) - chain0
    body_nodes = [n for n in ln if n.name in body]
    for dst, srcname in stage_in.items():
        new.append(h.make_node('Gather', [srcname, orders[0]], [dst], name=STEM + 'tr0_rows_' + dst.replace(STEM, ''), axis=0))
    for s0 in ('MatMul', 'MatMul_1', 'MatMul_2'):
        new.append(h.make_node('Gather', [banks[s0], orders[0]], [c_(f'W0_{s0}')], name=STEM + f'tr0_gather_{s0}', axis=0)); mm[s0].input[1] = c_(f'W0_{s0}')
    slice0, range0 = bn[STEM + 'Slice'], bn[STEM + 'Range_1']; assert slice0.op_type == 'Slice' and range0.op_type == 'Range' and slice0.name in body and range0.name in body
    if TIERS[0][1] < T: slice0.input[2] = i64(P + 'tr0_end', [TIERS[0][1]]); range0.input[1] = i32(P + 'tr0_stop', TIERS[0][1])
    datas, slots, counts = [out0], [STEM + 'Where_1_output_0'], [STEM + 'Einsum_1_output_0']
    # tiers 1..: clones of the stage-0 subgraph
    for t in range(1, NT):
        tag = f'tr{t}'; rmap = {}
        for dst, srcname in stage_in.items():
            rmap[dst] = c_(f'{tag}_' + dst.replace(STEM, ''))
            new.append(h.make_node('Gather', [srcname, orders[t]], [rmap[dst]], name=STEM + f'{tag}_rows_' + dst.replace(STEM, ''), axis=0))
        wmap = {}
        for s0 in ('MatMul', 'MatMul_1', 'MatMul_2'):
            wmap[c_(f'W0_{s0}')] = c_(f'W{t}_{s0}')
            new.append(h.make_node('Gather', [banks[s0], orders[t]], [c_(f'W{t}_{s0}')], name=STEM + f'{tag}_gather_{s0}', axis=0))
        for n in body_nodes:
            c = onnx.NodeProto(); c.CopyFrom(n); c.name = STEM + f'{tag}/' + n.name.replace(STEM, '')
            if n.op_type == 'If':   # static-shape guard 'squeeze the token table if its last dim is 1': the table is [lanes, T, 1], take the then branch
                then_g = next(x.g for x in n.attribute if x.name == 'then_branch')
                sq = next(x for x in then_g.node if x.op_type == 'Squeeze'); ax_node = next(x for x in then_g.node if x.op_type == 'Constant' and x.output[0] == sq.input[1])
                axes = nh.to_array(h.get_attribute_value(ax_node.attribute[0])).astype(np.int64).reshape(-1).tolist()
                rmap[n.output[0]] = c_(f'{tag}_' + n.output[0].replace(STEM, ''))
                new.append(h.make_node('Squeeze', [rmap[STEM + 'CtxScatter3DInt_output_0'], i64('tr_sq_axes_' + '_'.join(map(str, axes)).replace('-', 'm'), axes)],
                                       [rmap[n.output[0]]], name=STEM + f'{tag}/' + n.name.replace(STEM, ''))); continue
            for i, x in enumerate(c.input):
                if x in rmap: c.input[i] = rmap[x]
                elif x in wmap: c.input[i] = wmap[x]
            for i, o in enumerate(c.output): rmap[o] = c_(f'{tag}_' + o.replace(STEM, '')); c.output[i] = rmap[o]
            if n.name == slice0.name: c.input[2] = i64(P + f'{tag}_end', [TIERS[t][1]])
            if n.name == range0.name: c.input[1] = i32(P + f'{tag}_stop', TIERS[t][1])
            resize_lane_consts(c, 4 * TIERS[t][0]); new.append(c)
        datas.append(rmap[out0]); slots.append(rmap[STEM + 'Where_1_output_0']); counts.append(rmap[STEM + 'Einsum_1_output_0'])
    for n in body_nodes: resize_lane_consts(n, 4 * TIERS[0][0])   # after the clones, which copy the untouched 64-lane constants
    drop = chain0 | chain1 | {rg[0].name, rg[1].name, wg[0].name, wg[1].name} | {STEM + n for n in ('Reshape_1', 'Transpose_1', 'Unsqueeze_7', 'Transpose')}
    count_tensors[L] = counts
    return new, drop, routes_native, orders, datas, slots
def tokenowned(L, routes_native, orders, datas, slots):
    """Token-owned combine over NT tiers: position p = OFF[tier] + card * L_tier + local, row = local * C_tier + slot."""
    STEM, P = f'/model/layers.{L}/mlp/', f'L{L}_'; c_ = lambda nm: P + nm
    ln, bn, prod, cons = layer_view(STEM); W = T // 16; new = []
    e4 = bn[STEM + 'Einsum_4']; assert e4.op_type == 'Einsum' and h.get_attribute_value(e4.attribute[0]) == b'dth->th'
    new += [h.make_node('Concat', orders, [c_('to_order_all')], name=STEM + 'to_order_all', axis=0),
            h.make_node('Gather', [routes_native, c_('to_order_all')], [c_('to_routes_pos')], name=STEM + 'to_routes_pos', axis=1),
            h.make_node('TopK', [c_('to_routes_pos'), i64('to_k8', [8])], [c_('to_vals'), c_('to_idx64')], axis=1, largest=1, sorted=1, name=STEM + 'to_topk'),
            h.make_node('Cast', [c_('to_idx64')], [c_('to_pos')], to=TP.INT32, name=STEM + 'to_pos')]
    tier = None
    for t in range(1, NT):   # tier index = number of tier offsets <= pos
        new += [h.make_node('Greater', [c_('to_pos'), i32(f'to_offm1_{OFF[t]}', OFF[t] - 1)], [c_(f'to_ge{t}')], name=STEM + f'to_ge{t}'),
                h.make_node('Cast', [c_(f'to_ge{t}')], [c_(f'to_ge{t}_i')], to=TP.INT32, name=STEM + f'to_ge{t}_i')]
        if tier is None: tier = c_(f'to_ge{t}_i')
        else: new.append(h.make_node('Add', [tier, c_(f'to_ge{t}_i')], [c_(f'to_tier{t}')], name=STEM + f'to_tier{t}')); tier = c_(f'to_tier{t}')
    if tier is None: tier = c_('to_tier0'); new.append(h.make_node('Mul', [c_('to_pos'), i32('to_zero', 0)], [tier], name=STEM + 'to_tier0'))
    tabs = ('offs', 'lanes', 'caps'); vals = (OFF, [l for l, _ in TIERS], [c for _, c in TIERS])
    tabn = '_'.join(f'{l}x{c}' for l, c in TIERS)
    for nm, v in zip(tabs, vals):
        new.append(h.make_node('Gather', [i32(f'to_tab_{nm}_{tabn}', v), tier], [c_(f'to_{nm}')], name=STEM + f'to_{nm}', axis=0))
    new += [h.make_node('Sub', [c_('to_pos'), c_('to_offs')], [c_('to_lane')], name=STEM + 'to_lane'),
            h.make_node('Div', [c_('to_lane'), c_('to_lanes')], [c_('to_card')], name=STEM + 'to_card'),
            h.make_node('Mul', [c_('to_card'), c_('to_lanes')], [c_('to_cardL')], name=STEM + 'to_cardL'),
            h.make_node('Sub', [c_('to_lane'), c_('to_cardL')], [c_('to_local')], name=STEM + 'to_local'),
            h.make_node('Concat', slots, [c_('to_slot_all')], name=STEM + 'to_slot_all', axis=0),                  # [128, T] in position order
            h.make_node('Transpose', [c_('to_slot_all')], [c_('to_slotT')], perm=[1, 0], name=STEM + 'to_slotT'),
            h.make_node('GatherElements', [c_('to_slotT'), c_('to_pos')], [c_('to_slot')], axis=1, name=STEM + 'to_slot'),
            h.make_node('Mul', [c_('to_local'), c_('to_caps')], [c_('to_localrow')], name=STEM + 'to_localrow'),
            h.make_node('Add', [c_('to_localrow'), c_('to_slot')], [c_('to_row')], name=STEM + 'to_row')]
    gathered = []
    for t, (lanes, cap) in enumerate(TIERS):
        new += [h.make_node('Equal', [tier, i32(f'to_t{t}', t)], [c_(f'to_in{t}')], name=STEM + f'to_in{t}'),
                h.make_node('Reshape', [datas[t], i64(f'to_shape_d_{lanes}_{cap}', [4, lanes * cap, 2048])], [c_(f'to_data{t}')], name=STEM + f'to_data{t}')]
        idx_parts, mask_parts = [], []
        for k in range(4):
            new += [h.make_node('Equal', [c_('to_card'), i32(f'to_card{k}', k)], [c_(f'to_on{k}_t{t}')], name=STEM + f'to_on{k}_t{t}'),
                    h.make_node('And', [c_(f'to_in{t}'), c_(f'to_on{k}_t{t}')], [c_(f'to_m{k}_t{t}')], name=STEM + f'to_m{k}_t{t}'),
                    h.make_node('Where', [c_(f'to_m{k}_t{t}'), c_('to_row'), i32('to_zero', 0)], [c_(f'to_idx{k}_t{t}')], name=STEM + f'to_idx{k}_t{t}'),
                    h.make_node('Cast', [c_(f'to_m{k}_t{t}')], [c_(f'to_mf{k}_t{t}')], to=TP.FLOAT16, name=STEM + f'to_mf{k}_t{t}'),
                    h.make_node('Unsqueeze', [c_(f'to_idx{k}_t{t}'), i64('to_axis0', [0])], [c_(f'to_idx{k}_t{t}_u')], name=STEM + f'to_idx{k}_t{t}_u'),
                    h.make_node('Unsqueeze', [c_(f'to_mf{k}_t{t}'), 'to_axis0'], [c_(f'to_mf{k}_t{t}_u')], name=STEM + f'to_mf{k}_t{t}_u')]
            idx_parts.append(c_(f'to_idx{k}_t{t}_u')); mask_parts.append(c_(f'to_mf{k}_t{t}_u'))
        new += [h.make_node('Concat', idx_parts, [c_(f'to_idx4_t{t}')], axis=0, name=STEM + f'to_idx4_t{t}'), h.make_node('Concat', mask_parts, [c_(f'to_mf4_t{t}')], axis=0, name=STEM + f'to_mf4_t{t}'),
                h.make_node('Reshape', [c_(f'to_idx4_t{t}'), i64(f'to_shape_g_{T}', [4, T * 8])], [c_(f'to_idxr_t{t}')], name=STEM + f'to_idxr_t{t}'),
                h.make_node('Reshape', [c_(f'to_mf4_t{t}'), i64(f'to_shape_m_{T}', [4, T * 8, 1])], [c_(f'to_mask_t{t}')], name=STEM + f'to_mask_t{t}'),
                h.make_node('CtxGather3D', [c_(f'to_data{t}'), c_(f'to_idxr_t{t}')], [c_(f'to_g_t{t}')], name=STEM + f'to_g_t{t}', domain=DOM),
                h.make_node('Mul', [c_(f'to_g_t{t}'), c_(f'to_mask_t{t}')], [c_(f'to_gm_t{t}')], name=STEM + f'to_gm_t{t}')]
        gathered.append(c_(f'to_gm_t{t}'))
    acc = gathered[0]
    for t in range(1, NT):
        new.append(h.make_node('Add', [acc, gathered[t]], [c_(f'to_sum{t}')], name=STEM + f'to_sum{t}')); acc = c_(f'to_sum{t}')
    new += [h.make_node('Reshape', [acc, i64(f'to_shape4_{T}', [4, T, 8, 2048])], [c_('to_sum4')], name=STEM + 'to_sum4'),
            h.make_node('ReduceSum', [c_('to_sum4'), i64('to_reduce_axes', [2])], [c_('to_partials')], keepdims=0, name=STEM + 'to_partials')]
    def addtree(src, tag, out):
        ps = []
        for c in range(4):
            new.extend([h.make_node('Slice', [src, i64(f'fa_c{c}', [c]), i64(f'fa_c{c + 1}', [c + 1]), i64('fa_axis0', [0])], [c_(f'{tag}_s{c}')], name=STEM + f'{tag}_slice_{c}'),
                        h.make_node('Squeeze', [c_(f'{tag}_s{c}'), 'fa_axis0'], [c_(f'{tag}_p{c}')], name=STEM + f'{tag}_squeeze_{c}')])
            ps.append(c_(f'{tag}_p{c}'))
        new.extend([h.make_node('Add', [ps[0], ps[1]], [c_(f'{tag}_a01')], name=STEM + f'{tag}_add01'), h.make_node('Add', [ps[2], ps[3]], [c_(f'{tag}_a23')], name=STEM + f'{tag}_add23')])
        return h.make_node('Add', [c_(f'{tag}_a01'), c_(f'{tag}_a23')], [out], name=STEM + f'{tag}_add')
    if a.final == 'addtree': final = addtree(c_('to_partials'), 'fa', e4.output[0])
    else:
        outs = []
        for i in range(16):
            new.append(h.make_node('Slice', [c_('to_partials'), i64(f'tc_final_start_{i}_{T}', [i * W]), i64(f'tc_final_end_{i}_{T}', [(i + 1) * W]), i64('tc_final_axis', [1])], [c_(f'final_slice_{i}')], name=STEM + f'final_slice_{i}'))
            if a.final == 'einsum': new.append(h.make_node('Einsum', [c_(f'final_slice_{i}')], [c_(f'final_red_{i}')], equation='dth->th', name=STEM + f'final_tile_{i}'))
            else: new.append(addtree(c_(f'final_slice_{i}'), f'ft{i}', c_(f'final_red_{i}')))
            outs.append(c_(f'final_red_{i}'))
        final = h.make_node('Concat', outs, list(e4.output), axis=0, name=STEM + 'final_concat')
    return new + [final], {e4.name}
# ---------------- apply to every layer
added = []
for L in range(48):
    STEM = f'/model/layers.{L}/mlp/'
    anchor = next(n for n in nodes if n.name == STEM + 'Transpose'); base = key[id(anchor)]
    new, drop, routes_native, orders, datas, slots = tiered(L)
    for j, n in enumerate(new): key[id(n)] = base + j / (len(new) + 1)
    added += new
    new2, d2 = tokenowned(L, routes_native, orders, datas, slots); drop |= d2
    e4pos = key[id(next(n for n in nodes if n.name == STEM + 'Einsum_4'))]
    for j, n in enumerate(new2): key[id(n)] = e4pos + j / (len(new2) + 1)
    added += new2
    nodes = [n for n in nodes if n.name not in drop]
    if L % 8 == 0: print(f'layer {L} done; nodes {len(nodes) + len(added)}', flush=True)
nodes = nodes + added
# routing-counts output: per layer the tiers' per-lane counts in lane order (was stage 0 then stage 1)
cc = next(n for n in nodes if n.name == 'oracle_count_concat'); ins = []
for L in range(48): ins += count_tensors[L]
del cc.input[:]; cc.input.extend(ins)
# ---------------- prune dead nodes, stable topological order (heap on the original position)
needed = {o.name for o in g.output}; prod = {o: n for n in nodes for o in n.output}; live = set(); st = list(needed)
while st:
    t = st.pop(); n = prod.get(t)
    if n is None or id(n) in live: continue
    live.add(id(n)); st.extend(n.input)
nodes = [n for n in nodes if id(n) in live]
prod = {o: n for n in nodes for o in n.output}
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
json.dump(dict(src=a.src, mode='tiered', T=T, tiers=TIERS, final=a.final, placement=a.placement), open(f'{a.out}/info.json', 'w'))
print(f'{a.out}: tiered {TIERS}, final {a.final}, placement {"native" if not a.placement else a.placement}; nodes {len(order)}; initializers {len(keep)}')
