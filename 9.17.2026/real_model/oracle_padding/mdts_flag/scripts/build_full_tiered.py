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
A tier spec <E>x<C>s<S> splits each of the tier's E experts per card over S adjacent lanes of C/S rows (E*S lanes per card):
the token table [4E, C] is reshaped to [4E*S, C/S], each lane gets its own count, weights and routing-weight rows are
repeated per lane; the combine's row index is unchanged because the flat layout is the same.
--combine dense keeps the export's combine instead (the padding-only arm of the ablation): input is the export itself
(native_c128, no rewrites); per group of consecutive tiers with equal lanes per card one zeroed [lanes, T, 2048]
accumulator, each tier reads it at its token table (the export's zero read, or stage 1's read-modify-write), adds its
weighted rows and scatters them back, then per group the export's un-tiled lane Einsum 'dpth->dth', the groups' partials
added, and the export's card-rooted Einsum_4. With a single group (e.g. 16xT,16xT/8) this is the export's structure.
--blocks B replaces the tiers by fixed-size block scheduling (experiment 2 of RANKTIER_EXPERIMENT_PLAN.md): each card
packs its local assignments into N blocks of B rows, one expert per block, in native expert order; block metadata (expert
of each block slot, block offsets) is computed in the graph from the routing counts, each block gathers its expert's
weights from the bank by that runtime index, and the token-owned combine reads an assignment at row M_e * B + slot
(M_e the expert's first block on its card). The default budget N per card is the dropless bound for the worst local load:
min(ceil(8T/B) + 31, 32 * ceil(T/B)), rounded up to a multiple of 16 (even core mapping); --block-budget overrides it with
one integer or a [48] npy of per-layer budgets (calibrated or trace-specialized budgets).
--capacity-inputs makes the capacities of tiers 1.. run-time selectable (experiment 4a, overflow fallback): each such tier
t reads its capacity from the length of an int32 input cap_rows<t> [cap<t>] (values 0..cap-1, which replace the tier's row
Range), and an int32 input cap_tag [ptag] of zeros (its length identifies the profile, as in runtime_constraints/) enters
as a zero offset; a network-specialization file then lists one entry per capacity profile, so one program holds e.g. the
RankTier, a full-capacity and an undersized profile over the same weights and the same retained KV cache.
--direct-gather replaces each stage's activation read, CtxGather3D(Expand(hidden [1, T, 2048]) to [lanes, T, 2048], tokens),
by Gather(hidden [T, 2048], min(tokens, T - 1)) on axis 0 (padded rows read a valid token and stay masked).
Usage: build_full_tiered.py <src dir> <out dir> --T 512 --tiers 8x512,8x128,16x64 [--final einsum|addtree|tileadd]
       [--placement <[48,128] expert-to-card npy>] [--combine tokenowned|dense] [--blocks B [--block-budget N|npy]]"""
import sys, os, re, json, heapq, argparse, collections, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--T', type=int, required=True)
ap.add_argument('--tiers', required=True, help='comma list of <lanes per card>x<capacity>, e.g. 8x512,8x128,16x64')
ap.add_argument('--final', choices=['einsum', 'addtree', 'tileadd'], default='einsum')
ap.add_argument('--placement', default=None, help='[48,128] expert -> card assignment (32 per card); default native e // 32')
ap.add_argument('--combine', choices=['tokenowned', 'dense'], default='tokenowned', help='token-owned combine, or the export combine per tier group (see above)')
ap.add_argument('--order', choices=['rank', 'identity'], default='rank', help='rank: per-card sort by load (TopK); identity: each card keeps its experts in '
                'fixed order through the same gathers, the index made runtime data (min(position_ids, 0)) so the gathers are not folded')
ap.add_argument('--direct-gather', action='store_true', help='stage activations by Gather from [T, 2048] instead of CtxGather3D on the expanded hidden states')
ap.add_argument('--capacity-inputs', action='store_true', help='capacities of tiers 1.. from input lengths (see above)')
ap.add_argument('--blocks', type=int, default=None, help='fixed-size block scheduling with blocks of this many rows (see above)')
ap.add_argument('--block-pad', choices=['clamp', 'invalid'], default='clamp', help='rows past a block\'s count: clamp (the next tokens of the expert\'s list, masked) '
                'or invalid (an appended INT32_MAX token-table entry, the export\'s marker for an empty position, as in RankTier\'s padded rows)')
ap.add_argument('--block-budget', default=None, help='blocks per card: an integer, or a [48] npy of per-layer budgets; default the dropless bound')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
T, DOM = a.T, 'com.qualcomm.cloud'
def parse_tier(spec):
    mt = re.fullmatch(r'(\d+)x(\d+)(?:s(\d+))?', spec); assert mt, spec
    return int(mt.group(1)), int(mt.group(2)), int(mt.group(3) or 1)
BLOCK = a.blocks
if BLOCK:   # one tier over all 32 experts of a card, in native order through the identity-order gathers
    assert T % BLOCK == 0 and a.combine == 'tokenowned' and a.placement is None, 'blocks need B | T, the token-owned combine and native placement'
    a.tiers, a.order = f'32x{BLOCK}', 'identity'
    if a.block_budget is None: BUDGET = [-(-min(-(-8 * T // BLOCK) + 31, 32 * (T // BLOCK)) // 16) * 16] * 48
    elif a.block_budget.endswith('.npy'): BUDGET = [int(v) for v in np.load(a.block_budget)]
    else: BUDGET = [int(a.block_budget)] * 48
    assert len(BUDGET) == 48 and all(v >= 1 for v in BUDGET), BUDGET
    block_info = {}
TIERS3 = [parse_tier(x) for x in a.tiers.split(',')]
CAPIN = a.capacity_inputs
assert not CAPIN or (not BLOCK and a.combine == 'tokenowned' and all(sp == 1 for _, _, sp in TIERS3) and TIERS3[0][1] == T), 'capacity inputs: token-owned, unsplit tiers, tier 0 at T'
assert sum(e for e, _, _ in TIERS3) == 32 and (BLOCK or all(e * sp in (1, 2, 4, 8, 16) and c % sp == 0 and 0 < c <= T for e, c, sp in TIERS3)), TIERS3
TIERS = [(e, c) for e, c, _ in TIERS3]; SPLIT = [sp for _, _, sp in TIERS3]
assert a.combine == 'tokenowned' or all(sp == 1 for sp in SPLIT), 'split tiers need the token-owned combine'
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
count_tensors = {}; dense_info = {}
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
def block_meta(L, P, STEM, c_, counts128):
    """Per card: blocks per expert m = ceil(n / B), first block M = exclusive prefix sum of m, and for every block slot b < N
    its expert e (the number of experts whose blocks end at or before b, capped at 31), chunk j = b - M_e and row count
    clip(n_e - j * B, 0, B); unused slots fall on the last expert with count 0. Returns (nodes, global expert id per lane
    [4N], lane counts [4N], token-table row index [4N, B] into the flat [128 * T] native table, M as [128])."""
    B, N = BLOCK, BUDGET[L]; nd = []
    f32 = lambda nm, v: add_init(nh.from_array(np.array(v, np.float32), nm))
    upper = np.triu(np.ones((32, 32), np.float32), 1)                    # upper[i, j] = 1 for i < j: M_j = sum_{i<j} m_i
    nd += [h.make_node('Reshape', [counts128, i64('bk_shape_4_32', [4, 32])], [c_('bk_cnt')], name=STEM + 'bk_cnt'),
           h.make_node('Add', [c_('bk_cnt'), i32(f'bk_bm1_{B}', B - 1)], [c_('bk_cntp')], name=STEM + 'bk_cntp'),
           h.make_node('Div', [c_('bk_cntp'), i32(f'bk_b_{B}', B)], [c_('bk_m')], name=STEM + 'bk_m'),
           h.make_node('Cast', [c_('bk_m')], [c_('bk_mf')], to=TP.FLOAT, name=STEM + 'bk_mf'),
           h.make_node('MatMul', [c_('bk_mf'), f32('bk_upper32', upper)], [c_('bk_Mf')], name=STEM + 'bk_Mf'),
           h.make_node('Cast', [c_('bk_Mf')], [c_('bk_M')], to=TP.INT32, name=STEM + 'bk_M'),
           h.make_node('Add', [c_('bk_M'), c_('bk_m')], [c_('bk_end')], name=STEM + 'bk_end'),
           h.make_node('Unsqueeze', [c_('bk_end'), i64('bk_ax1', [1])], [c_('bk_end_u')], name=STEM + 'bk_end_u'),
           h.make_node('Less', [c_('bk_end_u'), i32(f'bk_slots_p1_{N}', np.arange(1, N + 1).reshape(1, N, 1))], [c_('bk_done')], name=STEM + 'bk_done'),   # [4, N, 32]: expert e's blocks end at or before slot b
           h.make_node('Cast', [c_('bk_done')], [c_('bk_done_f')], to=TP.FLOAT, name=STEM + 'bk_done_f'),
           h.make_node('ReduceSum', [c_('bk_done_f'), i64('bk_ax2', [2])], [c_('bk_ef')], keepdims=0, name=STEM + 'bk_ef'),
           h.make_node('Cast', [c_('bk_ef')], [c_('bk_e_raw')], to=TP.INT32, name=STEM + 'bk_e_raw'),
           h.make_node('Min', [c_('bk_e_raw'), i32('bk_31', 31)], [c_('bk_e')], name=STEM + 'bk_e'),
           h.make_node('GatherElements', [c_('bk_M'), c_('bk_e')], [c_('bk_Mb')], axis=1, name=STEM + 'bk_Mb'),
           h.make_node('Sub', [i32(f'bk_slots2_{N}', np.arange(N).reshape(1, N)), c_('bk_Mb')], [c_('bk_j')], name=STEM + 'bk_j'),
           h.make_node('GatherElements', [c_('bk_cnt'), c_('bk_e')], [c_('bk_nb')], axis=1, name=STEM + 'bk_nb'),
           h.make_node('Mul', [c_('bk_j'), f'bk_b_{B}'], [c_('bk_jB')], name=STEM + 'bk_jB'),
           h.make_node('Sub', [c_('bk_nb'), c_('bk_jB')], [c_('bk_rem')], name=STEM + 'bk_rem'),
           h.make_node('Max', [c_('bk_rem'), i32('bk_zero', 0)], [c_('bk_rem0')], name=STEM + 'bk_rem0'),
           h.make_node('Min', [c_('bk_rem0'), f'bk_b_{B}'], [c_('bk_count4')], name=STEM + 'bk_count4'),
           h.make_node('Reshape', [c_('bk_count4'), i64(f'bk_shape_{4 * N}', [4 * N])], [c_('bk_count')], name=STEM + 'bk_count'),
           h.make_node('Add', [c_('bk_e'), i32('bk_card_base', np.array([[0], [32], [64], [96]]))], [c_('bk_g4')], name=STEM + 'bk_g4'),
           h.make_node('Reshape', [c_('bk_g4'), f'bk_shape_{4 * N}'], [c_('bk_gid')], name=STEM + 'bk_gid'),
           h.make_node('Mul', [c_('bk_g4'), i32(f'bk_T_{T}', T)], [c_('bk_gT')], name=STEM + 'bk_gT'),
           h.make_node('Unsqueeze', [c_('bk_gT'), i64('bk_axm1', [-1])], [c_('bk_gT_u')], name=STEM + 'bk_gT_u'),
           h.make_node('Unsqueeze', [c_('bk_jB'), 'bk_axm1'], [c_('bk_jB_u')], name=STEM + 'bk_jB_u'),
           h.make_node('Add', [c_('bk_jB_u'), i32(f'bk_rows_{B}', np.arange(B).reshape(1, 1, B))], [c_('bk_pos')], name=STEM + 'bk_pos'),
           h.make_node('Min', [c_('bk_pos'), i32(f'bk_Tm1_{T}', T - 1)], [c_('bk_pos_c')], name=STEM + 'bk_pos_c'),
           *([h.make_node('Add', [c_('bk_gT_u'), c_('bk_pos_c')], [c_('bk_tidx4')], name=STEM + 'bk_tidx4')] if a.block_pad == 'clamp' else
             [h.make_node('Add', [c_('bk_gT_u'), c_('bk_pos_c')], [c_('bk_tidx4v')], name=STEM + 'bk_tidx4v'),
              h.make_node('Unsqueeze', [c_('bk_count4'), 'bk_axm1'], [c_('bk_count4_u')], name=STEM + 'bk_count4_u'),
              h.make_node('Less', [f'bk_rows_{B}', c_('bk_count4_u')], [c_('bk_rowvalid')], name=STEM + 'bk_rowvalid'),     # [4, N, B]
              h.make_node('Where', [c_('bk_rowvalid'), c_('bk_tidx4v'), i32(f'bk_invalid_row_{T}', 128 * T)], [c_('bk_tidx4')], name=STEM + 'bk_tidx4')]),
           h.make_node('Reshape', [c_('bk_tidx4'), i64(f'bk_shape_{4 * N}_{B}', [4 * N, B])], [c_('bk_tidx')], name=STEM + 'bk_tidx'),
           h.make_node('Reshape', [c_('bk_M'), i64('bk_shape_128', [128])], [c_('bk_Mflat')], name=STEM + 'bk_Mflat')]
    return nd, c_('bk_gid'), c_('bk_count'), c_('bk_tidx'), c_('bk_Mflat')
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
        if a.order == 'identity':   # the card's experts in fixed order, as runtime data
            new.append(h.make_node('Add', [i32(f'id_arange{k}', np.arange(32 * k, 32 * k + 32)), 'id_runtime_zero'], [c_(f'dc_gloc{k}')], name=STEM + f'dc_gloc{k}'))
        else:
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
    # stage subgraph: the nodes between the stage-0 inputs and its output Where_3
    stage_in = {STEM + 'CtxScatter3DInt_output_0': nat['CtxScatter3DInt_output_0'], STEM + 'Einsum_1_output_0': nat['Einsum_1_output_0'],
                STEM + 'Where_1_output_0': nat['Where_1_output_0'], wg[0].output[0]: c_('nat_wT'), STEM + 'Greater_output_0': nat['Greater_output_0']}
    out0 = STEM + 'Where_3_output_0'
    body = (ancestors([out0], prod) & descendants(list(stage_in) + [mm[k].input[1] for k in ('MatMul', 'MatMul_1', 'MatMul_2')], cons)) - chain0
    body_nodes = [n for n in ln if n.name in body]
    slice0, range0 = bn[STEM + 'Slice'], bn[STEM + 'Range_1']; assert slice0.op_type == 'Slice' and range0.op_type == 'Range' and slice0.name in body and range0.name in body
    datas, slots, counts = [], [], []; tts, zreads = [], []
    if a.direct_gather:
        dg_hidden = c_('dg_hidden'); x_in = bn[STEM + 'Expand_2'].input[0]
        new.append(h.make_node('Squeeze', [x_in, i64('dg_ax0', [0])], [dg_hidden], name=STEM + 'dg_hidden'))
    orig0 = SPLIT[0] == 1 and not BLOCK
    if orig0 and a.direct_gather:   # tier 0 keeps the export's nodes: turn its CtxGather3D into the Gather in place
        g0 = bn[STEM + 'CtxGather3D']; idx0 = c_('tr0_dg_idx')
        new.append(h.make_node('Min', [STEM + 'Slice_output_0', i32(f'dg_Tm1_{T}', T - 1)], [idx0], name=STEM + 'tr0_dg_idx'))
        g0.op_type = 'Gather'; g0.domain = ''; del g0.input[:]; g0.input.extend([dg_hidden, idx0]); del g0.attribute[:]; g0.attribute.extend([h.make_attribute('axis', 0)])
    if orig0:   # tier 0: the original stage-0 subgraph fed by rows of the native chain in tier-0 order
        for dst, srcname in stage_in.items():
            new.append(h.make_node('Gather', [srcname, orders[0]], [dst], name=STEM + 'tr0_rows_' + dst.replace(STEM, ''), axis=0))
        for s0 in ('MatMul', 'MatMul_1', 'MatMul_2'):
            new.append(h.make_node('Gather', [banks[s0], orders[0]], [c_(f'W0_{s0}')], name=STEM + f'tr0_gather_{s0}', axis=0)); mm[s0].input[1] = c_(f'W0_{s0}')
        if TIERS[0][1] < T: slice0.input[2] = i64(P + 'tr0_end', [TIERS[0][1]]); range0.input[1] = i32(P + 'tr0_stop', TIERS[0][1])
        datas.append(out0); slots.append(STEM + 'Where_1_output_0'); counts.append(STEM + 'Einsum_1_output_0')
        tts.append(STEM + 'Slice_output_0'); zreads.append(bn.get(STEM + 'CtxGather3D_2'))
    w_orig = {s0: mm[s0].input[1] for s0 in ('MatMul', 'MatMul_1', 'MatMul_2')}
    # further tiers (all tiers when tier 0 is split): clones of the stage-0 subgraph
    for t in range(1 if orig0 else 0, NT):
        tag = f'tr{t}'; rmap = {}; E, Ccap = TIERS[t]; Sp = SPLIT[t]; lanes = 4 * E * Sp
        order_lanes = orders[t]
        if BLOCK:
            bnodes, order_lanes, bk_count, bk_tidx, bk_Mflat = block_meta(L, P, STEM, c_, nat['Einsum_1_output_0']); new += bnodes; lanes = 4 * BUDGET[L]
            block_info[L] = (bk_Mflat, BUDGET[L])
            new += [h.make_node('Reshape', [nat['CtxScatter3DInt_output_0'], i64(f'bk_shape_flat_{T}', [128 * T, 1])], [c_('bk_tt_flat0' if a.block_pad == 'invalid' else 'bk_tt_flat')], name=STEM + 'bk_tt_flat'),
                    *([h.make_node('Concat', [c_('bk_tt_flat0'), i32('bk_int32max', np.array([[2147483647]]))], [c_('bk_tt_flat')], name=STEM + 'bk_tt_flat_inv', axis=0)] if a.block_pad == 'invalid' else []),
                    h.make_node('Gather', [nat['Einsum_1_output_0'], orders[t]], [c_('bk_expert_counts')], name=STEM + 'bk_expert_counts', axis=0)]
        if Sp > 1:   # lane order: each expert repeated Sp times (card-major, expert-major, half-minor)
            new += [h.make_node('Unsqueeze', [orders[t], i64('dc_axm1', [-1])], [c_(f'{tag}_ord_u')], name=STEM + f'{tag}_ord_u'),
                    h.make_node('Expand', [c_(f'{tag}_ord_u'), i64(f'tr_rep_shape_{4 * E}_{Sp}', [4 * E, Sp])], [c_(f'{tag}_ord_x')], name=STEM + f'{tag}_ord_x'),
                    h.make_node('Reshape', [c_(f'{tag}_ord_x'), i64(f'tr_lanes_shape_{lanes}', [lanes])], [c_(f'{tag}_ord_lanes')], name=STEM + f'{tag}_ord_lanes')]
            order_lanes = c_(f'{tag}_ord_lanes')
        for dst, srcname in stage_in.items():
            rmap[dst] = c_(f'{tag}_' + dst.replace(STEM, ''))
            per_lane = dst in (wg[0].output[0], STEM + 'Greater_output_0')     # rows needed once per lane (weights, lane count)
            if BLOCK and dst == STEM + 'CtxScatter3DInt_output_0':
                new.append(h.make_node('Gather', [c_('bk_tt_flat'), bk_tidx], [rmap[dst]], name=STEM + f'{tag}_block_tokens', axis=0)); continue
            if BLOCK and dst == STEM + 'Einsum_1_output_0': rmap[dst] = bk_count; continue
            if dst == STEM + 'Einsum_1_output_0' and Sp > 1:
                ce = c_(f'{tag}_expert_counts'); new.append(h.make_node('Gather', [srcname, orders[t]], [ce], name=STEM + f'{tag}_expert_counts', axis=0))
                half = Ccap // Sp
                new += [h.make_node('Unsqueeze', [ce, i64('dc_axm1', [-1])], [c_(f'{tag}_cnt_u')], name=STEM + f'{tag}_cnt_u'),
                        h.make_node('Sub', [c_(f'{tag}_cnt_u'), i32(f'tr_split_off_{Sp}_{half}', [[j * half for j in range(Sp)]])], [c_(f'{tag}_cnt_s')], name=STEM + f'{tag}_cnt_s'),
                        h.make_node('Max', [c_(f'{tag}_cnt_s'), i32('tr_zero_i32', 0)], [c_(f'{tag}_cnt_lo')], name=STEM + f'{tag}_cnt_lo'),
                        h.make_node('Min', [c_(f'{tag}_cnt_lo'), i32(f'tr_half_{half}', half)], [c_(f'{tag}_cnt_hi')], name=STEM + f'{tag}_cnt_hi'),
                        h.make_node('Reshape', [c_(f'{tag}_cnt_hi'), f'tr_lanes_shape_{lanes}'], [rmap[dst]], name=STEM + f'{tag}_lane_counts')]
                continue
            new.append(h.make_node('Gather', [srcname, order_lanes if per_lane else orders[t]], [rmap[dst]], name=STEM + f'{tag}_rows_' + dst.replace(STEM, ''), axis=0))
        wmap = {}
        for s0 in ('MatMul', 'MatMul_1', 'MatMul_2'):
            wmap[w_orig[s0]] = c_(f'W{t}_{s0}')
            new.append(h.make_node('Gather', [banks[s0], order_lanes], [c_(f'W{t}_{s0}')], name=STEM + f'{tag}_gather_{s0}', axis=0))
        zr = None
        for n in body_nodes:
            if a.direct_gather and n.name == STEM + 'CtxGather3D':   # activation read: plain Gather from the hidden states
                idx = c_(f'{tag}_dg_idx'); rmap[n.output[0]] = c_(f'{tag}_dg_x')
                new += [h.make_node('Min', [rmap[STEM + 'Slice_output_0'], i32(f'dg_Tm1_{T}', T - 1)], [idx], name=STEM + f'{tag}_dg_idx'),
                        h.make_node('Gather', [dg_hidden, idx], [rmap[n.output[0]]], name=STEM + f'{tag}_dg_x', axis=0)]
                continue
            c = onnx.NodeProto(); c.CopyFrom(n); c.name = STEM + f'{tag}/' + n.name.replace(STEM, '')
            if n.name == STEM + 'CtxGather3D_2': zr = c
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
            if n.name == slice0.name: c.input[2] = f'cap{t}_len64' if CAPIN else i64(P + f'{tag}_end', [Ccap])
            if n.name == range0.name:
                c.input[1] = i32(P + f'{tag}_stop', Ccap // Sp)
                if CAPIN: rmap[n.output[0]] = f'cap{t}_rows'   # consumers read the capacity input's rows instead
            resize_lane_consts(c, lanes); new.append(c)
            if n.name == slice0.name and Sp > 1:   # [4E, C] token table -> [4E*Sp, C/Sp] lanes
                sp_out = c_(f'{tag}_token_table_lanes')
                new.append(h.make_node('Reshape', [c.output[0], i64(f'tr_tt_shape_{lanes}_{Ccap // Sp}', [lanes, Ccap // Sp])], [sp_out], name=STEM + f'{tag}_token_table_lanes'))
                rmap[n.output[0]] = sp_out
        datas.append(rmap[out0]); slots.append(rmap[STEM + 'Where_1_output_0']); tts.append(rmap[STEM + 'Slice_output_0']); zreads.append(zr)
        counts.append(c_(f'{tag}_expert_counts') if Sp > 1 else c_('bk_expert_counts') if BLOCK else rmap[STEM + 'Einsum_1_output_0'])
    drop = chain0 | chain1 | {rg[0].name, rg[1].name, wg[0].name, wg[1].name} | {STEM + n for n in ('Reshape_1', 'Transpose_1', 'Unsqueeze_7', 'Transpose')}
    if orig0:
        for n in body_nodes: resize_lane_consts(n, 4 * TIERS[0][0])   # after the clones, which copy the untouched 64-lane constants
    count_tensors[L] = counts; dense_info[L] = (tts, zreads)
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
        new.append(h.make_node('Gather', ['cap_table' if CAPIN and nm == 'caps' else i32(f'to_tab_{nm}_{tabn}', v), tier], [c_(f'to_{nm}')], name=STEM + f'to_{nm}', axis=0))
    new += [h.make_node('Sub', [c_('to_pos'), c_('to_offs')], [c_('to_lane')], name=STEM + 'to_lane'),
            h.make_node('Div', [c_('to_lane'), c_('to_lanes')], [c_('to_card')], name=STEM + 'to_card'),
            h.make_node('Mul', [c_('to_card'), c_('to_lanes')], [c_('to_cardL')], name=STEM + 'to_cardL'),
            h.make_node('Sub', [c_('to_lane'), c_('to_cardL')], [c_('to_local')], name=STEM + 'to_local'),
            h.make_node('Concat', slots, [c_('to_slot_all')], name=STEM + 'to_slot_all', axis=0),                  # [128, T] in position order
            h.make_node('Transpose', [c_('to_slot_all')], [c_('to_slotT')], perm=[1, 0], name=STEM + 'to_slotT'),
            h.make_node('GatherElements', [c_('to_slotT'), c_('to_pos')], [c_('to_slot')], axis=1, name=STEM + 'to_slot'),
            h.make_node('Gather', [block_info[L][0], c_('to_pos')], [c_('to_blk')], name=STEM + 'to_blk', axis=0) if BLOCK else
            h.make_node('Mul', [c_('to_local'), c_('to_caps')], [c_('to_localrow')], name=STEM + 'to_localrow'),
            *([h.make_node('Mul', [c_('to_blk'), c_('to_caps')], [c_('to_localrow')], name=STEM + 'to_localrow')] if BLOCK else []),
            h.make_node('Add', [c_('to_localrow'), c_('to_slot')], [c_('to_row')], name=STEM + 'to_row')]
    gathered = []
    for t, (lanes, cap) in enumerate(TIERS):
        new += [h.make_node('Equal', [tier, i32(f'to_t{t}', t)], [c_(f'to_in{t}')], name=STEM + f'to_in{t}'),
                h.make_node('Reshape', [datas[t], f'cap{t}_dshape' if CAPIN and t > 0 else i64(f'to_shape_d_{lanes}_{cap}', [4, lanes * cap, 2048]) if not BLOCK else
                                        i64(f'to_shape_d_blk{block_info[L][1]}_{cap}', [4, block_info[L][1] * cap, 2048])], [c_(f'to_data{t}')], name=STEM + f'to_data{t}')]
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
def densecombine(L, datas):
    """The export's combine over the tiers (--combine dense); returns (nodes placed with the stages, nodes placed at Einsum_4)."""
    STEM, P = f'/model/layers.{L}/mlp/', f'L{L}_'; c_ = lambda nm: P + nm
    ln, bn, prod, cons = layer_view(STEM); tts, zreads = dense_info[L]
    assert all(z is not None and z.op_type == 'CtxGather3D' for z in zreads), 'dense combine needs the export graph (zero read present, no rewrites)'
    e4 = bn[STEM + 'Einsum_4']; assert e4.op_type == 'Einsum' and h.get_attribute_value(e4.attribute[0]) == b'dth->th'
    zval = next(x for x in bn[STEM + 'ConstantOfShape_1'].attribute if x.name == 'value').t
    head, tail, parts, t, grp = [], [], [], 0, 0
    while t < NT:
        E = TIERS[t][0]; members = [t]
        while t + len(members) < NT and TIERS[t + len(members)][0] == E: members.append(t + len(members))
        base = c_(f'dn_zeros{grp}')
        head.append(h.make_node('ConstantOfShape', [i64(f'dn_shape_{4 * E}_{T}', [4 * E, T, 2048])], [base], name=STEM + f'dn_zeros{grp}', value=zval))
        for u in members:   # first tier: the export's read of the zeroed accumulator; later tiers: stage 1's read-modify-write
            zreads[u].input[0] = base
            tail.append(h.make_node('CtxScatter3D', [base, tts[u], datas[u]], [c_(f'dn_acc{u}')], name=STEM + f'dn_scatter{u}', domain=DOM)); base = c_(f'dn_acc{u}')
        tail += [h.make_node('Reshape', [base, i64(f'dn_shape4_{E}_{T}', [4, E, T, 2048])], [c_(f'dn_acc4_{grp}')], name=STEM + f'dn_acc4_{grp}'),
                 h.make_node('Einsum', [c_(f'dn_acc4_{grp}')], [c_(f'dn_part{grp}')], equation='dpth->dth', name=STEM + f'dn_lanes{grp}')]
        parts.append(c_(f'dn_part{grp}')); t += len(members); grp += 1
    total = parts[0]
    for k in range(1, len(parts)):
        tail.append(h.make_node('Add', [total, parts[k]], [c_(f'dn_sum{k}')], name=STEM + f'dn_sum{k}')); total = c_(f'dn_sum{k}')
    e4.input[0] = total
    return head, tail
# ---------------- apply to every layer
added = []
if a.order == 'identity':   # a zero the compiler cannot fold: position ids are non-negative, but only at run time
    zero_nodes = [h.make_node('ReduceMin', ['position_ids'], ['id_pos_min'], name='id_pos_min', keepdims=0),
                  h.make_node('Min', ['id_pos_min', add_init(nh.from_array(np.array(0, np.int64), 'id_zero_i64'))], ['id_zero64'], name='id_zero64'),
                  h.make_node('Cast', ['id_zero64'], ['id_runtime_zero'], name='id_runtime_zero', to=TP.INT32)]
    for j, n in enumerate(zero_nodes): key[id(n)] = -10 + j
    added += zero_nodes
if CAPIN:   # capacity inputs: cap_rows<t> [cap<t>] int32 for tiers 1.., cap_tag [ptag] int32 zeros
    g.input.append(h.make_tensor_value_info('cap_tag', TP.INT32, ['ptag']))
    cap_nodes = [h.make_node('Gather', ['cap_tag', add_init(nh.from_array(np.array(0, np.int64), 'cap_idx0'))], ['cap_zero'], name='cap_zero', axis=0)]
    lens32 = [i32('cap_T_i32', [T])]
    for t in range(1, NT):
        g.input.append(h.make_tensor_value_info(f'cap_rows{t}', TP.INT32, [f'cap{t}']))
        cap_nodes += [h.make_node('Add', [f'cap_rows{t}', 'cap_zero'], [f'cap{t}_rows'], name=f'cap{t}_rows'),
                      h.make_node('Shape', [f'cap_rows{t}'], [f'cap{t}_len64'], name=f'cap{t}_len64'),
                      h.make_node('Cast', [f'cap{t}_len64'], [f'cap{t}_len32'], name=f'cap{t}_len32', to=TP.INT32),
                      h.make_node('Mul', [f'cap{t}_len64', i64(f'cap_lanes{t}', [TIERS[t][0]])], [f'cap{t}_lanerows'], name=f'cap{t}_lanerows'),
                      h.make_node('Concat', [i64('cap_four', [4]), f'cap{t}_lanerows', i64('cap_hidden', [2048])], [f'cap{t}_dshape'], name=f'cap{t}_dshape', axis=0)]
        lens32.append(f'cap{t}_len32')
    cap_nodes.append(h.make_node('Concat', lens32, ['cap_table'], name='cap_table', axis=0))
    for j, n in enumerate(cap_nodes): key[id(n)] = -5 + j / (len(cap_nodes) + 1)
    added += cap_nodes
for L in range(48):
    STEM = f'/model/layers.{L}/mlp/'
    anchor = next(n for n in nodes if n.name == STEM + 'Transpose'); base = key[id(anchor)]
    new, drop, routes_native, orders, datas, slots = tiered(L)
    for j, n in enumerate(new): key[id(n)] = base + j / (len(new) + 1)
    added += new
    if a.combine == 'dense':
        head, new2 = densecombine(L, datas)
        for j, n in enumerate(head): key[id(n)] = base - 0.5 + j / (len(head) + 1) * 0.1
        added += head
    else: new2, d2 = tokenowned(L, routes_native, orders, datas, slots); drop |= d2
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
for link in ('weights', 'weights_fp16', 'weights_native_fp16', 'regrouped', 'weights_cardmajor_fp16'):
    p = f'{a.src}/{link}'
    if os.path.islink(p) and not os.path.exists(f'{a.out}/{link}'): os.symlink(os.path.realpath(p), f'{a.out}/{link}')
onnx.save(m, f'{a.out}/model.onnx'); _c = os.getcwd(); os.chdir(a.out); onnx.checker.check_model('model.onnx'); os.chdir(_c)
json.dump(dict(src=a.src, mode='tiered', T=T, tiers=TIERS3, final=a.final, placement=a.placement, **({'combine': 'dense'} if a.combine == 'dense' else {}),
               **({'order': 'identity'} if a.order == 'identity' else {}), **({'blocks': BLOCK, 'block_budget': BUDGET, 'block_pad': a.block_pad} if BLOCK else {}), **({'capacity_inputs': True} if CAPIN else {}), **({'direct_gather': True} if a.direct_gather else {})), open(f'{a.out}/info.json', 'w'))
print(f'{a.out}: ' + (f'blocks of {BLOCK} rows, budget per card {sorted(set(BUDGET))}; ' if BLOCK else '') + f'tiered {TIERS3}, {"dense combine" if a.combine == "dense" else "final " + a.final}, placement {"native" if not a.placement else a.placement}; nodes {len(order)}; initializers {len(keep)}')
