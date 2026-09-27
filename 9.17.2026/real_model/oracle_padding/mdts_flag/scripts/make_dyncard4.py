#!/usr/bin/env python3
"""Per-card dynamic hot/cold split with native-order routing chains.
Each card owns experts 32c..32c+31 (four per-card bank constants per matrix). At run time each card sorts its 32 experts
by token count; its 16 largest fill its hot lanes, the other 16 its cold lanes, and the banks are gathered from the
card's own constant. The mask / prefix-scan / slot-table chain runs once over all 128 experts in native order, so it does
not wait for the sort; only the per-stage row gathers, the routing permutation and the bank gathers depend on it.
Usage: make_dyncard.py <src dir (E7/E8 hc anchor)> <out dir> <C_hot> <C_cold>"""
import sys, os, json, collections, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
S = '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad'
src, out, C_HOT, C_COLD = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True)
m = onnx.load(f'{src}/model.onnx', load_external_data=False); g = m.graph; STEM = '/model/layers.2/mlp/'
T = g.input[0].type.tensor_type.shape.dim[0].dim_value
bn = {n.name: n for n in g.node}; prod = {o: n for n in g.node for o in n.output}
init = {t.name: t for t in g.initializer}
def consumers(t): return [n for n in g.node if t in n.input]
# ---------- 1. per-card bank constants
nbytes = 64 * 2048 * 768 * 2; per_card = 32 * 2048 * 768 * 2
BANKS = {'onnx::MatMul_31532': (0, (32, 2048, 768), 0), 'onnx::MatMul_31540': (0, (32, 2048, 768), 1), 'onnx::MatMul_31533': (2 * nbytes, (32, 2048, 768), 0), 'onnx::MatMul_31541': (2 * nbytes, (32, 2048, 768), 1),
         'onnx::MatMul_31534': (4 * nbytes, (32, 768, 2048), 0), 'onnx::MatMul_31542': (4 * nbytes, (32, 768, 2048), 1)}
if not os.path.exists(f'{out}/native_weights.bin'): os.symlink(f'{S}/e4/native_weights.bin', f'{out}/native_weights.bin')
keep = []; bank_consts = {}
for t in g.initializer:
    if t.name in BANKS:
        off, shape, stage = BANKS[t.name]; key = f'bank_all_{off}'
        if key not in bank_consts:
            ta = TP(); ta.name = key; ta.data_type = TP.FLOAT16; ta.dims.extend([128, shape[1], shape[2]]); ta.data_location = TP.EXTERNAL
            for k, v in (('location', 'native_weights.bin'), ('offset', str(off)), ('length', str(4 * per_card))): e = ta.external_data.add(); e.key = k; e.value = v
            bank_consts[key] = ta
        continue
    if t.name == 'probe_stage0_end': t.CopyFrom(nh.from_array(np.array([C_HOT], np.int64), t.name))
    if t.name == 'probe_stage0_stop': t.CopyFrom(nh.from_array(np.array(C_HOT, np.int32), t.name))
    if t.name == 'probe_stage1_end': t.CopyFrom(nh.from_array(np.array([C_COLD], np.int64), t.name))
    if t.name == 'probe_stage1_stop': t.CopyFrom(nh.from_array(np.array(C_COLD, np.int32), t.name))
    keep.append(t)
del g.initializer[:]; g.initializer.extend(keep); g.initializer.extend(bank_consts.values())
i64 = lambda n, v: nh.from_array(np.array(v, np.int64), n); i32 = lambda n, v: nh.from_array(np.array(v, np.int32), n)
new_inits = [i64('dc_k32', [32]), i64('dc_0', [0]), i64('dc_16', [16]), i64('dc_32', [32]), i64('dc_ax0', [0]), i64('dc_ax1', [1]), nh.from_array(np.array(0, np.float16), 'dc_zero16')]
new_inits += [i64(f'dc_lo{c}', [32 * c]) for c in range(4)] + [i64(f'dc_hi{c}', [32 * c + 32]) for c in range(4)] + [i32(f'dc_base{c}', 32 * c) for c in range(4)]
g.initializer.extend(new_inits)
# ---------- 2. native-order routing chain: clone stage 0's chain from Gather_6_output_0 to {slot table, counts, slot index}
sl = bn.get(STEM + 'probe_input_routes') or bn['probe_input_routes']; sl.output[0] = 'routes_native'; sl.name = STEM + 'probe_input_routes_native'
targets0 = {STEM + 'CtxScatter3DInt_output_0', STEM + 'Einsum_1_output_0', STEM + 'Where_1_output_0'}
targets1 = {STEM + 'CtxScatter3DInt_1_output_0', STEM + 'Einsum_2_output_0', STEM + 'Where_5_output_0'}
def ancestors(tensors):
    seen = set(); stack = list(tensors)
    while stack:
        t = stack.pop(); n = prod.get(t)
        if n is None or n.name in seen: continue
        seen.add(n.name); stack.extend(n.input)
    return seen
def descendants(tensor):
    seen = set(); stack = [tensor]
    while stack:
        t = stack.pop()
        for n in consumers(t):
            if n.name not in seen: seen.add(n.name); stack.extend(n.output)
    return seen
chain0 = ancestors(targets0) & descendants(STEM + 'Gather_6_output_0'); chain1 = ancestors(targets1) & descendants(STEM + 'Gather_13_output_0')
assert STEM + 'CtxScatter3DInt' in chain0 and STEM + 'Where_1' in chain0 and STEM + 'Einsum_1' in chain0, sorted(chain0)[:10]
order_nodes = [n for n in g.node if n.name in chain0]          # in original topological order
ren = {STEM + 'Gather_6_output_0': 'nat_routesT'}; clone = []
for n in order_nodes:
    c = onnx.NodeProto(); c.CopyFrom(n); c.name = STEM + 'nat/' + n.name.replace(STEM, '')
    for i, x in enumerate(c.input):
        if x in ren: c.input[i] = ren[x]
        elif x in init and list(init[x].dims)[:1] == [64] and 'retune_zero' in x:
            nz = 'nat_' + x.split('/')[-1]; z = nh.to_array(init[x])
            if nz not in init: t = nh.from_array(np.zeros((128, z.shape[1]), z.dtype), nz); g.initializer.append(t); init[nz] = t
            c.input[i] = nz
    for i, o in enumerate(c.output): ren[o] = 'nat_' + o.replace(STEM, ''); c.output[i] = ren[o]
    clone.append(c)
nat_scatter, nat_counts, nat_slot, nat_mask = ren[STEM + 'CtxScatter3DInt_output_0'], ren[STEM + 'Einsum_1_output_0'], ren[STEM + 'Where_1_output_0'], ren[STEM + 'Greater_output_0']
# ---------- 3. per-card sort, permutations and bank gathers
new = [h.make_node('Transpose', ['routes_native'], ['nat_routesT'], name=STEM + 'nat_transpose', perm=[1, 0]),
       h.make_node('Unsqueeze', ['nat_routesT', 'dc_ax1'], ['nat_wT'], name=STEM + 'nat_unsq')]     # [128, T] and [128, T, 1]
# NB: Unsqueeze at axis 1 of [128,T] gives [128,1,T]; we need [128,T,1] -> use axis -1 via a separate constant
new[-1] = h.make_node('Unsqueeze', ['nat_routesT', 'dc_axm1'], ['nat_wT'], name=STEM + 'nat_unsq'); g.initializer.append(i64('dc_axm1', [-1]))
new += [h.make_node('Greater', ['routes_native', 'dc_zero16'], ['dc_routed'], name=STEM + 'dc_routed'),
        h.make_node('Cast', ['dc_routed'], ['dc_routed16'], name=STEM + 'dc_routed16', to=TP.FLOAT16),
        h.make_node('ReduceSum', ['dc_routed16', 'dc_ax0'], ['dc_counts'], name=STEM + 'dc_counts', keepdims=0)]
hot_g, cold_g = [], []
for c in range(4):
    new += [h.make_node('Slice', ['dc_counts', f'dc_lo{c}', f'dc_hi{c}', 'dc_ax0'], [f'dc_cnt{c}'], name=STEM + f'dc_cnt{c}'),
            h.make_node('TopK', [f'dc_cnt{c}', 'dc_k32'], [f'dc_sorted{c}', f'dc_loc64_{c}'], name=STEM + f'dc_topk{c}', axis=0, largest=1, sorted=1),
            h.make_node('Cast', [f'dc_loc64_{c}'], [f'dc_loc{c}'], name=STEM + f'dc_loc{c}', to=TP.INT32),
            h.make_node('Slice', [f'dc_loc{c}', 'dc_0', 'dc_16', 'dc_ax0'], [f'dc_hot{c}'], name=STEM + f'dc_hot{c}'),
            h.make_node('Slice', [f'dc_loc{c}', 'dc_16', 'dc_32', 'dc_ax0'], [f'dc_cold{c}'], name=STEM + f'dc_cold{c}'),
            h.make_node('Add', [f'dc_hot{c}', f'dc_base{c}'], [f'dc_ghot{c}'], name=STEM + f'dc_ghot{c}'),
            h.make_node('Add', [f'dc_cold{c}', f'dc_base{c}'], [f'dc_gcold{c}'], name=STEM + f'dc_gcold{c}')]
    hot_g.append(f'dc_ghot{c}'); cold_g.append(f'dc_gcold{c}')
new += [h.make_node('Concat', hot_g, ['dc_order0'], name=STEM + 'dc_order0', axis=0), h.make_node('Concat', cold_g, ['dc_order1'], name=STEM + 'dc_order1', axis=0),
        h.make_node('Concat', ['dc_order0', 'dc_order1'], ['dc_order'], name=STEM + 'dc_order', axis=0),
        h.make_node('Gather', ['routes_native', 'dc_order'], ['oracle_L2_routing'], name=STEM + 'probe_input_routes', axis=1)]
for s, ord_s, names in ((0, 'dc_order0', ('CtxScatter3DInt_output_0', 'Einsum_1_output_0', 'Where_1_output_0', 'Gather_7_output_0', 'Greater_output_0')), (1, 'dc_order1', ('CtxScatter3DInt_1_output_0', 'Einsum_2_output_0', 'Where_5_output_0', 'Gather_14_output_0', 'Greater_1_output_0'))):
    for srcname, dst in zip((nat_scatter, nat_counts, nat_slot, 'nat_wT', nat_mask), names):
        new.append(h.make_node('Gather', [srcname, ord_s], [STEM + dst], name=STEM + f'dc_rows_s{s}_{dst}', axis=0))
for name, (off, shape, stage) in BANKS.items():
    new.append(h.make_node('Gather', [f'bank_all_{off}', 'dc_order0' if stage == 0 else 'dc_order1'], [name], name=STEM + f'dc_gather_{name[-5:]}', axis=0))
# ---------- 4. assemble: drop both original chains and the stage split, add clone + new, topologically sort
drop = chain0 | chain1 | {STEM + n for n in ('Gather_6', 'Gather_13', 'Gather_7', 'Gather_14', 'Reshape_1', 'Transpose_1', 'Unsqueeze_7', 'Transpose')}
nodes = [n for n in g.node if n.name not in drop] + clone + new
avail = {i.name for i in g.input} | {t.name for t in g.initializer}; pending = list(nodes); ordered = []
while pending:
    progressed = False
    for n in list(pending):
        if all(x in avail or x == '' for x in n.input): ordered.append(n); avail.update(n.output); pending.remove(n); progressed = True
    assert progressed, 'cannot order: ' + ', '.join(f"{n.name}<-{[x for x in n.input if x not in avail]}" for n in pending[:5])
del g.node[:]; g.node.extend(ordered)
_c = os.getcwd(); os.chdir(out); onnx.checker.check_model(m); onnx.save(m, 'model.onnx'); os.chdir(_c)
json.dump(dict(T=T, mode='dyncard', C_hot=C_HOT, C_cold=C_COLD, cards='native blocks of 32'), open(f'{out}/info.json', 'w'))
print(f'{out}: per-card dynamic split, T={T}, capacities {C_HOT}/{C_COLD}, cloned chain {len(clone)} nodes, dropped {len(drop)}, total {len(ordered)}')
