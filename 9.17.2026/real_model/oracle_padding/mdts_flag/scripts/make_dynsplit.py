#!/usr/bin/env python3
"""Dynamic hot/cold split: experts are sorted by their token count at run time (TopK over the counts), the 64 largest fill
the hot stage and the rest the cold stage, all six banks are gathered from the 128-expert banks by that order, and the
routing columns are permuted by it. Input routing is in native expert order. Capacities stay static (calibrated).
Usage: make_dynsplit.py <src dir (E8 hc anchor)> <out dir> <C_hot> <C_cold>"""
import sys, os, json, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
S = '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad'
src, out, C_HOT, C_COLD = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True)
m = onnx.load(f'{src}/model.onnx', load_external_data=False); g = m.graph; STEM = '/model/layers.2/mlp/'
T = g.input[0].type.tensor_type.shape.dim[0].dim_value
nbytes = 64 * 2048 * 768 * 2
BANKS = {'onnx::MatMul_31532': (0, (128, 2048, 768), 0), 'onnx::MatMul_31540': (0, (128, 2048, 768), 1), 'onnx::MatMul_31533': (2 * nbytes, (128, 2048, 768), 0), 'onnx::MatMul_31541': (2 * nbytes, (128, 2048, 768), 1),
         'onnx::MatMul_31534': (4 * nbytes, (128, 768, 2048), 0), 'onnx::MatMul_31542': (4 * nbytes, (128, 768, 2048), 1)}
if not os.path.exists(f'{out}/native_weights.bin'): os.symlink(f'{S}/e4/native_weights.bin', f'{out}/native_weights.bin')
keep = []; seen_all = {}
for t in g.initializer:
    if t.name in BANKS:
        off, shape, stage = BANKS[t.name]; key = f'bank_all_{off}'
        if key not in seen_all:
            ta = TP(); ta.name = key; ta.data_type = TP.FLOAT16; ta.dims.extend(shape); ta.data_location = TP.EXTERNAL
            for k, v in (('location', 'native_weights.bin'), ('offset', str(off)), ('length', str(2 * nbytes))): e = ta.external_data.add(); e.key = k; e.value = v
            seen_all[key] = ta
        continue
    if t.name == 'probe_stage0_end': t.CopyFrom(nh.from_array(np.array([C_HOT], np.int64), t.name))
    if t.name == 'probe_stage0_stop': t.CopyFrom(nh.from_array(np.array(C_HOT, np.int32), t.name))
    if t.name == 'probe_stage1_end': t.CopyFrom(nh.from_array(np.array([C_COLD], np.int64), t.name))
    if t.name == 'probe_stage1_stop': t.CopyFrom(nh.from_array(np.array(C_COLD, np.int32), t.name))
    keep.append(t)
del g.initializer[:]; g.initializer.extend(keep); g.initializer.extend(seen_all.values())
i64 = lambda n, v: nh.from_array(np.array(v, np.int64), n)
g.initializer.extend([i64('ds_k128', [128]), i64('ds_0', [0]), i64('ds_64', [64]), i64('ds_128', [128]), i64('ds_ax0', [0]), i64('ds_ax1', [1]), nh.from_array(np.array(0, np.float16), 'ds_zero16')])
# routes: the Slice now yields native-order routes; sort experts by count; permute routes into position order
sl = next(n for n in g.node if n.name == STEM + 'probe_input_routes' or n.name == 'probe_input_routes'); sl.output[0] = 'routes_native'; sl.name = STEM + 'probe_input_routes_native'
new = [h.make_node('Greater', ['routes_native', 'ds_zero16'], ['ds_routed'], name=STEM + 'ds_routed'),
       h.make_node('Cast', ['ds_routed'], ['ds_routed16'], name=STEM + 'ds_routed16', to=TP.FLOAT16),
       h.make_node('ReduceSum', ['ds_routed16', 'ds_ax0'], ['ds_counts'], name=STEM + 'ds_counts', keepdims=0),          # [128] fp16, exact up to 2048
       h.make_node('TopK', ['ds_counts', 'ds_k128'], ['ds_sorted_counts', 'ds_order64'], name=STEM + 'ds_topk', axis=0, largest=1, sorted=1),
       h.make_node('Cast', ['ds_order64'], ['ds_order'], name=STEM + 'ds_order', to=TP.INT32),
       h.make_node('Gather', ['routes_native', 'ds_order'], ['oracle_L2_routing'], name=STEM + 'probe_input_routes', axis=1),
       h.make_node('Slice', ['ds_order', 'ds_0', 'ds_64', 'ds_ax0'], ['ds_hot'], name=STEM + 'ds_hot'),
       h.make_node('Slice', ['ds_order', 'ds_64', 'ds_128', 'ds_ax0'], ['ds_cold'], name=STEM + 'ds_cold')]
for name, (off, shape, stage) in BANKS.items():
    new.append(h.make_node('Gather', [f'bank_all_{off}', 'ds_hot' if stage == 0 else 'ds_cold'], [name], name=STEM + f'ds_gather_{name[-5:]}', axis=0))
nodes = list(g.node); pos = next(i for i, n in enumerate(nodes) if n.name == STEM + 'probe_input_routes_native'); nodes[pos + 1:pos + 1] = new
del g.node[:]; g.node.extend(nodes)
_c = os.getcwd(); os.chdir(out); onnx.checker.check_model(m); onnx.save(m, 'model.onnx'); os.chdir(_c)
json.dump(dict(T=T, mode='dynsplit', C_hot=C_HOT, C_cold=C_COLD), open(f'{out}/info.json', 'w'))
print(f'{out}: dynamic split, T={T}, capacities {C_HOT}/{C_COLD}')
