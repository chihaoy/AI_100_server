#!/usr/bin/env python3
"""Runtime expert->lane selection probe: the hot stage's three banks become Gather(bank_all[128,...] fp16, hot_idx[64]) where
hot_idx arrives with the input (64 extra fp16 columns, row 0). The cold stage stays static. Usage: make_dyngather.py <src dir> <out dir>"""
import sys, os, json, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
S = '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad'
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
m = onnx.load(f'{src}/model.onnx', load_external_data=False); g = m.graph; STEM = '/model/layers.2/mlp/'
T = g.input[0].type.tensor_type.shape.dim[0].dim_value; NC = g.input[0].type.tensor_type.shape.dim[1].dim_value
g.input[0].type.tensor_type.shape.dim[1].dim_value = NC + 64
nbytes = 64 * 2048 * 768 * 2
BANKS = {'onnx::MatMul_31532': (0, (128, 2048, 768)), 'onnx::MatMul_31533': (2 * nbytes, (128, 2048, 768)), 'onnx::MatMul_31534': (4 * nbytes, (128, 768, 2048))}
if not os.path.exists(f'{out}/native_weights.bin'): os.symlink(f'{S}/e4/native_weights.bin', f'{out}/native_weights.bin')
if not os.path.exists(f'{out}/weights.bin'): os.symlink(os.path.realpath(f'{src}/weights.bin'), f'{out}/weights.bin')
new_inits = []
for name, (off, shape) in BANKS.items():
    t = next(t for t in g.initializer if t.name == name); g.initializer.remove(t)
    ta = TP(); ta.name = name + '_all'; ta.data_type = TP.FLOAT16; ta.dims.extend(shape); ta.data_location = TP.EXTERNAL
    for k, v in (('location', 'native_weights.bin'), ('offset', str(off)), ('length', str(2 * nbytes))): e = ta.external_data.add(); e.key = k; e.value = v
    new_inits.append(ta)
g.initializer.extend(new_inits)
i64 = lambda n, v: nh.from_array(np.array(v, np.int64), n)
g.initializer.extend([i64('dg_r0', [0]), i64('dg_r1', [1]), i64('dg_c0', [NC]), i64('dg_c1', [NC + 64]), i64('dg_ax0', [0]), i64('dg_ax1', [1]), i64('dg_sh64', [64])])
pre = [h.make_node('Slice', ['probe_input', 'dg_c0', 'dg_c1', 'dg_ax1'], ['dg_cols'], name=STEM + 'dg_cols'),
       h.make_node('Slice', ['dg_cols', 'dg_r0', 'dg_r1', 'dg_ax0'], ['dg_row'], name=STEM + 'dg_row'),
       h.make_node('Reshape', ['dg_row', 'dg_sh64'], ['dg_idx16'], name=STEM + 'dg_idx16'),
       h.make_node('Cast', ['dg_idx16'], ['hot_idx'], name=STEM + 'dg_cast', to=TP.INT32)]
for name in BANKS: pre.append(h.make_node('Gather', [name + '_all', 'hot_idx'], [name], name=STEM + f'dg_gather_{name[-5:]}', axis=0))
# the probe_input_routes slice end must stay at NC (routes are columns 2048..NC)
nodes = list(g.node); nodes[0:0] = pre; del g.node[:]; g.node.extend(nodes)
_c = os.getcwd(); os.chdir(out); onnx.checker.check_model(m); onnx.save(m, 'model.onnx'); os.chdir(_c)
order = json.load(open(f'{src}/info.json'))['order']
x = np.fromfile(f'{src}/input_f16.bin', np.float16).reshape(T, NC); xin = np.zeros((T, NC + 64), np.float16); xin[:, :NC] = x; xin[0, NC:] = np.array(order[:64], np.float16)
xin.tofile(f'{out}/input_f16.bin')
for f in ('expected_counts_i32.bin', 'info.json'):
    if not os.path.exists(f'{out}/{f}'): os.symlink(os.path.realpath(f'{src}/{f}'), f'{out}/{f}')
print(f'{out}: hot banks gathered at run time from the 128-expert fp16 banks; input [{T}, {NC + 64}]')
