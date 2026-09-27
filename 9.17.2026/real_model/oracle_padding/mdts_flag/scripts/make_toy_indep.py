#!/usr/bin/env python3
"""Partition probe: per card c, W_c = Gather(bank_c [32, 2048, 768], idx_c [16]) from that card's quarter of the real
layer-2 gate bank; h_c = MatMul(Expand(x) [16, T, 2048], W_c); r_c = ReduceSum over lanes; y = r_0 + r_1 + r_2 + r_3.
mode 'percard': four per-card constants; mode 'single': one [128, ...] constant and card-major global indices.
Usage: make_toy.py <out dir> <percard|single> [T]"""
import sys, os, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
S = '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad'
out, mode = sys.argv[1], sys.argv[2]; T = int(sys.argv[3]) if len(sys.argv) > 3 else 128; os.makedirs(out, exist_ok=True)
per_card = 32 * 2048 * 768 * 2
if not os.path.exists(f'{out}/native_weights.bin'): os.symlink(f'{S}/e4/native_weights.bin', f'{out}/native_weights.bin')
def ext(name, shape, off, ln):
    t = TP(); t.name = name; t.data_type = TP.FLOAT16; t.dims.extend(shape); t.data_location = TP.EXTERNAL
    for k, v in (('location', 'native_weights.bin'), ('offset', str(off)), ('length', str(ln))): e = t.external_data.add(); e.key = k; e.value = v
    return t
i64 = lambda n, v: nh.from_array(np.array(v, np.int64), n)
inits = [i64('c_ax0', [0]), i64('c_sh16', [16, 1, 1])] + [i64(f'c_lo{c}', [16 * c]) for c in range(4)] + [i64(f'c_hi{c}', [16 * c + 16]) for c in range(4)]
nodes = []
if mode == 'percard': inits += [ext(f'bank_c{c}', [32, 2048, 768], c * per_card, per_card) for c in range(4)]
else: inits += [ext('bank_all', [128, 2048, 768], 0, 4 * per_card)]
parts = []
for c in range(4):
    P = f'/toy/c{c}/'
    nodes += [h.make_node('Cast', ['idx_in'], [f'idx32_c{c}'], name=P + 'cast', to=TP.INT32), h.make_node('Unsqueeze', ['x', 'c_ax0'], [f'xu_c{c}'], name=P + 'unsq'),
              h.make_node('Slice', [f'idx32_c{c}', f'c_lo{c}', f'c_hi{c}', 'c_ax0'], [f'idx_c{c}'], name=P + 'slice_idx'),
              h.make_node('Gather', [f'bank_c{c}' if mode == 'percard' else 'bank_all', f'idx_c{c}'], [f'W_c{c}'], name=P + 'gather', axis=0),
              h.make_node('Expand', [f'xu_c{c}', 'c_sh16'], [f'xe_c{c}'], name=P + 'expand'),
              h.make_node('MatMul', [f'xe_c{c}', f'W_c{c}'], [f'h_c{c}'], name=P + 'matmul'),
              h.make_node('ReduceSum', [f'h_c{c}', 'c_ax0'], [f'r_c{c}'], name=P + 'reduce', keepdims=0)]
    parts.append(f'r_c{c}')

g = h.make_graph(nodes, 'toy', [h.make_tensor_value_info('x', TP.FLOAT16, [T, 2048]), h.make_tensor_value_info('idx_in', TP.INT64, [64])],
                 [h.make_tensor_value_info(f'r_c{c}', TP.FLOAT16, [T, 768]) for c in range(4)], initializer=inits)
m = h.make_model(g, opset_imports=[h.make_opsetid('', 17)]); m.ir_version = 8
_c = os.getcwd(); os.chdir(out); onnx.checker.check_model(m); onnx.save(m, 'model.onnx'); os.chdir(_c)
# inputs + fp32 reference (local index j on card c = expert 32c + j; 'single' mode gets the global index)
rng = np.random.default_rng(1); loc = np.stack([rng.permutation(32)[:16] for _ in range(4)])
idx = (loc if mode == 'percard' else loc + 32 * np.arange(4)[:, None]).reshape(64).astype(np.int64); idx.tofile(f'{out}/idx_i64.bin')
x = np.fromfile(f'{S}/e4/input_native_f16.bin', np.float16).reshape(128, 2176)[:, :2048]; x = np.tile(x, (T // 128, 1)); x.tofile(f'{out}/x_f16.bin')
bank = np.memmap(f'{S}/e4/native_weights.bin', np.float16, 'r', shape=(128, 2048, 768))
experts = (loc + 32 * np.arange(4)[:, None]).reshape(64)
y = x.astype(np.float32) @ bank[experts].astype(np.float32).sum(0); y.astype(np.float32).tofile(f'{out}/y_ref_f32.bin')
print(f'{out}: mode {mode}, T={T}, experts per card {[list(map(int, experts[16*c:16*c+16]))[:4] for c in range(4)]}...')
