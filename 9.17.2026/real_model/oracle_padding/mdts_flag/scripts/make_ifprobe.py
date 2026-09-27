#!/usr/bin/env python3
"""Data-dependent If probe: y = If(sel == 1) ? heavy batched MatMul (64 lanes, 192 MB fp16 weights) : cheap branch.
Input probe_input [128, 2049] fp16: columns 0..2047 = x, column 2048 (row 0) = sel. Output y [128, 768]."""
import sys, os, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
out = sys.argv[1]; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(0)
W = (rng.standard_normal((64, 2048, 768)) * 0.02).astype(np.float16); Wc = (rng.standard_normal((2048, 768)) * 0.02).astype(np.float16)
W.tofile(f'{out}/W.bin')
tW = TP(); tW.name = 'W'; tW.data_type = TP.FLOAT16; tW.dims.extend([64, 2048, 768]); tW.data_location = TP.EXTERNAL
for k, v in (('location', 'W.bin'), ('offset', '0'), ('length', str(W.nbytes))): e = tW.external_data.add(); e.key = k; e.value = v
i64 = lambda n, v: nh.from_array(np.array(v, np.int64), n)
then_g = h.make_graph([h.make_node('Expand', ['x', 'sh64'], ['xe']), h.make_node('MatMul', ['xe', 'W'], ['ye']), h.make_node('ReduceSum', ['ye', 'ax0'], ['y_then'], keepdims=0)], 'then', [], [h.make_tensor_value_info('y_then', TP.FLOAT16, [128, 768])])
else_g = h.make_graph([h.make_node('MatMul', ['x', 'Wc'], ['y_else'])], 'else', [], [h.make_tensor_value_info('y_else', TP.FLOAT16, [128, 768])])
nodes = [h.make_node('Slice', ['probe_input', 'z', 'c2048', 'ax1'], ['x']),
         h.make_node('Slice', ['probe_input', 'c2048', 'c2049', 'ax1'], ['selcol']),
         h.make_node('Slice', ['selcol', 'z', 'one', 'ax0'], ['sel1']),
         h.make_node('Reshape', ['sel1', 'sh_scalar'], ['self16']),
         h.make_node('Cast', ['self16'], ['seli'], to=TP.INT32),
         h.make_node('Equal', ['seli', 'one32'], ['cond']),
         h.make_node('If', ['cond'], ['y'], then_branch=then_g, else_branch=else_g)]
g = h.make_graph(nodes, 'ifprobe', [h.make_tensor_value_info('probe_input', TP.FLOAT16, [128, 2049])], [h.make_tensor_value_info('y', TP.FLOAT16, [128, 768])],
                 initializer=[tW, nh.from_array(Wc, 'Wc'), i64('z', [0]), i64('c2048', [2048]), i64('c2049', [2049]), i64('ax1', [1]), i64('ax0', [0]), i64('one', [1]), i64('sh64', [64, 1, 1]), i64('sh_scalar', []), nh.from_array(np.array(1, np.int32), 'one32')])
m = h.make_model(g, opset_imports=[h.make_opsetid('', 17)]); m.ir_version = 8
_c = os.getcwd(); os.chdir(out); onnx.checker.check_model(m); onnx.save(m, 'model.onnx'); os.chdir(_c)
x = (rng.standard_normal((128, 2048)) * 0.5).astype(np.float16)
for sel in (0, 1):
    inp = np.zeros((128, 2049), np.float16); inp[:, :2048] = x; inp[0, 2048] = sel; inp.tofile(f'{out}/input_sel{sel}.bin')
    y = (x.astype(np.float32) @ W.astype(np.float32).sum(0)) if sel else (x.astype(np.float32) @ Wc.astype(np.float32)); y.astype(np.float32).tofile(f'{out}/ref_sel{sel}.bin')
print('ifprobe written')
