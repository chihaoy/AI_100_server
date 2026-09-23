#!/usr/bin/env python3
"""Build a layer-2 replay graph + input for an arbitrary expert placement (bank order) from the native-order banks/input.
Usage: make_perm_graph.py <base model.onnx> <order.json (128 expert ids by bank position)> <hot cap> <cold cap> <out dir> <weights file to write or reuse>"""
import sys, os, json, numpy as np, onnx
from onnx import numpy_helper as nh, TensorProto as TP
S = '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad'
base, order_path, hot, cold, out, wfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
order = np.array(json.load(open(order_path))['order']); assert sorted(order.tolist()) == list(range(128))
NAT_W = f'{S}/e4/native_weights.bin'; BANKS = [('onnx::MatMul_31532', 'onnx::MatMul_31540', (64, 2048, 768)), ('onnx::MatMul_31533', 'onnx::MatMul_31541', (64, 2048, 768)), ('onnx::MatMul_31534', 'onnx::MatMul_31542', (64, 768, 2048))]
nbytes = 64 * 2048 * 768 * 2
def native_bank(proj_index):  # [128, ...] expert-id order
    with open(NAT_W, 'rb') as f:
        f.seek(proj_index * 2 * nbytes); buf = f.read(2 * nbytes)
    return np.frombuffer(buf, np.float16).reshape((128,) + BANKS[proj_index][2][1:])
os.makedirs(out, exist_ok=True); newext = {}
if not os.path.exists(wfile):
    with open(wfile, 'wb') as w:
        for i, (n0, n1, shape) in enumerate(BANKS):
            nat = native_bank(i)
            for stage, name in ((0, n0), (1, n1)):
                arr = np.ascontiguousarray(nat[order[64 * stage:64 * (stage + 1)]]); off = w.tell(); w.write(arr.tobytes()); newext[name] = (off, arr.nbytes)
    json.dump(newext, open(wfile + '.index.json', 'w'))
else:
    newext = {k: tuple(v) for k, v in json.load(open(wfile + '.index.json')).items()}
m = onnx.load(base, load_external_data=False)
for t in m.graph.initializer:
    if t.name in newext:
        off, ln = newext[t.name]; t.ClearField('external_data'); t.data_location = TP.EXTERNAL
        for k, v in (('location', os.path.basename(wfile)), ('offset', str(off)), ('length', str(ln))): e = t.external_data.add(); e.key = k; e.value = v
    for nm, val, dt in (('probe_stage0_end', [hot], np.int64), ('probe_stage0_stop', hot, np.int32), ('probe_stage1_end', [cold], np.int64), ('probe_stage1_stop', cold, np.int32)):
        if t.name == nm: t.CopyFrom(nh.from_array(np.array(val, dt), nm))
link = os.path.join(out, os.path.basename(wfile))
if not os.path.exists(link): os.symlink(os.path.abspath(wfile), link)
onnx.save(m, f'{out}/model.onnx'); onnx.checker.check_model(f'{out}/model.onnx')
x = np.fromfile(f'{S}/e4/input_native_f16.bin', np.float16).reshape(128, 2176); xp = x.copy(); xp[:, 2048:] = x[:, 2048 + order]; xp.tofile(f'{out}/input_f16.bin')
cn = np.fromfile(f'{S}/e4/expected_counts_native_i32.bin', np.int32); cp = cn[order].astype(np.int32); cp.tofile(f'{out}/expected_counts_i32.bin')
assert np.array_equal((xp[:, 2048:] > 0).sum(0), cp)
g0, g1 = cp[:64].max(), cp[64:].max(); card_active = [int(((cp[16*c:16*c+16] > 0).sum() + (cp[64+16*c:64+16*c+16] > 0).sum())) for c in range(4)]
print(f'{out}: group maxima {g0}/{g1} (caps {hot}/{cold}), active experts per card {card_active}, per stage {[int((cp[:64]>0).sum()), int((cp[64:]>0).sum())]}')
