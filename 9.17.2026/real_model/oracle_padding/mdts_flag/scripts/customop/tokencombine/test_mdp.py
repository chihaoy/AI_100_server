#!/usr/bin/env python3
"""E49 step 1: does a custom combine op compile and run inside the four-SoC expert-parallel program?
Builds a small graph shaped like one MoE stage and its combine, in two forms that differ only in the combine:
  custom : y = MatMul(x[64, R, H], W[64, H, 2048]) -> Reshape [4, 16R, 2048] -> TokenCombine(y, idx[4, T*k]) -> [4, T, 2048] -> (p0+p1)+(p2+p3)
  graph  : the same, with the token-owned graph combine (CtxGather3D on [4, 16R, 2048] with a safe index, mask, ReduceSum over k)
writes model.onnx, inputs and the NumPy reference output into <out dir>/<form>/, and compiles each with the flags of the real
programs (16 cores per SoC, the four-SoC partition, -mdts-mos=1). Usage: test_mdp.py <out dir> [--R 64] [--H 256] [--T 128] [--compile]"""
import os, sys, json, argparse, subprocess, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--R', type=int, default=64); ap.add_argument('--H', type=int, default=256)
ap.add_argument('--T', type=int, default=128); ap.add_argument('--k', type=int, default=8); ap.add_argument('--compile', action='store_true'); a = ap.parse_args()
PKG = os.path.dirname(os.path.abspath(__file__)); C = '/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag/full_model/configs'
R, H, T, k, D = a.R, a.H, a.T, a.k, 2048; rng = np.random.default_rng(0)
CTXG3D = next(f for f in onnx.load(f'{C}/../trunc2_512_e45Eos/model.onnx', load_external_data=False).functions if f.name == 'CtxGather3D')
x = (rng.standard_normal((64, R, H)) * 0.1).astype(np.float16); W = (rng.standard_normal((64, H, D)) * 0.05).astype(np.float16)
# routing: each token picks k distinct experts (lanes) and a row in each; lane e lives on card e // 16 at local lane e % 16
idx = np.full((4, T * k), -1, np.int32); picks = np.zeros((T, k, 2), np.int64)
for t in range(T):
    es = rng.choice(64, k, replace=False); rows = rng.integers(0, R, k)
    for j, (e, r) in enumerate(zip(es, rows)): idx[e // 16, t * k + j] = (e % 16) * R + r; picks[t, j] = (e, r)
y = np.einsum('erh,ehd->erd', x.astype(np.float32), W.astype(np.float32))
ref = np.stack([y[picks[t, :, 0], picks[t, :, 1]].sum(0) for t in range(T)]).astype(np.float32)
def build(form):
    nodes, inits = [], [nh.from_array(W, 'W'), nh.from_array(np.array([4, 16 * R, D], np.int64), 'shape4')]
    nodes += [h.make_node('MatMul', ['x', 'W'], ['y'], name='stage_matmul'), h.make_node('Reshape', ['y', 'shape4'], ['y4'], name='to_card_blocks')]
    if form == 'custom':
        nodes.append(h.make_node('TokenCombine', ['y4', 'idx'], ['part'], domain='QAic', k=k, name='token_combine'))
    else:
        inits += [nh.from_array(np.array(0, np.int32), 'zero'), nh.from_array(np.array([4, T, k, D], np.int64), 'shape_sum'), nh.from_array(np.array([2], np.int64), 'axis2'),
                  nh.from_array(np.array(-1, np.int32), 'neg1'), nh.from_array(np.array([-1], np.int64), 'axm1')]
        nodes += [h.make_node('Greater', ['idx', 'neg1'], ['valid'], name='valid'), h.make_node('Where', ['valid', 'idx', 'zero'], ['safe'], name='safe'),
                  h.make_node('Cast', ['valid'], ['maskh'], to=TP.FLOAT16, name='maskh'), h.make_node('Unsqueeze', ['maskh', 'axm1'], ['mask'], name='mask'),
                  h.make_node('CtxGather3D', ['y4', 'safe'], ['g'], domain='com.qualcomm.cloud', name='gather'), h.make_node('Mul', ['g', 'mask'], ['gm'], name='masked'),
                  h.make_node('Reshape', ['gm', 'shape_sum'], ['g4'], name='per_token'), h.make_node('ReduceSum', ['g4', 'axis2'], ['part'], keepdims=0, name='sum_k')]
    for c in range(4):
        inits += [nh.from_array(np.array([c], np.int64), f'c{c}'), nh.from_array(np.array([c + 1], np.int64), f'c{c + 1}x')] if c == 0 else [nh.from_array(np.array([c + 1], np.int64), f'c{c + 1}x')]
    inits.append(nh.from_array(np.array([0], np.int64), 'ax0'))
    ps = []
    for c in range(4):
        lo = 'c0' if c == 0 else f'c{c}x'
        nodes += [h.make_node('Slice', ['part', lo, f'c{c + 1}x', 'ax0'], [f's{c}'], name=f'slice{c}'), h.make_node('Squeeze', [f's{c}', 'ax0'], [f'p{c}'], name=f'squeeze{c}')]; ps.append(f'p{c}')
    nodes += [h.make_node('Add', ['p0', 'p1'], ['a01'], name='add01'), h.make_node('Add', ['p2', 'p3'], ['a23'], name='add23'), h.make_node('Add', ['a01', 'a23'], ['out'], name='add')]
    g = h.make_graph(nodes, f'tc_{form}', [h.make_tensor_value_info('x', TP.FLOAT16, [64, R, H]), h.make_tensor_value_info('idx', TP.INT32, [4, T * k])],
                     [h.make_tensor_value_info('out', TP.FLOAT16, [T, D])], inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid('', 17), h.make_opsetid('QAic', 1), h.make_opsetid('com.qualcomm.cloud', 1)]); m.ir_version = 8
    if form == 'graph': m.functions.append(CTXG3D)   # the export defines CtxGather3D as a local function the compiler knows
    d = f'{a.out}/{form}'; os.makedirs(d, exist_ok=True); onnx.save(m, f'{d}/model.onnx')
    x.tofile(f'{d}/x.raw'); idx.tofile(f'{d}/idx.raw'); ref.tofile(f'{d}/reference_out_f32.raw')
    json.dump({'IO-files': [[{'path': f'{d}/x.raw', 'io-direction': 'in', 'elem-size': 2, 'map-to': 'x'}, {'path': f'{d}/idx.raw', 'io-direction': 'in', 'elem-size': 4, 'map-to': 'idx'}]]},
              open(f'{d}/io.json', 'w'), indent=1)
    if a.compile:
        cmd = ['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', f'-m={d}/model.onnx', '-aic-num-cores=16', '-mos=1', '-aic-enable-depth-first',
               f'-mdp-load-partition-config={C}/mdp_ts_4.json', '-mdts-mos=1', f'-aic-binary-dir={d}/qpc']
        if form == 'custom': cmd.append(f'-register-custom-op={PKG}/custom_op_config.yaml')
        subprocess.run(['rm', '-rf', f'{d}/qpc']); r = subprocess.run(cmd, capture_output=True, text=True, cwd=PKG)
        open(f'{d}/compile.log', 'w').write(r.stdout + r.stderr)
        print(f'{form}: compile rc={r.returncode}, program {"present" if os.path.exists(f"{d}/qpc/programqpc.bin") else "missing"}')
        if r.returncode: print('\n'.join(l for l in (r.stdout + r.stderr).splitlines() if l.strip())[-1500:])
for form in ('custom', 'graph'): build(form)
print(f'wrote {a.out}: R={R} H={H} T={T} k={k}; reference |out| = {np.linalg.norm(ref):.3f}')
