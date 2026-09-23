#!/usr/bin/env python3
"""Build a T-token variant of the c2_tree layer-2 replay with separate hot/cold capacities and a chosen placement.
Same construction as make_chunk.py (tree scans and reduction tiles regenerated for T, synthetic routing realizing the
given per-expert counts with seed 0, real hidden rows tiled to T), but:
  * stage 0 capacity = C_hot, stage 1 capacity = C_cold (probe_stage{0,1}_{end,stop})
  * placement 'hc': stage 0 = the 64 largest-count experts spread round-robin over the cards, stage 1 = the rest with the
    active ones spread round-robin starting at card 3 (card 0 hosts the final combine), inactive experts fill the gaps
  * placement 'lpt': the make_chunk.py placement (LPT over active experts, alternating a card's two stages)
Usage: make_cap.py <T> <counts.npy> <out dir> <C_hot> <C_cold> <hc|lpt>"""
import sys, os, json, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
S = '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad'
BASE = '/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/routing_retune/c2_tree/model.onnx'
T, counts, out, C_HOT, C_COLD, MODE = int(sys.argv[1]), np.load(sys.argv[2]).astype(np.int64), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
os.makedirs(out, exist_ok=True)
assert counts.sum() == 8 * T and counts.max() <= T and len(counts) == 128 and MODE in ('hc', 'lpt')
ranked = [int(e) for e in np.argsort(-counts, kind='stable')]
if MODE == 'lpt':
    active = [e for e in ranked if counts[e] > 0]; empty = [e for e in ranked if counts[e] == 0]
    cards = [[] for _ in range(4)]; load = [0] * 4
    for e in active:
        c = min(range(4), key=lambda c: (load[c], -c)); cards[c].append(e); load[c] += 1
    idx = np.argsort(load, kind='stable'); cards = [cards[i] for i in idx]
    for c in range(4):
        need = 32 - len(cards[c]); cards[c] += empty[:need]; empty = empty[need:]
    order = [None] * 128
    for c in range(4):
        act = [e for e in cards[c] if counts[e] > 0]; emp = [e for e in cards[c] if counts[e] == 0]
        s0 = act[0::2]; s1 = act[1::2]; s0 += emp[:16 - len(s0)]; s1 += emp[16 - len(act[0::2]):]
        order[16 * c:16 * c + 16] = s0; order[64 + 16 * c:64 + 16 * c + 16] = s1
else:
    hot, cold = ranked[:64], ranked[64:]
    slots0 = [[] for _ in range(4)]; slots1 = [[] for _ in range(4)]
    for i, e in enumerate(hot): slots0[i % 4].append(e)
    cold_act = [e for e in cold if counts[e] > 0]; cold_emp = [e for e in cold if counts[e] == 0]
    for j, e in enumerate(cold_act): slots1[[3, 2, 1, 0][j % 4]].append(e)
    for c in range(4):
        need = 16 - len(slots1[c]); slots1[c] += cold_emp[:need]; cold_emp = cold_emp[need:]
    order = sum(slots0, []) + sum(slots1, [])
order = np.array(order); assert sorted(order.tolist()) == list(range(128))
hot_max = int(counts[order[:64]].max()); cold_max = int(counts[order[64:]].max())
assert hot_max <= C_HOT and cold_max <= C_COLD, f'capacity would drop tokens: hot max {hot_max} > {C_HOT} or cold max {cold_max} > {C_COLD}'
# ---- weights in the new bank order (from the native-order banks)
NAT = f'{S}/e4/native_weights.bin'; BANKS = [('onnx::MatMul_31532', 'onnx::MatMul_31540', (2048, 768)), ('onnx::MatMul_31533', 'onnx::MatMul_31541', (2048, 768)), ('onnx::MatMul_31534', 'onnx::MatMul_31542', (768, 2048))]
nbytes = 64 * 2048 * 768 * 2; ext = {}
with open(f'{out}/weights.bin', 'wb') as w, open(NAT, 'rb') as f:
    for i, (n0, n1, shape) in enumerate(BANKS):
        f.seek(i * 2 * nbytes); nat = np.frombuffer(f.read(2 * nbytes), np.float16).reshape((128,) + shape)
        for stage, name in ((0, n0), (1, n1)):
            arr = np.ascontiguousarray(nat[order[64 * stage:64 * (stage + 1)]]); off = w.tell(); w.write(arr.tobytes()); ext[name] = (off, arr.nbytes)
# ---- graph edits
m = onnx.load(BASE, load_external_data=False); g = m.graph; STEM = '/model/layers.2/mlp/'
g.input[0].type.tensor_type.shape.dim[0].dim_value = T
for o in g.output:
    if o.name == 'y': o.type.tensor_type.shape.dim[0].dim_value = T
CAPS = {'probe_stage0_end': ([C_HOT], np.int64), 'probe_stage0_stop': (C_HOT, np.int32), 'probe_stage1_end': ([C_COLD], np.int64), 'probe_stage1_stop': (C_COLD, np.int32)}
keep = []; seen = set()
for t in g.initializer:
    if 'retune' in t.name: continue
    if t.name in ext:
        off, ln = ext[t.name]; t.ClearField('external_data'); t.data_location = TP.EXTERNAL
        for k, v in (('location', 'weights.bin'), ('offset', str(off)), ('length', str(ln))): e = t.external_data.add(); e.key = k; e.value = v
    if t.name in CAPS: val, dt = CAPS[t.name]; t.CopyFrom(nh.from_array(np.array(val, dt), t.name)); seen.add(t.name)
    if t.name.startswith('ablation_start_'): i = int(t.name.split('_')[-1]); t.CopyFrom(nh.from_array(np.array([i * T // 16], np.int64), t.name))
    if t.name.startswith('ablation_end_'): i = int(t.name.split('_')[-1]); t.CopyFrom(nh.from_array(np.array([(i + 1) * T // 16], np.int64), t.name))
    keep.append(t)
assert seen == set(CAPS), f'capacity constants missing: {set(CAPS) - seen}'
del g.initializer[:]; g.initializer.extend(keep)
new_nodes = [n for n in g.node if 'retune' not in n.name]
def scan_chain(prefix, src, final_out):
    stem = STEM + prefix + '_retune_'; nodes = []; inits = []
    def const(name, data): inits.append(nh.from_array(np.asarray(data, np.int64), stem + name)); return stem + name
    axis, start = const('token_axis', [1]), const('start', [0]); cur = src; shifts = []
    s = 1
    while s < T: shifts.append(s); s *= 2
    for k, s in enumerate(shifts):
        zero = stem + f'zero_{s}'; inits.append(nh.from_array(np.zeros((64, s), np.int32), zero))
        sliced, shifted = stem + f'slice_{s}', stem + f'shift_{s}'; result = final_out if k == len(shifts) - 1 else stem + f'stage_{s}_out'
        nodes += [h.make_node('Slice', [cur, start, const(f'end_{s}', [T - s]), axis], [sliced], name=sliced),
                  h.make_node('Concat', [zero, sliced], [shifted], axis=1, name=shifted),
                  h.make_node('Add', [cur, shifted], [result], name=stem + f'add_stage_{s}')]
        cur = result
    return nodes, inits
final = {n.name: n.input[0] for n in new_nodes if n.op_type == 'Identity' and n.name in (STEM + 'CumSum', STEM + 'CumSum_1')}
for prefix, src in (('CumSum', STEM + 'Cast_4_output_0'), ('CumSum_1', STEM + 'Cast_16_output_0')):
    nodes, inits = scan_chain(prefix, src, final[STEM + prefix]); g.initializer.extend(inits)
    pos = next(i for i, n in enumerate(new_nodes) if n.name == STEM + prefix); new_nodes[pos:pos] = nodes
del g.node[:]; g.node.extend(new_nodes)
_cwd = os.getcwd(); os.chdir(out); onnx.checker.check_model(m); os.chdir(_cwd); onnx.save(m, f'{out}/model.onnx'); onnx.checker.check_model(f'{out}/model.onnx')
# ---- synthetic routing realizing the counts (identical token->expert assignment for a given counts vector)
rem = counts.copy(); rows = np.zeros((T, 128), np.float16); rng = np.random.default_rng(0)
for t in range(T):
    pick = np.argsort(-(rem + rng.random(128) * 1e-3))[:8]; assert (rem[pick] > 0).all(), f'infeasible at token {t}'
    rem[pick] -= 1; rows[t, pick] = np.float16(0.125)
assert (rem == 0).all() and np.array_equal((rows > 0).sum(0), counts)
x128 = np.fromfile(f'{S}/e4/input_native_f16.bin', np.float16).reshape(128, 2176)[:, :2048]
xin = np.zeros((T, 2176), np.float16); xin[:, :2048] = np.tile(x128, (T // 128, 1))[:T]; xin[:, 2048:] = rows[:, order]
xin.tofile(f'{out}/input_f16.bin'); counts[order].astype(np.int32).tofile(f'{out}/expected_counts_i32.bin')
per_stage = [[int((counts[order[64 * s + 16 * c:64 * s + 16 * c + 16]] > 0).sum()) for c in range(4)] for s in range(2)]
rows_padded = int((counts[order[:64]] > 0).sum()) * C_HOT + int((counts[order[64:]] > 0).sum()) * C_COLD
info = dict(T=T, mode=MODE, C_hot=C_HOT, C_cold=C_COLD, hot_max_count=hot_max, cold_max_count=cold_max, active_experts=int((counts > 0).sum()),
            assignments=int(counts.sum()), active_per_card=[per_stage[0][c] + per_stage[1][c] for c in range(4)], active_per_card_stage=per_stage,
            padded_rows=rows_padded, padded_over_real=round(rows_padded / int(counts.sum()), 2), weight_MiB_per_token=int((counts > 0).sum()) * 9 / T, order=order.tolist())
json.dump(info, open(f'{out}/info.json', 'w')); print({k: v for k, v in info.items() if k != 'order'})
