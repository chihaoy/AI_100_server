#!/usr/bin/env python3
"""Hot/cold replay (as make_cap.py) with the hot experts SPLIT across lanes: a lane is (expert, row chunk of C_hot rows).
Hot experts whose count exceeds C_hot occupy ceil(count / C_hot) lanes, each holding a consecutive chunk of the expert's
tokens (chunk = position of the token among the expert's tokens // C_hot); the smallest hot experts are demoted to the cold
stage until the 64 hot lanes suffice, empty cold experts are dropped to make room, and the cold capacity is the largest
demoted or cold count. Routing is presented to the graph as 'virtual lanes': the split expert's routing column is copied
into each of its lanes masked to that lane's chunk, so the unchanged graph (mask, scan, slots, token-owned combine) sees
64 ordinary hot lanes with at most C_hot tokens each. Weight DMA per stage is unchanged (64 lanes stream one bank each).
Usage: make_split.py <T> <counts.npy> <out dir> <C_hot> [C_cold_min]"""
import sys, os, json, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
S = '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad'
BASE = '/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/routing_retune/c2_tree/model.onnx'
T, counts, out, C_HOT = int(sys.argv[1]), np.load(sys.argv[2]).astype(np.int64), sys.argv[3], int(sys.argv[4]); C_COLD_MIN = int(sys.argv[5]) if len(sys.argv) > 5 else 1
os.makedirs(out, exist_ok=True); assert counts.sum() == 8 * T and len(counts) == 128
ranked = [int(e) for e in np.argsort(-counts, kind='stable')]
hot, cold = ranked[:64], ranked[64:]
def lanes_needed(hs): return sum(int(-(-counts[e] // C_HOT)) for e in hs)
demoted = []
while lanes_needed(hot) > 64:
    e = hot.pop(); cold.append(e); demoted.append(e)          # smallest hot count goes cold
cold_act = sorted([e for e in cold if counts[e] > 0], key=lambda e: -counts[e]); cold_emp = [e for e in cold if counts[e] == 0]
assert len(cold_act) <= 64, f'{len(cold_act)} active cold experts do not fit 64 lanes'
C_COLD = max(C_COLD_MIN, int(counts[cold_act].max()) if cold_act else 1)
# hot lanes (expert, chunk), spread round-robin over cards like make_cap hc; cold active spread from card 3, empties fill
hot_lanes = [(e, k) for e in hot for k in range(int(-(-counts[e] // C_HOT)))]
while len(hot_lanes) < 64: hot_lanes.append((cold_emp.pop(), 0)) if cold_emp else hot_lanes.append((cold_act[-1], 0))
assert len(hot_lanes) == 64
s0 = [[] for _ in range(4)]
for i, l in enumerate(hot_lanes): s0[i % 4].append(l)
s1 = [[] for _ in range(4)]
for j, e in enumerate(cold_act): s1[[3, 2, 1, 0][j % 4]].append((e, 0))
dropped = []
for c in range(4):
    need = 16 - len(s1[c])
    s1[c] += [(e, 0) for e in cold_emp[:need]]; cold_emp = cold_emp[need:]
dropped = cold_emp; lanes = sum(s0, []) + sum(s1, []); assert len(lanes) == 128, len(lanes)
order = np.array([e for e, k in lanes]); chunk = np.array([k for e, k in lanes])
# ---- banks (duplicates allowed)
NAT = f'{S}/e4/native_weights.bin'; BANKS = [('onnx::MatMul_31532', 'onnx::MatMul_31540', (2048, 768)), ('onnx::MatMul_31533', 'onnx::MatMul_31541', (2048, 768)), ('onnx::MatMul_31534', 'onnx::MatMul_31542', (768, 2048))]
nbytes = 64 * 2048 * 768 * 2; ext = {}
with open(f'{out}/weights.bin', 'wb') as w, open(NAT, 'rb') as f:
    for i, (n0, n1, shape) in enumerate(BANKS):
        f.seek(i * 2 * nbytes); nat = np.frombuffer(f.read(2 * nbytes), np.float16).reshape((128,) + shape)
        for stage, name in ((0, n0), (1, n1)):
            arr = np.ascontiguousarray(nat[order[64 * stage:64 * (stage + 1)]]); off = w.tell(); w.write(arr.tobytes()); ext[name] = (off, arr.nbytes)
# ---- graph edits (identical to make_cap.py)
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
assert seen == set(CAPS)
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
_cwd = os.getcwd(); os.chdir(out); onnx.checker.check_model(m); os.chdir(_cwd); onnx.save(m, f'{out}/model.onnx')
# ---- synthetic routing (same generator as make_cap.py) and its virtual-lane presentation
rem = counts.copy(); rows = np.zeros((T, 128), np.float16); rng = np.random.default_rng(0)
for t in range(T):
    pick = np.argsort(-(rem + rng.random(128) * 1e-3))[:8]; assert (rem[pick] > 0).all()
    rem[pick] -= 1; rows[t, pick] = np.float16(0.125)
assert (rem == 0).all() and np.array_equal((rows > 0).sum(0), counts)
rank_in_expert = np.cumsum(rows > 0, axis=0) - 1                    # [T,128]: position of token t among expert e's tokens
vrows = np.zeros((T, 128), np.float16)
for p, (e, k) in enumerate(lanes):
    sel = (rows[:, e] > 0) & ((rank_in_expert[:, e] // C_HOT) == k) if p < 64 else (rows[:, e] > 0)
    vrows[:, p] = np.where(sel, rows[:, e], 0)
lane_counts = (vrows > 0).sum(0).astype(np.int32)
assert lane_counts[:64].max() <= C_HOT and lane_counts[64:].max() <= C_COLD and lane_counts.sum() == 8 * T, (lane_counts[:64].max(), lane_counts[64:].max(), lane_counts.sum())
x128 = np.fromfile(f'{S}/e4/input_native_f16.bin', np.float16).reshape(128, 2176)[:, :2048]
xin = np.zeros((T, 2176), np.float16); xin[:, :2048] = np.tile(x128, (T // 128, 1))[:T]; xin[:, 2048:] = vrows
xin.tofile(f'{out}/input_f16.bin'); lane_counts.tofile(f'{out}/expected_counts_i32.bin')
padded = 64 * C_HOT + 64 * C_COLD
info = dict(T=T, mode='split', C_hot=C_HOT, C_cold=C_COLD, hot_experts=len(hot), split_experts=sorted({int(e) for e, k in hot_lanes if k > 0}), extra_lanes=int(sum(1 for e, k in hot_lanes if k > 0)),
            demoted=[int(e) for e in demoted], demoted_counts=[int(counts[e]) for e in demoted], dropped_empty=len(dropped), cold_active=len(cold_act),
            padded_rows=padded, padded_over_real=round(padded / int(counts.sum()), 2), hot_lane_counts_max=int(lane_counts[:64].max()), order=order.tolist(), chunk=chunk.tolist())
json.dump(info, open(f'{out}/info.json', 'w')); print({k: v for k, v in info.items() if k not in ('order', 'chunk')})
