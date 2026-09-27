#!/usr/bin/env python3
"""Deployable lane-split layout: all 128 experts kept. Hot stage = 64 lanes (top experts, the largest split into chunks of
C_hot rows, the smallest hot experts demoted); cold stage = 64 + k lanes (k = extra chunk lanes, rounded up to a multiple
of 4 with duplicate empty experts), i.e. some cores run two cold lanes. Routing enters the graph as 64 + L1 virtual lanes;
stage 1 of the graph is re-batched from 64 to L1 lanes (routes slicing, scan-chain zeros, banks, counts output).
Usage: make_split_wide.py <T> <counts.npy> <out dir> <C_hot> [C_cold_min]"""
import sys, os, json, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
S = '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad'
BASE = '/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/routing_retune/c2_tree/model.onnx'
T, counts, out, C_HOT = int(sys.argv[1]), np.load(sys.argv[2]).astype(np.int64), sys.argv[3], int(sys.argv[4]); C_COLD_MIN = int(sys.argv[5]) if len(sys.argv) > 5 else 1
os.makedirs(out, exist_ok=True); assert counts.sum() == 8 * T and len(counts) == 128
ranked = [int(e) for e in np.argsort(-counts, kind='stable')]; hot, cold = ranked[:64], ranked[64:]
def lanes_needed(hs): return sum(int(-(-counts[e] // C_HOT)) for e in hs)
demoted = []
while lanes_needed(hot) > 64: e = hot.pop(); cold.append(e); demoted.append(e)
hot_lanes = [(e, k) for e in hot for k in range(int(-(-counts[e] // C_HOT)))]; assert len(hot_lanes) == 64, len(hot_lanes)
cold_act = sorted([e for e in cold if counts[e] > 0], key=lambda e: -counts[e]); cold_emp = [e for e in cold if counts[e] == 0]
L1 = int(sys.argv[6]) if len(sys.argv) > 6 else -(-len(cold) // 4) * 4; PER = L1 // 4              # cold lanes (optional override), PER per card
spare = L1 - len(cold); C_COLD = max(C_COLD_MIN, int(counts[cold_act].max()))
# hot lanes round-robin over cards (16 per card); cold: active experts first on each card (round-robin from card 3), then
# empties, then spare duplicates of an empty expert, so the lanes beyond 16 on a card are empty ones when possible
s0 = [[] for _ in range(4)]
for i, l in enumerate(hot_lanes): s0[i % 4].append(l)
s1 = [[] for _ in range(4)]
for j, e in enumerate(cold_act): s1[[3, 2, 1, 0][j % 4]].append((e, 0))
emp = list(cold_emp) + [cold_emp[0] if cold_emp else cold_act[-1]] * spare
for c in range(4):
    need = PER - len(s1[c]); assert need >= 0, f'card {c} has {len(s1[c])} active cold experts > {PER} lanes'
    s1[c] += [(e, 0) for e in emp[:need]]; emp = emp[need:]
assert not emp
lanes = sum(s0, []) + sum(s1, []); order = np.array([e for e, k in lanes]); chunk = np.array([k for e, k in lanes]); NL = 64 + L1
# ---- banks: stage 0 [64,...], stage 1 [L1,...]
NAT = f'{S}/e4/native_weights.bin'; BANKS = [('onnx::MatMul_31532', 'onnx::MatMul_31540', (2048, 768)), ('onnx::MatMul_31533', 'onnx::MatMul_31541', (2048, 768)), ('onnx::MatMul_31534', 'onnx::MatMul_31542', (768, 2048))]
nbytes = 64 * 2048 * 768 * 2; ext = {}
with open(f'{out}/weights.bin', 'wb') as w, open(NAT, 'rb') as f:
    for i, (n0, n1, shape) in enumerate(BANKS):
        f.seek(i * 2 * nbytes); nat = np.frombuffer(f.read(2 * nbytes), np.float16).reshape((128,) + shape)
        for stage, name, sl in ((0, n0, slice(0, 64)), (1, n1, slice(64, NL))):
            arr = np.ascontiguousarray(nat[order[sl]]); off = w.tell(); w.write(arr.tobytes()); ext[name] = (off, arr.nbytes, arr.shape)
# ---- graph
m = onnx.load(BASE, load_external_data=False); g = m.graph; STEM = '/model/layers.2/mlp/'
g.input[0].type.tensor_type.shape.dim[0].dim_value = T; g.input[0].type.tensor_type.shape.dim[1].dim_value = 2048 + NL
for o in g.output:
    if o.name == 'y': o.type.tensor_type.shape.dim[0].dim_value = T
    if o.name == 'counts': o.type.tensor_type.shape.dim[0].dim_value = NL
CAPS = {'probe_stage0_end': ([C_HOT], np.int64), 'probe_stage0_stop': (C_HOT, np.int32), 'probe_stage1_end': ([C_COLD], np.int64), 'probe_stage1_stop': (C_COLD, np.int32), 'probe_end': ([2048 + NL], np.int64)}
keep = []; seen = set()
for t in g.initializer:
    if 'retune' in t.name: continue
    if t.name in ext:
        off, ln, shape = ext[t.name]; t.ClearField('external_data'); t.data_location = TP.EXTERNAL; del t.dims[:]; t.dims.extend(shape)
        for k, v in (('location', 'weights.bin'), ('offset', str(off)), ('length', str(ln))): e = t.external_data.add(); e.key = k; e.value = v
    if t.name in CAPS: val, dt = CAPS[t.name]; t.CopyFrom(nh.from_array(np.array(val, dt), t.name)); seen.add(t.name)
    if t.name.startswith('ablation_start_'): i = int(t.name.split('_')[-1]); t.CopyFrom(nh.from_array(np.array([i * T // 16], np.int64), t.name))
    if t.name.startswith('ablation_end_'): i = int(t.name.split('_')[-1]); t.CopyFrom(nh.from_array(np.array([(i + 1) * T // 16], np.int64), t.name))
    keep.append(t)
assert seen == set(CAPS), set(CAPS) - seen
del g.initializer[:]; g.initializer.extend(keep)
new_nodes = [n for n in g.node if 'retune' not in n.name]
# stage routes: replace Reshape_1 [2,64,T] / Transpose_1 / Gather_6, Gather_13 (routes per stage) and Unsqueeze_7 / Gather_7, Gather_14 (weights per stage) by slices of the [NL, T] transposed routes
bn = {n.name: n for n in new_nodes}
_made = set()
def i64(name, v):
    if STEM + name not in _made: g.initializer.append(nh.from_array(np.array(v, np.int64), STEM + name)); _made.add(STEM + name)
    return STEM + name
tr = bn[STEM + 'Transpose'].output[0]                                  # [NL, T]
repl = []
for s, (a, b) in enumerate(((0, 64), (64, NL))):
    routes_s = bn[STEM + ('Gather_6' if s == 0 else 'Gather_13')].output[0]; weights_s = bn[STEM + ('Gather_7' if s == 0 else 'Gather_14')].output[0]
    repl += [h.make_node('Slice', [tr, i64(f'wide_start{s}', [a]), i64(f'wide_end{s}', [b]), i64('wide_axis0', [0])], [routes_s], name=STEM + f'wide_routes_s{s}'),
             h.make_node('Unsqueeze', [routes_s, i64('wide_axm1', [-1])], [weights_s], name=STEM + f'wide_weights_s{s}')]
drop = {STEM + n for n in ('Reshape_1', 'Transpose_1', 'Gather_6', 'Gather_13', 'Unsqueeze_7', 'Gather_7', 'Gather_14')}
pos = next(i for i, n in enumerate(new_nodes) if n.name == STEM + 'Reshape_1'); new_nodes[pos:pos] = repl
new_nodes = [n for n in new_nodes if n.name not in drop]
def scan_chain(prefix, src, final_out, L):
    stem = STEM + prefix + '_retune_'; nodes = []; inits = []
    def const(name, data): inits.append(nh.from_array(np.asarray(data, np.int64), stem + name)); return stem + name
    axis, start = const('token_axis', [1]), const('start', [0]); cur = src; shifts = []
    s = 1
    while s < T: shifts.append(s); s *= 2
    for k, s in enumerate(shifts):
        zero = stem + f'zero_{s}'; inits.append(nh.from_array(np.zeros((L, s), np.int32), zero))
        sliced, shifted = stem + f'slice_{s}', stem + f'shift_{s}'; result = final_out if k == len(shifts) - 1 else stem + f'stage_{s}_out'
        nodes += [h.make_node('Slice', [cur, start, const(f'end_{s}', [T - s]), axis], [sliced], name=sliced),
                  h.make_node('Concat', [zero, sliced], [shifted], axis=1, name=shifted),
                  h.make_node('Add', [cur, shifted], [result], name=stem + f'add_stage_{s}')]
        cur = result
    return nodes, inits
final = {n.name: n.input[0] for n in new_nodes if n.op_type == 'Identity' and n.name in (STEM + 'CumSum', STEM + 'CumSum_1')}
for prefix, src, L in (('CumSum', STEM + 'Cast_4_output_0', 64), ('CumSum_1', STEM + 'Cast_16_output_0', L1)):
    nodes, inits = scan_chain(prefix, src, final[STEM + prefix], L); g.initializer.extend(inits)
    p = next(i for i, n in enumerate(new_nodes) if n.name == STEM + prefix); new_nodes[p:p] = nodes
del g.node[:]; g.node.extend(new_nodes)
_cwd = os.getcwd(); os.chdir(out); onnx.checker.check_model(m); os.chdir(_cwd); onnx.save(m, f'{out}/model.onnx')
# ---- routing (same generator as make_cap) -> virtual lanes over NL positions
rem = counts.copy(); rows = np.zeros((T, 128), np.float16); rng = np.random.default_rng(0)
for t in range(T):
    pick = np.argsort(-(rem + rng.random(128) * 1e-3))[:8]; assert (rem[pick] > 0).all()
    rem[pick] -= 1; rows[t, pick] = np.float16(0.125)
assert (rem == 0).all() and np.array_equal((rows > 0).sum(0), counts)
rank = np.cumsum(rows > 0, axis=0) - 1; vrows = np.zeros((T, NL), np.float16); seen_e = set()
for p, (e, k) in enumerate(lanes):
    if p < 64: sel = (rows[:, e] > 0) & ((rank[:, e] // C_HOT) == k)
    else: sel = (rows[:, e] > 0) & (e not in seen_e); seen_e.add(e)          # duplicate empties get no tokens
    vrows[:, p] = np.where(sel, rows[:, e], 0)
lane_counts = (vrows > 0).sum(0).astype(np.int32)
assert lane_counts[:64].max() <= C_HOT and lane_counts[64:].max() <= C_COLD and lane_counts.sum() == 8 * T
x128 = np.fromfile(f'{S}/e4/input_native_f16.bin', np.float16).reshape(128, 2176)[:, :2048]
xin = np.zeros((T, 2048 + NL), np.float16); xin[:, :2048] = np.tile(x128, (T // 128, 1))[:T]; xin[:, 2048:] = vrows
xin.tofile(f'{out}/input_f16.bin'); lane_counts.tofile(f'{out}/expected_counts_i32.bin')
percard = [[int((lane_counts[64 + PER * c: 64 + PER * (c + 1)] > 0).sum()), PER] for c in range(4)]
info = dict(T=T, mode='split_wide', C_hot=C_HOT, C_cold=C_COLD, lanes_total=NL, cold_lanes=L1, cold_lanes_per_card=PER, hot_experts=len(hot), split_experts=sorted({int(e) for e, k in hot_lanes if k > 0}),
            extra_lanes=int(sum(1 for e, k in hot_lanes if k > 0)), demoted=[int(e) for e in demoted], demoted_counts=[int(counts[e]) for e in demoted], spare_lanes=spare, cold_active_per_card=[a for a, b in percard],
            padded_rows=64 * C_HOT + L1 * C_COLD, padded_over_real=round((64 * C_HOT + L1 * C_COLD) / int(counts.sum()), 2), order=order.tolist(), chunk=chunk.tolist())
json.dump(info, open(f'{out}/info.json', 'w')); print({k: v for k, v in info.items() if k not in ('order', 'chunk')})
