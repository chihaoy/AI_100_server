#!/usr/bin/env python3
"""Port of the layer-2 replay stack to all 48 layers of the full prefill graph.
Usage: build_full_stack.py <src dir> <out dir> <counts_i32.bin> --regroup native|hc --caps 128|oracle --rewrites 0|1 [--banks <dir>]
  src dir     : a rebuilt full-model graph dir (model.onnx + weights symlinks), e.g. native_c128_kvfree
  regroup hc  : per layer, stage 0 = 64 largest-count experts round-robin over cards, stage 1 = rest (active ones
                round-robin from card 3); routing Gather before the Transpose, banks materialized under <banks>/weights.bin
  caps oracle : per-stage capacity = largest count in the stage (>=1); edits Slice/Slice_1 end and Range_1/Range_3 stop
  rewrites 1  : zero-read removal (CtxGather3D_2 + Add), 16 token tiles for Einsum_3, Hillis-Steele tree scans for CumSum/CumSum_1"""
import sys, os, json, copy, argparse, numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
from onnx.reference import ReferenceEvaluator
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('counts'); ap.add_argument('--regroup', default='native'); ap.add_argument('--caps', default='128'); ap.add_argument('--rewrites', type=int, default=1); ap.add_argument('--banks', default=None); ap.add_argument('--poscounts', default=None, help='counts in position order from a run of the same layout; capacities are taken from these')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
O = '/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding'; F = f'{O}/mdts_flag/full_model'
m = onnx.load(f'{a.src}/model.onnx', load_external_data=False); g = m.graph
counts = np.fromfile(a.counts, np.int32).reshape(48, 128); assert (counts.sum(1) == 1024).all()
poscounts = np.fromfile(a.poscounts, np.int32).reshape(48, 128) if a.poscounts else None
native_idx = json.load(open(f'{F}/weights_native_fp16/index.json'))['tensors']
def by_name(): return {n.name: n for n in g.node}
plan = {}
# ---------------- placement + capacities
def hc_order(cnt):
    ranked = [int(e) for e in np.argsort(-cnt, kind='stable')]; hot, cold = ranked[:64], ranked[64:]
    s0 = [[] for _ in range(4)]; s1 = [[] for _ in range(4)]
    for i, e in enumerate(hot): s0[i % 4].append(e)
    act = [e for e in cold if cnt[e] > 0]; emp = [e for e in cold if cnt[e] == 0]
    for j, e in enumerate(act): s1[[3, 2, 1, 0][j % 4]].append(e)
    for c in range(4): need = 16 - len(s1[c]); s1[c] += emp[:need]; emp = emp[need:]
    return sum(s0, []) + sum(s1, [])
for L in range(48):
    cnt = counts[L]; order = hc_order(cnt) if a.regroup == 'hc' else list(range(128)); assert sorted(order) == list(range(128))
    if a.caps == '128': caps = [128, 128]
    elif poscounts is not None: caps = [max(1, int(poscounts[L][:64].max())), max(1, int(poscounts[L][64:].max()))]
    else: caps = [max(1, int(cnt[order[:64]].max())), max(1, int(cnt[order[64:]].max()))]
    plan[str(L)] = dict(order=order, capacities=caps, active=int((cnt > 0).sum()),
                        active_per_card=[int(sum(int((cnt[order[s*64+16*c:s*64+16*c+16]] > 0).sum()) for s in range(2))) for c in range(4)])
# ---------------- banks (materialize permuted native banks)
bank_names = {}
for L in range(48):
    p = f'/model/layers.{L}/mlp/'; bn = by_name()
    bank_names[L] = [[i for i in bn[p + ('MatMul' if k == 0 else f'MatMul_{k}')].input if i in native_idx][0] for k in range(6)]
if a.regroup == 'hc':
    assert a.banks, '--banks required for hc'; os.makedirs(a.banks, exist_ok=True)
    src_bin = np.memmap(f'{F}/weights_native_fp16/weights.bin', np.float16, 'r'); total = sum(b['length'] for b in native_idx.values())
    if not os.path.exists(f'{a.banks}/index.json'):
        dst = np.memmap(f'{a.banks}/weights.bin', np.float16, 'w+', shape=(total // 2,))
        for L in range(48):
            order = np.array(plan[str(L)]['order']); names = bank_names[L]   # MatMul..MatMul_5 = gate0, up0, down0, gate1, up1, down1
            for kind in range(3):
                b0, b1 = native_idx[names[kind]], native_idx[names[kind + 3]]; shp = tuple(b0['shape']); n = int(np.prod(shp))
                nat = np.concatenate([src_bin[b0['offset']//2:b0['offset']//2+n].reshape(shp), src_bin[b1['offset']//2:b1['offset']//2+n].reshape(shp)], 0)
                perm = nat[order]; dst[b0['offset']//2:b0['offset']//2+n] = perm[:64].reshape(-1); dst[b1['offset']//2:b1['offset']//2+n] = perm[64:].reshape(-1)
            print('banks layer', L, flush=True)
        dst.flush(); json.dump(dict(layout='hc per layer', tensors=native_idx, plan_orders={k: v['order'] for k, v in plan.items()}), open(f'{a.banks}/index.json', 'w'))
    else:
        saved = json.load(open(f'{a.banks}/index.json'))['plan_orders']; assert all(saved[k] == plan[k]['order'] for k in plan), 'existing bank dir has a different order'
    inits = {t.name: t for t in g.initializer}; bname = os.path.basename(a.banks.rstrip('/'))
    for L in range(48):
        for name in bank_names[L]:
            t = inits[name]; e = native_idx[name]; t.ClearField('external_data')
            for k, v in (('location', f'{bname}/weights.bin'), ('offset', str(e['offset'])), ('length', str(e['length']))): x = t.external_data.add(); x.key = k; x.value = v
    if not os.path.islink(f'{a.out}/{bname}'): os.symlink(os.path.relpath(a.banks, a.out), f'{a.out}/{bname}')
# ---------------- capacity edits + routing gather
insert_before = {}
for L in range(48):
    p = f'/model/layers.{L}/mlp/'; bn = by_name(); spec = plan[str(L)]
    if a.caps != '128':
        for stage, cap in enumerate(spec['capacities']):
            end = nh.from_array(np.array([cap], np.int64), f'stack_L{L}_S{stage}_end'); stop = nh.from_array(np.array(cap, np.int32), f'stack_L{L}_S{stage}_stop'); g.initializer.extend([end, stop])
            sl = bn[p + ('Slice' if stage == 0 else 'Slice_1')]; rg = bn[p + ('Range_1' if stage == 0 else 'Range_3')]; assert sl.op_type == 'Slice' and rg.op_type == 'Range'
            sl.input[2] = end.name; rg.input[1] = stop.name
    if spec['order'] != list(range(128)):
        oname = f'stack_L{L}_order'; g.initializer.append(nh.from_array(np.array(spec['order'], np.int64), oname))
        tr = bn[p + 'Transpose']; orig = tr.input[0]; re = f'stack_L{L}_routing'
        insert_before[tr.name] = [h.make_node('Gather', [orig, oname], [re], axis=1, name=f'stack_L{L}_routing_gather')]; tr.input[0] = re
if insert_before:
    nodes = []
    for n in g.node: nodes.extend(insert_before.get(n.name, [])); nodes.append(copy.deepcopy(n))
    del g.node[:]; g.node.extend(nodes)
# ---------------- rewrites
def scan_nodes(original, stem):
    nodes, inits = [], []
    def const(name, data): inits.append(nh.from_array(np.asarray(data, np.int64), stem + name)); return stem + name
    axis, start = const('token_axis', [1]), const('start', [0]); cur = original.input[0]
    for shift in [1, 2, 4, 8, 16, 32, 64]:
        zero, sliced, shifted, result = [stem + f'{part}_{shift}' for part in ['zero', 'slice', 'shift', 'add']]
        inits.append(nh.from_array(np.zeros((64, shift), np.int32), zero))
        nodes += [h.make_node('Slice', [cur, start, const(f'end_{shift}', [128 - shift]), axis], [sliced], name=sliced),
                  h.make_node('Concat', [zero, sliced], [shifted], axis=1, name=shifted), h.make_node('Add', [cur, shifted], [result], name=result)]
        cur = result
    nodes.append(h.make_node('Identity', [cur], list(original.output), name=original.name)); return nodes, inits
checked = False
if a.rewrites:
    for L in range(48):
        p = f'/model/layers.{L}/mlp/'; bn = by_name()
        # zero-read removal
        gather, add, zero = bn[p + 'CtxGather3D_2'], bn[p + 'Add'], bn[p + 'ConstantOfShape_1']
        assert gather.op_type == 'CtxGather3D' and gather.input[0] == zero.output[0] and list(add.input) == [gather.output[0], bn[p + 'Mul_5'].output[0]]
        assert sum(gather.output[0] in n.input for n in g.node) == 1
        old, new = add.output[0], add.input[1]; nodes = []
        for n in g.node:
            if n.name in (gather.name, add.name): continue
            for i, v in enumerate(n.input):
                if v == old: n.input[i] = new
            nodes.append(n)
        del g.node[:]; g.node.extend(nodes); bn = by_name()
        # tiled local reduction
        e3 = bn[p + 'Einsum_3']; assert e3.op_type == 'Einsum' and h.get_attribute_value(e3.attribute[0]) == b'dpth->dth'
        inits = [nh.from_array(np.array([2], np.int64), f'stack_L{L}_tile_axis')]; rep, outs = [], []
        for i in range(16):
            s, e = f'stack_L{L}_tile_start_{i}', f'stack_L{L}_tile_end_{i}'; inits += [nh.from_array(np.array([i * 8], np.int64), s), nh.from_array(np.array([(i + 1) * 8], np.int64), e)]
            sl, rd = f'stack_L{L}_tile_slice_{i}', f'stack_L{L}_tile_reduced_{i}'
            rep += [h.make_node('Slice', [e3.input[0], s, e, f'stack_L{L}_tile_axis'], [sl], name=p + f'reduce_slice_{i}'), h.make_node('Einsum', [sl], [rd], equation='dpth->dth', name=p + f'reduce_tile_{i}')]
            outs.append(rd)
        rep.append(h.make_node('Concat', outs, list(e3.output), axis=1, name=e3.name))
        nodes = []
        for n in g.node: nodes.extend(rep if n.name == e3.name else [n])
        del g.node[:]; g.node.extend(nodes); g.initializer.extend(inits); bn = by_name()
        # tree scans
        for cs in ('CumSum', 'CumSum_1'):
            original = bn[p + cs]; assert original.op_type == 'CumSum' and not original.attribute
            axis_node = next(n for n in g.node if original.input[1] in n.output); assert int(nh.to_array(h.get_attribute_value(axis_node.attribute[0]))) == 1
            rep, inits = scan_nodes(original, original.name + '_retune_')
            if not checked:   # CPU semantic check once
                cg = h.make_graph(rep, 'scan_check', [h.make_tensor_value_info(original.input[0], TP.INT32, [64, 128])], [h.make_tensor_value_info(original.output[0], TP.INT32, [64, 128])], inits)
                ev = ReferenceEvaluator(h.make_model(cg, opset_imports=[h.make_opsetid('', 17)])); src = np.random.default_rng(0).integers(0, 2, (64, 128), dtype=np.int32)
                assert np.array_equal(ev.run(None, {original.input[0]: src})[0], np.cumsum(src, axis=1, dtype=np.int32)); checked = True
            nodes = []
            for n in g.node: nodes.extend(rep if n.name == original.name else [n])
            del g.node[:]; g.node.extend(nodes); g.initializer.extend(inits); bn = by_name()
        if L % 8 == 0: print('rewrites layer', L, flush=True)
del g.value_info[:]
for link in ('weights', 'weights_fp16', 'weights_native_fp16', 'regrouped'):
    ps = f'{a.src}/{link}'
    if os.path.islink(ps) and not os.path.exists(f'{a.out}/{link}'): os.symlink(os.path.realpath(ps), f'{a.out}/{link}')
onnx.save(m, f'{a.out}/model.onnx'); _c = os.getcwd(); os.chdir(a.out); onnx.checker.check_model('model.onnx'); os.chdir(_c)
json.dump(dict(src=a.src, regroup=a.regroup, caps=a.caps, rewrites=a.rewrites, layers=plan), open(f'{a.out}/plan.json', 'w'))
busiest = [max(v['active_per_card']) for v in plan.values()]; padded = sum(sum(v['capacities'][s] * sum(1 for e in v['order'][s*64:(s+1)*64] if counts[int(k)][e] > 0) for s in range(2)) for k, v in plan.items())
print(f'{a.out}: regroup={a.regroup} caps={a.caps} rewrites={a.rewrites}; nodes {len(g.node)}; busiest-card active experts mean {np.mean(busiest):.1f}; padded rows total {padded} (real {int(counts.sum())})')
