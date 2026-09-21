#!/usr/bin/env python3
"""Static, input-specific padding experiments on the native 48-layer Qwen3 graph."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP

from moe_qwen3_baseline import EXPORT

SOURCE = EXPORT / 'Qwen3MoeForCausalLM.onnx'
CONFIG = EXPORT / 'qpc-6ff7c7675b1fcae0'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_graph(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'weights').symlink_to(EXPORT, target_is_directory=True)
    model = onnx.load(SOURCE, load_external_data=False)
    for value in model.graph.initializer:
        for entry in value.external_data:
            if entry.key == 'location':
                entry.value = 'weights/' + entry.value
    return model


def add_counts(model):
    """Reuse the graph's exact per-expert counts before capacity truncation."""
    nodes = {n.name: n for n in model.graph.node}
    counts = []
    for layer in range(48):
        for stage in range(2):
            node = nodes[f'/model/layers.{layer}/mlp/Einsum_{stage + 1}']
            if node.op_type != 'Einsum':
                raise ValueError('Unexpected source graph routing structure')
            counts.append(node.output[0])
    model.graph.initializer.append(nh.from_array(np.array([48, 128], np.int64), 'oracle_counts_shape'))
    model.graph.node.extend([
        h.make_node('Concat', counts, ['oracle_flat_counts'], axis=0, name='oracle_count_concat'),
        h.make_node('Reshape', ['oracle_flat_counts', 'oracle_counts_shape'], ['routing_counts'], name='oracle_count_reshape'),
    ])
    model.graph.output.append(h.make_tensor_value_info('routing_counts', TP.INT32, [48, 128]))


def diagnostic(args):
    model = source_graph(args.out)
    add_counts(model)
    path = args.out / 'model.onnx'
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    dump(args.out / 'info.json', dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        kind='native C128 diagnostic', layers=48, experts=128, stages=2, experts_per_stage=64,
        added_output='routing_counts [48,128], original expert order', weights='unchanged external data via symlink'))
    print('DIAGNOSTIC_EXPORT_OK', path, flush=True)


def make_plan(args):
    counts = np.fromfile(args.counts, np.int32).reshape(48, 128)
    if np.any(counts < 0) or np.any(counts > 128) or not np.all(counts.sum(1) == 1024):
        raise ValueError('Invalid native device routing counts')
    calibrated = json.loads(args.calibrate.read_text()) if args.calibrate else None
    plan = dict(input_counts=str(args.counts), counts_sha256=hashlib.sha256(args.counts.read_bytes()).hexdigest(),
                order_policy='preserve calibrated layout' if calibrated else args.order,
                alignment=args.alignment, rounding=args.rounding, layers={})
    if calibrated:
        plan['calibration_layout'] = str(args.calibrate.resolve())
        plan['calibration_layout_sha256'] = hashlib.sha256(args.calibrate.read_bytes()).hexdigest()
    for layer, row in enumerate(counts):
        if calibrated:
            order = np.array(calibrated['layers'][str(layer)]['order'])
            required = row.reshape(2, 64).max(1)
        else:
            order = np.lexsort((np.arange(128), -row)) if args.order == 'sorted' else np.arange(128)
            required = row[order].reshape(2, 64).max(1)
        if args.rounding == 'power-of-two':
            caps = np.array([1 << (max(1, int(n)) - 1).bit_length() for n in required])
        else:
            caps = (np.maximum(required, 1) + args.alignment - 1) // args.alignment * args.alignment
        if np.any(caps > 128):
            raise ValueError('Aligned capacity exceeds token length')
        plan['layers'][str(layer)] = dict(order=order.tolist(), capacities=caps.tolist(),
            required=required.tolist(), padded_rows=int(caps.sum() * 64))
    plan['padded_rows'] = sum(v['padded_rows'] for v in plan['layers'].values())
    plan['uniform_padded_rows'] = 48 * 128 * 128
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError(f'Will not overwrite {args.out}')
    dump(args.out, plan)
    print('PLAN_OK', args.out, 'padded_fraction', plan['padded_rows'] / plan['uniform_padded_rows'], flush=True)


def layout_digest(plan):
    orders = [plan['layers'][str(layer)]['order'] for layer in range(48)]
    return hashlib.sha256(json.dumps(orders, separators=(',', ':')).encode()).hexdigest()


def materialize_weights(args):
    """Stream exact FP16 expert bytes once, avoiding large compiler folding files."""
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    model = onnx.load(SOURCE, load_external_data=False)
    nodes = {node.name: node for node in model.graph.node}
    initializers = {value.name: value for value in model.graph.initializer}
    manifest = dict(layout_sha256=layout_digest(plan), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), tensors={})
    start = time.monotonic()
    with (args.out / 'weights.bin').open('xb') as output:
        for layer in range(48):
            order = plan['layers'][str(layer)]['order']
            if sorted(order) != list(range(128)):
                raise ValueError('Invalid expert permutation')
            for projection in range(3):
                names = [nodes[f'/model/layers.{layer}/mlp/' + ('MatMul' if n == 0 else f'MatMul_{n}')].input[1]
                         for n in [projection, projection + 3]]
                banks = []
                for name in names:
                    value = initializers[name]
                    metadata = {entry.key: entry.value for entry in value.external_data}
                    if value.data_type != TP.FLOAT16 or value.dims[0] != 64 or 'location' not in metadata:
                        raise ValueError('Unexpected source expert tensor')
                    banks.append(np.memmap(EXPORT / metadata['location'], dtype=np.uint16, mode='r',
                        offset=int(metadata.get('offset', '0')), shape=tuple(value.dims)))
                for stage, name in enumerate(names):
                    offset, checksum = output.tell(), hashlib.sha256()
                    for expert in order[stage*64:(stage+1)*64]:
                        data = memoryview(banks[expert // 64][expert % 64]).cast('B')
                        output.write(data)
                        checksum.update(data)
                    manifest['tensors'][name] = dict(offset=offset, length=output.tell()-offset,
                        shape=list(initializers[name].dims), sha256=checksum.hexdigest())
            if layer % 8 == 7:
                print('WEIGHTS_LAYERS_DONE', layer + 1, flush=True)
        manifest['bytes'] = output.tell()
    manifest['seconds'] = time.monotonic() - start
    dump(args.out / 'index.json', manifest)
    print('WEIGHTS_READY', args.out, 'bytes', manifest['bytes'], flush=True)


def oracle(args):
    plan = json.loads(args.plan.read_text())
    model = source_graph(args.out)
    by_name = {n.name: n for n in model.graph.node}
    initializers = {v.name: v for v in model.graph.initializer}
    cache = getattr(args, 'weights_cache', None)
    cached = json.loads((cache / 'index.json').read_text()) if cache else None
    if cached:
        if cached['layout_sha256'] != layout_digest(plan) or cached['source_sha256'] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
            raise ValueError('Reordered weight cache does not match this layout/source')
        if (cache / 'weights.bin').stat().st_size != cached['bytes']:
            raise ValueError('Incomplete reordered weight cache')
        (args.out / 'regrouped').symlink_to(cache.resolve(), target_is_directory=True)
    insert_before = {}
    changes = []
    for layer in range(48):
        prefix = f'/model/layers.{layer}/mlp/'
        spec = plan['layers'][str(layer)]
        order, capacities = spec['order'], spec['capacities']
        if sorted(order) != list(range(128)) or len(capacities) != 2 or any(not 1 <= c <= 128 for c in capacities):
            raise ValueError(f'Invalid plan for layer {layer}')
        for stage, capacity in enumerate(capacities):
            stem = f'oracle_L{layer}_S{stage}'
            end = nh.from_array(np.array([capacity], np.int64), stem + '_end')
            stop = nh.from_array(np.array(capacity, np.int32), stem + '_stop')
            model.graph.initializer.extend([end, stop])
            slice_node = by_name[prefix + ('Slice' if stage == 0 else 'Slice_1')]
            range_node = by_name[prefix + ('Range_1' if stage == 0 else 'Range_3')]
            if slice_node.op_type != 'Slice' or range_node.op_type != 'Range':
                raise ValueError('Unexpected source capacity structure')
            changes.append(dict(layer=layer, stage=stage, capacity=capacity,
                slice=slice_node.name, old_slice_end=slice_node.input[2],
                range=range_node.name, old_range_stop=range_node.input[1]))
            slice_node.input[2] = end.name
            range_node.input[1] = stop.name
        if order != list(range(128)):
            order_name = f'oracle_L{layer}_order'
            model.graph.initializer.append(nh.from_array(np.array(order, np.int64), order_name))
            transpose = by_name[prefix + 'Transpose']
            original = transpose.input[0]
            reordered = f'oracle_L{layer}_routing'
            insert_before[transpose.name] = [h.make_node('Gather', [original, order_name], [reordered], axis=1,
                name=f'oracle_L{layer}_routing_gather')]
            transpose.input[0] = reordered
            for projection in range(3):
                matmuls = [by_name[prefix + ('MatMul' if n == 0 else f'MatMul_{n}')]
                           for n in [projection, projection + 3]]
                banks = [n.input[1] for n in matmuls]
                if any(b not in initializers or list(initializers[b].dims)[0] != 64 for b in banks):
                    raise ValueError('Unexpected native expert weight layout')
                if cached:
                    for bank in banks:
                        entry = cached['tensors'][bank]
                        value = initializers[bank]
                        if list(value.dims) != entry['shape']:
                            raise ValueError('Reordered tensor shape mismatch')
                        del value.external_data[:]
                        for key, val in dict(location='regrouped/weights.bin', offset=entry['offset'], length=entry['length']).items():
                            value.external_data.add(key=key, value=str(val))
                    continue
                combined = f'oracle_L{layer}_W{projection}_full'
                nodes = [h.make_node('Concat', banks, [combined], axis=0, name=combined)]
                for stage, matmul in enumerate(matmuls):
                    ids_name = f'oracle_L{layer}_W{projection}_S{stage}_ids'
                    model.graph.initializer.append(nh.from_array(np.array(order[stage*64:(stage+1)*64], np.int64), ids_name))
                    output = f'oracle_L{layer}_W{projection}_S{stage}'
                    nodes.append(h.make_node('Gather', [combined, ids_name], [output], axis=0, name=output))
                    matmul.input[1] = output
                insert_before.setdefault(matmuls[0].name, []).extend(nodes)
    if insert_before:
        nodes = []
        for node in model.graph.node:
            nodes.extend(insert_before.get(node.name, []))
            nodes.append(copy.deepcopy(node))
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    # Shapes of affected intermediate values must be inferred from the new slices.
    del model.graph.value_info[:]
    timing = args.out / 'timing.onnx'
    onnx.save(model, timing)
    onnx.checker.check_model(str(timing))
    add_counts(model)
    onnx.save(model, args.out / 'diagnostic.onnx')
    onnx.checker.check_model(str(args.out / 'diagnostic.onnx'))
    dump(args.out / 'plan.json', plan)
    dump(args.out / 'changes.json', changes)
    print('ORACLE_EXPORT_OK', args.out, 'capacity_edits', len(changes) * 2, flush=True)


def selftest(args):
    """Exercise native packing slices and masks without loading the expert weights."""
    from onnx.reference import ReferenceEvaluator
    from onnx.utils import Extractor
    from moe_capacity_audit import CtxScatter3DInt, CtxGather3D

    args.out.mkdir(parents=True, exist_ok=False)
    plan = dict(layers={str(layer): dict(order=list(range(128)), capacities=[17, 9]) for layer in range(48)})
    dump(args.out / 'test_plan.json', plan)
    oracle(argparse.Namespace(out=args.out / 'graph', plan=args.out / 'test_plan.json'))
    prefix = '/model/layers.0/mlp/'
    x_name, routes_name = prefix + 'Reshape_output_0', prefix + 'ScatterElements_output_0'
    outputs = [prefix + s for s in ['CtxGather3D_output_0', 'CtxGather3D_3_output_0', 'Less_output_0', 'Less_1_output_0']]
    rng = np.random.default_rng(8103)
    inputs = {x_name: rng.normal(size=(128, 4)).astype(np.float16), routes_name: np.zeros((128, 128), np.float16)}
    for expert in range(128):
        count = int(rng.integers(0, (17 if expert < 64 else 9) + 1))
        tokens = rng.choice(128, count, replace=False)
        inputs[routes_name][tokens, expert] = rng.uniform(.1, 1, count).astype(np.float16)
    results = []
    for path in [SOURCE, args.out / 'graph/timing.onnx']:
        model = onnx.load(path, load_external_data=False)
        del model.graph.value_info[:]
        model.graph.value_info.extend([
            h.make_tensor_value_info(x_name, TP.FLOAT16, [128, 4]),
            h.make_tensor_value_info(routes_name, TP.FLOAT16, [128, 128]),
            *[h.make_tensor_value_info(name, TP.FLOAT16 if i < 2 else TP.BOOL,
               [64, None, 4] if i < 2 else [64, None]) for i, name in enumerate(outputs)],
        ])
        subset = Extractor(model).extract_model([x_name, routes_name], outputs)
        if any(v.external_data for v in subset.graph.initializer):
            raise RuntimeError('Packing self-test unexpectedly depends on external weights')
        results.append(ReferenceEvaluator(subset, new_ops=[CtxScatter3DInt, CtxGather3D]).run(None, inputs))
    comparisons = []
    for i, capacity in enumerate([17, 9, 17, 9]):
        expected = results[0][i][:, :capacity]
        actual = results[1][i]
        if not np.array_equal(actual, expected):
            raise RuntimeError(f'Native packing/mask mismatch for output {outputs[i]}')
        comparisons.append(dict(name=outputs[i], shape=list(actual.shape), exact=True))
    dump(args.out / 'result.json', dict(passed=True, comparisons=comparisons,
        scope='Native ONNX packing and valid-row masks on bounded synthetic routing; no expert GEMMs'))
    print('NATIVE_PACKING_SELFTEST_PASSED', flush=True)


def regroup_selftest(args):
    """Check native routing and constant-bank permutation with small synthetic tensors."""
    from onnx.reference import ReferenceEvaluator
    from onnx.utils import Extractor
    from moe_capacity_audit import CtxScatter3DInt, CtxGather3D

    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(12864)
    order = rng.permutation(128)
    plan = dict(layers={str(layer): dict(order=order.tolist(), capacities=[17, 9]) for layer in range(48)})
    dump(args.out / 'test_plan.json', plan)
    oracle(argparse.Namespace(out=args.out / 'graph', plan=args.out / 'test_plan.json'))
    prefix = '/model/layers.0/mlp/'
    x_name, routes_name = prefix + 'Reshape_output_0', prefix + 'ScatterElements_output_0'
    packed = [prefix + s for s in ['CtxGather3D_output_0', 'CtxGather3D_3_output_0', 'Less_output_0', 'Less_1_output_0']]
    x = rng.normal(size=(128, 4)).astype(np.float16)
    routes = np.zeros((128, 128), np.float16)
    expected_packed, expected_masks = [], []
    for stage, capacity in enumerate([17, 9]):
        values = np.zeros((64, capacity, 4), np.float16)
        masks = np.zeros((64, capacity), bool)
        for lane, expert in enumerate(order[stage*64:(stage+1)*64]):
            count = int(rng.integers(capacity + 1))
            tokens = np.sort(rng.choice(128, count, replace=False))
            routes[tokens, expert] = rng.uniform(.1, 1, count).astype(np.float16)
            values[lane, :count] = x[tokens]
            masks[lane, :count] = True
        expected_packed.append(values)
        expected_masks.append(masks)
    model = onnx.load(args.out / 'graph/timing.onnx', load_external_data=False)
    model.graph.value_info.extend([
        h.make_tensor_value_info(x_name, TP.FLOAT16, [128, 4]),
        h.make_tensor_value_info(routes_name, TP.FLOAT16, [128, 128]),
        *[h.make_tensor_value_info(name, TP.FLOAT16 if i < 2 else TP.BOOL,
           [64, None, 4] if i < 2 else [64, None]) for i, name in enumerate(packed)],
    ])
    subset = Extractor(model).extract_model([x_name, routes_name], packed)
    actual = ReferenceEvaluator(subset, new_ops=[CtxScatter3DInt, CtxGather3D]).run(None, {x_name: x, routes_name: routes})
    # Invalid packed indices can contain duplicates, so only valid rows matter.
    for stage in range(2):
        mask = expected_masks[stage]
        if not np.array_equal(actual[stage][mask], expected_packed[stage][mask]) or not np.array_equal(actual[stage+2], mask):
            raise RuntimeError('Regrouped native packing does not match original expert IDs')
    by_name = {node.name: node for node in model.graph.node}
    for projection in range(3):
        combined = f'oracle_L0_W{projection}_full'
        sources = by_name[combined].input
        shape = (64, 4, 3) if projection < 2 else (64, 3, 4)
        banks = [rng.normal(size=shape).astype(np.float16) for _ in range(2)]
        replacements = {name: nh.from_array(bank, name) for name, bank in zip(sources, banks)}
        for index, value in enumerate(model.graph.initializer):
            if value.name in replacements:
                model.graph.initializer[index].CopyFrom(replacements[value.name])
        outputs = [f'oracle_L0_W{projection}_S{stage}' for stage in range(2)]
        model.graph.value_info.extend(h.make_tensor_value_info(name, TP.FLOAT16, list(shape)) for name in outputs)
        subset = Extractor(model).extract_model([], outputs)
        actual = ReferenceEvaluator(subset).run(None, {})
        expected = np.concatenate(banks, axis=0)[order]
        if not all(np.array_equal(actual[stage], expected[stage*64:(stage+1)*64]) for stage in range(2)):
            raise RuntimeError('Expert weight banks do not match the routing permutation')
    dump(args.out / 'result.json', dict(passed=True, routing_packing=True, valid_masks=True,
        gate_up_down_weight_banks=True, scope='Native permutation nodes with synthetic bounded activations and small weight tensors'))
    print('NATIVE_REGROUP_SELFTEST_PASSED', flush=True)


def audit(args):
    plan = json.loads(args.plan.read_text())
    counts = np.fromfile(args.run / 'counts_i32.bin', np.int32).reshape(48, 128)
    caps = np.array([plan['layers'][str(layer)]['capacities'] for layer in range(48)])
    if not np.all(counts.sum(1) == 1024) or np.any(counts < 0) or np.any(counts > 128):
        raise ValueError('Invalid observed routing counts')
    required = counts.reshape(48, 2, 64).max(2)
    overflow = np.maximum(counts.reshape(48, 2, 64) - caps[:, :, None], 0)
    result = dict(plan=str(args.plan.resolve()), run=str(args.run.resolve()),
        counts_sha256=hashlib.sha256((args.run / 'counts_i32.bin').read_bytes()).hexdigest(),
        overflow_assignments=int(overflow.sum()), overflow_experts=int((overflow > 0).sum()),
        required_capacities=required.tolist(), capacities=caps.tolist(),
        padded_rows=int(caps.sum() * 64), uniform_padded_rows=48*128*128)
    if args.control:
        actual = np.fromfile(args.run / 'logits_f32.bin', np.float32).astype(np.float64)
        expected = np.fromfile(args.control / 'logits_f32.bin', np.float32).astype(np.float64)
        if actual.shape != (151936,) or expected.shape != actual.shape or not np.isfinite(actual).all():
            raise ValueError('Invalid logits')
        result['control'] = str(args.control.resolve())
        result['logits_vs_control'] = dict(bit_exact=bool(np.array_equal(actual, expected)),
            relative_l2=float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
            max_abs=float(np.abs(actual - expected).max()), argmax_matches=bool(actual.argmax() == expected.argmax()))
        result['counts_same_bank_order_exact'] = bool(np.array_equal(counts,
            np.fromfile(args.control / 'counts_i32.bin', np.int32).reshape(48, 128)))
    if args.out.exists():
        raise ValueError(f'Will not overwrite {args.out}')
    dump(args.out, result)
    print('AUDIT', json.dumps({key: value for key, value in result.items() if key not in ['required_capacities', 'capacities']}), flush=True)
    if result['overflow_assignments']:
        raise RuntimeError('Unsafe capacity plan: observed token truncation')


def compile_graph(args):
    args.out.mkdir(parents=True, exist_ok=False)
    command = ['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100',
        f'-m={args.graph}', '-retained-state', '-convert-to-fp16', '-aic-num-cores=16',
        '-mos=1', '-aic-enable-depth-first', f'-mdp-load-partition-config={CONFIG}/mdp_ts_4.json',
        f'-network-specialization-config={CONFIG}/specializations_flat.json',
        f'-custom-IO-list-file={CONFIG}/custom_io.yaml', '-compile-only', f'-aic-binary-dir={args.out}/qpc']
    if args.precision == 'mxfp6':
        command.append('-mxfp6-matmul')
    if args.stats_level:
        command.append(f'-stats-level={args.stats_level}')
    dump(args.out / 'command.json', command)
    start = time.monotonic()
    with (args.out / 'compile.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report = dict(returncode=result.returncode, seconds=time.monotonic() - start,
        qpc_exists=(args.out / 'qpc/programqpc.bin').exists(), graph=str(args.graph), precision=args.precision,
        stats_level=args.stats_level)
    dump(args.out / 'result.json', report)
    print('COMPILE', json.dumps(report), flush=True)
    if result.returncode or not report['qpc_exists']:
        raise RuntimeError('Compilation failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['diagnostic', 'compile', 'plan', 'oracle', 'weights', 'selftest', 'regroup-selftest', 'audit'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--precision', choices=['fp16', 'mxfp6'], default='fp16')
    parser.add_argument('--counts', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--weights-cache', type=Path, help='For oracle: reuse exact reordered expert bytes produced by weights')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--calibrate', type=Path,
                        help='For plan: counts are in this existing plan\'s bank order; preserve its layout')
    parser.add_argument('--order', choices=['identity', 'sorted'], default='identity')
    parser.add_argument('--alignment', type=int, choices=[1, 8, 16, 32, 64, 128], default=1)
    parser.add_argument('--rounding', choices=['multiple', 'power-of-two'], default='multiple')
    parser.add_argument('--stats-level', type=int, choices=[0, 70], default=0,
                        help='For compile: 70 enables SDK per-operation profiling; 0 preserves timing builds')
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.graph:
        args.graph = args.graph.resolve()
    if args.action == 'compile' and args.graph is None:
        parser.error('compile requires --graph')
    if args.action == 'plan' and args.counts is None:
        parser.error('plan requires --counts')
    if args.rounding == 'power-of-two' and args.alignment != 1:
        parser.error('power-of-two rounding requires --alignment 1')
    if args.action in ['oracle', 'weights', 'audit'] and args.plan is None:
        parser.error('oracle/weights/audit requires --plan')
    if args.action == 'audit' and args.run is None:
        parser.error('audit requires --run')
    {'diagnostic': diagnostic, 'compile': compile_graph, 'plan': make_plan, 'oracle': oracle,
     'weights': materialize_weights, 'selftest': selftest, 'regroup-selftest': regroup_selftest, 'audit': audit}[args.action](args)


if __name__ == '__main__':
    main()
