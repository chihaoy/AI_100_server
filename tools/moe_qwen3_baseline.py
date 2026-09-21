#!/usr/bin/env python3
"""Run and analyze fixed-T=128 full-model Qwen3-30B-A3B prefill on four AI 100s.

Uses a compiled EP model and an explicit FP32 reference. Supports C128 controls
and static oracle variants, optionally with pre-truncation routing counts.
This runner does not select padding, measure decode, or establish cross-chunk KV correctness.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

MODEL = Path('/home/chihao/models/qwen3_30b_a3b/hf')
REFERENCE = Path('/home/chihao/mllm/9.17.2026/e2e/ref')
EXPORT = Path('/home/chihao/models/qwen3_30b_a3b/ep/qeff_cache/Qwen3MoeForCausalLM/'
              'Qwen3MoeForCausalLM-ed085dabb01c623b')
CACHED_QPC = EXPORT / 'qpc-6ff7c7675b1fcae0/qpc'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(args):
    if args.out.exists():
        raise ValueError(f'Will not overwrite {args.out}')
    sys.path.extend(['/opt/qti-aic/dev/lib/x86_64', '/opt/qti-aic/dev/python'])
    import qaicrt
    import QAicApi_pb2 as api
    qpc = qaicrt.Qpc(str(args.qpc))
    status, data = qpc.getIoDescriptor()
    if status != qaicrt.QStatus.QS_SUCCESS:
        raise RuntimeError('Cannot read QPC IO descriptor')
    descriptor = api.IoDesc()
    descriptor.ParseFromString(bytes(data))
    actual = {b.name: (b.type, tuple(b.dims)) for b in descriptor.selected_set.bindings}
    expected = {'input_ids': (api.INT64_I_TYPE, (1, 128)),
                'position_ids': (api.INT64_I_TYPE, (1, 128)),
                'logits': (api.FLOAT_TYPE, (1, 1, 151936))}
    if args.routing_counts:
        expected['routing_counts'] = (api.INT32_I_TYPE, (48, 128))
    if actual != expected:
        raise ValueError(f'Unexpected QPC IO: {actual}')
    del qpc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='qwen3-baseline-') as temp:
        host = Path(temp) / 'host'
        subprocess.run([
            'g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror',
            '-I/opt/qti-aic/dev/inc', str(Path(__file__).with_name('moe_qwen3_baseline_host.cpp')),
            '-o', str(host), '-L/opt/qti-aic/dev/lib/x86_64', '-lQAic',
            '-Wl,-rpath,/opt/qti-aic/dev/lib/x86_64',
        ], check=True)
        with args.out.with_suffix('.log').open('x') as log:
            capacity_args = []
            if args.capacity_plan:
                plan = json.loads(args.capacity_plan.read_text())
                capacities = np.array([plan['layers'][str(layer)]['capacities'] for layer in range(48)], np.int32)
                if capacities.shape != (48, 2) or np.any(capacities < 1) or np.any(capacities > 128):
                    raise ValueError('Invalid capacity plan')
                capacity_file = Path(temp) / 'capacities.bin'
                capacities.tofile(capacity_file)
                capacity_args = [str(capacity_file)]
            command = [
                str(host), str(args.qpc), str(args.reference / 'input_ids_i64.bin'),
                str(args.reference / 'position_ids_i64.bin'), str(args.out),
                str(args.iterations), str(args.rounds), str(args.warmup),
            ] + (['--counts'] if args.routing_counts else []) + capacity_args
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    analyze(args)


def analyze(args):
    from transformers import AutoTokenizer

    actual = np.fromfile(args.out / 'logits_f32.bin', np.float32).astype(np.float64)
    expected = np.load(args.reference / 'logits.npy', mmap_mode='r')[-1].astype(np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid logits shape or nonfinite values')
    rows = list(csv.DictReader((args.out / 'samples.csv').open()))
    if not rows:
        raise ValueError('Missing timing samples')
    times = np.array([float(row['host_ms']) for row in rows])
    if not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError('Invalid timing samples')
    resources = list(csv.DictReader((args.out / 'resources.csv').open()))
    allocations = []
    for device in range(4):
        records = [r for r in resources if int(r['device']) == device]
        before = next(r for r in records if r['stage'] == 'before')
        active = next(r for r in records if r['stage'] == 'active')
        released = next(r for r in records if r['stage'] == 'released')
        rounds = [r for r in records if r['stage'].startswith('round')]
        allocations.append(dict(
            device=device,
            active_gib=(int(before['dram_free_kib']) - int(active['dram_free_kib'])) / 2**20,
            stable=all((r['dram_free_kib'], r['nsp_free']) ==
                       (active['dram_free_kib'], active['nsp_free']) for r in rounds),
            released=(released['dram_free_kib'], released['nsp_free']) ==
                     (before['dram_free_kib'], before['nsp_free']),
        ))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = np.fromfile(args.reference / 'input_ids_i64.bin', np.int64)
    positions = np.fromfile(args.reference / 'position_ids_i64.bin', np.int64)
    if len(ids) != 128 or not np.array_equal(positions, np.arange(128)):
        raise ValueError('This baseline requires the first 128 contiguous token positions')
    relative_l2 = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    argmax, reference_argmax = int(actual.argmax()), int(expected.argmax())
    result = dict(
        scope=args.scope,
        weight_precision=args.precision, model=str(args.model), qpc=str(args.qpc),
        qpc_bytes=(args.qpc / 'programqpc.bin').stat().st_size,
        reference=str(args.reference), input_ids_sha256=digest(args.reference / 'input_ids_i64.bin'),
        reference_logits_sha256=digest(args.reference / 'logits.npy'),
        checkpoint_config_sha256=digest(args.model / 'config.json'),
        checkpoint_index_sha256=digest(args.model / 'model.safetensors.index.json'),
        input_text=tokenizer.decode(ids),
        logits=dict(relative_l2=relative_l2, max_abs=float(np.abs(actual - expected).max()),
                    correlation=float(np.corrcoef(actual, expected)[0, 1]),
                    argmax=argmax, reference_argmax=reference_argmax,
                    next_token=tokenizer.decode([argmax]), reference_next_token=tokenizer.decode([reference_argmax]),
                    top5=np.argsort(-actual)[:5].tolist(), reference_top5=np.argsort(-expected)[:5].tolist()),
        reference_accuracy_threshold=.05, reference_accuracy_pass=relative_l2 < .05,
        next_token_matches=argmax == reference_argmax,
        timing=dict(samples=len(rows), median_ms=float(np.median(times)),
                    p10_ms=float(np.percentile(times, 10)), p90_ms=float(np.percentile(times, 90)),
                    tokens_per_second=128000 / float(np.median(times)),
                    round_medians_ms={r: float(np.median([float(x['host_ms']) for x in rows if x['round'] == r]))
                                      for r in sorted({row['round'] for row in rows})}),
        setup=list(csv.DictReader((args.out / 'setup.csv').open())), device_memory=allocations,
    )
    previous = REFERENCE.parent / 'naive/logits_naive.bin'
    if args.precision == 'mxfp6' and previous.exists():
        result['bit_exact_historical_mxfp6'] = bool(np.array_equal(actual, np.fromfile(previous, np.float32)))
    if args.routing_counts:
        counts = np.fromfile(args.out / 'counts_i32.bin', np.int32).reshape(48, 128)
        if (counts < 0).any() or (counts > 128).any() or not np.all(counts.sum(1) == 1024):
            raise ValueError('Invalid saved routing counts')
        result['routing_counts'] = counts.tolist()
        if args.capacity_plan:
            plan = json.loads(args.capacity_plan.read_text())
            capacities = np.array([plan['layers'][str(layer)]['capacities'] for layer in range(48)], np.int32)
            overflow = np.maximum(counts.reshape(48, 2, 64) - capacities[:, :, None], 0)
            result['capacity_plan'] = str(args.capacity_plan.resolve())
            result['capacity_plan_sha256'] = digest(args.capacity_plan)
            result['overflow_assignments'] = int(overflow.sum())
            if overflow.any():
                raise ValueError('Observed expert counts exceed compiled capacities')
    write_json(args.out / 'result.json', result)
    print(json.dumps({key: result[key] for key in ['weight_precision', 'logits', 'timing', 'device_memory']}, indent=2))
    if not all(x['stable'] and x['released'] for x in allocations):
        raise RuntimeError('Device resource stability/release check failed')
    # Accuracy is reported independently; a successful hardware run is not an accuracy pass.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'analyze'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qpc', type=Path, default=CACHED_QPC)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--precision', choices=['fp16', 'mxfp6'], required=True)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--routing-counts', action='store_true')
    parser.add_argument('--capacity-plan', type=Path, help='With routing counts, enforce the static capacities on every invocation')
    parser.add_argument('--scope', default='48-layer EP model, batch 1, T=128, first chunk; fixed conservative padding')
    args = parser.parse_args()
    for key in ['out', 'qpc', 'model', 'reference']:
        setattr(args, key, getattr(args, key).resolve())
    if args.precision == 'fp16' and args.qpc == CACHED_QPC.resolve():
        parser.error('The default QPC uses MXFP6; supply --qpc for the FP16 control')
    if args.capacity_plan and not args.routing_counts:
        parser.error('--capacity-plan requires --routing-counts')
    {'run': run, 'analyze': analyze}[args.action](args)


if __name__ == '__main__':
    main()
