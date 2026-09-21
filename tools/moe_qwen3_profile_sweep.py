#!/usr/bin/env python3
"""Run native C128, minimum, and power-of-two full-model profiling builds.

Requires the completed oracle experiment, including power-of-two plans/graphs.
Two precision workers may compile concurrently; an advisory lock serializes
their device runs. QPCs created by this sweep are disposable after validation.
"""
import argparse
import fcntl
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def dump(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Completed oracle experiment directory')
    parser.add_argument('--scratch', type=Path, required=True, help='Directory for disposable compilation outputs')
    parser.add_argument('--precision', choices=['fp16', 'mxfp6'], required=True)
    parser.add_argument('--adopt-compile-pid', type=int,
                        help='Wait for an already started native128 compile with this PID')
    parser.add_argument('--resume', action='store_true', help='Reuse validated completed steps after a failure')
    args = parser.parse_args()
    root, scratch = args.root.resolve(), args.scratch.resolve()
    profile = root / 'layer_profile'
    tools = Path(__file__).resolve().parent
    fmt = args.precision
    progress = profile / f'{fmt}_progress.json'
    completed = []

    def status(case, phase, **more):
        value = dict(precision=fmt, case=case, phase=phase, completed=completed,
                     updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **more)
        dump(progress, value)
        print(json.dumps(value), flush=True)

    def command(script, *arguments):
        subprocess.run([sys.executable, str(tools / script), *map(str, arguments)], check=True)

    try:
        for variant in ['native128', 'min', 'pow2']:
            case = f'{variant}_{fmt}'
            build = scratch / f'{case}_compile'
            archive = profile / f'{case}_compile'
            analysis = profile / f'{case}_analysis'
            if args.resume and (analysis / 'summary.json').exists() and (archive / 'qpc_removed.json').exists():
                if len(json.loads((analysis / 'summary.json').read_text())['layers']) != 48:
                    raise ValueError('Invalid saved analysis')
                completed.append(case)
                status(case, 'reused_completed_case')
                continue
            plan = None
            if variant == 'native128':
                graph = root / 'diagnostic_graph/model.onnx'
                control = root / f'baseline_counts_{fmt}'
            elif variant == 'min':
                graph = root / f'min_{fmt}_graph/diagnostic.onnx'
                plan = root / f'min_{fmt}_plan.json'
                control = root / f'sorted128_{fmt}_run'
            else:
                graph = profile / f'pow2_{fmt}_graph/diagnostic.onnx'
                plan = profile / f'pow2_{fmt}_plan.json'
                control = root / f'sorted128_{fmt}_run'
            status(case, 'compile', graph=str(graph), build=str(build))
            if args.resume and (build / 'result.json').exists():
                previous = json.loads((build / 'result.json').read_text())
                if (previous.get('graph'), previous.get('precision'), previous.get('stats_level')) != (str(graph), fmt, 70):
                    raise ValueError('Existing build does not match this profiling case')
            elif variant == 'native128' and args.adopt_compile_pid:
                started = time.monotonic()
                while not (build / 'result.json').exists():
                    proc = Path(f'/proc/{args.adopt_compile_pid}/cmdline')
                    if not proc.exists() or str(build).encode() not in proc.read_bytes():
                        # Allow the compile wrapper to write its final report.
                        time.sleep(2)
                        if not (build / 'result.json').exists():
                            raise RuntimeError('Adopted compiler ended without a result')
                    if time.monotonic() - started > 7200:
                        raise TimeoutError('Compile wait exceeded two hours')
                    time.sleep(10)
            else:
                command('moe_qwen3_oracle.py', 'compile', '--graph', graph, '--out', build,
                        '--precision', fmt, '--stats-level', 70)
            result = json.loads((build / 'result.json').read_text())
            if result['returncode'] != 0 or not (build / 'qpc/programqpc.bin').is_file():
                raise RuntimeError(f'Compilation failed: {build}')
            if archive != build:
                archive.mkdir(exist_ok=args.resume)
                for name in ['command.json', 'result.json', 'compile.log']:
                    shutil.copy2(build / name, archive / name)
            run = profile / f'{case}_run'
            trace = profile / f'{case}_profile'
            with (profile / '.device.lock').open('a') as lock:
                status(case, 'waiting_for_devices')
                fcntl.flock(lock, fcntl.LOCK_EX)
                status(case, 'timing')
                extra = ['--capacity-plan', plan] if plan else []
                if not (args.resume and (run / 'result.json').exists()):
                    command('moe_qwen3_baseline.py', 'run', '--qpc', build / 'qpc', '--out', run,
                            '--precision', fmt, '--routing-counts', '--iterations', 20,
                            '--rounds', 3, '--warmup', 2, '--scope',
                            f'Full model, stats-level=70, {variant}, first 128-token chunk', *extra)
                for name in ['logits_f32.bin', 'counts_i32.bin']:
                    if (run / name).read_bytes() != (control / name).read_bytes():
                        raise ValueError(f'Instrumented {case} differs from previous control: {name}')
                if plan and not (args.resume and (profile / f'{case}_audit.json').exists()):
                    command('moe_qwen3_oracle.py', 'audit', '--plan', plan, '--run', run,
                            '--control', control, '--out', profile / f'{case}_audit.json')
                status(case, 'capture')
                if not (args.resume and (trace / 'capture.json').exists()):
                    command('moe_qwen3_profile.py', 'capture', '--qpc', build / 'qpc',
                            '--expected', run, '--out', trace, '--samples', 3)
            status(case, 'analyze')
            if args.resume and analysis.exists():
                analysis.rename(analysis.with_name(analysis.name + '_previous_' + str(time.time_ns())))
            command('moe_qwen3_profile.py', 'analyze', '--trace-dir', trace / 'trace',
                    '--out', analysis)
            # All deletion is limited to a successful QPC generated by this sweep.
            binary = build / 'qpc/programqpc.bin'
            dump(archive / 'qpc_removed.json', dict(path=str(binary), bytes=binary.stat().st_size,
                 reason='Disposable instrumented QPC; successful timing, exact outputs, traces and per-layer analysis retained'))
            shutil.rmtree(build / 'qpc')
            status(case, 'compress_traces')
            compressed = []
            for path in sorted((trace / 'trace').glob('*.trace.json')):
                destination = path.with_suffix(path.suffix + '.gz')
                with path.open('rb') as source, gzip.open(destination, 'wb', compresslevel=1) as output:
                    shutil.copyfileobj(source, output, length=4*1024*1024)
                compressed.append(dict(original=str(path), compressed=str(destination),
                                       original_bytes=path.stat().st_size, compressed_bytes=destination.stat().st_size))
                path.unlink()
            dump(trace / 'compression.json', compressed)
            relocated = {entry['original']: entry['compressed'] for entry in compressed}
            for manifest in [trace / 'capture.json', analysis / 'summary.json']:
                value = json.loads(manifest.read_text())
                value['traces'] = [relocated.get(name, name) for name in value['traces']]
                dump(manifest, value)
            completed.append(case)
            status(case, 'complete')
        status('all', 'complete')
    except Exception as error:
        status(locals().get('case', 'setup'), 'failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
