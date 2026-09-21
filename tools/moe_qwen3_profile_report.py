#!/usr/bin/env python3
"""Write tables and publication-quality plots from completed Qwen3 traces."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='layer_profile directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--precisions', nargs='+', choices=['fp16', 'mxfp6'], default=['fp16', 'mxfp6'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases, rows, aggregate = {}, [], []
    for fmt in args.precisions:
        for variant in ['native128', 'min', 'pow2']:
            case = f'{variant}_{fmt}'
            analysis = json.loads((args.root / f'{case}_analysis/summary.json').read_text())
            run = json.loads((args.root / f'{case}_run/result.json').read_text())
            capture = json.loads((args.root / f'{case}_profile/output_validation.json').read_text())
            if (not capture['bit_exact'] or capture['samples'] != 3 or
                    len(analysis['layers']) != 48 or
                    any(row['samples'] != 3 for row in analysis['layers']) or
                    run['timing']['samples'] != 60 or
                    not all(device['stable'] and device['released'] for device in run['device_memory'])):
                raise ValueError(f'Incomplete or invalid case {case}')
            if variant != 'native128' and run['overflow_assignments'] != 0:
                raise ValueError(f'Capacity overflow in {case}')
            cases[case] = dict(analysis=analysis, run=run)
        native = cases[f'native128_{fmt}']
        for layer in range(48):
            baseline_record = native['analysis']['layers'][layer]
            baseline = baseline_record['moe_ms']
            row = dict(precision=fmt, layer=layer, native128_moe_ms=baseline,
                       native128_moe_sample_min_ms=baseline_record['moe_ms_min'],
                       native128_moe_sample_max_ms=baseline_record['moe_ms_max'])
            for variant in ['min', 'pow2']:
                measured = cases[f'{variant}_{fmt}']['analysis']['layers'][layer]
                plan_root = args.root.parent if variant == 'min' else args.root
                plan = json.loads((plan_root / f'{variant}_{fmt}_plan.json').read_text())
                caps = plan['layers'][str(layer)]['capacities']
                value = measured['moe_ms']
                row.update({f'{variant}_hot': caps[0], f'{variant}_cold': caps[1],
                            f'{variant}_moe_ms': value,
                            f'{variant}_moe_sample_min_ms': measured['moe_ms_min'],
                            f'{variant}_moe_sample_max_ms': measured['moe_ms_max'],
                            f'{variant}_speedup': baseline/value,
                            f'{variant}_saved_ms': baseline-value,
                            f'{variant}_reduction_pct': 100*(1-value/baseline)})
            row['pow2_speedup_vs_min'] = row['min_moe_ms']/row['pow2_moe_ms']
            rows.append(row)
        current = [r for r in rows if r['precision'] == fmt]
        for variant in ['native128', 'min', 'pow2']:
            values = [r[f'{variant}_moe_ms'] for r in current]
            base_sum = sum(r['native128_moe_ms'] for r in current)
            speeds = [r['native128_moe_ms']/r[f'{variant}_moe_ms'] for r in current]
            measured = cases[f'{variant}_{fmt}']['run']
            host_ms = measured['timing']['median_ms']
            baseline_host_ms = native['run']['timing']['median_ms']
            with (args.root / f'{variant}_{fmt}_analysis/samples.csv').open() as stream:
                trace_samples = {int(r['sample']): float(r['execution_ms']) for r in csv.DictReader(stream)}
            aggregate.append(dict(precision=fmt, variant=variant,
                mean_moe_ms=float(np.mean(values)), sum_layer_median_moe_ms=sum(values),
                speedup_ratio_of_sums=base_sum/sum(values),
                speedup_layer_min=min(speeds), speedup_layer_max=max(speeds),
                faster_layers=sum(s>1 for s in speeds),
                trace_device_median_ms=float(np.median(list(trace_samples.values()))),
                full_model_host_median_ms=host_ms,
                full_model_host_speedup=baseline_host_ms/host_ms,
                full_model_host_reduction_pct=100*(1-host_ms/baseline_host_ms),
                full_model_host_p10_ms=measured['timing']['p10_ms'],
                full_model_host_p90_ms=measured['timing']['p90_ms']))
    with (args.out / 'layers.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (args.out / 'aggregate.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)
    report = dict(scope='Matched stats-level=70 full-model traces, 48 layers, 4 cards, T128, one input',
                  boundary=cases[f'native128_{args.precisions[0]}']['analysis']['moe_boundary'],
                  aggregate=aggregate, layers=rows)
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(len(args.precisions), 2, figsize=(14, 4*len(args.precisions)),
                             sharex=True, constrained_layout=True, squeeze=False)
    palette = {'native128': '#687786', 'min': '#087f8c', 'pow2': '#dc7633'}
    names = {'native128': 'C128', 'min': 'Minimum capacity', 'pow2': 'Power of two'}
    for index, fmt in enumerate(args.precisions):
        current = [r for r in rows if r['precision'] == fmt]
        for variant in ['native128', 'min', 'pow2']:
            axes[index, 0].plot(range(48), [r[f'{variant}_moe_ms'] for r in current],
                                label=names[variant], color=palette[variant], marker='.', markersize=4)
            axes[index, 0].fill_between(range(48),
                [r[f'{variant}_moe_sample_min_ms'] for r in current],
                [r[f'{variant}_moe_sample_max_ms'] for r in current], color=palette[variant], alpha=.1)
        for variant in ['min', 'pow2']:
            axes[index, 1].plot(range(48), [r[f'{variant}_speedup'] for r in current],
                                label=names[variant], color=palette[variant], marker='.', markersize=4)
        axes[index, 1].axhline(1, color=palette['native128'], linewidth=.8, linestyle='--')
        axes[index, 0].set_ylabel(f'{fmt.upper()} MoE elapsed time (ms)')
        axes[index, 1].set_ylabel('Speedup relative to C128 (×)')
        for axis in axes[index]:
            axis.grid(alpha=.2)
            axis.set_xlim(-.5, 47.5)
            axis.set_xticks([0, 8, 16, 24, 32, 40, 47])
            axis.legend(loc='best', frameon=False)
    axes[-1, 0].set_xlabel('Decoder layer (zero based)')
    axes[-1, 1].set_xlabel('Decoder layer (zero based)')
    fig.suptitle('Qwen3-30B-A3B: per-layer MoE, 128-token prefill on four AI 100s\n'
                 'Matched instrumented builds; median of three traces (shading: observed range); router HMX → final combine')
    fig.savefig(args.out / 'layer_moe.png', dpi=180)
    fig.savefig(args.out / 'layer_moe.pdf')
    fig.savefig(args.out / 'layer_moe.svg')
    plt.close(fig)
    lines = ['# Measured per-layer MoE timings', '', report['scope'] + '.', '',
             'Interval: ' + report['boundary'] + '.', '',
             'Each number is the median of three detailed traces. These are instrumented, in-model '
             'elapsed times, not isolated-operator or uninstrumented latencies. Native C128 uses '
             'the original expert order; minimum and power-of-two use the same calibrated regrouping.', '',
             'Host timings come from a separate 60-invocation run of the same instrumented QPC. '
             'Host timings and detailed-capture device timings have different measurement conditions; '
             'do not subtract a trace interval from the separate host measurement.', '',
             '![Per-layer MoE timings](layer_moe.png)', '',
             '| Precision | Policy | Mean MoE ms/layer | MoE speedup, ratio of sums | Per-layer speedup range | Full-model host median ms |',
             '|---|---|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['precision']} | {r['variant']} | {r['mean_moe_ms']:.4f} | "
                     f"{r['speedup_ratio_of_sums']:.4f}× | {r['speedup_layer_min']:.4f}–"
                     f"{r['speedup_layer_max']:.4f}× | {r['full_model_host_median_ms']:.3f} |")
    lines += ['', '## Capacity-policy comparison', '']
    for fmt in args.precisions:
        current = [r for r in rows if r['precision'] == fmt]
        wins = sum(r['pow2_moe_ms'] < r['min_moe_ms'] for r in current)
        minimum = next(r for r in aggregate if r['precision'] == fmt and r['variant'] == 'min')
        power2 = next(r for r in aggregate if r['precision'] == fmt and r['variant'] == 'pow2')
        moe_change = 100*(power2['sum_layer_median_moe_ms']/minimum['sum_layer_median_moe_ms']-1)
        host_change = 100*(power2['full_model_host_median_ms']/minimum['full_model_host_median_ms']-1)
        lines.append(f"- {fmt.upper()}: power-of-two has a lower MoE median than minimum in {wins}/48 layers. "
                     f"Relative latency change versus minimum: {moe_change:+.2f}% for summed MoE medians "
                     f"and {host_change:+.2f}% for full-model host latency (negative means faster).")
    lines += ['', 'The smallest safe capacity is not consistently the fastest compiled shape. '
              'Keep both policies as candidates for per-layer tuning. These measurements do not '
              'establish the performance of a mixed-policy graph or an exhaustive optimum.', '',
              'All minimum and power-of-two runs had zero overflow and bit-identical counts/logits '
              'against their corresponding regrouped C128 controls; every saved profiler output '
              'also matched its host timing run. Padding equivalence is separate from FP32 accuracy.']
    for fmt in args.precisions:
        measured = cases[f'min_{fmt}']['run']
        error = 100*measured['logits']['relative_l2']
        passed = 'passes' if measured['reference_accuracy_pass'] else 'fails'
        lines.append(f"The regrouped {fmt.upper()} result has {error:.4f}% relative L2 logit error "
                     f"against the saved FP32 reference and {passed} the unchanged 5% diagnostic gate.")
    lines += ['', 'This is the first chunk of one calibration input. Runtime selection, regrouping, '
              'weight movement, loading and compilation costs are outside the measured intervals.', '']
    for fmt in args.precisions:
        lines += ['', f'## {fmt.upper()}', '',
                  '| Layer | C128 ms | Minimum caps | Minimum ms | Speedup | Power-of-two caps | Power-of-two ms | Speedup |',
                  '|---:|---:|---|---:|---:|---|---:|---:|']
        for r in [r for r in rows if r['precision'] == fmt]:
            lines.append(f"| {r['layer']} | {r['native128_moe_ms']:.4f} | "
                         f"{r['min_hot']}/{r['min_cold']} | {r['min_moe_ms']:.4f} | {r['min_speedup']:.4f}× | "
                         f"{r['pow2_hot']}/{r['pow2_cold']} | {r['pow2_moe_ms']:.4f} | {r['pow2_speedup']:.4f}× |")
    lines += ['', 'Raw numbers and the range across three samples are in [layers.csv](layers.csv). '
              'Engine-level measurements remain in each case’s analysis directory. '
              'Do not add engine spans or busy times across cores to estimate elapsed time.', '']
    (args.out / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
