#!/usr/bin/env python3
"""Build an additive elapsed-time breakdown from validated Qwen3 MoE traces."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import re
import statistics

from moe_qwen3_routing_profile import read_json, write_json, max_core_busy


PHASES = [
    ('initial_pack_ms', 'Routing / initial packing / waits'),
    ('group0_ms', 'Group 0 expert execution span'),
    ('between_groups_ms', 'Group 0 unpack / group 1 packing / waits'),
    ('group1_ms', 'Group 1 expert execution span'),
    ('last_unpack_ms', 'Group 1 unpack before reduction'),
    ('combine_ms', 'Final dense combination'),
]
NODE = re.compile(r'/model/layers\.(\d+)/mlp/(MatMul(?:_[1-5])?|Einsum_[34]|CumSum(?:_1)?|CtxGather3D(?:_3)?/n14)$')


def selected_rows(path):
    with path.open() as stream:
        header = next(csv.reader([next(stream)]))
        lines = (line for line in stream if '/mlp/' in line and
                 (',HMX,aicconvolutiond32,' in line or '/mlp/Einsum_' in line or
                  '/mlp/CumSum' in line or '/mlp/CtxGather3D' in line))
        for row in csv.DictReader(lines, fieldnames=header):
            match = NODE.fullmatch(row['node'])
            if not match:
                continue
            for name in ['sample', 'card', 'core', 'events']:
                row[name] = int(row[name])
            for name in ['first_us', 'last_us', 'busy_us']:
                row[name] = float(row[name])
            row['layer'], row['short_name'] = int(match[1]), match[2]
            yield row


def boundary(rows):
    if not rows:
        raise ValueError('Missing boundary events')
    return min(r['first_us'] for r in rows)/1000, max(r['last_us'] for r in rows)/1000


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Completed layer_profile directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='Requires Matplotlib')
    args = parser.parse_args()
    args.root, args.out = args.root.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    all_samples, layers, aggregate, layouts = [], [], [], []
    metric_names = [name for name, _ in PHASES] + ['moe_ms', 'group0_hmx_busy_max_core_ms',
        'group1_hmx_busy_max_core_ms', 'cumsum0_hvx_busy_max_core_ms',
        'cumsum1_hvx_busy_max_core_ms', 'local_reduce_hvx_busy_max_core_ms',
        'final_reduce_hvx_busy_max_core_ms', 'input_gather0_busy_max_core_ms']
    for fmt in ['fp16', 'mxfp6']:
        for variant in ['native128', 'min', 'pow2']:
            case = f'{variant}_{fmt}'
            analysis = args.root / f'{case}_analysis'
            summary = read_json(analysis / 'summary.json')
            validation = read_json(args.root / f'{case}_profile/output_validation.json')
            if len(summary['layers']) != 48 or validation['samples'] != 3 or not validation['bit_exact']:
                raise ValueError('Incomplete validated capture')
            grouped = defaultdict(list)
            for row in selected_rows(analysis / 'nodes.csv'):
                grouped[row['sample'], row['layer']].append(row)
            with (analysis / 'samples.csv').open() as stream:
                intervals = list(csv.DictReader(stream))
            if {(int(r['sample']), int(r['layer'])) for r in intervals} != {(s, l) for s in range(3) for l in range(48)}:
                raise ValueError('Missing MoE intervals')
            counts = read_json(args.root / f'{case}_run/result.json')['routing_counts']
            current = []
            zero_stages = []
            for interval in intervals:
                sample, layer = int(interval['sample']), int(interval['layer'])
                records = grouped[sample, layer]
                hot = [r for r in records if r['short_name'] in ['MatMul', 'MatMul_1', 'MatMul_2']
                       and r['engine'] == 'HMX' and r['kind'] == 'aicconvolutiond32']
                cold = [r for r in records if r['short_name'] in ['MatMul_3', 'MatMul_4', 'MatMul_5']
                        and r['engine'] == 'HMX' and r['kind'] == 'aicconvolutiond32']
                reduce = [r for r in records if r['short_name'] == 'Einsum_3'
                          and r['engine'] == 'HVX' and r['kind'] == 'aicbatchedreduceadd']
                for bank, names in [(hot, {'MatMul', 'MatMul_1', 'MatMul_2'}),
                                    (cold, {'MatMul_3', 'MatMul_4', 'MatMul_5'})]:
                    if bank and {r['short_name'] for r in bank} != names:
                        raise ValueError('Incomplete expert GEMM coverage')
                start, end = float(interval['router_start_ms']), float(interval['combine_end_ms'])
                hot_first, hot_last = boundary(hot)
                reduction_first, _ = boundary(reduce)
                if cold:
                    cold_first, cold_last = boundary(cold)
                else:
                    if sum(counts[layer][64:]) != 0 or [layer, 1] not in summary['zero_assignment_stages_without_gemm_events']:
                        raise ValueError('Missing cold GEMMs without validated zero assignments')
                    # No cold GEMM interval exists. Assign the intervening work to
                    # the transition, then start the final reduction directly.
                    cold_first = cold_last = reduction_first
                    zero_stages.append([sample, layer, 1])
                times = [start, hot_first, hot_last, cold_first, cold_last, reduction_first, end]
                if any(b < a - 1e-8 for a, b in zip(times, times[1:])):
                    raise ValueError(f'Overlapping phase boundaries in {case}, sample {sample}, layer {layer}')
                row = dict(case=case, precision=fmt, variant=variant, sample=sample, layer=layer)
                row.update({name: b-a for (name, _), a, b in zip(PHASES, times, times[1:])})
                row['moe_ms'] = float(interval['moe_ms'])
                if not math.isclose(sum(row[name] for name, _ in PHASES), row['moe_ms'], abs_tol=1e-8):
                    raise ValueError('Phase sum does not match the original MoE interval')
                row['group0_hmx_busy_max_core_ms'] = max_core_busy(hot)/1000
                row['group1_hmx_busy_max_core_ms'] = max_core_busy(cold)/1000 if cold else 0.
                for name, metric, engine, kind in [
                    ('CumSum', 'cumsum0_hvx_busy_max_core_ms', 'HVX', 'cumsum'),
                    ('CumSum_1', 'cumsum1_hvx_busy_max_core_ms', 'HVX', 'cumsum'),
                    ('Einsum_3', 'local_reduce_hvx_busy_max_core_ms', 'HVX', 'aicbatchedreduceadd'),
                    ('Einsum_4', 'final_reduce_hvx_busy_max_core_ms', 'HVX', 'aicbatchedreduceadd'),
                    ('CtxGather3D/n14', 'input_gather0_busy_max_core_ms', 'HVX_DMA', 'aicgather'),
                ]:
                    selected = [r for r in records if r['short_name'] == name and r['engine'] == engine and r['kind'] == kind]
                    if not selected and not (name == 'CumSum_1' and not cold):
                        raise ValueError(f'Missing {name} activity')
                    if selected and (min(r['first_us'] for r in selected)/1000 < start - 1e-8 or
                                     max(r['last_us'] for r in selected)/1000 > end + 1e-8):
                        raise ValueError('Selected compute event lies outside its MoE interval')
                    row[metric] = max_core_busy(selected)/1000 if selected else 0.
                current.append(row)
                all_samples.append(row)
            for layer in range(48):
                # Use all components from the same median-total invocation:
                # independently taking component medians would lose additivity.
                ordered = sorted((r for r in current if r['layer'] == layer), key=lambda r: (r['moe_ms'], r['sample']))
                chosen = dict(ordered[1])
                chosen['moe_sample_min_ms'], chosen['moe_sample_max_ms'] = ordered[0]['moe_ms'], ordered[-1]['moe_ms']
                if not math.isclose(chosen['moe_ms'], summary['layers'][layer]['moe_ms'], abs_tol=1e-8):
                    raise ValueError('Representative invocation differs from saved median')
                layers.append(chosen)
            representative = [r for r in layers if r['case'] == case]
            result = dict(case=case, precision=fmt, variant=variant)
            result.update({key: statistics.mean(r[key] for r in representative) for key in metric_names})
            if not math.isclose(sum(result[k] for k, _ in PHASES), result['moe_ms'], abs_tol=1e-8):
                raise ValueError('Aggregate breakdown does not add up')
            result['zero_assignment_stages_without_gemm'] = zero_stages
            aggregate.append(result)
            if variant == 'native128':
                capacities = [[128, 128] for _ in range(48)]
            else:
                parent = args.root.parent if variant == 'min' else args.root
                plan = read_json(parent / f'{variant}_{fmt}_plan.json')
                capacities = [plan['layers'][str(l)]['capacities'] for l in range(48)]
            layouts.append(dict(case=case, group0_mean_capacity=statistics.mean(c[0] for c in capacities),
                group1_mean_capacity=statistics.mean(c[1] for c in capacities),
                mean_padded_expert_rows=statistics.mean(sum(c)*64 for c in capacities),
                real_token_expert_assignments=1024,
                logical_fp16_expert_weight_bytes=128*3*2048*768*2,
                logical_dense_output_bytes=64*128*2048*2))
            print('BREAKDOWN_OK', case, {k: round(result[k], 4) for k in metric_names}, flush=True)
    for name, rows in [('samples.csv', all_samples), ('layers.csv', layers), ('aggregate.csv', aggregate), ('layouts.csv', layouts)]:
        write_csv(args.out / name, rows)
    write_json(args.out / 'summary.json', dict(aggregate=aggregate, layouts=layouts,
        validation=dict(layer_sample_intervals=len(all_samples), ordered_nonoverlapping_phases=True,
                        phase_sums_match_original_moe=True, medians_match_original_report=True),
        aggregation='For each layer choose the capture with median total MoE time; average its components across 48 layers'))
    lines = ['# Additive MoE elapsed-time breakdown', '',
        'Saved Qwen3-30B-A3B traces, 48 layers, first 128-token chunk, four AI 100 cards, '
        'matched stats-level=70 builds. All 864 layer/sample intervals were checked.', '',
        'The six disjoint time intervals below add up to the existing MoE measurement. '
        'For each layer, all components come from the capture with median total MoE latency; '
        'the table averages those 48 representative captures. Component medians are not taken independently.', '',
        '| Case | Routing / initial packing | Group 0 execution | Between groups | Group 1 execution | Last unpack | Final combine | Total MoE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        values = [r[key] for key, _ in PHASES] + [r['moe_ms']]
        lines.append('| ' + r['case'] + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', 'All values above are milliseconds. Group 0/1 mean hot/cold in regrouped cases; '
        'native C128 retains its original expert order.', '',
        '## Boundaries and interpretation', '',
        '1. Router HMX start → first group-0 expert HMX event: routing, first packing and any readiness waits.',
        '2. First → last group-0 expert HMX event: its three GEMMs, interleaved activation work, transfers and waits.',
        '3. Last group-0 → first group-1 HMX event: first output unpacking, second packing and waits.',
        '4. First → last group-1 expert HMX event: its three GEMMs and intervening work/waits.',
        '5. Last group-1 HMX event → first final local-reduction HVX event: remaining unpacking and waits.',
        '6. First final local reduction → last final cross-card-combine compute event: dense combination and waits.', '',
        'These are execution phases, not exclusive causal costs of named operators. '
        'Asynchronous copies may overlap phase boundaries, and an expert execution span is not pure matrix arithmetic. '
        'The table does not claim an exact split of memory transfer, communication and synchronization costs.', '',
        'For MXFP6 layer 31 in min/pow2, validated cold-group counts are zero and no GEMM events exist. '
        'Its group-1 execution/last-unpack intervals are zero; all work between group-0 completion '
        'and final reduction belongs to the transition. This avoids inventing a missing-GEMM timestamp.', '',
        '## Recorded engine work, separate from wall time', '',
        'The next table is the maximum summed recorded busy time on one core/engine within each '
        'named category. Parallel cores are never summed. These values cannot be added to the '
        'elapsed phases above or subtracted to obtain a precise memory-stall budget.', '',
        '| Case | Group 0 HMX | Group 1 HMX | First CumSum HVX | Second CumSum HVX | Local reduction HVX | Final reduction HVX | First input Gather |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        keys = metric_names[7:]
        lines.append('| ' + r['case'] + ' | ' + ' | '.join(f'{r[k]:.4f}' for k in keys) + ' |')
    lines += ['', 'All values are milliseconds, using the same representative captures as the additive table.', '',
        '## What padding changes', '',
        '| Case | Mean group 0 capacity | Mean group 1 capacity | Mean padded expert rows | Actual assignments |',
        '|---|---:|---:|---:|---:|']
    for r in layouts:
        lines.append(f"| {r['case']} | {r['group0_mean_capacity']:.2f} | {r['group1_mean_capacity']:.2f} | "
                     f"{r['mean_padded_expert_rows']:.2f} | 1024 |")
    lines += ['', 'Minimum padding still leaves about six padded rows per actual token-expert assignment. '
        'The hot group averages capacity about 94, so it is reduced much less than the cold group, '
        'which averages about 2. Both groups retain 64 experts.', '',
        'The graph changes packed expert activations `[64,C,2048]` and intermediate activations '
        '`[64,C,768]`. It retains the 128-token routing/packing scans and the logical dense output '
        'accumulator `[64,128,2048]`, reshaped to `[4,16,128,2048]` for final reductions. '
        'That accumulator is 32 MiB at FP16, independent of C. The graph also retains all '
        '`128 × 3 × 2048 × 768` expert parameters per layer: 1.125 GiB in FP16. '
        'These are logical sizes, not measurements of traffic or physical compiler allocations.', '',
        'Consequently, the 62% reduction in logical padded rows is not a 62% reduction in all MoE work. '
        'The cold execution interval shrinks strongly, but substantial dense combination remains, '
        'and the hot execution/packing intervals can regress for smaller irregular shapes. '
        'Power-of-two shapes improve aggregate timing despite more padded rows. The traces show '
        'the scheduling outcome; they do not establish its precise compiler or hardware cause.', '',
        'A targeted next experiment is to optimize the full-size scatter/reduction path and '
        'inspect transfers/waits in the hot expert interval. Changing only the routing-column '
        'permutation targets less than 1% of the current interval.', '',
        '## Reproduction', '', '```bash', 'python tools/moe_qwen3_breakdown.py \\',
        '  --root 9.17.2026/real_model/oracle_padding/layer_profile \\',
        '  --out 9.17.2026/real_model/oracle_padding/layer_profile/new_breakdown --plot', '```', '',
        '`--plot` requires Matplotlib. The analysis otherwise uses the standard library and '
        'the helper functions in `moe_qwen3_routing_profile.py`. Outputs must be new. '
        '[samples.csv](samples.csv) contains every capture; [layers.csv](layers.csv) contains '
        'the chosen per-layer capture; [summary.json](summary.json) records validation. '
        'No new model compilation or device execution was needed.', '']
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors = ['#879caf', '#087f8c', '#b6cfca', '#dc7633', '#edc4a3', '#7865a7']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
        for axis, fmt in zip(axes, ['fp16', 'mxfp6']):
            data = [r for r in aggregate if r['precision'] == fmt]
            bottom = [0., 0., 0.]
            for (key, label), color in zip(PHASES, colors):
                values = [r[key] for r in data]
                axis.bar(range(3), values, bottom=bottom, color=color, label=label)
                bottom = [a+b for a, b in zip(bottom, values)]
            for index, total in enumerate(bottom):
                axis.text(index, total+.12, f'{total:.2f}', ha='center')
            axis.set(title=fmt.upper(), xticks=range(3), xticklabels=['Native C128', 'Minimum', 'Power of two'],
                     ylabel='Mean per-layer elapsed time (ms)', ylim=(0, 13))
            axis.grid(axis='y', alpha=.2)
            axis.set_axisbelow(True)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncols=2, frameon=False, fontsize=9)
        fig.suptitle('Qwen3 MoE elapsed-time breakdown: four cards, 128-token prefill\n'
                     'Disjoint execution phases; expert spans include transfers and waits')
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(args.out / f'breakdown.{suffix}', dpi=180)
        plt.close(fig)
        lines += ['![Additive MoE breakdown](breakdown.png)', '']
    (args.out / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
