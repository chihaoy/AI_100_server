#!/usr/bin/env python3
"""Report measured layer-2 overhead ablations without adding overlapping costs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABELS = {'baseline': 'C2 control', 'drop_zero': 'Remove zero read/add',
          'tile_t16': '16 token tiles', 'tile_h16': '16 hidden tiles',
          'drop_zero_tile_t16': 'Zero removal + token tiles'}
STEM = '/model/layers.2/mlp/'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def representative(root, case):
    return [r for r in rows(root/case['case']/'analysis/work.csv') if int(r['sample'])==case['sample']]


def report(root):
    data, screen = read(root/'summary.json'), read(root/'initial.json')
    plain = read(root/'uninstrumented.json')
    plain_base = next(c for c in plain['cases'] if c['case']=='baseline')
    plain_best = next(c for c in plain['cases'] if c['case']=='drop_zero_tile_t16')
    cases = data['cases']
    base = next(c for c in cases if c['case']=='baseline')
    best = min(cases, key=lambda c: c['host_median_ms'])
    if any(not c['validation']['bit_exact'] for c in cases+screen['cases']):
        raise ValueError('Review numerical differences before publishing this report')
    if len({c['export']['input_sha256'] for c in cases+screen['cases']}) != 1:
        raise ValueError('Input changed between cases')
    if any(c['export']['trained_weight_references'] != base['export']['trained_weight_references'] for c in cases):
        raise ValueError('Weights changed')
    lines = ['# Qwen3 layer-2 overhead ablations', '',
        'Measured 2026-09-22 on four AI 100 cards, 16 cores each; SDK 1.21.6, FP16. '
        'Stats-level-70 ablations are followed by a separate stats-level-0 confirmation.', '',
        f'**With profiling instrumentation disabled, zero-read removal plus token-tiled '
        f'reduction lowers isolated MoE host latency from {plain_base["host_median_ms"]:.3f} '
        f'to {plain_best["host_median_ms"]:.3f} ms '
        f'({100*(1-plain_best["host_median_ms"]/plain_base["host_median_ms"]):.2f}% lower latency; '
        f'{plain_base["host_median_ms"]/plain_best["host_median_ms"]:.3f}× speedup).** '
        'All saved timing and profiling outputs match the original C2 output bit for bit.', '',
        '## Controlled changes', '',
        'The replay uses the trained Qwen3-30B-A3B zero-based layer 2, captured expert inputs '
        'and already-regrouped routing from the same GSM8K prompt 41, batch 1 / 128 tokens. '
        'Hot capacity stays 128 and cold capacity stays 2. The six trained weight banks, '
        'expert order, GEMM nodes, capacities, final cross-card reduction and four-card '
        'partition configuration stay identical. Physical tiling, placement and scheduling '
        'can change when the compiler sees a rewritten graph.', '',
        '- **Zero-read removal:** the first `CtxGather3D_2` reads the all-zero accumulator. '
        'Its only consumer, `Add`, adds those zeros to the weighted hot-group outputs. '
        'Remove these two nodes and feed the weighted outputs directly into the existing '
        'valid-row mask and scatter. Retain dense zero initialization, the hot scatter '
        'and the cold read/modify/write path, which must preserve hot contributions.',
        '- **Token tiling:** replace the local `dpth->dth` reduction with 16 independent '
        'reductions of `[4,16,8,2048]` slices and concatenate their token outputs. '
        'Every output still reduces the same 16 expert-lane contributions; no tree of '
        'partial sums is introduced.',
        '- **Hidden tiling:** screen the alternative of 16 `[4,16,128,128]` slices, '
        'concatenating along hidden features.',
        '- **Combined:** apply zero-read removal and the token-tiled local reduction '
        'together, after measuring each separately.', '',
        'The source dense accumulator is still `[64,128,2048]` FP16 (32 MiB). '
        'This experiment changes its access and reduction schedule; it does not implement '
        'sparse token-oriented combination or omit inactive experts. The replay starts '
        'after routing/permutation. Full-model latency, runtime selection and adaptation '
        'overhead are outside this measurement.', '',
        '## Initial screen', '',
        'Three rounds of 100 timed invocations per case, 10 warmups each; capacity and '
        'weights are fixed. Reverse case order in the middle round.', '',
        '| Candidate | Host median, ms | Per-round medians, ms |', '|---|---:|---|']
    for c in screen['cases']:
        lines.append(f'| {LABELS[c["case"]]} | {c["host_median_ms"]:.3f} | '+
                     ', '.join(f'{v:.3f}' for v in c['host_round_medians_ms'])+' |')
    lines += ['', 'Token tiling was selected for the combined candidate from this screen. '
        'A fresh timing cohort compares the control, each selected individual change and '
        'their combination; its results below do not mix old and new host timing samples.', '',
        '## Instrumented confirmation cohort', '',
        '| Variant | Host median, ms | Host P10–P90, ms | Reduction vs control | SDK device interval, ms |',
        '|---|---:|---:|---:|---:|']
    for c in cases:
        lines.append(f'| {LABELS[c["case"]]} | {c["host_median_ms"]:.3f} | '
                     f'{c["host_p10_ms"]:.3f}–{c["host_p90_ms"]:.3f} | '
                     f'{100*(1-c["host_median_ms"]/base["host_median_ms"]):.2f}% | {c["device_ms"]:.3f} |')
    lines += ['', 'Host time includes set-data, enqueue, input/output transfer and wait. '
        'Compilation, loading, activation and warmup are excluded. Host P10/P90 are invocation '
        'spread, not confidence intervals. Device metrics use the same median whole-device '
        'duration trace for every metric in a row, from three validated captures per variant.', '',
        '## Confirmation without profiling instrumentation', '',
        'Recompile the unchanged control and combined graphs with `-stats-level=0`; '
        'all other compiler flags stay the same. Run another three rounds × 100 invocations '
        'per graph, reversing order in round 2. These host timings have their own matched '
        'control and are kept separate from the instrumented cohort.', '',
        '| Variant | Host median, ms | P10–P90, ms | Per-round medians, ms |',
        '|---|---:|---:|---|']
    for c in plain['cases']:
        lines.append(f'| {LABELS[c["case"]]} | {c["host_median_ms"]:.3f} | '
                     f'{c["host_p10_ms"]:.3f}–{c["host_p90_ms"]:.3f} | '+
                     ', '.join(f'{v:.3f}' for v in c['host_round_medians_ms'])+' |')
    lines += ['',
        '![Measured overhead changes](overhead_results.png)', '',
        '## What the compiler actually changed', '',
        '| Variant | Hot span, ms | Between groups, ms | Cold span, ms | Combine span, ms | Local reduction max-core work, ms | DDR-backed reduction cores | Total outgoing P2P, MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        lines.append(f'| {LABELS[c["case"]]} | {c["hot_span_ms"]:.3f} | '
                     f'{c["cold_start_ms"]-c["hot_end_ms"]:.3f} | {c["cold_span_ms"]:.3f} | '
                     f'{c["combine_span_ms"]:.3f} | {c["local_all_max_core_ms"]:.3f} | '
                     f'{c["local_DDR_cores"]} / 64 | {c["p2p_bytes"]/2**20:.4f} |')
    lines += ['', 'Expert spans run from first to last HMX event in each group. Combine spans '
        'run from first local reduction to last final-combine computation. These are '
        'observational windows and may overlap other work. Max-core work sums HVX compute '
        'within each core, then takes the largest core total; it is not elapsed latency. '
        'The DDR-backed-core count describes the lowered reduction kernels, not an exact '
        'memory-traffic counter. P2P payload counts outgoing endpoints once.', '',
        'Removing the redundant zero read/add shortens the between-group interval '
        'from 1.654 to 0.766 ms in its isolated ablation, while leaving the combination '
        'span almost unchanged. Token tiling shortens the combination span from '
        '2.377 to 0.802 ms. The combined graph has its own compiled schedule; '
        'individual phase savings should not be added to predict its latency.', '',
        '**The compiler did not distribute the final within-card merge across 16 cores.** '
        'Each card still runs that merge on core 0. The original one-shot DDR-backed merge '
        'becomes 16 small TCM-backed merge kernels, interleaved with partial reductions. '
        f'The largest per-card total for those merge kernels drops from '
        f'{base["core0_merge_max_ms"]:.3f} to {best["core0_merge_max_ms"]:.3f} ms. '
        'A DDR-backed HVX kernel duration can include operand-access stalls; this is '
        'recorded kernel time, not a pure arithmetic measurement. Tiling improved '
        'locality and pipelining while retaining the core-0 merge bottleneck.', '',
        'All variants send the same 63.9851 MiB of P2P payload per replay. Removing '
        '`Add` moves the hot-stage 24 MiB exchange to the first scatter; it does not '
        'eliminate that exchange. The observed speedups therefore cannot be explained '
        'as fewer P2P bytes. Reduction tiling also changes the attribution of the final '
        '1.5 MiB partial-result exchange from `Einsum_3` to `reduce_tile_0`.', '',
        'The following timeline shows local reduction work on every core of card 0. '
        'Each panel starts at that trace’s first local reduction across all cards. '
        'This directly tests whether the intended parallelism survived compilation.', '',
        '![Local reduction placement](reduction_cores.png)', '',
        '## Validation and limits', '',
        '- Original graph initializers are unchanged; all six expert projections, '
        'capacity Slice/Range nodes and final cross-card reduction pass structural checks.',
        '- Each tiling rewrite is checked using its actual ONNX replacement subgraph '
        'against the original reduction on a deterministic random `[4,16,128,2048]` '
        'FP32 input. Both tiled candidates are exactly equal in this check.',
        '- Across 3,000 timed invocations, every saved timing-round output is byte-identical '
        'to the original C2 replay. '
        'All routing counts match and remain within hot128/cold2. The timing host saves '
        'the last output of each round (30 total), not every intermediate invocation.',
        '- Three saved profiler outputs per candidate (15 total) also match its timing output '
        'bit for bit. All hot/cold expert stages still execute on all 64 cores.',
        '- Devices are checked Ready with 16 free cores, no loaded networks and no '
        'active networks before and after every timing/profile run.',
        '- This is one layer, prompt and precision. The result does not establish '
        'the same saving for all 48 layers or uninstrumented full-model execution. '
        'It also does not establish a global optimum over tile sizes or placements.', '',
        '## Artifacts and reproduction', '',
        '`initial.json`, `summary.json` and `uninstrumented.json` hold the three timing cohorts. Each case retains '
        'the rewritten ONNX, exact changes, compiler command/log, timing samples, output '
        'comparisons, SDK traces and analysis CSVs. QPCs are in RAM scratch and can be rebuilt. '
        'PNG/PDF/SVG charts and large model/trace artifacts stay local; scripts and this '
        'report are committed.', '',
        'Use a new output and scratch directory; the original cold-capacity capture and '
        'C2 control QPC must be available. Run `export`, `compile`, `timing` for the default '
        'four cases with `tools/moe_qwen3_overhead_ablation.py`. Then export/compile '
        '`drop_zero_tile_t16`. Run `timing --timing-prefix confirm` on '
        '`baseline drop_zero tile_t16 drop_zero_tile_t16`. Profile all five cases using '
        'their existing timing prefix (`confirm` for the combined-only case). Analyze '
        'the initial four with `--summary-name initial` and the confirmation four with '
        '`--timing-prefix confirm --summary-name summary`.', '',
        'Finally, run `compile` and `timing` on `baseline drop_zero_tile_t16` with '
        '`--stats-level 0 --timing-prefix uninstrumented`. Use `timing-summary` with '
        'those same arguments and `--summary-name uninstrumented`; no device trace '
        'is requested from uninstrumented QPCs. The stats-level-0 binaries and logs '
        'have separate paths, preserving the instrumented artifacts.', '',
        'Shared arguments for the analysis script:', '',
        '```bash',
        '--cold 9.17.2026/real_model/oracle_padding/cold_capacity_control \\',
        '--out 9.17.2026/real_model/oracle_padding/overhead_ablation \\',
        '--scratch /dev/shm/qwen3_overhead_wentao_20260922 \\',
        '--base-scratch /dev/shm/qwen3_cold_wentao_20260922',
        '```', '',
        'Use `/home/chihao/qeff-venv/bin/python` for export/compile/run/analyze and '
        '`/tmp/qwen3_profile_plot_wentao_20260921/bin/python '
        'tools/moe_qwen3_overhead_report.py --root OUTPUT` for this report and plots.', '',
    ]
    (root/'README.md').write_text('\n'.join(lines))
    return cases


def plots(root, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.dpi':180, 'svg.fonttype':'none'})

    def save(fig, stem):
        for ext in ['png','pdf','svg']:
            fig.savefig(root/f'{stem}.{ext}')
        plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
    x = np.arange(len(cases))
    labels = [LABELS[c['case']].replace(' + ', '\n+ ').replace(' read/add', '\nread/add') for c in cases]
    host = np.array([c['host_median_ms'] for c in cases])
    axes[0].bar(x-.18, host, width=.36, color='#397fad', label='Host median')
    axes[0].errorbar(x-.18,host,yerr=[host-np.array([c['host_p10_ms'] for c in cases]),
        np.array([c['host_p90_ms'] for c in cases])-host],fmt='none',ecolor='#333',capsize=3)
    axes[0].bar(x+.18,[c['device_ms'] for c in cases],width=.36,color='#79b4c0',label='SDK device interval')
    axes[1].bar(x-.18,[c['combine_span_ms'] for c in cases],width=.36,color='#9477af',label='Combine elapsed span')
    axes[1].bar(x+.18,[c['local_all_max_core_ms'] for c in cases],width=.36,color='#d7b1df',label='Local reduction max-core work')
    for ax,title in zip(axes,['Complete MoE replay','Reduction observations (not additive)']):
        ax.set(xticks=x,xticklabels=labels,ylabel='Time (ms)',title=title)
        ax.tick_params(axis='x',labelsize=8)
        ax.grid(axis='y',alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle('Instrumented Qwen3 layer 2 · hot128/cold2 · identical inputs and weights',fontsize=13)
    save(fig,'overhead_results')

    fig, axes = plt.subplots(len(cases),1,figsize=(12,2.7*len(cases)),sharex=True,layout='constrained')
    width = max(c['combine_span_ms'] for c in cases)*1.08
    for ax,case in zip(np.atleast_1d(axes),cases):
        origin = case['local_start_ms']
        for r in representative(root,case):
            if r['card']!='0' or r['engine']!='HVX' or r['kind']!='aicbatchedreduceadd':
                continue
            if r['node']==STEM+'Einsum_4':
                color='#d68639'
            elif r['node']==STEM+'Einsum_3' or r['node'].startswith(STEM+'reduce_tile_'):
                color='#397fad' if r['memory']=='DDR' else '#36896e'
            else:
                continue
            ax.broken_barh([(float(r['start_us'])/1000-origin,float(r['duration_us'])/1000)],
                           (int(r['core'])-.34,.68),facecolors=color,linewidth=0)
        ax.set(yticks=range(0,16,2),ylim=(15.7,-.7),xlim=(0,width),ylabel='Card 0 core',
               title=f'{LABELS[case["case"]]} — combine span {case["combine_span_ms"]:.3f} ms')
        ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlabel('Time since earliest local reduction across cards (ms)')
    fig.legend(handles=[Patch(color=c,label=l) for c,l in [('#397fad','DDR-backed local reduction'),
        ('#36896e','TCM local reduction'),('#d68639','Final card-0 reduction')]],
        loc='outside lower center',ncol=3,fontsize=9)
    save(fig,'reduction_cores')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    plots(args.root,report(args.root))


if __name__=='__main__':
    main()
