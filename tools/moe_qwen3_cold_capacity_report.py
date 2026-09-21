#!/usr/bin/env python3
"""Render the controlled Qwen3 cold-capacity sweep from saved measurements."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def report(root):
    data = read(root/'summary.json')
    cases = data['cases']
    by_capacity = {c['capacity']:c for c in cases}
    base, small, best = by_capacity[128], by_capacity[2], min(cases,key=lambda c:c['host_median_ms'])
    packages = read(root/'package_audit.json')
    constants = {c:sum(p['bytes'] for p in v.values()) for c,v in packages.items()}
    graph = read(root/'graph_validation.json')
    if not graph['identical_except_cold_constants'] or constants['128']!=constants['2']:
        raise ValueError('Missing structural/package controls')
    if any(not c['output_validation']['bit_exact'] or c['hot_cores']!=64 or c['cold_cores']!=64 for c in cases):
        raise ValueError('Report assumptions need review')
    lines = [
        '# Controlled cold-group capacity sweep', '',
        'Measured 2026-09-22, SDK/runtime/firmware 1.21.6, cards 0–3, 16 cores per card.', '',
        f'**With weights, routing and expert order fixed, shrinking cold capacity from 128 to 2 '
        f'reduces isolated MoE host latency from {base["host_median_ms"]:.3f} to '
        f'{small["host_median_ms"]:.3f} ms '
        f'({(1-small["host_median_ms"]/base["host_median_ms"])*100:.2f}% reduction, '
        f'{base["host_median_ms"]/small["host_median_ms"]:.3f}×).** '
        f'C{best["capacity"]} is the fastest tested capacity at {best["host_median_ms"]:.3f} ms. '
        'The smallest logical capacity is not the fastest compiled graph.', '',
        '## Controlled scope', '',
        f'- Trained Qwen3-30B-A3B FP16 MoE, zero-based layer {data["layer"]}, batch 1, 128 tokens, '
        'hidden size 2048, intermediate size 768, top 8 of 128 experts.',
        '- Capture the actual expert input and already-regrouped routing by running the trained model '
        'prefix on the same saved GSM8K prompt 41. Both captures repeat bit for bit; '
        'all 128 expert counts exactly match the earlier full-model calibration.',
        '- Keep two sequential groups of 64 experts, the same bank order and four-card partition, '
        'and hot capacity C128. The captured maxima are hot 92 / cold 2. '
        'Both groups execute HMX work on all 64 cores in every trace.',
        '- Sweep cold C128, C64, C32, C16, C8, C4, C2. After normalizing the cold Slice end '
        'and valid-row Range stop, all exported ONNX graphs are byte-identical. '
        'All six trained weight-bank references and the replay input hash are identical.',
        '- This controls logical expert placement and compilation options; physical kernel tiling, '
        'packing and scheduling remain compiler decisions and may change with capacity.',
        '- The replay begins after routing and routing-column permutation. It includes native packing, '
        'both expert groups, unpacking and dense combination, plus a single-input ABI split '
        'and diagnostic count output. Host timing includes input/output transfer, enqueue and wait; '
        'it excludes compilation, loading, activation, warmups and prefix capture.', '',
        'This is an isolated layer replay with hot C128, not another full-model measurement. '
        'Its cold-stage timings must not replace the earlier 48-layer averages '
        '(approximately 3.2 → 1.7 ms), which used layer-specific hot capacities and a different '
        'surrounding execution schedule.', '',
        '## Results', '',
        'All QPCs use `-stats-level=70`. Host values are medians of 300 invocations '
        '(three rounds × 100, 10 warmups per round, reversed capacity order in round 2). '
        'P10/P90 describe individual invocation spread, not confidence intervals. '
        'Each case also has three independent SDK traces; all trace metrics in a row come '
        'from the same median-device-duration trace.', '',
        '| Cold capacity | Host median, ms | Host P10–P90, ms | SDK device interval, ms | Cold execution span, ms |',
        '|---:|---:|---:|---:|---:|',
    ]
    for c in cases:
        lines.append(f'| {c["capacity"]} | {c["host_median_ms"]:.3f} | '
                     f'{c["host_p10_ms"]:.3f}–{c["host_p90_ms"]:.3f} | '
                     f'{c["device_ms"]:.3f} | {c["cold_span_ms"]:.3f} |')
    lines += ['', '![Capacity sweep](capacity_sweep.png)', '',
        '## Kernels and waits', '',
        'Cold execution span runs from the earliest cold HMX convolution start to the latest '
        'cold HMX convolution end across the four cards. It includes intervening dependencies '
        'and transfers; it is not pure GEMM arithmetic.', '',
        '| Cold capacity | Cold HMX events per core | Median-core HMX compute, ms | Median-core HMX dependency wait within cold span, ms |',
        '|---:|---:|---:|---:|']
    for c in cases:
        if c['cold_hmx_events_min_core']!=c['cold_hmx_events_max_core']:
            raise ValueError('Nonuniform kernel event count')
        lines.append(f'| {c["capacity"]} | {c["cold_hmx_events_min_core"]} | '
                     f'{c["cold_hmx_busy_median_core_ms"]:.4f} | '
                     f'{c["cold_hmx_wait_clipped_median_core_ms"]:.4f} |')
    lines += ['',
        f'At C2, the cold span is {small["cold_span_ms"]:.3f} ms, while median-core HMX '
        f'compute is only {small["cold_hmx_busy_median_core_ms"]:.4f} ms and its clipped '
        f'dependency waits are {small["cold_hmx_wait_clipped_median_core_ms"]:.3f} ms. '
        'The corresponding busiest-core HMX compute is '
        f'{small["cold_hmx_busy_max_core_ms"]:.4f} ms. The trace therefore supports a '
        'dependency/transfer floor, rather than millisecond-scale HMX arithmetic at C2.', '',
        'The SDK emits `sync HMX` with zero visible duration and records its actual duration in '
        '`opSyncDurUs`. We reconstruct each wait as `[ts, ts + opSyncDurUs]`, clip it to the '
        'cold-stage span, and sum only within each individual HMX thread. Every wait is paired '
        'with its compute event by core and lowered operation ID; wait-end/compute-start '
        'discrepancies are recorded and required to be below 1 µs. We also reject overlapping '
        'waits on a thread. Pre-stage waits are excluded, preventing earlier hot work from '
        'being charged to the cold stage. This follows the SDK analyzer’s treatment of '
        '`opSyncDurUs` (`/opt/qti-aic/tools/opstats-profiling/qaic-opstats-analyzer.py`).', '',
        'Core-local compute and wait totals are separate observations, not values to add across '
        '64 parallel cores or subtract from global wall time. A sync event identifies a blocked '
        'dependency; it does not establish whether that entire wait is caused by DDR, '
        'communication, vector work, or scheduling.', '',
        'The following fixed card-0/core-0 examples show HMX waits alongside recorded DDR→TCM '
        'DMA intervals attributed to the three cold projections. DMA intervals can include '
        'asynchronous completion/queueing and overlap HMX waits. They are not additive '
        'exclusive memory costs. The illustrated core is chosen in advance, not selected for '
        'the largest wait.', '',
        '![Cold-stage example timelines](cold_timelines.png)', '',
        'C128 executes 26 cold HMX convolution events per core; every C64–C2 case executes six. '
        'The cold stage remains on all 64 cores even though 42 of its 64 experts receive no '
        'tokens in this captured input. The source graph still describes all 64 experts; '
        'this capacity specialization does not remove empty experts from the topology. '
        'Testing the benefit of omitting them is a separate ablation. The available trace '
        'does not establish useful arithmetic for every padded row or expose enough physical matrix '
        'tile dimensions to claim an exact hardware minimum row size.', '',
        '## Work that capacity does not remove', '',
        f'- The compiled static constants occupy **{constants["128"]:,} bytes** in both '
        'C128 and C2, with identical per-card sizes. Hashes differ, so the packages are not '
        'byte-identical; the unchanged source bank references establish trained-weight identity. '
        'Each cold group still contains 64 × 3 × 2048 × 768 × 2 bytes = **576 MiB** of '
        'logical FP16 weights. Constant storage size is not a measurement of runtime DDR traffic.',
        '- Native dense combination still has a `[64,128,2048]` FP16 accumulator, **32 MiB**, '
        'independent of cold capacity. The final local/global combination remains expensive.',
        '- Hot capacity stays C128; this work is intentionally unchanged by the sweep.', '',
        '| Observed phase, ms | Cold C128 | Cold C2 |', '|---|---:|---:|']
    for name,key in [('Hot group','hot_span_ms'),('Between groups','between_groups_ms'),
                     ('Cold group','cold_span_ms'),('Last unpack','last_unpack_ms'),('Final combine','combine_ms')]:
        lines.append(f'| {name} | {base[key]:.3f} | {small[key]:.3f} |')
    lines += ['',
        'These five nonoverlapping phases omit initial packing/input handling and trailing '
        'device work; they are not a complete accounting of host latency. Phase boundaries '
        'do not isolate exclusive causal memory/communication costs.', '',
        '## Algorithm implications', '',
        '1. Select from measured compiled capacities. Minimum-safe capacity alone is not a '
        'latency optimizer: C64 is slower than C128 here, and C32 has the best measured host '
        'median. The roughly 1% advantage of C32 over C2 is specific to this input and build.',
        '2. Further large gains need to address work that survives small padding: inactive '
        'expert execution, weight movement/dependencies, and dense scatter/reduction. '
        'The next useful ablation is to omit empty experts or replace dense combination '
        'with a sparse token-oriented path while preserving numerical output.',
        '3. Keep runtime selection, grouping changes, additional prompts/layers, and other '
        'precisions as separate experiments. This FP16 layer-2 result does not establish '
        'their overhead or generalization.', '',
        '## Validation and artifacts', '',
        '- Seven structurally controlled graphs; identical captured input and weight references.',
        '- 2,100 timed invocations, with the saved final output from every timing round '
        'bit-identical to C128. The host does not save or compare every intermediate invocation.',
        '- 21 profiled invocations, every saved output bit-identical to its timing reference.',
        '- All saved expert counts exact; top-8 assignments sum to 1024; no capacity overflow.',
        '- All 21 traces cover four cards × 16 cores in both expert stages. Devices return '
        'Ready with all 16 cores free and no loaded or active networks after every host/profile run.',
        '- Equality validates padding equivalence to the trained FP16 control; it is not an '
        'independent FP32 accuracy test.', '',
        '`summary.json`, `samples.csv`, `graph_validation.json`, `package_audit.json` and '
        'each `c*/analysis/` contain the measurements. Each case retains ONNX, export '
        'metadata, compiler logs, timing runs, raw SDK stats and merged traces. Prefix '
        'captures and input hashes are in `prefix/`. QPCs and extracted constant packages '
        'are in the RAM scratch directory and are reproducible, not persistent artifacts.', '',
        'The charts are also saved as PDF and SVG. Large experiment artifacts stay local '
        'under the repository’s existing policy; scripts and this report are committed.', '',
        '## Reproduce', '',
        'Use a fresh output and scratch directory. The SDK/QEff Python environment needs '
        'NumPy and ONNX; the report environment needs NumPy and Matplotlib. The full-model '
        'oracle graphs, materialized weights and saved reference prompt must already exist.', '',
        '```bash',
        'experiment_out=9.17.2026/real_model/oracle_padding/cold_capacity_repeat',
        'experiment_scratch=/dev/shm/qwen3_cold_capacity_repeat',
        'for action in prefix export compile timing profile analyze; do',
        '  /home/chihao/qeff-venv/bin/python tools/moe_qwen3_cold_capacity.py "$action" \\',
        '    --oracle 9.17.2026/real_model/oracle_padding \\',
        '    --out "$experiment_out" --scratch "$experiment_scratch"',
        'done',
        '/home/chihao/qeff-venv/bin/python tools/moe_qwen3_cold_capacity.py package \\',
        '  --oracle 9.17.2026/real_model/oracle_padding \\',
        '  --out "$experiment_out" --scratch "$experiment_scratch" --capacities 128 2',
        '/tmp/qwen3_profile_plot_wentao_20260921/bin/python \\',
        '  tools/moe_qwen3_cold_capacity_report.py --root "$experiment_out"',
        '```', '',
    ]
    (root/'README.md').write_text('\n'.join(lines))
    return cases


def plots(root,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180,'svg.fonttype':'none'})
    x=np.arange(len(cases));labels=[str(c['capacity']) for c in cases]
    fig,ax=plt.subplots(1,2,figsize=(11.2,4.1),layout='constrained')
    host=np.array([c['host_median_ms'] for c in cases])
    lower=host-np.array([c['host_p10_ms'] for c in cases])
    upper=np.array([c['host_p90_ms'] for c in cases])-host
    ax[0].errorbar(x,host,yerr=[lower,upper],fmt='o-',capsize=4,color='#2066a8',label='Host median; P10–P90')
    ax[0].plot(x,[c['device_ms'] for c in cases],'s--',color='#a76524',label='SDK device interval')
    ax[0].set(title='Whole isolated MoE replay',ylabel='Time (ms)',ylim=(0,14))
    for key,label,color in [('cold_span_ms','Cold execution span','#2066a8'),
                           ('cold_hmx_wait_clipped_median_core_ms','Median-core HMX wait','#d28a32'),
                           ('cold_hmx_busy_median_core_ms','Median-core HMX compute','#28856a')]:
        ax[1].plot(x,[c[key] for c in cases],'o-',label=label,color=color)
    ax[1].set(title='Cold group: elapsed span and core-local observations',ylabel='Time (ms)',ylim=(0,4.3))
    for panel in ax:
        panel.set_xticks(x,labels);panel.set_xlabel('Cold capacity (hot capacity fixed at 128)')
        panel.grid(axis='y',alpha=.2);panel.legend(fontsize=8,loc='upper right')
    fig.suptitle('Qwen3 FP16 layer 2 · same real input, routing and weights · four cards',fontsize=13)
    for ext in ['png','pdf','svg']:fig.savefig(root/f'capacity_sweep.{ext}')
    plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(11.2,5.5),layout='constrained',sharex=True)
    for panel,capacity in zip(axes,[128,2]):
        case=next(c for c in cases if c['capacity']==capacity)
        directory=root/f'c{capacity}/analysis';sample=case['sample']
        nodes=list(csv.DictReader((directory/'nodes.csv').open()))
        cold=[r for r in nodes if int(r['sample'])==sample and r['engine']=='HMX' and
              r['kind']=='aicconvolutiond32' and r['node'].rsplit('/',1)[-1] in ['MatMul_3','MatMul_4','MatMul_5']]
        start=min(float(r['first_us']) for r in cold);end=max(float(r['last_us']) for r in cold)
        detail=list(csv.DictReader((directory/f'card0_core0_sample{sample}.csv').open()))
        for row in detail:
            if row['stage']!='cold':continue
            if row['name']=='sync HMX':y,color,dur=1,'#d9a052',float(row['sync_us'])
            elif row['engine']=='HMX' and row['name']=='aicconvolutiond32':y,color,dur=1,'#168265',float(row['duration_us'])
            elif row['engine']=='DMAIssue_DMA' and row['memory']=='DDR' and not row['name'].startswith(('sync ','barrier ')):
                y,color,dur=0,'#447eaf',float(row['duration_us'])
            else:continue
            lo=max(float(row['start_us']),start);hi=min(float(row['start_us'])+dur,end)
            if hi>lo:panel.broken_barh([((lo-start)/1000,(hi-lo)/1000)],(y-.25,.5),facecolors=color)
        panel.axvline((end-start)/1000,color='#555',ls=':',lw=1)
        panel.set(title=f'C{capacity}: cold span {case["cold_span_ms"]:.3f} ms; card 0 / core 0 shown',
                  yticks=[0,1],yticklabels=['DDR → TCM DMA','HMX'],ylim=(-.6,1.6))
        panel.grid(axis='x',alpha=.2)
    axes[-1].set_xlabel('Time since earliest cold HMX computation across all cores (ms)')
    axes[0].legend(handles=[Patch(color='#d9a052',label='Dependency wait'),Patch(color='#168265',label='Compute'),
                           Patch(color='#447eaf',label='Recorded DMA interval')],loc='upper right',fontsize=8)
    fig.suptitle('Waits dominate the small-capacity cold stage; overlapping rows are not additive',fontsize=12)
    for ext in ['png','pdf','svg']:fig.savefig(root/f'cold_timelines.{ext}')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    cases=report(args.root)
    plots(args.root,cases)


if __name__=='__main__':
    main()
