#!/usr/bin/env python3
"""Plot measured routing/capacity cohorts; leave interpretation in the report."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def label(name):
    capacity, _, scan = name.partition('_')
    return capacity.upper() + {'': ' / original scan', 'rows4': ' / 4 row slices',
                              'rows16': ' / 16 row slices', 'tree': ' / shift-add scan'}.get(scan, ' / '+scan)


def save(fig, path):
    for suffix in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


def report(root):
    cohorts = {key:read(root/f'{key}.json') for key in ['screen','combined','confirm']}
    profiles = read(root/'profiles.json')['cases']
    c2_profiles = [c for c in profiles if c['case'] in ['c2','c2_rows16','c2_tree']]
    if len(c2_profiles)!=3 or {s['p2p_bytes'] for c in c2_profiles for s in c['samples']}!={67093248}:
        raise ValueError('Matched C2 P2P claim no longer holds')
    timed, saved = 0, 0
    for cohort in cohorts.values():
        if cohort['stats_level'] != 0:
            raise ValueError('Host cohorts must have instrumentation disabled')
        for case in cohort['cases']:
            actual = []
            for i in range(cohort['rounds']):
                path = root/case['case']/f'{cohort["timing_prefix"]}_r{i}'
                validation = read(path/'validation.json')
                if not all(validation[k] for k in ['bit_exact','counts_exact','no_overflow']):
                    raise ValueError('Unvalidated timing output')
                actual.extend(read(path/'timing.json')['samples_ms'])
                saved += 1
            if len(actual) != case['host_samples'] or float(np.median(actual)) != case['host_median_ms']:
                raise ValueError('Summary differs from raw timings')
            timed += len(actual)
    confirm = cohorts['confirm']['cases']
    base = next(c for c in confirm if c['case']=='c2')
    best = next(c for c in confirm if c['case']=='c2_tree')
    padded = next(c for c in confirm if c['case']=='c32_tree')
    c2_round_wins = sum(a<b for a,b in zip(best['host_round_medians_ms'],padded['host_round_medians_ms']))
    lines = ['# Qwen3 layer-2 routing scans and padding retuning', '',
        'Measured 2026-09-22 on four AI 100 cards, 16 cores per card, SDK 1.21.6, FP16.', '',
        f'**The fresh confirmation lowers isolated MoE host latency from {base["host_median_ms"]:.3f} '
        f'to {best["host_median_ms"]:.3f} ms ({100*(1-best["host_median_ms"]/base["host_median_ms"]):.2f}% '
        f'lower; {base["host_median_ms"]/best["host_median_ms"]:.3f}×).** '
        f'The reported default is `{best["case"]}`. The control already includes '
        'the previous zero-read removal and local reduction split into 16 token tiles; this is an additional gain.', '',
        '## Scope and graph changes', '',
        'The source is `../overhead_ablation/drop_zero_tile_t16/model.onnx`: trained zero-based layer 2, '
        'batch 1, 128 tokens, hidden 2048, intermediate 768, 128 experts, top-8 routing. '
        'The captured input remains the first 128 tokens of GSM8K prompt 41. '
        'The six FP16 weight banks, expert order, hot capacity 128, expert projections, accumulator '
        'and output reductions are unchanged. Both 64-expert groups span all four cards. '
        'The replay excludes router/TopK, expert sorting and routing-column permutation, but includes '
        'the integer prefix sums and token packing/unpacking.', '',
        '- **Row slices:** split each `[64,128]` integer mask along the expert dimension into '
        '4 or 16 slices, independently run the original token-axis `CumSum`, and concatenate rows. '
        'This exposes independent work without changing the scan order within each row.',
        '- **Shift/add scan:** replace each inclusive `CumSum` with seven int32 stages. At distance '
        '`d = 1,2,4,8,16,32,64`, add the previous stage to itself shifted right by `d` token positions, '
        'with zeros on the left. This is an inclusive parallel prefix scan. Values remain exact '
        'integers in `[0,128]`; no FP16/FP32 conversion is involved.',
        '- **Padding:** vary only the cold Slice endpoint and Range limit over 2,4,8,16,32,64,128. '
        'The captured hot/cold maxima remain 92/2, so every tested capacity is safe for this input.', '',
        'The routing mask is still a runtime input. No captured prefix sums or token indices are '
        'embedded into the graph. These are separately compiled static capacities; runtime selection '
        'and changing expert placement are outside this experiment.', '',
        '## First screen: routing at cold capacity 2', '',
        '| Scan | Host median, ms | P10–P90, ms | Per-round medians, ms |', '|---|---:|---:|---|']
    for c in cohorts['screen']['cases']:
        if c['case']=='c2' or '_' in c['case']:
            lines.append(f'| {label(c["case"])} | {c["host_median_ms"]:.3f} | '
                         f'{c["host_p10_ms"]:.3f}–{c["host_p90_ms"]:.3f} | '+
                         ', '.join(f'{v:.3f}' for v in c['host_round_medians_ms'])+' |')
    lines += ['', '## Capacity screens', '',
        '| Cold capacity | Original scan, ms | Shift/add scan, ms |', '|---|---:|---:|']
    plain = {c['case']:c for c in cohorts['screen']['cases']}
    tree = {c['case']:c for c in cohorts['combined']['cases']}
    for cap in [2,4,8,16,32,64,128]:
        lines.append(f'| {cap} | {plain[f"c{cap}"]["host_median_ms"]:.3f} | '
                     f'{tree[f"c{cap}_tree"]["host_median_ms"]:.3f} |')
    small = [tree[f'c{cap}_tree']['host_median_ms'] for cap in [2,4,8,16,32]]
    lines += ['', f'With the shift/add scan, capacities 2–32 span only {min(small):.3f}–{max(small):.3f} ms '
        'in this screen. This narrow range does not support a strong preference for the lowest '
        'screening median alone.', '',
        'These columns are separate screening cohorts. Each variant has three rounds of '
        '100 timed invocations and 10 warmups per round, with reversed case order in the middle round. '
        'Selection from a screen is followed by a fresh matched confirmation; small differences among '
        'capacities should be assessed against the round-to-round spread.', '',
        '## Fresh confirmation, profiling disabled', '',
        '| Variant | Host median, ms | P10–P90, ms | Reduction vs control | Per-round medians, ms |',
        '|---|---:|---:|---:|---|']
    for c in confirm:
        lines.append(f'| {label(c["case"])} | {c["host_median_ms"]:.3f} | '
                     f'{c["host_p10_ms"]:.3f}–{c["host_p90_ms"]:.3f} | '
                     f'{100*(1-c["host_median_ms"]/base["host_median_ms"]):.2f}% | '+
                     ', '.join(f'{v:.3f}' for v in c['host_round_medians_ms'])+' |')
    lines += ['', f'C2 and C32 with the new scan differ by only '
        f'{1000*abs(best["host_median_ms"]-padded["host_median_ms"]):.3f} µs in pooled median. '
        f'C2 is faster in {c2_round_wins} of {cohorts["confirm"]["rounds"]} paired round medians. '
        'There is no consistent C32 latency advantage; prefer C2 for its smaller communication footprint.', '',
        f'The confirmation has {cohorts["confirm"]["rounds"]} rounds × '
        f'{cohorts["confirm"]["iterations"]} timed invocations per variant, reversing order in alternate rounds.', '',
        'Host timings use `-stats-level=0` and include set-data, enqueue, input/output transfer '
        'and wait. Compilation, loading, activation and warmup are excluded. P10/P90 describe invocation '
        'spread, not confidence intervals. Compilation of these variants does not overlap device measurements.', '',
        '![Routing and capacity results](routing_retune.png)', '',
        '## Placement and remaining scheduling cost', '',
        'The following metrics come from separate `-stats-level=70` builds. Each row uses the same '
        'median whole-device-duration sample among three validated captures.', '',
        '| Variant | Device, ms | Hot/cold scan cores | Hot/cold max-core scan work, µs | '
        'Hot/cold max-core scan span, µs | P2P payload, MiB |', '|---|---:|---:|---:|---:|---:|']
    for c in profiles:
        lines.append(f'| {label(c["case"])} | {c["device_ms"]:.3f} | '
                     f'{c["hot_scan_cores"]}/{c["cold_scan_cores"]} | '
                     f'{1000*c["hot_scan_max_core_ms"]:.3f}/{1000*c["cold_scan_max_core_ms"]:.3f} | '
                     f'{1000*c["hot_scan_max_core_span_ms"]:.3f}/{1000*c["cold_scan_max_core_span_ms"]:.3f} | '
                     f'{c["p2p_bytes"]/2**20:.4f} |')
    lines += ['', 'The original scan runs on core 0 of each card. The 16-row-slice graph places four '
        'scans on each card (card 0 cores 0–3, card 1 cores 4–7, card 2 cores 8–11, card 3 cores 12–15). '
        'The shift/add graph executes all seven integer-add stages on all 64 cores. This is observed '
        'kernel placement, not just potential parallelism in the source graph.', '',
        'Max-core work sums recorded HVX compute durations on each core and takes the largest sum. '
        'Max-core span includes the gaps between that core’s first and last scan kernels. Neither '
        'is the end-to-end routing phase: copies, masks, packing, dependencies and staggered core/card '
        'readiness also matter. In particular, the cold scan can have a wide global envelope even '
        'when each individual core finishes its scan quickly.', '',
        '| Variant | Hot expert span, ms | Between groups, ms | Cold expert span, ms | Combine span, ms |',
        '|---|---:|---:|---:|---:|']
    for c in profiles:
        lines.append(f'| {label(c["case"])} | {c["hot_span_ms"]:.3f} | {c["between_groups_ms"]:.3f} | '
                     f'{c["cold_span_ms"]:.3f} | {c["combine_span_ms"]:.3f} |')
    c2_tree = next(c for c in profiles if c['case']=='c2_tree')
    c32_tree = next(c for c in profiles if c['case']=='c32_tree')
    lines += ['', 'These are observational windows, not additive arithmetic costs. Changing routing '
        'also changes the compiled execution schedule. All three matched C2 routing variants retain '
        'the same 63.9851 MiB outgoing P2P payload. Both expert stages still execute on all 64 cores.', '',
        f'C32 with the shift/add scan sends {c32_tree["p2p_bytes"]/2**20:.3f} MiB, '
        f'{100*(c32_tree["p2p_bytes"]/c2_tree["p2p_bytes"]-1):.1f}% more than C2, without a large '
        'latency advantage. C2 is therefore a useful default when latency is comparable and '
        'communication volume matters. This is a choice within the measured cases, not a '
        'universal optimum for other layers or workloads.', '',
        '![Routing scan placement](routing_cores.png)', '',
        '## Validation and limitations', '',
        f'- {timed:,} timed invocations across the three cohorts. All {saved} saved timing-round '
        'outputs are byte-identical to the original FP16 C2 replay; only the final output of each '
        'round is saved, not every invocation.',
        f'- {sum(len(c["samples"]) for c in profiles)} saved profiling outputs also match their '
        'uninstrumented timing reference bit for bit. Counts are exact, sum to 1024 and never overflow.',
        '- Every routing rewrite is checked using the actual replacement ONNX subgraph against an '
        'int32 CPU prefix sum for empty, full, deterministic random and captured masks, separately '
        'for hot and cold groups. Export checks preserve all other original nodes and all original '
        'initializers except the two cold-capacity constants.',
        '- Every device run checks all four cards Ready, 16 free cores, zero loaded and active '
        'networks before and after execution.',
        '- This remains one captured layer/prompt/precision. Full-model gain and behavior across '
        'other routing distributions have not been measured. No empty experts are removed, and '
        'the dense accumulator, expert group widths and four-card partition configuration remain unchanged. '
        'Physical tiling and scheduling remain compiler decisions.', '',
        '## Artifacts and reproduction', '',
        '`screen.json`, `combined.json`, `confirm.json` and `profiles.json` hold the cohorts. '
        'Each case retains graph/export checks, exact compile commands, logs, timing samples and '
        'output validation. Profiled cases also retain raw SDK captures, decoded full-flow traces '
        'and per-core analysis CSVs. QPCs are in `/dev/shm/qwen3_retune_wentao_20260922`; the C2 '
        'control reuses the byte-identical graph and binaries from the overhead experiment.', '',
        'Use `/home/chihao/qeff-venv/bin/python tools/moe_qwen3_routing_retune.py ACTION` with:', '',
        '```bash', '--cold 9.17.2026/real_model/oracle_padding/cold_capacity_control \\\n'
        '--overhead 9.17.2026/real_model/oracle_padding/overhead_ablation \\\n'
        '--out NEW_OUTPUT_DIRECTORY --scratch NEW_RAM_SCRATCH_DIRECTORY \\\n'
        '--base-scratch /dev/shm/qwen3_cold_wentao_20260922 \\\n'
        '--overhead-scratch /dev/shm/qwen3_overhead_wentao_20260922', '```', '',
        'Run `export`, `compile`, `timing`, then `timing-summary`. Case names are `c2` through '
        '`c128`, optionally suffixed `_rows4`, `_rows16`, `_rows64` or `_tree`; rows64 is supported '
        'but was not measured here. The default stats level is 0. Use distinct timing prefixes '
        'and summary names for each cohort. Export each graph once; use `compile --stats-level 70` '
        'for separately stored profiling binaries, then `profile` and `analyze` with stats level 70. '
        'Profiling may validate against saved stats-level-0 timing outputs from the same graph.', '',
        'Recorded timing cohorts (pass the name as both `--timing-prefix` and `--summary-name`):', '']
    for name,cohort in cohorts.items():
        lines.append(f'- `{name}`: `--cases '+ ' '.join(c['case'] for c in cohort['cases'])+
                     f' --rounds {cohort["rounds"]} --iterations {cohort["iterations"]}`.')
    lines += ['', 'Profile `c2 c2_rows16 c2_tree` against `--timing-prefix screen` and `c32_tree` '
        'against `--timing-prefix combined`; analyze all four with `--summary-name profiles`.', '',
        'Regenerate this report and PNG/PDF/SVG plots with '
        '`/tmp/qwen3_profile_plot_wentao_20260921/bin/python tools/moe_qwen3_routing_retune_report.py '
        '--root OUTPUT`. Large artifacts remain local; scripts and Markdown are committed.', '']
    (root/'README.md').write_text('\n'.join(lines))


def plots(root):
    screen = read(root/'screen.json')['cases']
    combined = read(root/'combined.json')['cases']
    confirm = read(root/'confirm.json')['cases']
    profiles = read(root/'profiles.json')['cases']
    caps = sorted([c for c in screen if '_' not in c['case']],key=lambda c:int(c['case'][1:]))
    fig,axes = plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    ax = axes[0]
    x = np.arange(len(caps))
    ax.plot(x,[c['host_median_ms'] for c in caps],marker='o',color='#246a91',label='Original scan')
    ax.fill_between(x,[c['host_p10_ms'] for c in caps],[c['host_p90_ms'] for c in caps],
                    color='#246a91',alpha=.15)
    tree = sorted(combined,key=lambda c:int(c['case'].split('_')[0][1:]))
    ax.plot(x,[c['host_median_ms'] for c in tree],marker='o',color='#217f70',label='Shift/add scan')
    ax.fill_between(x,[c['host_p10_ms'] for c in tree],[c['host_p90_ms'] for c in tree],
                    color='#217f70',alpha=.15)
    ax.set_xticks(x,[c['case'][1:] for c in caps])
    ax.set(xlabel='Cold capacity (hot capacity = 128)',ylabel='Host latency (ms)',
           title='Separate capacity screens; bands = P10–P90')
    ax.legend(frameon=False)
    ax = axes[1]
    y = np.arange(len(confirm))
    bars = ax.barh(y,[c['host_median_ms'] for c in confirm],
                  color=['#7c8994' if c['case']=='c2' else '#217f70' for c in confirm])
    ax.set_yticks(y,[label(c['case']) for c in confirm])
    ax.invert_yaxis()
    ax.set(xlabel='Host latency (ms)',title='Fresh confirmation: profiling disabled')
    ax.set_xlim(0,max(c['host_p90_ms'] for c in confirm)*1.16)
    for bar,c in zip(bars,confirm):
        ax.text(bar.get_width()+.05,bar.get_y()+bar.get_height()/2,
                f'{c["host_median_ms"]:.3f}',va='center')
    for ax in axes:
        ax.grid(axis='y' if ax is axes[0] else 'x',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Qwen3 trained layer 2 · FP16 · four cards × 16 cores · 128 tokens')
    save(fig,root/'routing_retune')

    # Zoom into hot routing; the cold scan has the same placement but substantial
    # readiness skew between cores. Retain true widths and show all 16 core rows.
    fig,axes = plt.subplots(len(profiles),1,figsize=(11,2.3*len(profiles)),
                            squeeze=False,layout='constrained')
    for ax,case in zip(axes[:,0],profiles):
        with (root/case['case']/'analysis/work.csv').open() as stream:
            work = [r for r in csv.DictReader(stream) if int(r['sample'])==case['sample']
                    and int(r['card'])==0 and r['engine']=='HVX'
                    and 'CumSum' in r['node'] and 'CumSum_1' not in r['node']
                    and r['kind']!='aicendcyclestats']
        origin = min(float(r['start_us']) for r in work)
        end = max(float(r['end_us']) for r in work)-origin
        for r in work:
            ax.broken_barh([(float(r['start_us'])-origin,float(r['duration_us']))],
                          (int(r['core'])-.35,.7),facecolors='#246a91')
        ax.set(ylim=(-.7,15.7),yticks=[0,4,8,12,15],ylabel='Card 0 core',
               title=label(case['case']),xlim=(-end*.02,end*1.02),
               xlabel='Microseconds from this panel’s first hot-scan kernel')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Hot routing on card 0 · median-duration captures · horizontal scales differ')
    save(fig,root/'routing_cores')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    root = parser.parse_args().root
    report(root)
    plots(root)
