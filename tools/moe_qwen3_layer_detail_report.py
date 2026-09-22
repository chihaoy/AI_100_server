#!/usr/bin/env python3
"""Render card/core mechanisms from audited Qwen3 layer-2 replay traces."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np


STEM = '/model/layers.2/mlp/'
PHASES = [
    ('initial_pack_ms', 'Routing / initial packing', '#80b1d3'),
    ('group0_ms', 'Hot expert span', '#1b9e77'),
    ('between_groups_ms', 'Between groups', '#b3de69'),
    ('group1_ms', 'Cold expert span', '#d95f02'),
    ('last_unpack_ms', 'Last unpack', '#fdb462'),
    ('combine_ms', 'Dense combination', '#7570b3'),
    ('tail_ms', 'Device tail', '#999999'),
]


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def directory(root, case):
    return root/f'c{case["capacity"]}/sample{case["sample"]}'


def span(case, key):
    values = [s[key] for s in case['samples']]
    return f'{min(values):.3f}–{max(values):.3f}'


def core_work(work, node, kind):
    selected = [r for r in work if r['node'] == STEM+node and r['kind'] == kind
                and r['engine'] in ['HMX', 'HVX']]
    by_core = defaultdict(float)
    for r in selected:
        by_core[int(r['card']), int(r['core'])] += float(r['duration_us'])/1000
    return len(by_core), statistics.median(by_core.values()), max(by_core.values())


def idle_gaps(root, case):
    """Audit the C2 idle-looking intervals using the same clock origin as the figure."""
    src = directory(root, case)
    origin = case['device_start_ms']
    work, waits = rows(src/'work.csv'), rows(src/'waits.csv')

    def interval(r):
        return dict(node=r['node'], kind=r['kind'], core=int(r['core']),
                    start_ms=float(r['start_us'])/1000-origin,
                    end_ms=float(r['end_us'])/1000-origin,
                    duration_ms=float(r['duration_us'])/1000)

    result = dict(capacity=case['capacity'], sample=case['sample'], origin_sdk_ms=origin,
                  time_origin='device execution start')
    for name, node, memory in [('zero_accumulator', 'ConstantOfShape_1', 'DDR'),
                               ('cold_prefix_sum', 'CumSum_1', 'TCM'),
                               ('dense_reduction', 'Einsum_3', 'DDR')]:
        selected = [r for r in work if r['card']=='0' and r['core']=='0' and
                    r['engine']=='HVX' and r['node']==STEM+node and r['memory']==memory]
        if len(selected) != 1:
            raise ValueError(f'Ambiguous core-0 interval: {name}')
        result[name] = interval(selected[0])
    result['last_gemm_end_by_card0_core_ms'] = [
        max(float(r['end_us'])/1000-origin for r in work if r['card']=='0' and
            int(r['core'])==core and r['engine']=='HMX' and r['kind']=='aicconvolutiond32')
        for core in range(16)]
    result['core8_terminal_waits'] = [interval(r) for r in waits if r['card']=='0' and
        r['core']=='8' and r['engine']=='HMX' and r['kind'] in ['aicendcyclestats', 'aicoutputsemaphoreinc']]
    result['final_receive_events'] = [dict(port=r['port'], **interval(r)) for r in work
        if r['node']==STEM+'Einsum_3' and r['port'].startswith('0<-')]
    (root/'idle_gap_check.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def report(root, data):
    cases = data['cases']
    base, small = cases
    work = {c['capacity']: rows(directory(root, c)/'work.csv') for c in cases}
    producers = {c['capacity']: rows(directory(root, c)/'producers.csv') for c in cases}
    p2p = {c['capacity']: {r['node'].removeprefix(STEM): r for r in rows(directory(root, c)/'p2p_summary.csv')} for c in cases}
    gaps = idle_gaps(root, small)
    lines = [
        '# Qwen3 layer 2: card and core profiling', '',
        'Analyzed 2026-09-22; trained Qwen3-30B-A3B FP16, batch 1, prefill 128, '
        'top 8 of 128 experts, four AI 100 cards × 16 cores; SDK 1.21.6.', '',
        '**The large-padding cold stage exposes cross-card activation dependencies. '
        'Small padding nearly removes the stagger in down-projection starts, but retains '
        'weight-sized DDR copies, core-0 routing/reduction work and dense combination. '
        'These are distinct mechanisms; the traces do not support calling every wait a weight wait.**', '',
        '## Scope and timing', '',
        'Two views of the same zero-based layer are kept separate:', '',
        '- **Inside the full model:** the router-inclusive MoE interval for native C128/128, '
        'regrouped minimum C92/2, and regrouped power-of-two C128/2. Each row uses its '
        'median layer-2 duration among three existing validated traces. Expert regrouping '
        'changes between native and the two oracle cases.',
        '- **Controlled replay:** the same captured layer input, already-regrouped routing, '
        'weights and expert order; hot C128 is fixed and only cold capacity changes C128 → C2. '
        'The replay excludes router and routing-column permutation; it includes packing, both '
        'groups, dense combination, an input ABI split and a diagnostic count output. '
        'Each row uses the median whole-device duration among three saved captures.', '',
        'The input is the first 128 tokens of saved GSM8K prompt 41. These measurements are '
        'input-specific. Both sequential expert groups span **all four cards** and every '
        'core; they are not assigned to separate pairs of cards. The compiler chooses '
        'physical tiling and intermediate placement.', '',
        '### Layer inside the full model', '',
        '| FP16 policy | Capacities | Routing / initial pack | Hot span | Between groups | Cold span | Last unpack | Dense combine | Total MoE |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    labels = {'native128': ('Native', '128 / 128'), 'min': ('Minimum + regrouping', '92 / 2'),
              'pow2': ('Power-of-two + regrouping', '128 / 2')}
    for r in data['full_model']:
        label, capacity = labels[r['variant']]
        values = [float(r[k]) for k, _, _ in PHASES[:-1]]+[float(r['moe_ms'])]
        lines.append(f'| {label} | {capacity} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines += ['', 'All table values are milliseconds. Hot/cold label group 0/1 in the '
        'regrouped graphs; native groups retain their original expert order. '
        'The minimum-policy hot stage is slower '
        'in this layer; the smaller cold group alone does not determine the complete layer speedup. '
        'The full-model interval starts at router HMX work and ends at final-combine compute; '
        'it excludes earlier speculative prefetch and the following residual.', '',
        '### Controlled replay: complete device interval', '',
        '| Observational phase, ms | Cold C128 | Cold C2 |', '|---|---:|---:|']
    for key, label, _ in PHASES:
        if key == 'initial_pack_ms':
            label = 'Input handling / initial packing (no router)'
        lines.append(f'| {label} | {base["phases"][key]:.3f} | {small["phases"][key]:.3f} |')
    lines += [f'| **Device total** | **{base["device_ms"]:.3f}** | **{small["device_ms"]:.3f}** |',
        f'| Host median, 300 invocations | {base["host_median_ms"]:.3f} | {small["host_median_ms"]:.3f} |', '',
        'These phases add exactly to the device interval. Boundaries are device start, '
        'first/last hot HMX, first/last cold HMX, first local reduction, last final-combine '
        'compute, and device end. DMA, packing and scheduling can overlap a labeled expert span; '
        'these are elapsed phases, not exclusive causal costs. Local and global reductions '
        'also overlap between cards and remain one combination phase.', '',
        '![Layer budgets](layer_budgets.png)', '',
        '## Between cards: activation exchange releases groups of cores', '',
        'Cold `Mul_9` is the activation after gate/up and SiLU, before the down projection. '
        'Its exchange has 12 directed edges: each card sends to all three peers. '
        'The down projection then consumes locally available and remotely produced shards. '
        'The source-node mapping comes from the compiled graph and SDK dependencies.', '',
        '| Outgoing P2P payload, counted once per directed edge | C128 | C2 |',
        '|---|---:|---:|']
    for name, label in [('Mul_4', 'Hot intermediate activations'), ('Mul_9', 'Cold intermediate activations'),
                        ('Add', 'Hot accumulator update'), ('Add_2', 'Cold accumulator update'),
                        ('CtxGather3D_3/n14', 'Cold input packing'), ('Einsum_3', 'Final partial results')]:
        values = [int(p2p[c].get(name, {}).get('bytes', 0))/2**20 for c in [128, 2]]
        lines.append(f'| {label} (`{name}`) | {values[0]:.4f} MiB | {values[1]:.4f} MiB |')
    lines += [f'| All replay P2P sites, including index exchange | {base["p2p_send_bytes"]/2**20:.4f} MiB | {small["p2p_send_bytes"]/2**20:.4f} MiB |', '',
        'Cold intermediate payload falls from **3 MiB to 48 KiB per edge (64×)**. '
        'The complete cold activation exchange falls from 36 MiB to 0.5625 MiB. '
        'C2 introduces an additional 1.5 MiB P2P site during input packing; smaller '
        'logical shapes can change the compiler communication plan.', '',
        '| First cold down-GEMM start across all 64 cores | C128 | C2 |', '|---|---:|---:|',
        f'| Representative spread, ms | {base["cold_first_down_spread_ms"]:.6f} | {small["cold_first_down_spread_ms"]:.6f} |',
        f'| Range across three captures, ms | {span(base, "cold_first_down_spread_ms")} | {span(small, "cold_first_down_spread_ms")} |', '',
        'The clearest direct dependencies are on card 3 in C128. Core groups 0–3, '
        '4–7 and 8–11 consume activation shards from cards 0, 1 and 2 respectively; '
        'cores 12–15 start earlier on the locally produced shard. Representative edges:', '',
        '| Card 3 consumer | Remote activation producer | Receive event ends, ms | Down GEMM starts, ms | Gap, µs |',
        '|---|---|---:|---:|---:|']
    for core, port in [(0, '3<-0'), (4, '3<-1'), (8, '3<-2')]:
        r = next(r for r in producers[128] if r['data_dependency'] == 'True' and
                 r['target_card'] == '3' and r['target_core'] == str(core) and
                 r['producer_port'] == port and r['producer_node'] == STEM+'Mul_9')
        lines.append(f'| Core {core} | `{port}` | {float(r["producer_end_ms"]):.6f} | '
                     f'{float(r["target_start_ms"]):.6f} | {float(r["visible_gap_us"]):.3f} |')
    lines += ['', 'Times in this dependency table are the original SDK trace timestamps. '
        'These edges directly establish activation readiness gating those GEMMs. '
        'They do **not** establish a universal hardware receive-then-send order: '
        'card-3 send markers are already open while receive intervals are active. '
        'Other cores can wait beyond a particular receive end because of additional '
        'dependencies and collective scheduling.', '',
        '![Cross-card exchange and down-GEMM starts](cross_card.png)', '',
        'The chart shows each send and receive separately. Their recorded intervals '
        'need not have identical endpoints; they can include queueing/collective completion. '
        'Payload divided by these durations is not a reliable physical link bandwidth.', '',
        '## Within a card: vector preparation, local exchange and HMX waits', '',
        'Each card has 16 active HMX cores. The trace also separates vector work (HVX), '
        'DDR-to-local-memory DMA, and local-memory multicast/copy operations. '
        'A local multicast can fan out, so descriptor output bytes are not physical '
        'on-chip fabric traffic after replication.', '',
        '- Both routing prefix sums (`CumSum`, `CumSum_1`) execute on **core 0 of '
        'each card**, about 0.399 ms of HVX work apiece. The second prefix sum starts '
        'at different times on the four cards because preceding work has different readiness.',
        '- There are **242 explicit true cross-core dependency edges** per trace in '
        'both variants. Of these, 236 originate in local multicast operations, mainly '
        'core-0 input/routing preparation feeding other cores. This counts compiler '
        'dependencies, not all hardware transfers; multicast fanout can be implicit.',
        '- An explicit within-card example is the C2 card-0/core-0 `Where_5` multicast '
        '(a one-byte routing predicate). It finishes at SDK timestamp 0.183678 ms '
        'and feeds `Sub_3` on cores 4, 8 and 12, whose copy operations start around '
        '4.712–4.721 ms. This establishes the broadcast dependency, but its early '
        'completion means it cannot explain those later multi-millisecond waits.',
        '- Cold intermediate local multicast descriptors change from 180 operations / '
        '22.5 MiB of output-size fields at C128 to 60 / 0.1758 MiB at C2. '
        'The cold HMX execution still uses all 64 cores, including when 42 of the '
        '64 cold experts receive no tokens.',
        '- Dense combination first reduces local contributions, then exchanges partial '
        'results and combines them on card 0. A long DDR-backed HVX local reduction '
        'runs on core 0 of each card: up to 1.650 ms at C128 and 1.612 ms at C2. '
        'The final card-0 reduction uses cores 0–3, with up to about 0.242 ms per core. '
        'The source dense accumulator remains `[64,128,2048]` FP16 (32 MiB).', '',
        '![All 16 cores within card 0](within_card.png)', '',
        'Each core row has four sublanes: HMX wait/compute, HVX work, DDR DMA, '
        'and local-memory multicast. The cold HMX span is lightly shaded. '
        'GEMM dependency waits and other HMX thread synchronization use separate colors. '
        'The additional bottom row shows recorded P2P endpoint intervals involving card 0; '
        'these include waiting and are not continuous link utilization. '
        'Rows execute concurrently; the figure must not be read as a sum of costs. '
        'All cards and individual lowered operations are available in the CSV artifacts.', '',
        '### What the idle-looking gaps mean', '',
        'White space in the original chart was not a complete core-idle measurement: '
        'it omitted P2P, HVX/DMA-issue waits, some local copies and cacheable-DDR gathers. '
        'The original gray HMX lane also included end-of-program synchronization, '
        'not just waits preceding GEMMs. The updated chart separates that synchronization '
        'and adds P2P intervals.', '',
        'For the C2 representative capture, the main card-0 intervals are below. '
        'Times are milliseconds since device execution starts, matching the chart.', '',
        '| Interval | What is happening | Why many compute engines have no work |',
        '|---|---|---|',
        '| Approximately 2.0–4.5 ms | Hot-group activation/index exchange, unpacking and accumulator update | '
        'Many cores wait on P2P and packing dependencies before cold-stage inputs are ready; '
        'brief down-GEMM and vector work still occurs within this window. |',
        f'| {gaps["zero_accumulator"]["start_ms"]:.3f}–{gaps["zero_accumulator"]["end_ms"]:.3f} ms | '
        f'Core 0 zeroes the dense accumulator ({gaps["zero_accumulator"]["duration_ms"]:.3f} ms) | '
        'This vector stage has limited core participation. |',
        f'| {gaps["cold_prefix_sum"]["start_ms"]:.3f}–{gaps["cold_prefix_sum"]["end_ms"]:.3f} ms | '
        f'Core 0 computes the cold routing prefix sum ({gaps["cold_prefix_sum"]["duration_ms"]:.3f} ms) | '
        'Other cores cannot use the resulting packing indices until ready. |',
        f'| {gaps["dense_reduction"]["start_ms"]:.3f}–{gaps["dense_reduction"]["end_ms"]:.3f} ms | '
        f'Core 0 performs the dense local reduction ({gaps["dense_reduction"]["duration_ms"]:.3f} ms) | '
        'Expert GEMMs have finished; other cores wait for final combination and output synchronization. |', '',
        f'All 16 card-0 cores finish their last GEMM by '
        f'{max(gaps["last_gemm_end_by_card0_core_ms"]):.3f} ms, although the complete replay '
        f'lasts {small["device_ms"]:.3f} ms. Core 8, for example, waits in '
        '`aicendcyclestats` around 7.128–7.731 ms and then in an output semaphore '
        'around 7.739–9.392 ms. This is waiting for graph completion, not additional '
        'expert arithmetic or evidence of continuous weight DMA. The short stats operation '
        'itself must not be charged the full preceding wait as profiling overhead.', '',
        'Remote final-result receive events on card 0 remain open until 9.246–9.391 ms. '
        'Their long intervals include waiting for remote producers; they do not show '
        'that the links transfer continuously. The profile exposes limited parallelism '
        'and dependency sequencing in the compiled route/unpack/combine path. It does '
        'not establish that every such gap can be removed or overlapped safely.', '',
        '### Actual compute work, separate from elapsed spans', '',
        'Values below sum compute durations within each participating core and then take '
        'the maximum core. They are **not operator latency** and cannot be added to '
        'the phase table. Reductions aggregate their TCM and DDR portions on each core.', '',
        '| Operation | Active cores C128 / C2 | C128 max-core work, ms | C2 max-core work, ms |',
        '|---|---:|---:|---:|']
    for node, label, kind in [
        ('MatMul', 'Hot gate', 'aicconvolutiond32'), ('MatMul_1', 'Hot up', 'aicconvolutiond32'),
        ('MatMul_2', 'Hot down', 'aicconvolutiond32'), ('MatMul_3', 'Cold gate', 'aicconvolutiond32'),
        ('MatMul_4', 'Cold up', 'aicconvolutiond32'), ('MatMul_5', 'Cold down', 'aicconvolutiond32'),
        ('CumSum', 'Hot prefix sum', 'cumsum'), ('CumSum_1', 'Cold prefix sum', 'cumsum'),
        ('Einsum_3', 'Local dense reduction', 'aicbatchedreduceadd'),
        ('Einsum_4', 'Final card-0 reduction', 'aicbatchedreduceadd')]:
        a, b = [core_work(work[c], node, kind) for c in [128, 2]]
        lines.append(f'| {label} (`{node}`) | {a[0]} / {b[0]} | {a[2]:.4f} | {b[2]:.4f} |')
    lines += ['', '### What remains at small capacity', '',
        '| Cold-stage observation | C128 | C2 |', '|---|---:|---:|',
        f'| HMX compute, median core, ms | {base["cold_compute_median_core_ms"]:.5f} | {small["cold_compute_median_core_ms"]:.5f} |',
        f'| HMX dependency wait clipped to cold span, median core, ms | {base["cold_wait_median_core_ms"]:.5f} | {small["cold_wait_median_core_ms"]:.5f} |',
        f'| Repeated projection DDR-copy payload | {base["cold_copy_bytes"]/2**20:.0f} MiB | {small["cold_copy_bytes"]/2**20:.0f} MiB |',
        f'| Repeated projection DDR-copy events | {base["cold_copy_events"]} | {small["cold_copy_events"]} |', '',
        'The repeated copies total exactly 64 × 3 × 2048 × 768 × 2 bytes = **576 MiB** '
        'in both cases. C128 uses twelve 256 KiB copies per core for each gate/up '
        'projection and two 1.5 MiB copies for down; C2 uses two 1.5 MiB copies '
        'per projection per core. Matching sizes and dataflow strongly identify these '
        'as weight tiles. The stats-level-70 descriptor omits operand buffer names, '
        'so exact source-buffer attribution remains an inference. C128 has one additional '
        '384 KiB `MatMul_5`-attributed copy on card 3/core 15; it is retained as '
        'unclassified and excluded from the repeated-weight total.', '',
        'Two C2 card-0/core-0 examples connect delayed GEMMs directly to projection '
        'DDR-copy completion:', '',
        '| Consumer | Copy OC | Payload | Recorded copy end, ms | GEMM start, ms | Gap, µs |',
        '|---|---:|---:|---:|---:|---:|']
    for node in ['MatMul_4', 'MatMul_5']:
        candidates = [r for r in producers[2] if r['data_dependency'] == 'True' and
                      r['target_card'] == '0' and r['target_core'] == '0' and
                      r['target_node'] == STEM+node and r['producer_kind'] == 'aiccopytovtcm']
        r = max(candidates, key=lambda r: float(r['target_start_ms']))
        lines.append(f'| Last `{node}` | {r["producer_oc"]} | 1.5 MiB | '
                     f'{float(r["producer_end_ms"]):.6f} | {float(r["target_start_ms"]):.6f} | '
                     f'{float(r["visible_gap_us"]):.3f} |')
    lines += ['', 'Earlier copies in the same batch can have sub-microsecond visible durations '
        'while their dependent HMX operations remain blocked for hundreds of microseconds. '
        'The SDK event representation includes asynchronous issue/completion behavior; '
        'do not interpret every visible copy end as physical data-ready time. These examples '
        'support a remaining weight-transfer/dependency floor, but do not prove DDR bandwidth '
        'saturation or partition all waiting time between weights, activations and scheduling.', '',
        '## Implications for the MoE algorithm', '',
        '1. **Padding helps activation exchange and unpacking.** For this replay, the '
        'cold span falls 3.168 → 2.152 ms, unpacking 0.639 → 0.130 ms, and combination '
        '2.883 → 2.366 ms. The between-group interval grows 1.419 → 1.726 ms. '
        'Compiler tiling and communication changes matter alongside logical tensor size.',
        '2. **Capacity alone preserves the expert/weight topology.** Skipping empty experts '
        'or changing active expert placement is a separate experiment needed to reduce '
        'the persistent weight payload. This profile does not measure its speedup.',
        '3. **The dense route-and-combine path is a second target.** Prefix sums and '
        'core-0 reductions survive tiny cold padding. A sparse token-oriented combine '
        'or different partition could address them, but requires its own correctness '
        'and timing comparison.',
        '4. **Optimize measured dependencies, not an assumed card order.** C128 shows '
        'remote activations releasing core groups; C2 has a different placement/schedule '
        'and nearly simultaneous down starts. Neither implies a fixed architectural '
        'ordering across all invocations.', '',
        '## Validation and artifacts', '',
        '- Reused six existing bit-exact captures; no new hardware execution or model changes. '
        'Both hot and cold stages cover four cards × 16 cores in every capture.',
        '- Full-flow decode reproduces the original device durations and representative samples. '
        'C128 device range is '+span(base, 'device_ms')+' ms; C2 is '+span(small, 'device_ms')+' ms.',
        f'- Paired {base["flow_pairs"]:,} C128 and {small["flow_pairs"]:,} C2 dependency flows '
        'per capture with no missing endpoints. IDs are paired in event-file order because '
        'the decoder reuses them across cards; work is keyed by process/thread/operation ID.',
        '- Every P2P send is matched to a receive by source, destination and transaction ID; '
        'payload sizes agree. Only outgoing payloads contribute to byte totals.',
        '- HMX waits use `opSyncDurUs`, not the zero visible sync duration; waits are '
        'clipped to the cold interval and checked for per-thread overlap. All wait/compute '
        'pairs agree within 1 µs. Overlapping DMA, waits and parallel core work are not added.',
        '- Compiler descriptors are copied locally and decoded with the installed SDK binary’s '
        'embedded `AICOpstatsDesc.proto` schema. Schema, descriptor and trace hashes are saved.', '',
        '`summary.json` and `full_model_layer2.csv` hold the main results. Each '
        '`c*/sample*/` contains `work.csv`, `waits.csv`, `cores.csv`, `nodes.csv`, '
        '`p2p.csv`, `p2p_summary.csv`, `flows.csv`, `cross_core.csv`, `producers.csv`, '
        'and `cold_copies.csv`. `idle_gap_check.json` records the figure-relative gap audit. '
        'Full-flow traces are in `c*/trace/`; metadata is in '
        '`c*/metadata/`. Charts are available as PNG, PDF and SVG. Large local artifacts '
        'follow the repository ignore policy; scripts and this report are committed.', '',
        '## Reproduce from the saved captures', '',
        'The QEff environment supplies protobuf; the plot environment supplies NumPy and '
        'Matplotlib. `--decode` needs the sweep QPCs in scratch. Once decoded and descriptors '
        'extracted, rerun without `--decode` to use the persistent local artifacts.', '',
        '```bash',
        '/home/chihao/qeff-venv/bin/python tools/moe_qwen3_layer_detail.py \\',
        '  --cold 9.17.2026/real_model/oracle_padding/cold_capacity_control \\',
        '  --profile 9.17.2026/real_model/oracle_padding/layer_profile \\',
        '  --out 9.17.2026/real_model/oracle_padding/layer2_detail \\',
        '  --scratch /dev/shm/qwen3_cold_wentao_20260922 --decode',
        '/tmp/qwen3_profile_plot_wentao_20260921/bin/python \\',
        '  tools/moe_qwen3_layer_detail_report.py \\',
        '  --root 9.17.2026/real_model/oracle_padding/layer2_detail',
        '```', '',
    ]
    (root/'README.md').write_text('\n'.join(lines))


def plots(root, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 190, 'svg.fonttype': 'none'})

    def save(fig, name):
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(root/f'{name}.{ext}')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout='constrained')
    values = [data['full_model'], [c['phases'] for c in data['cases']]]
    labels = [['Native 128/128', 'Minimum 92/2', 'Power-of-two 128/2'], ['Cold C128', 'Cold C2']]
    for ax, entries, ticks, title in zip(axes, values, labels,
            ['Layer 2 inside full model (router included)', 'Controlled replay (router excluded; hot C128)']):
        left = np.zeros(len(entries))
        for key, label, color in PHASES:
            v = np.array([float(r.get(key, 0)) for r in entries])
            ax.barh(range(len(entries)), v, left=left, color=color, label=label, height=.55)
            for y, (a, b) in enumerate(zip(left, v)):
                if b > .55:
                    ax.text(a+b/2, y, f'{b:.2f}', ha='center', va='center', fontsize=8)
            left += v
        for y, total in enumerate(left):
            ax.text(total+.12, y, f'{total:.3f}', va='center', fontsize=9)
        ax.set(yticks=range(len(entries)), yticklabels=ticks, xlabel='Elapsed device time (ms)',
               xlim=(0, 12.5), title=title)
        ax.invert_yaxis()
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in PHASES], loc='outside lower center', ncol=3, fontsize=9)
    save(fig, 'layer_budgets')

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout='constrained', height_ratios=[1, 2.1])
    for col, case in enumerate(data['cases']):
        src = directory(root, case)
        origin = case['device_start_ms']
        matrix = np.zeros((4, 16))
        for r in rows(src/'cores.csv'):
            if r['stage'] == 'cold':
                matrix[int(r['card']), int(r['core'])] = float(r['first_down_ms'])-origin
        ax = axes[0, col]
        im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=5.8, vmax=7.85)
        ax.set(xticks=range(16), yticks=range(4), xlabel='Core within card', ylabel='Card',
               title=f'C{case["capacity"]}: first cold down-GEMM start\nspread {case["cold_first_down_spread_ms"]:.3f} ms')
        fig.colorbar(im, ax=ax, label='Time since device start (ms)', shrink=.9)
        ax = axes[1, col]
        p2p = [r for r in rows(src/'p2p.csv') if r['node'] == STEM+'Mul_9']
        ports = [(a, b) for a in range(4) for b in range(4) if a != b]
        for y, (a, b) in enumerate(ports):
            for port, off, color in [(f'{a}->{b}', -.28, '#4783b3'), (f'{b}<-{a}', .02, '#d98735')]:
                r = next(r for r in p2p if r['port'] == port)
                ax.broken_barh([(float(r['start_us'])/1000-origin, float(r['duration_us'])/1000)],
                               (y+off, .24), facecolors=color)
        ax.set(yticks=range(12), yticklabels=[f'{a} → {b}' for a, b in ports], ylim=(-.6, 11.6),
               xlim=(5.45, 7.9), xlabel='Time since device start (ms)',
               title=f'Cold activation P2P: {"3 MiB" if col == 0 else "48 KiB"} per edge')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
        ax.legend(handles=[Patch(color='#4783b3', label='Send event'), Patch(color='#d98735', label='Receive event')],
                  loc='lower left', fontsize=8)
    fig.suptitle('Cross-card readiness: smaller payload collapses the stagger in down-GEMM starts')
    save(fig, 'cross_card')

    fig, axes = plt.subplots(1, 2, figsize=(15, 11), sharey=True, layout='constrained')
    colors = {'wait': '#d8d8d8', 'sync': '#e4d6aa', 'HMX': '#13876a', 'HVX': '#8c62a6',
              'DDR': '#4783b3', 'local': '#d98735', 'P2P': '#b66c75'}
    for ax, case in zip(axes, data['cases']):
        src = directory(root, case)
        origin = case['device_start_ms']
        bars = defaultdict(list)
        for r in rows(src/'waits.csv'):
            if r['card'] == '0' and r['engine'] == 'HMX':
                label = 'wait' if r['kind']=='aicconvolutiond32' else 'sync'
                bars[int(r['core']), label].append((float(r['start_us'])/1000-origin, float(r['duration_us'])/1000))
        for r in rows(src/'work.csv'):
            if r['card'] != '0':
                continue
            kind = None
            if r['engine'] == 'HMX' and r['kind'] == 'aicconvolutiond32':
                kind = 'HMX'
            elif r['engine'] == 'HVX' and r['kind'] != 'aicendcyclestats':
                kind = 'HVX'
            elif r['engine'].endswith('_DMA') and r['memory'] == 'DDR':
                kind = 'DDR'
            elif r['kind'] in ['aicmulticastvtcm', 'aicmulticastvtcm2d']:
                kind = 'local'
            if kind:
                bars[int(r['core']), kind].append((float(r['start_us'])/1000-origin, float(r['duration_us'])/1000))
        offsets = {'wait': -.38, 'sync': -.38, 'HMX': -.38, 'HVX': -.18, 'DDR': .02, 'local': .22}
        for kind in ['wait', 'sync', 'HMX', 'HVX', 'DDR', 'local']:
            for core in range(16):
                ax.broken_barh(bars[core, kind], (core+offsets[kind], .16), facecolors=colors[kind], linewidth=0)
        p2p_intervals = [(float(r['start_us'])/1000-origin, float(r['duration_us'])/1000)
                         for r in rows(src/'p2p.csv') if r['port'].startswith(('0->', '0<-'))]
        ax.broken_barh(p2p_intervals, (16-.2, .4), facecolors=colors['P2P'], linewidth=0)
        bounds = case['phase_boundaries_ms']
        ax.axvspan(bounds[3]-origin, bounds[4]-origin, alpha=.08, color='#d95f02')
        for y in np.arange(.5, 16, 1):
            ax.axhline(y, color='#dddddd', lw=.5)
        ax.set(yticks=range(17), yticklabels=[f'Core {c}' for c in range(16)]+['Card 0 P2P'],
               xlabel='Time since device start (ms)', xlim=(0, 11.6), ylim=(16.6, -.6),
               title=f'Card 0 · cold C{case["capacity"]} · {case["device_ms"]:.3f} ms device interval')
        ax.grid(axis='x', alpha=.15)
    fig.legend(handles=[Patch(color=colors[k], label=l) for k, l in [
        ('wait', 'GEMM dependency wait'), ('sync', 'Other HMX thread sync'),
        ('HMX', 'HMX compute'), ('HVX', 'HVX compute'), ('DDR', 'Recorded DDR DMA interval'),
        ('local', 'Local-memory multicast'), ('P2P', 'Recorded P2P wait / transfer interval')]],
        loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('Within one card: all 16 cores and four concurrent activity lanes per core')
    save(fig, 'within_card')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    data = json.loads((args.root/'summary.json').read_text())
    report(args.root, data)
    plots(args.root, data)


if __name__ == '__main__':
    main()
