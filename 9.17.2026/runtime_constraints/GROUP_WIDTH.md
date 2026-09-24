# Constraint 3: expert group count and width

Measured 2026-09-21 on AI 100 card 0, SDK 1.21.6, with 16 cores allocated.

**Group width is a useful tuning variable even when total padded work is
unchanged.** All six tested widths compile and execute with runtime expert IDs
in FP16 and MXFP6. Eight groups of four experts give the lowest latency in this
sweep: 19.9% below the two-group FP16 control and 26.4% below its MXFP6 control.
Further splitting into groups of two or one makes execution slower again.

This is a sweep of separate compiled topologies. It does not demonstrate
changing group count inside one resident QPC or combining that change with
runtime capacity specialization.

## Controlled workload

The probe reuses the [resident-selection experiment](RESIDENT_SELECTION.md)'s
synthetic Qwen-dimension weight bank, activation bytes, expert-ID cases and CPU
references. Weight-bank SHA256 hashes and activation bytes match the source.

- H=2048, I=768, T=128; global router has 128 experts and top-8 routing.
- The local graph returns only the partial contribution of 32 resident experts.
  Its FP16 gate/up/down bank contains 288 MiB of weights.
- Widths 32, 16, 8, 4, 2 and 1 imply respectively 1, 2, 4, 8, 16 and 32
  sequential groups. Every group has capacity 32.
- Every layout processes all 32 experts and exactly 1,024 logical padded
  expert-token rows. The audit executes the pruned ONNX packing dependencies
  and checks every group's activation shape `[width,32,2048]`.
- Each graph receives `expert_ids: int32[32]`; identity, half-swap and shuffle
  all cover the bank. A duplicate-half diagnostic tests that IDs affect the
  result, and is excluded from timing summaries.
- The same input produces 266 assignments to resident experts, maximum 13 per
  expert. All normal cases have zero overflow in every layout.
- Each QPC uses `-aic-num-cores=16 -mos=1 -aic-enable-depth-first`, with the
  same other compiler options as the resident-selection probe. The quantized
  variants additionally use `-mxfp6-matmul`.

The comparison isolates the effects of group topology, including the resulting
packing, accumulation, tiling and compiler placement. Equal logical padded work
does not guarantee equal physical memory traffic or HMX tile scheduling.
The earlier resident-selection baseline used capacities 64/32; this sweep's
control uses 32/32 and is compiled and measured afresh.

## Latency

Milliseconds of host wall-clock time for one invocation. The timer includes
copying ID values, binding buffers, enqueueing and waiting. Each variant loads
and activates once; those costs are excluded. Variants run sequentially on the
same card with no competing device benchmark.

Each reported median pools 450 samples: three rounds, 50 samples per normal
ID case, with warmups. Cycle mode adds 450 samples while changing IDs each
invocation. Case order alternates across rounds. Per-case medians and pooled
p10/p90 values are retained in the result JSON.

| Experts/group | Groups | FP16, held IDs | FP16, cycling IDs | MXFP6, held IDs | MXFP6, cycling IDs |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 8.306 | 8.313 | 7.225 | 7.243 |
| 16 | 2 | 6.599 | 6.602 | 4.854 | 4.836 |
| 8 | 4 | 5.468 | 5.428 | 3.862 | 3.863 |
| 4 | 8 | **5.287** | **5.292** | **3.571** | **3.568** |
| 2 | 16 | 6.045 | 6.000 | 4.131 | 4.120 |
| 1 | 32 | 7.291 | 7.256 | 5.703 | 5.679 |

Width 4's pooled p10–p90 range is 5.170–5.384 ms in FP16 and 3.474–3.702 ms
in MXFP6. The width-16 control ranges are 6.504–6.753 ms and 4.727–5.016 ms.
The observed benefit is not caused by reducing padded rows or dropping routed
tokens. It is specific to this graph, input, dimensions, capacity and SDK;
width 4 is a measured candidate, not a universal optimum.

## Correctness

All variants pass routing-count, overflow, diagnostic-selection and repeated
output checks. Each timed output is byte-identical to that variant/case's
initial output. Grouping changes the accumulation order, so cross-width
outputs are not generally bitwise identical.

- FP16: maximum relative L2 against the independent CPU reference is 0.0691%
  across all widths and cases, including the diagnostic. Maximum difference
  from width 16 is 0.0252% relative L2, below the 0.5% grouping-equivalence limit.
- MXFP6: maximum cross-width difference is 0.0251% relative L2. Normal cases
  retain about 6.033% CPU-reference error, and the diagnostic about 6.075%.
  All widths pass grouping equivalence but fail the unchanged 5% absolute
  reference-accuracy threshold. `--equivalence-only` retains
  `validation_pass=false` rather than relabeling quantization error a pass.

There is no attention, KV continuation, trained-weight accuracy test, adaptive
policy, current-layer dispatch boundary or overflow recovery in this probe.

## Core participation and compression

All variants allocate 16 cores, but the HMX traces show a different number
executing expert GEMMs. These counts apply to each expert projection in every
group, in both precision modes:

| Experts/group | HMX cores participating in each expert projection | FP16 packed constants, MiB | MXFP6 packed constants, MiB |
|---:|---:|---:|---:|
| 32 | 4 | 304.584 | 128.779 |
| 16 | 16 | 304.562 | 128.758 |
| 8 | 16 | 296.552 | 120.747 |
| 4 | 16 | 292.548 | 116.744 |
| 2 | 16 | 290.545 | 114.740 |
| 1 | 16 | 289.542 | 113.738 |

**A group does not require one expert per core.** For width 1, each group
contains a single expert and each of its projections executes across all 16
HMX cores. The compiler can partition computation within that expert. At the
other extreme, allocating 16 cores and supplying a 32-expert batch does not
guarantee that its expert GEMMs use all 16. Physical expert/core assignment
must be measured for each compiled shape and topology.

The audit also records per-core expert HMX kernel durations, peak simultaneous
participation and an occupancy proxy: the sum of each core's union of expert
HMX kernel intervals divided by `16 * device_kernel_span`. This proxy ranges
from about 0.8% to 3.3% in these traces. It excludes router/HVX/copy work and is
not a measurement of FLOP efficiency or total accelerator utilization.
Participation counts also do not mean the participating cores compute
continuously or simultaneously throughout a projection.

Traces are single profiled invocations. Their spans include profiling effects
and differ from the unprofiled host medians, especially for many small groups.
Use the host measurements for latency comparisons; use traces to inspect
placement and operator paths. The data do not isolate a causal time penalty
for each additional group or quantify DDR/PCIe traffic.

Each MXFP6 trace contains expert `blockdequantize_mxfp6` kernels for every
projection, and packed constants shrink for every width. Runtime expert IDs
therefore keep the compressed bank path even with one expert per group.
Packed constants include router/auxiliary data and layout padding. The
observed decrease with narrower groups is not removal of expert parameters:
all source weight-bank hashes match and all outputs pass the grouping checks.

The host still uses one program load, one activation and one ExecObj per
variant. Device resource snapshots are unchanged across ID cases after
activation. This establishes stable residency for each individual graph;
the sweep does not measure switching residency between different width QPCs.

## Consequences for algorithm design

1. Keep group width/count as a measured design variable. Do not constrain a
   group to 16 experts merely because the card has 16 cores.
2. Use latency measurements indexed by shape/topology/precision, not only
   `sum(group_size * capacity)`. All six layouts have the same padding proxy
   here but substantially different runtimes. Maximum HMX participation alone
   also fails to predict which layout is fastest.
3. Retain widths 4 and 8 as useful candidates for this workload. More groups
   can improve capacity granularity, but widths 2 and 1 already incur higher
   latency at equal padded work. They could still pay off if a future workload
   saves enough padding; this uniform-capacity sweep does not decide that.
4. Membership can change at runtime within each tested topology. Width/count
   are still compiled choices, and expert IDs still address only the local
   resident bank. No inter-card placement freedom is added by this result.

The [combined-adaptation probe](COMBINED_ADAPTATION.md) now validates runtime
membership plus five capacity profiles in one resident QPC at width 4, including
stateless overflow replay. It also finds that profile-library memory depends
strongly on the chosen capacities and precision. Current-count observation,
full-model state and multi-card behavior remain separate constraints.
The broader [placement and adaptation design space](../multilayer/RUNTIME_ADAPTATION.md)
remains open.

## Reproduce and inspect

Use a fresh output directory and run hardware commands sequentially. A source
resident-selection export is required; its QPCs are not required for this sweep.

```bash
PY=/home/chihao/qeff-venv/bin/python
$PY tools/moe_resident_selection.py --out /tmp/resident_source \
  --stage export --hidden 2048 --intermediate 768
$PY tools/moe_group_width_probe.py --out /tmp/group_width \
  --source /tmp/resident_source
$PY tools/moe_group_width_probe.py --out /tmp/group_width \
  --stage compile --precision mxfp6
$PY tools/moe_group_width_probe.py --out /tmp/group_width \
  --stage run --precision mxfp6 --equivalence-only
$PY tools/moe_group_width_probe.py --out /tmp/group_width --stage audit --profile
$PY tools/moe_group_width_probe.py --out /tmp/group_width \
  --stage audit --precision mxfp6 --profile
```

The script accepts `--capacity` and `--widths` for new exports; width 16 must be
included as the numerical control. `--run-name` and `--reverse` support retained
additional timing runs. Generated artifacts are ignored by git.

- [Sweep exporter, compiler, runner, validator and trace audit](../../tools/moe_group_width_probe.py)
- [Shared resident-program host](../../tools/moe_resident_selection_host.cpp)
- [Configuration and source hashes](group_width_qwen/info.json), [graph audit](group_width_qwen/graph_audit.json)
- [FP16 results](group_width_qwen/fp16/run/result.json), [MXFP6 results](group_width_qwen/mxfp6/run/result.json)
- [FP16 package/shape/trace audit](group_width_qwen/fp16/package_audit.json), [MXFP6 audit](group_width_qwen/mxfp6/package_audit.json)
- [FP16 traces](group_width_qwen/fp16/profiling/), [MXFP6 traces](group_width_qwen/mxfp6/profiling/)
