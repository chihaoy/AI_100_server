# Constraint 2: runtime regrouping of resident experts

Measured 2026-09-21, Qualcomm AI 100, SDK 1.21.6, card 0, 16 NSP cores.

**Runtime expert-ID values can change membership between two padding groups
inside one resident QPC.** FP16 and MXFP6 both compile and execute at Qwen's
matrix dimensions. The tested Gather-from-constant-bank pattern preserves
MXFP6 compression, and dynamic outputs match the corresponding fixed-layout
outputs exactly. Its measured latency is close to the fixed-layout controls.

This establishes local regrouping as an available algorithm choice. It does
not yet establish combined capacity/membership adaptation, inter-card expert
movement, current-layer routing decisions, or full-model chunked prefill.

## Scope and implementation

- Global router: 128 experts, top-8, T=128. The graph computes only the partial
  contribution of resident experts 0–31; it is not a complete 128-expert layer.
- Resident bank: 32 synthetic experts. Main test H=2048, I=768, matching Qwen3
  30B A3B matrix dimensions. Three FP16 projection banks contain 288 MiB of
  weights. A smaller H=256/I=128 test checks the same mechanism first.
- Two sequential groups of 16 experts use capacities 64 and 32. Group sizes,
  capacities and input shapes remain fixed throughout this probe.
- `expert_ids: int32[32]` is an ordinary graph input. Its first 16 IDs select
  group 0 and its remaining 16 select group 1. The graph gathers both each
  expert's routing column and its gate/up/down matrices using those IDs.
- The host changes **128 bytes of ID values**, binds fixed-size buffers using
  `qaicExecObjSetData`, enqueues and waits. Each dynamic run creates one program,
  loads once, activates once and reuses one ExecObj. There is no shape selector,
  extra compiled layout per permutation, or host weight upload in the loop.
- Normal cases: identity, swap the two halves, and a seeded shuffle. Each is a
  full permutation of the same resident bank. Three separately compiled graphs
  with those fixed assignments supply the paired controls.
- A duplicate-half diagnostic selects experts 0–15 twice. Its expected partial
  output differs from the normal output. This proves that ID values affect the
  weight/token association; normal permutations alone could conceal ignored
  IDs because they preserve the mathematical MoE sum. This diagnostic is not a
  legal full-bank grouping and is excluded from latency summaries.

The ONNX audit finds six weight Gather nodes in the dynamic graph, with
nonconstant expert-MatMul right operands. All six expert-MatMul right operands
are initializers in each fixed graph. The compiler nevertheless preserves the
compressed constant-bank path through the dynamic Gathers.

Controlled router logits give exactly representable normalized weights of
1/8. Both sizes have 266 assignments to the resident bank, maximum 13 for any
expert, and zero capacity overflow for every normal grouping. Counts are
reported in canonical resident-expert order, before grouping; the diagnostic
also reports these original counts, not counts multiplied by its duplicate IDs.

## Correctness

An independent CPU reference computes FP32 expert projections from the same
FP16 source weights and uses the controlled router assignments. Every timed
invocation is also checked byte-for-byte against its own case's initial device
output, outside the timed region.

| Check | H=256/I=128 | H=2048/I=768 |
|---|---:|---:|
| FP16 dynamic vs matching static, all three normal cases | Exact | Exact |
| MXFP6 dynamic vs matching static, all three normal cases | Exact | Exact |
| FP16 normal-case relative L2 vs CPU | 0.0687–0.0689% | 0.0672–0.0673% |
| MXFP6 normal-case relative L2 vs CPU | 6.0150% | 6.0326% |
| Canonical counts vs CPU | Exact | Exact |
| Repetitions, including changing IDs every invocation | Exact per case | Exact per case |
| Duplicate-half diagnostic changes output | Yes | Yes |

MXFP6's error is shared by dynamic and fixed layouts. It exceeds the unchanged
5% absolute-reference threshold: `selection_equivalence_pass=true`,
`reference_accuracy_pass=false`, `validation_pass=false`. `--equivalence-only`
allows collecting/reporting that result without relabeling it an accuracy pass.
FP16 passes both checks, including the independent diagnostic reference. These
are synthetic-weight mechanism results, not trained-model accuracy measurements.

## Latency at Qwen matrix dimensions

Host wall-clock milliseconds, including copying ID values, binding, enqueueing
and waiting for completion. Weight/program load and activation are excluded.
One invocation executes both groups and returns their combined partial output.

Each solo median contains 300 samples, taken as three rounds of 100 with
warmups. Dynamic cycle mode adds 300 samples per normal case while changing
identity/swap/shuffle each invocation. Case order alternates across rounds.
Static controls run as separate programs, sequentially on the same card; no
other device benchmark runs concurrently. These are single-workload medians,
not universal cost guarantees.

| Precision | Assignment | Fixed layout | Runtime IDs, held fixed | Runtime/fixed | Runtime IDs, changing each invocation |
|---|---|---:|---:|---:|---:|
| FP16 | Identity | 6.626 | 6.670 | 1.007× | 6.649 |
| FP16 | Swap halves | 6.574 | 6.626 | 1.008× | 6.639 |
| FP16 | Shuffle | 6.688 | 6.675 | 0.998× | 6.663 |
| MXFP6 | Identity | 5.071 | 4.888 | 0.964× | 4.833 |
| MXFP6 | Swap halves | 5.024 | 4.851 | 0.966× | 4.820 |
| MXFP6 | Shuffle | 4.728 | 4.865 | 1.029× | 4.832 |

FP16 paired medians differ by less than 1%; MXFP6 by less than 4%. The data show
no substantial extra cost specifically when IDs change. They do not show that
weight selection itself is free: the Gathers execute even when IDs are held
fixed, and the compiler's packing/copy schedule also differs between graphs.
Subtracting these end-to-end medians is not an isolated Gather measurement.

The small test gives FP16 dynamic medians 0.800–0.833 ms versus fixed
0.828–1.050 ms, and MXFP6 dynamic 0.788–0.809 ms versus fixed 0.809–0.996 ms.
These noisier small-matrix results are supporting mechanism evidence, not the
basis for a claim that dynamic selection accelerates execution.

## Compression, residency and device traces

Packed static-constant storage, including router and auxiliary constants:

| Matrix dimensions | Precision | Dynamic | Static identity/swap | Static shuffle |
|---|---|---:|---:|---:|
| H=256/I=128 | FP16 | 8,523,776 B | 8,523,776 B | 8,524,288 B |
| H=256/I=128 | MXFP6 | 4,649,984 B | 4,649,984 B | 4,650,496 B |
| H=2048/I=768 | FP16 | 319,361,024 B | 319,361,024 B | 319,361,536 B |
| H=2048/I=768 | MXFP6 | 135,016,448 B | 135,016,448 B | 135,016,960 B |

At Qwen dimensions, dynamic packed constants shrink from 304.57 MiB to
128.76 MiB under `-mxfp6-matmul`. They have the same size as the identity and
swap static controls, rather than retaining a full uncompressed bank in
addition to compressed expert weights.

`qaic-runner` records raw device stats, and `qaic-opstats` produces traces for
each of the four variants in each precision and size. Runner also compares
outputs against those saved by the benchmark. The traces show runtime weight
Gather kernels in the dynamic graph, none in its static controls, and
`blockdequantize_mxfp6` kernels associated with all six expert MatMuls in the
MXFP6 dynamic and static graphs. Thus the compressed expert path is actually
executed, not merely requested with a compiler flag. Per-core kernel durations
in the audit are sums across cores and must not be read as wall-clock time.

The host's device-resource snapshots stay unchanged across all ID choices
after activation. We verify one program load and no host weight input; this is
not a PCIe DMA audit or proof that there are no on-device weight copies.

## Consequences for the algorithm

1. **Local group membership can be a per-invocation decision.** For this graph,
   the host can supply a valid resident-bank permutation without compiling
   that permutation or reloading the weight bank.
2. **The tested dynamic selection keeps MXFP6 available.** Do not impose a
   blanket rule that runtime Gathers necessarily force FP16 weights. Other
   graph patterns and runtime-provided weight tensors still need testing.
3. **Group shapes remain compiled constraints.** We have independently shown
   capacity specialization and runtime membership, but have not combined them
   in one QPC. The number of groups and experts per group also remain fixed.
4. **The residency boundary still matters.** An ID refers only to this local
   bank. Selecting an expert owned solely by another card requires a tested
   activation-routing, replication or migration mechanism. This experiment
   does not assign physical expert ownership across cards or prove a specific
   lane-to-core placement.
5. **Decision timing remains separate.** Host IDs are supplied before routing
   executes in this graph. They can use previous observations. Exact current
   counts require splitting after routing or implementing an on-device
   selection policy. No attention, KV continuation or overflow recovery is
   included here.

The [group-count/width sweep](GROUP_WIDTH.md) now measures that constraint:
widths 1–32 compile and execute; width 4 is fastest for the tested equal-work
case, and even width 1 uses all 16 HMX cores. The [combined-adaptation probe](COMBINED_ADAPTATION.md)
now validates ID selection plus five capacity profiles in one resident QPC,
with stateless replay and a measured profile-library storage cost. Keep both four-card and
two-card-pair placement candidates in the [design space](../multilayer/RUNTIME_ADAPTATION.md).

## Reproduce and inspect

Use fresh output directories. Generated ONNX, QPC, binary, JSON, CSV and trace
artifacts are ignored by git; the scripts and this report are retained.

```bash
PY=/home/chihao/qeff-venv/bin/python
$PY tools/moe_resident_selection.py --out /tmp/resident_small
$PY tools/moe_resident_selection.py --out /tmp/resident_qwen \
  --hidden 2048 --intermediate 768
$PY tools/moe_resident_selection.py --out /tmp/resident_qwen \
  --stage compile --precision mxfp6
$PY tools/moe_resident_selection.py --out /tmp/resident_qwen \
  --stage run --precision mxfp6 --equivalence-only
python3 tools/moe_resident_audit.py --out /tmp/resident_qwen --profile
python3 tools/moe_resident_audit.py --out /tmp/resident_qwen --precision mxfp6 --profile
```

Run hardware benchmarks/profiles sequentially. The default target is card 0;
both tools accept `--device`. `--run-name` retains additional benchmark runs.

- [Exporter, compiler and validator](../../tools/moe_resident_selection.py)
- [C++ resident-program host](../../tools/moe_resident_selection_host.cpp)
- [Storage and trace auditor](../../tools/moe_resident_audit.py)
- Qwen dimensions: [configuration](resident_qwen_dims/info.json), [graph audit](resident_qwen_dims/graph_audit.json), [FP16 result](resident_qwen_dims/fp16/run/result.json), [MXFP6 result](resident_qwen_dims/mxfp6/run/result.json)
- Qwen dimensions: [FP16 storage/trace audit](resident_qwen_dims/fp16/package_audit.json), [MXFP6 storage/trace audit](resident_qwen_dims/mxfp6/package_audit.json), [FP16 traces](resident_qwen_dims/fp16/profiling/), [MXFP6 traces](resident_qwen_dims/mxfp6/profiling/)
- Small test: [FP16 result](resident_small/fp16/run/result.json), [MXFP6 result](resident_small/mxfp6/run/result.json), [FP16 audit](resident_small/fp16/package_audit.json), [MXFP6 audit](resident_small/mxfp6/package_audit.json)
