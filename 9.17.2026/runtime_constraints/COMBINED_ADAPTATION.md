# Constraint 4: combined capacity and resident-membership adaptation

Measured 2026-09-21, AI 100 card 0, SDK 1.21.6, synthetic Qwen matrix dimensions
H=2048/I=768, T=128, 32 resident experts in eight groups of four.

**Runtime expert IDs and shape-specialized capacities work together in one
resident QPC.** All five profiles and three normal expert permutations execute
correctly in FP16 and MXFP6. Changing both each invocation does not require a
program reload. Stateless overflow rejection and full-capacity replay also
work in that same program.

The significant new constraint is storage: sharing is highly dependent on the
chosen capacity profiles and precision. Three profiles using capacities 16/32
add almost no packed-constant storage. Adding capacity 128 nearly doubles it;
adding the tested capacity-64 profile adds another bank-sized increment in
FP16, while MXFP6 storage barely changes at that step.

## Interface and tested profiles

The graph retains the [group-width probe](GROUP_WIDTH.md)'s width of four.
It receives `x: FP16[128,2048]`, `expert_ids: int32[32]`, eight row vectors
`rows_g: int32[C_g]`, and a zero-valued `tag: int32[P]` whose shape uniquely
identifies the compiled profile. Row values are `0..C_g-1` and participate in
the packing mask. Expert-ID **values** choose membership; row/tag **shapes**
choose a precompiled capacity vector. This is not arbitrary scalar control of
an uncompiled capacity or graph topology.

| Profile | Capacities of the eight groups | Tag length | Padded expert-token rows |
|---|---|---:|---:|
| full128 | 128,128,128,128,128,128,128,128 | 1 | 4,096 |
| uniform32 | 32,32,32,32,32,32,32,32 | 2 | 1,024 |
| uniform16 | 16,16,16,16,16,16,16,16 | 3 | 512 |
| alternating | 32,16,32,16,32,16,32,16 | 4 | 768 |
| mixed | 64,32,16,16,32,16,16,16 | 5 | 832 |

All profiles use the same source bank, verified by SHA256 against the
[resident-selection probe](RESIDENT_SELECTION.md). All use identical normal
activation bytes and independent CPU references. Identity, half-swap and
shuffle ID lists cover every resident expert exactly once. The duplicate-half
diagnostic additionally verifies that ID values affect the output; it is not
a valid production grouping and is excluded from normal timings.

The host loads and activates once, reuses one ExecObj, and binds all inputs
with `qaicExecObjSetDataExt`. It allocates maximum-sized buffers once and
changes their active sizes/shapes. Unlisted capacity vectors, invalid profile
tags and out-of-range expert IDs are rejected by host checks before enqueue.
The earlier unsupported-shape device error is not repeated here.

## Correctness and execution evidence

- All 15 normal profile/permutation combinations have zero overflow and exact
  CPU routing counts: 266 resident assignments, maximum 13 for an expert.
- For each ID case, all capacity profiles produce **bitwise-identical outputs**
  within each precision. This also holds for the duplicate-half diagnostic.
- Every repeated output is byte-checked against that profile/case's initial
  device output, outside the timed region.
- The uniform32 output in the five-profile QPC is exactly equal to its
  separately compiled single-profile control, including the diagnostic.
- FP16 relative L2 against CPU is below 0.069% across normal and stress checks.
  MXFP6 has the shared roughly 6.033% normal-reference error (about 6.075% for
  the diagnostic and 5.582–5.584% for stress). Its mechanism checks pass, but
  the unchanged 5% accuracy gate fails: `combined_equivalence_pass=true`,
  `reference_accuracy_pass=false`, `validation_pass=false`.
- The audit executes the actual ONNX packing dependencies for every profile
  and verifies all eight `[4,C_g,2048]` activation shapes. These are logical
  GEMM input shapes, not assertions about physical HMX tile geometry.
- Runner/opstats traces show runtime weight Gathers for each profile and
  MXFP6 dequantization at all 24 expert projections in each quantized profile.
  Runner also compares its outputs with the C++ benchmark's saved outputs.

## Latency and switching

Host wall-clock milliseconds, excluding load/activation. Each median pools
180 samples: three rounds of 20 for each of three normal ID cases. Every
profile is measured in four modes: held, IDs-only changes, shapes-only changes,
and both changing. Case/profile order is varied across rounds and all hardware
runs are sequential on card 0. There are 3,600 timed invocations of the
five-profile QPC per precision, with warmups and correctness checks in addition.

| Profile | FP16 held | FP16 both changing | MXFP6 held | MXFP6 both changing |
|---|---:|---:|---:|---:|
| full128 | 15.969 | 15.850 | 5.017 | 5.101 |
| uniform32 | 5.310 | 5.365 | 3.751 | 3.851 |
| uniform16 | 5.237 | 5.380 | 3.657 | 3.801 |
| alternating | 5.245 | 5.336 | 3.694 | 3.838 |
| mixed | 6.562 | 6.529 | 3.816 | 3.913 |

Binding medians in the both-changing mode are 1.9–3.4 microseconds in FP16 and
1.0–1.2 microseconds in MXFP6. Total execution can still change: cycling both
adds up to about 0.14 ms relative to the corresponding held medians here.
These differences include cache/scheduling and measurement effects; they are
not an isolated API switching cost.

The single-profile uniform32 control takes 5.269 ms in FP16 and 3.527 ms in
MXFP6. The same held profile inside the five-profile QPC takes 5.310 ms and
3.751 ms, respectively: about 0.8% and 6.3% higher. Specialization therefore
need not have zero recurring execution cost even when the profile is held.

Reducing capacity 32 to 16 halves logical padded work but makes little latency
difference. The mixed profile has fewer padded rows than uniform32 yet is
slower, particularly in FP16. Profile choice must use measured cost, not only
the padding sum. These results concern a single controlled workload and local
partial MoE; they are not full-model throughput measurements.

## Profile-library storage

Packed static constants in MiB, excluding other QPC sections and runtime
workspace. The two subset libraries are **compile/storage probes only**;
their separate QPCs were not benchmarked or used for the replay test.

| Library | Included profiles | FP16 | MXFP6 |
|---|---|---:|---:|
| single32 | uniform32 | 292.546 | 116.742 |
| single128 | full128 | 292.548 | 116.743 |
| compact3 | uniform32, uniform16, alternating | 292.620 | 116.815 |
| safe4 | compact3 + full128 | 582.658 | 231.353 |
| multi | safe4 + mixed | 870.696 | 231.391 |

Three profiles are not intrinsically three weight copies: compact3 adds only
0.073 MiB over single32. Conversely, the fifth tested profile adds roughly
288 MiB to FP16 storage, while adding only 0.038 MiB in MXFP6. Thus neither
profile count alone nor the ONNX bank count predicts compiled storage. The
increments are consistent with additional packed representations; the audit
does not reverse-engineer every byte into an exact weight-copy inventory.

Measured device free-memory deltas after load/activation are 1,303.17 MiB for
the FP16 five-profile program versus 623.02 MiB for single32; the corresponding
MXFP6 deltas are 570.07 MiB and 447.30 MiB. These include allocations beyond
packed constants. Free-memory snapshots remain unchanged across all ID/shape
choices and after replay. No host weight inputs or program reloads occur in
the loop; this is not a PCIe DMA trace or proof of zero on-device copying.

## Overflow rejection and replay

The stress input repeats source token 55, causing five resident experts to
receive all 128 tokens: 640 resident assignments. Counts remain in canonical
expert order. The host computes overflow using the current ID-to-group map:
`sum_j max(count[expert_ids[j]] - capacity[j // 4], 0)`.

Uniform16 overflows by 560 assignments for each permutation. The mixed profile
overflows by 496, 464 and 448 assignments for identity, half-swap and shuffle,
respectively. All counts and overflow amounts match the independent CPU data.
This exercises the interaction between capacity and membership rather than
assuming an expert's original group still determines its capacity.

For each of those six stress cases, the host rejects the truncated output and
replays the same activation input and expert IDs using full128, in the same
program. All 120 replay transactions per precision recover exactly the output
of direct full128 execution for that permutation. A subsequent normal-input
execution also passes, showing that replay leaves the program usable.

| Rejected profile | FP16 transaction median | MXFP6 transaction median |
|---|---:|---:|
| uniform16 | 11.561 ms | 5.985 ms |
| mixed | 12.963 ms | 6.082 ms |

These totals include the failed-capacity execution, host overflow check and
full-capacity replay. Each median pools 60 stress transactions across the three
permutations. Validation/file writes are outside the timer. The input differs
from the normal benchmark; these totals must not be compared with normal-input
full128 latency to claim replay is faster than conservative execution.

This is **stateless partial-MoE recovery**. It does not implement KV rollback,
undo downstream layers, merge results across cards or supply an accepted
full-model routing trace after a failed chunk. Those remain separate work.

## Consequences and next constraint

The available local decision is now a pair: a resident-expert permutation and
one compiled eight-group capacity vector. Both can change each invocation.
The algorithm must account for measured execution time, profile-library memory,
and overflow/replay cost. It cannot assume all profiles share one packed bank,
that fewer padded rows always execute faster, or that capacity follows expert
identity after regrouping.

The host supplies both choices before this graph computes routing. Decisions
can therefore use earlier observations, but this implementation does not yet
choose from the current invocation's router counts. The next constraint is
the cost of obtaining current-layer counts before expert dispatch, compared
with prediction from previous chunks. Multi-card placement, full-model scale
and correct KV continuation remain open in the [design space](../multilayer/RUNTIME_ADAPTATION.md).

## Reproduce and inspect

Use fresh output directories. The source is a resident-selection export; its
compiled QPCs are not needed. Run hardware benchmarks/profiles sequentially.

```bash
PY=/home/chihao/qeff-venv/bin/python
$PY tools/moe_resident_selection.py --out /tmp/resident_source \
  --stage export --hidden 2048 --intermediate 768
$PY tools/moe_adaptive_probe.py --out /tmp/adaptive --source /tmp/resident_source
$PY tools/moe_adaptive_probe.py --out /tmp/adaptive --stage compile --precision mxfp6
$PY tools/moe_adaptive_probe.py --out /tmp/adaptive --stage run --precision mxfp6 --equivalence-only
$PY tools/moe_adaptive_probe.py --out /tmp/adaptive --stage sharing
$PY tools/moe_adaptive_probe.py --out /tmp/adaptive --stage sharing --precision mxfp6
$PY tools/moe_adaptive_probe.py --out /tmp/adaptive --stage audit --profile
$PY tools/moe_adaptive_probe.py --out /tmp/adaptive --stage audit --precision mxfp6 --profile
```

`--device`, `--iterations` and `--run-name` control retained runs. Generated
ONNX/QPC, binaries, JSON/CSV and traces remain local and are ignored by git.

- [Exporter, compiler, validator and auditor](../../tools/moe_adaptive_probe.py)
- [C++ resident-program host and stateless replay](../../tools/moe_adaptive_host.cpp)
- [Configuration/source hashes](adaptive_qwen/info.json), [specializations](adaptive_qwen/multi_specializations.json), [graph audit](adaptive_qwen/graph_audit.json)
- [FP16 results](adaptive_qwen/fp16/run/result.json), [MXFP6 results](adaptive_qwen/mxfp6/run/result.json)
- [FP16 shape/storage/trace audit](adaptive_qwen/fp16/package_audit.json), [MXFP6 audit](adaptive_qwen/mxfp6/package_audit.json)
- [FP16 replay transactions](adaptive_qwen/fp16/run/multi/recovery.csv), [MXFP6 transactions](adaptive_qwen/mxfp6/run/multi/recovery.csv)
