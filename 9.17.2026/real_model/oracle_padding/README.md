# Full-model static oracle padding

Experiment started 2026-09-21; SDK/runtime/firmware 1.21.6, cards 0–3.
The user authorized static per-layer capacities and expert regrouping to measure
an input-specific optimization bound before implementing runtime selection.

**Static per-layer/group minimum capacities reduce measured full-model latency
by 12.35% in FP16 and 14.30% in MXFP6.** Expert groups are fixed offline, then
the entire model executes in one resident QPC without runtime selection.

| Format | Matched native C128, ms | Static oracle, ms | Speedup | Latency reduction |
|---|---:|---:|---:|---:|
| FP16 | 572.298 | 501.623 | 1.1409× | 12.3494% |
| MXFP6 | 496.480 | 425.481 | 1.1669× | 14.3005% |

Both minimum-capacity variants have zero overflow and reproduce the regrouped
C128 routing counts and last-token logits bit for bit. Each result contains
60 measured prefills and 6 warmups. This is an input-specific oracle for the
current two-stage, 64-expert-per-stage topology, not a global latency optimum.

The follow-up [per-layer profiling and power-of-two capacity experiment](layer_profile/README.md)
uses matched instrumented builds to measure individual MoE intervals and test
a third capacity policy. Its timings are kept separate from this uninstrumented comparison.

The [controlled cold-group sweep](cold_capacity_control/README.md) replays trained
FP16 layer 2 with identical activations, routing and expert order, hot C128, and
seven cold capacities. Cold C128 → C2 reduces isolated MoE host latency by
13.93%; C32 is the fastest tested variant. The traces show substantial HMX
dependency waits at small capacities. These isolated-layer timings are kept
separate from full-model results.

The [detailed layer-2 card/core profile](layer2_detail/README.md) decodes the
saved C128/C2 captures with dependency flows and compiler byte counts. It shows
remote activations releasing groups of cores, persistent weight-sized DDR
copies at small capacity, and core-0 vector reductions. It includes the actual
full-model layer budget, cross-card timelines and all 16 cores within card 0.

The [measured overhead ablations](overhead_ablation/README.md) remove the first
group's redundant zero-accumulator read/add and tile the local reduction. At
fixed hot128/cold2, the combined rewrite reduces uninstrumented layer-2 replay
latency from 10.305 to 7.776 ms (24.54%), with bit-exact saved outputs. The
profile shows smaller TCM merge kernels on core 0 and unchanged P2P payload.

## Scope and method

The benchmark uses the trained Qwen3-30B-A3B model, all 48 decoder layers,
embedding, attention, final normalization and LM head in one resident four-card
QPC. Batch size is 1, prefill length 128, context length 256. The input is the
same first 128 tokens of saved GSM8K prompt 41 as the [original baseline](../README.md).
There is no host dispatch between layers and no runtime selection of padding.

Routing calibration, sorting, weight reordering, compilation, loading and
activation occur outside the measured interval. The timing assumes the chosen
layout is already resident. The measured saving is 70.7 ms in FP16 and 71.0 ms
in MXFP6 for the whole 128-token prefill; an online implementation must fit its added costs
within that saving to retain a net benefit for this case.

This experiment retains the native topology: each layer has two sequential
stages of 64 experts. Both stages span the four-card partition; the hot and cold
groups are not assigned to separate pairs of cards. It does not search all
possible group widths, partition layouts, or expert-to-core placements.

1. Run native C128 with an added `[48,128]` int32 routing-count output. These
   counts are taken before packing truncates any expert's token list. Verify
   the diagnostic model reproduces the original C128 logits exactly.
2. Sort each layer's experts by those device counts, with expert ID breaking
   ties. The first 64 experts form the hot stage and the remaining 64 form the
   cold stage. Apply the same permutation to the dense routing matrix and to
   all gate, up and down weight banks. Router weights and TopK remain unchanged.
3. Run the regrouped model at C128 and recalibrate using its actual routing.
   Changing reduction order can change downstream FP16 routing, so counts from
   the original layout alone are insufficient for a tight capacity guarantee.
4. For each layer/stage, set capacity to `max(1, maximum observed expert count)`.
   Compare this minimum-safe candidate with capacities rounded upward to a
   multiple of 16. Smaller logical tensors need not produce faster kernels.
5. Check observed pre-truncation counts against every capacity, compare logits
   with the regrouped C128 control, and measure full-model latency.

For fixed expert counts and two equal groups of 64, sorting minimizes the sum
of group maxima: one group must contain the largest count, and the other group's
maximum cannot be smaller than the 65th-largest count. This minimizes logical
padded rows for that topology and routing trace. It is not a proof of the
globally fastest compiled graph, nor a bound covering arbitrary group widths.

The final layout is frozen before capacity calibration so numerical changes do
not silently invalidate the plan. Consequently it is not an exact optimizer
over regrouping-dependent routing: re-sorting the calibrated counts while
hypothetically holding them fixed would save another 704 FP16 rows or 384
MXFP6 rows (about 0.24%/0.13% of the chosen padded work). We do not perform that
additional regrouping or claim it would preserve routing. Empty groups retain
capacity 1; expert pruning and zero-capacity elimination are outside this test.

The graph editor changes each stage's packed-index Slice end and valid-row
Range stop. Native packing, attention, residuals and trained values are retained.
The first regrouped control represents weight reordering with constant
Concat/Gather nodes. Subsequent builds reuse materialized external weight banks:
the exporter streams the original FP16 bytes into the requested expert order,
without converting or changing their values. Both precision-specific layouts
passed bitwise spot checks of 864 expert matrices each, covering all 48 layers,
all three projections, both stages, and lanes 0/31/63. Original weight files are
accessed read-only.
The exporter writes both `timing.onnx` and a count-producing `diagnostic.onnx`.

## Control measurements

Both rebuilt controls use the same compilation flags and routing-count output
as the oracle diagnostic candidates. Each has 60 timed invocations in three
rounds, with two warmups per round.

| Control | Median ms | P10–P90 ms | Logits vs original C128 |
|---|---:|---:|---|
| FP16 C128, original expert order | 572.298 | 569.939–574.242 | Bit-identical |
| MXFP6 C128, original expert order | 496.480 | 494.540–499.296 | Bit-identical |
| MXFP6 C128, regrouped expert order | 496.578 | 494.466–499.091 | 2.3927% relative L2; same next token |
| FP16 C128, regrouped expert order | 573.328 | 571.428–575.057 | 3.3700% relative L2; same next token |

The older cached MXFP6 QPC measured 525.103 ms and included compiler profiling
instrumentation. Its number is not the denominator for the matched oracle
comparison. Adding count outputs does not change logits in either format.

The initial device counts suggest minimum padded-row fractions of 37.622% for
FP16 and 37.557% for MXFP6 after sorting, before recalibration. Cold-stage maximum
counts are at most 6 and hot-stage maxima range from 46 to 122. These are work
estimates, not measured speedups.

After actual MXFP6 regrouping, 602 expert counts change, with a largest count
change of 4. Its calibrated minimum/rounded-to-16 padded fractions are
37.5895%/45.9635%. The regrouped C128 run has zero overflow, stable/released
resources, and repeatable counts/logits. Its FP32 reference error is 17.2008%,
still above the unchanged 5% diagnostic gate.

The FP16 regrouped C128 control also completed 60 measured prefills, with
zero overflow and stable/released resources. Its FP32 relative L2 error is
3.0959%, below the unchanged 5% diagnostic gate for this input. This is an
observed effect of the changed reduction order, not a general accuracy fix.
Its calibrated minimum/rounded-to-16 padded fractions are 37.6790%/45.8333%.

## Optimized measurements

| Format | Capacity policy | Padded rows / C128 | Median ms | P10–P90 ms | Speedup vs native C128 |
|---|---|---:|---:|---:|---:|
| FP16 | Minimum safe | 37.6790% | 501.623 | 500.810–502.387 | 1.1409× |
| FP16 | Round up to 16 | 45.8333% | 513.963 | 513.146–514.502 | 1.1135× |
| MXFP6 | Minimum safe | 37.5895% | 425.481 | 424.666–426.598 | 1.1669× |
| MXFP6 | Round up to 16 | 45.9635% | 444.950 | 444.052–445.896 | 1.1158× |

**All four candidates completed 60 measured prefills plus 6 warmups each.**
Every invocation passed the capacity guard. Saved routing counts and all
151,936 last-token logits are bit-identical to the corresponding regrouped
C128 control. Every capacity audit reports zero overflow assignments.
Minimum capacities are 2.40% faster than rounding to 16 in FP16 and 4.38% faster
in MXFP6. This is a comparison of two capacity policies, not an exhaustive
latency search over every possible integer capacity vector.

The minimum-capacity variants remove 62.3210%/62.4105% of logical padded rows
in FP16/MXFP6, while reducing end-to-end latency by 12.3494%/14.3005%.
Logical row savings should not be treated as a latency prediction.

| Minimum-capacity result | FP16 | MXFP6 |
|---|---:|---:|
| Input tokens/s | 255.17 | 300.84 |
| Active device GiB, four-card total | 61.333 | 26.785 |
| Native C128 active device GiB | 60.748 | 26.551 |
| Last-token relative L2 vs FP32 | 3.0959% | 17.2008% |
| Unchanged 5% diagnostic gate | Pass on this input | Fail |
| Next token vs FP32 | Match | Match |

Active allocation increases slightly despite fewer padded rows. Resource
snapshots are stable and all four cards release their allocations after every
run. The complete comparison contains 480 timed invocations across eight cases
(native C128, regrouped C128, minimum and rounded capacities in each precision),
plus the warmups. Machine-readable results are in [comparison.json](comparison.json)
and [comparison.csv](comparison.csv).

## Validation and timing boundaries

The C++ runner times buffer binding, enqueue and completion wait. Loading,
activation, input preparation, output validation, and file writes are excluded.
Every invocation checks finite and repeatable last-token logits and routing
counts; each layer must have exactly `128 * 8 = 1024` routing assignments.
The capacity audit reports the number of dropped assignments as
`sum(max(count - capacity, 0))` and rejects a nonzero result.
Hardware runs were serialized. The cases were measured sequentially rather
than in randomized order; host compilation overlapped several cases. These are
descriptive measurements for this server and input.

Count-producing comparisons include the same extra 24 KiB output in both
control and candidate. Accuracy relative to the saved FP32 reference is reported
separately from padding equivalence. The original FP16/MXFP6 controls have
5.0293%/18.8420% relative L2 logit error; neither passes the inherited 5% diagnostic
gate. No threshold is relaxed by this experiment.

This is one repeatedly executed first chunk of one known input, calibrated on
that same input. It does not establish unseen-prompt accuracy, multiple-chunk KV
continuity, generation quality, or an online selector's attainable speedup.

## Reproduction and artifacts

Use `/home/chihao/qeff-venv/bin/python` from the repository root. The graph editor
is [`tools/moe_qwen3_oracle.py`](../../../tools/moe_qwen3_oracle.py); hardware runs
use [`tools/moe_qwen3_baseline.py`](../../../tools/moe_qwen3_baseline.py) and its
C++ host. Every export/compile/run output directory must be new.

To rerun the retained minimum-capacity QPCs on the saved input:

```bash
QWEN_PY=/home/chihao/qeff-venv/bin/python
QWEN_ORACLE=9.17.2026/real_model/oracle_padding
for QWEN_FORMAT in fp16 mxfp6; do
  $QWEN_PY tools/moe_qwen3_baseline.py run \
    --qpc "$QWEN_ORACLE/min_${QWEN_FORMAT}_compile/qpc" \
    --out "$QWEN_ORACLE/repeat_min_${QWEN_FORMAT}" \
    --precision "$QWEN_FORMAT" --routing-counts \
    --capacity-plan "$QWEN_ORACLE/min_${QWEN_FORMAT}_plan.json" \
    --scope 'Static regrouped oracle, minimum capacities, 128-token first chunk'
done
```

The saved [FP16](calibrated_fp16_capacities.csv) and
[MXFP6](calibrated_mxfp6_capacities.csv) tables contain all 48 layers' capacities.
Plans also record the full expert permutations and hashes of their calibration
inputs. To build a new calibrated layout:

```bash
QWEN_PY=/home/chihao/qeff-venv/bin/python
QWEN_ORACLE=9.17.2026/real_model/oracle_padding

# Original layout, with pre-truncation routing counts.
$QWEN_PY tools/moe_qwen3_oracle.py diagnostic --out "$QWEN_ORACLE/new_diagnostic_graph"
$QWEN_PY tools/moe_qwen3_oracle.py compile \
  --graph "$QWEN_ORACLE/new_diagnostic_graph/model.onnx" \
  --out "$QWEN_ORACLE/new_diagnostic_compile" --precision fp16
$QWEN_PY tools/moe_qwen3_baseline.py run \
  --qpc "$QWEN_ORACLE/new_diagnostic_compile/qpc" \
  --out "$QWEN_ORACLE/new_capture" --precision fp16 --routing-counts \
  --scope 'Native C128 oracle calibration control'

# Reorder experts, retaining C128 for safe recalibration.
$QWEN_PY tools/moe_qwen3_oracle.py plan \
  --counts "$QWEN_ORACLE/new_capture/counts_i32.bin" \
  --order sorted --alignment 128 --out "$QWEN_ORACLE/new_sorted128_plan.json"
$QWEN_PY tools/moe_qwen3_oracle.py weights \
  --plan "$QWEN_ORACLE/new_sorted128_plan.json" --out "$QWEN_ORACLE/new_weights"
$QWEN_PY tools/moe_qwen3_oracle.py oracle \
  --plan "$QWEN_ORACLE/new_sorted128_plan.json" --weights-cache "$QWEN_ORACLE/new_weights" \
  --out "$QWEN_ORACLE/new_sorted128_graph"
# Compile/run diagnostic.onnx as above, then preserve that layout when reading
# its counts: those counts are indexed by the new bank position, not expert ID.
$QWEN_PY tools/moe_qwen3_oracle.py plan \
  --counts "$QWEN_ORACLE/new_sorted128_run/counts_i32.bin" \
  --calibrate "$QWEN_ORACLE/new_sorted128_plan.json" --alignment 1 \
  --out "$QWEN_ORACLE/new_minimum_plan.json"
$QWEN_PY tools/moe_qwen3_oracle.py oracle \
  --plan "$QWEN_ORACLE/new_minimum_plan.json" --weights-cache "$QWEN_ORACLE/new_weights" \
  --out "$QWEN_ORACLE/new_minimum_graph"
# Compile diagnostic.onnx and run with --routing-counts and
# --capacity-plan "$QWEN_ORACLE/new_minimum_plan.json" to enforce bounds on every invocation.
```

`selftest` executes actual native packing and mask dependencies and compares
reduced capacities with the C128 prefixes. `regroup-selftest` additionally checks
the routing permutation, valid packed rows, and all three weight-bank
permutations with small synthetic tensors. Both passed; they complement the
full-model hardware checks and are not trained-model accuracy tests.

All raw artifacts are local and ignored by Git. This directory retains plans,
graph exports, compile commands/logs, counts, logits, samples, resource snapshots,
and JSON reports. Disposable calibration QPC binaries may be removed after
verified captures to make room for subsequent builds; `qpc_removed.json` records
each removal. Original baseline QPCs and checkpoint files are retained.

The initial FP16 Concat/Gather compile was stopped when temporary constant files
nearly exhausted disk space. An overlapping, incomplete MXFP6 weight-cache write
failed with ENOSPC and was discarded; `weights_mxfp6_v2` is the completed cache.
Early materialized FP16/rounded-MXFP6 builds were also stopped for storage
rescheduling before producing any QPC. These are not timing results. Some later
QPCs used `/dev/shm/qwen3_oracle_wentao_20260921/` as temporary build storage;
hardware timing excludes file loading and activation. The two winning QPCs are
retained under `min_fp16_compile/qpc/` and `min_mxfp6_compile/qpc/` in this
workspace. Their `relocation.json` files record byte counts and matching SHA256
hashes before and after copying from RAM. Slower candidate QPCs were discarded
after validation; their source graphs, plans, compile logs and measurements remain.
