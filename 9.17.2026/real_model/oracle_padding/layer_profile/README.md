# Per-layer profiling and power-of-two capacities

Completed 2026-09-21. **Power-of-two capacities improve aggregate MoE latency
over minimum capacities in both precisions**, while minimum capacities remain
faster in some layers. The [full 48-layer tables and plots](results/README.md)
compare all six matched profiling builds.

| Precision | C128 mean MoE ms/layer | Minimum mean MoE ms/layer | Power-of-two mean MoE ms/layer | Minimum / power-of-two speedup vs C128 |
|---|---:|---:|---:|---:|
| FP16 | 11.5466 | 10.0475 | 9.7871 | 1.1492× / 1.1798× |
| MXFP6 | 10.1564 | 8.5871 | 8.0593 | 1.1828× / 1.2602× |

Each layer's value is the median of three traces; the table averages those
48 layer medians. Power-of-two has a lower median than minimum in 24/48 FP16
layers and 40/48 MXFP6 layers. Compared with minimum, summed MoE medians fall
2.59%/6.15%, and full-model host medians fall 2.33%/5.39% in FP16/MXFP6.
The profiling-enabled host medians are 605.044 → 526.944 → 514.675 ms for FP16
and 537.920 → 453.995 → 429.529 ms for MXFP6 (C128 → minimum → power of two).

The 2026-09-22 [routing-column permutation profile](routing_permutation/README.md)
isolates the graph's added `Gather` using these same saved traces. Its broad
elapsed envelope, including associated copies and intervening gaps, averages
about 60–66 µs per layer across the four regrouped cases, below 1% of the MoE
interval. The gather itself has about 24 µs of busiest-core execution. These
direct costs do not explain the remaining milliseconds per layer; removing the
permutation has not been measured as a causal graph ablation.

The [additive elapsed-time breakdown](breakdown/README.md) partitions all 864
layer/sample intervals into six ordered phases and reproduces the original
MoE medians exactly. In FP16, minimum padding reduces the second expert-group
span from 3.1905 to 1.7133 ms, but the first-group span grows from 3.0459 to
3.4418 ms and initial routing/packing/waits grow from 0.6333 to 1.1615 ms.
Final dense combination still takes 2.4612 ms. The report also records the
unchanged logical dense output buffer and expert weight dimensions, and keeps
engine busy times separate from the additive elapsed phases.

Keep both policies as candidates for per-layer tuning. For example, FP16
layer 0 favors minimum (9.0895 vs 10.0676 ms), while layer 2 favors power of
two (10.9753 vs 9.8683 ms). These are descriptive comparisons of three traces,
not significance tests. A mixed-policy full-model graph has not been measured.

The comparison uses the same Qwen3-30B-A3B, first 128-token chunk, 48 layers,
four AI 100 cards and native two stages of 64 experts as the
[completed uninstrumented oracle experiment](../README.md).
All six new builds use `-stats-level=70`: native expert order with C128,
regrouped minimum capacities, and the same regrouping with capacities rounded
up to powers of two, in FP16 and MXFP6.

Capacity is shared by a group, not selected independently for every expert.
For layer `l` and group `g`, minimum capacity is
`max(1, max(tokens_assigned_to_expert[e] for e in group[l,g]))`.
Power-of-two capacity is the smallest of `1,2,4,8,16,32,64,128` that covers
that minimum. Layer 0 changes from `(46,6)` to `(64,8)` in both precisions.
The expert permutations stay fixed, so this comparison isolates the capacity
policy between the two regrouped candidates.

| Precision | Minimum padded rows / C128 | Power-of-two padded rows / C128 | Distinct capacity pairs, minimum → power of two |
|---|---:|---:|---:|
| FP16 | 37.6790% | 48.3073% | 43 → 6 |
| MXFP6 | 37.5895% | 48.2340% | 42 → 5 |

These are logical work and shape counts, not measured speedups. A smaller
shape set could help a future runtime graph cache, but this experiment uses
one static full-model graph for each candidate. It does not measure runtime
selection or regrouping costs.

Each candidate collects 60 host timings (three rounds, two warmups each) and
three SDK traces after two warmups. The host validates repeatability, device
allocation and release, routing assignment totals, and any supplied capacity
plan. Instrumented outputs must match the corresponding prior control exactly;
every saved profiler output must also match the new host run byte for byte.

All six cases passed: **360 timed prefills and 18 detailed traces**, zero
overflow, exact outputs against the corresponding native or regrouped C128
control, and stable/released device resources. The final audit is retained in
[final_validation.json](final_validation.json). All four cards ended Ready,
with 16 free cores each and no loaded or active networks. Disposable profiling
QPCs were removed after successful analysis; original QPCs remain available.

Padding equivalence does not establish FP32 accuracy. Regrouped FP16/MXFP6
relative L2 logit errors remain 3.0959%/17.2008% against the saved FP32 reference;
only FP16 passes the unchanged 5% diagnostic gate on this input. No unseen
input, multiple-chunk KV continuity or generation-quality claim is made.

The MoE trace interval starts at the earliest router HMX operation
and ends at the latest final cross-card `Einsum_4` compute operation. This
observable interval covers routing computation, packing, expert execution,
combination and intervening waits. It excludes earlier router-input conversion
or speculative weight prefetch, and the following residual operation.
The real traces were inspected to validate those boundaries; per-layer
intervals are ordered and do not overlap within each captured execution.
Expert HMX busy times and spans are reported separately; they overlap across
cores and engines and must not be summed to obtain operator latency.

In the minimum and power-of-two MXFP6 traces, layer 31's cold group has zero
routing assignments and no recorded expert GEMM events on any engine. The
analyzer records a zero GEMM span for that stage only after checking the
independently validated routing counts and the absence of all three GEMM nodes.
It still measures the full router-to-combine interval, including packing and
combination work. This observation does not establish a general runtime
expert-skipping mechanism.

Profiling instrumentation affects code and timing. Compare these candidates
with one another; the new absolute times must not be mixed with the earlier
uninstrumented baseline times.
Host timing and detailed trace capture are separate runs of the same QPC.
The trace interval should not be subtracted from the separate host measurement.
Compilation and CPU trace decoding can overlap device runs; the six cases are
sequential comparisons, not a randomized benchmark.

Scripts:

- [`moe_qwen3_oracle.py`](../../../../tools/moe_qwen3_oracle.py):
  `plan --rounding power-of-two`, `compile --stats-level 70`.
- [`moe_qwen3_profile.py`](../../../../tools/moe_qwen3_profile.py):
  checked SDK capture and per-layer/engine trace analysis.
- [`moe_qwen3_profile_sweep.py`](../../../../tools/moe_qwen3_profile_sweep.py):
  precision-specific compile/run/profile sequence, with device runs serialized
  between the two workers. Disposable QPCs are removed only after validation
  and successful trace analysis; original QPCs and checkpoints are preserved.
- [`moe_qwen3_profile_report.py`](../../../../tools/moe_qwen3_profile_report.py):
  both precisions' 48-layer tables and standalone PNG/PDF/SVG plots.
- [`moe_qwen3_routing_profile.py`](../../../../tools/moe_qwen3_routing_profile.py):
  isolated routing permutation analysis with independent raw-trace checks.
- [`moe_qwen3_breakdown.py`](../../../../tools/moe_qwen3_breakdown.py):
  six additive MoE phases, separate engine work, logical capacity/weight sizes
  and stacked timing plots from the saved traces.

[capacity_policies.csv](capacity_policies.csv) contains both capacity policies
for every layer and precision. Precision-specific `*_progress.json` and
`*_sweep.log` files record completion and any failures. The initial MXFP6
analysis stopped on layer 31's zero-assignment stage; the checked analyzer
handling described above allowed analysis to resume using the saved traces.
This was an analysis coverage check, not an inference or output failure.

## Reproduce

The model and SDK tools use `/home/chihao/qeff-venv/bin/python`. Power-of-two
plans preserve the regrouped C128 calibration layout:

```bash
QWEN_PY=/home/chihao/qeff-venv/bin/python
QWEN_ORACLE=9.17.2026/real_model/oracle_padding
QWEN_PROFILE="$QWEN_ORACLE/layer_profile"
$QWEN_PY tools/moe_qwen3_oracle.py plan \
  --counts "$QWEN_ORACLE/sorted128_fp16_run/counts_i32.bin" \
  --calibrate "$QWEN_ORACLE/sorted128_fp16_plan.json" \
  --rounding power-of-two --out "$QWEN_PROFILE/repeat_pow2_fp16_plan.json"

# Reanalyze retained compressed traces without using the devices.
$QWEN_PY tools/moe_qwen3_profile.py analyze \
  --trace-dir "$QWEN_PROFILE/pow2_fp16_profile/trace" \
  --out "$QWEN_PROFILE/repeat_pow2_fp16_analysis"
```

Graph export uses the oracle script's `oracle --plan ... --weights-cache ...`
action; reuse the corresponding precision's existing reordered weight cache.
Compile each candidate with `compile --stats-level 70`; the sweep requires
the native diagnostic graph, calibrated minimum graphs, and power-of-two
graphs/plans in the directory layout shown above. Run each precision with
`moe_qwen3_profile_sweep.py --root ORACLE --scratch NEW_COMPILE_ROOT
--precision fp16|mxfp6`. Use a fresh experiment root for a new sweep; `--resume`
continues an interrupted run by reusing completed steps. The sweep's device
lock coordinates workers sharing the same experiment root.

The report requires NumPy and Matplotlib (this run used Matplotlib 3.10.8 in a
separate local plotting environment). To regenerate tables and plots from the
saved analyses, use that environment's Python:

```bash
python tools/moe_qwen3_profile_report.py \
  --root 9.17.2026/real_model/oracle_padding/layer_profile \
  --out 9.17.2026/real_model/oracle_padding/layer_profile/repeat_results
```

All output directories must be new. Markdown records the numerical results in
Git; compiled binaries, graphs, raw traces, CSV/JSON data and plot files remain
local under the repository's artifact policy.
