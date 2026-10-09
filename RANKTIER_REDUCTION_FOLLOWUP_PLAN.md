# RankTier Reduction: Follow-up Experiments and Claim Revision

October 6, 2026

This plan incorporates the added motivation and E44–E46 results reported in `RANKTIER_REDUCTION_RESEARCH_PLAN(1).md`. The underlying experiment logs and traces were not attached, so these results are treated as reported evidence rather than independently verified measurements.

## 1. Main assessment

The added evidence strengthens the importance of the reduction optimizations. It also identifies two issues that should change the immediate experimental priorities:

1. The attention head-group reduction confounded the previous explanation of expert-parallel performance.
2. The closest graph-level analog of a GPU combine—a direct gather of each token's top-k expert outputs followed by reduction over k—has not yet been built.

The research question is now:

**After expressing the correct sparse operation, what additional graph structure is necessary to preserve locality and parallel execution, and does that requirement remain when expert capacities become heterogeneous?**

The incremental gain of tiering over an optimized static program does not measure the complete contribution: that static program already contains the reduction improvements.

## 2. P0: Rebaseline with the attention reduction fixed

### Why this comes first

The updated notes attribute the previously observed slowdown of expert-parallel compilation at larger chunks to attention's head-group sum, rather than to the MoE combine. This changes the causal explanation in the paper.

The reported reduction speedups of 2.21×, 2.72×, and 2.94× at T = 128, 256, and 512 make the routing/combine work look more important, but the attention rewrite must remain separately identifiable in the attribution.

### Tasks

- [ ] Use the corrected attention head-group sum in every controlled expert-parallel comparison.
- [ ] Retime the relevant ladder variants in comparable sessions, with repeated anchors.
- [ ] Report the attention rewrite separately from routing metadata construction and MoE output aggregation.
- [ ] Recheck numerical agreement against the chosen same-precision reference after changing the attention summation order.
- [ ] Remove the old causal claim from the introduction, motivation, and evaluation.
- [ ] Keep old measurements labeled as diagnostic history; do not mix them into the final ladder without identifying configuration differences.

### Interpretation

The attention result suggests that problematic reduction lowering occurs in more than one part of the model. It motivates investigating a reusable graph-transformation principle. It does not establish that the problem is unique to MoE routing or inherent to every AOT compiler.

## 3. P0: Separate sparse selection, device locality, and parallel execution

The existing variants expose three different obstacles:

| Property | Reported observation | Remaining question |
|---|---|---|
| Sparse selection | The token-centric version gathers all 32 lanes for each token. | Does directly selecting only top-k contributions eliminate the remaining overhead? |
| Device locality | Explicit per-SoC slices cause substantial cross-SoC transfers. | Which graph structure makes locality survive compilation? |
| Parallel execution | A locality-preserving token-centric version still performs cross-core lane sums and serializes token tiles. | Does assigning token reduction locally remove those dependencies? |

The working hypothesis is that an efficient representation must expose selected contributions, preserve their device placement, and give independent token outputs independent execution resources.

These properties must be isolated. The current variants change more than one property at a time.

## 4. P0: Build the direct top-k gather/reduce baseline

### Variants

| Variant | Definition | Purpose |
|---|---|---|
| C-slices | Sparse combine expressed through explicit per-SoC slices | Diagnose placement changes introduced by that formulation |
| C-lanes | Token-indexed gather over all local lanes, followed by lane reduction | Test whether retaining locality is sufficient |
| C-topk | Gather each token's k expert contributions directly, then reduce over k | Test the most direct sparse-combine formulation |
| D | Proposed token-owned implementation | Isolate the benefit of its additional layout and work-distribution choices |

Include the repaired dense combine as a reference. The main novelty comparison is **C-topk versus D**; C-slices and C-lanes explain intermediate failure modes.

### Controls

- [ ] Start with uniform capacity T and static expert weights.
- [ ] Hold the attention fix, expert placement, routing metadata construction, input routing, precision, and cache configuration constant.
- [ ] Use matched tile settings for the primary attribution comparison; report independently tuned settings separately if they differ.
- [ ] Include all packing, concatenation, or layout-conversion costs required by a variant.
- [ ] Verify output agreement under the same numerical criterion.
- [ ] Give each natural formulation reasonable tuning; do not use a deliberately poor translation as the decisive baseline.

### Measurements

Record absolute combine latency, full-layer latency, DRAM traffic, cross-SoC traffic, intermediate buffers, core work distribution, and serialized intervals. Preserve the compiler settings and traces that explain each difference.

### Decision rule

- If C-topk performs similarly to D, emphasize integration with static partitioning and heterogeneous tiers. A stronger claim for the standalone combine would need additional evidence.
- If C-topk fails in a reproducible way that D resolves, identify the exact redistribution, materialization, or execution dependency responsible. This supports a distinct compilation problem and transformation.

## 5. P0: Test custom-op integration feasibility

The updated notes report an HVX custom-op toolchain in the SDK. Therefore, do not motivate the graph approach by assuming that custom kernels cannot be written.

The unresolved issue is whether a custom combine can integrate into the four-SoC graph while preserving local inputs and efficient execution.

Start with a small correctness-first integration experiment:

- [ ] Can the custom operation compile inside the four-SoC partitioned graph?
- [ ] What placement and input layouts does it receive?
- [ ] Are expert outputs supplied locally, or copied/redistributed?
- [ ] Does integration introduce serialization or additional buffers?
- [ ] Can it produce the required local token partials and feed the final cross-device operation correctly?

If integration works, implement a reasonably tuned custom combine as a baseline and measure its complete integration cost. If it fails or introduces unavoidable overhead, document the observed limitation and its scope. Do not infer a hardware limitation from one unsupported graph configuration.

## 6. P0: Run the controlled combine × capacity experiment

Once the baseline and direct sparse formulation are understood, compare:

| Capacities | Repaired dense combine | Proposed combine |
|---|---|---|
| Uniform capacities | Reference | Reduction benefit without capacity savings |
| Ranked tiers | Does combine overhead absorb saved expert work? | Combined benefit |

- [ ] Use the same weight-access mechanism across configurations.
- [ ] Keep sorting present in both capacity configurations where feasible, or account for it separately.
- [ ] Include the attention fix everywhere.
- [ ] Separate expert computation, routing metadata construction, and combine work in the measurements.
- [ ] Distinguish resource busy time from elapsed critical-path time when operations overlap; do not sum overlapping intervals as a latency breakdown.
- [ ] Test whether the previously reported tiering slowdown under the exported combine persists after reasonable dense-path repairs.

This experiment establishes whether reduction is an enabling component of heterogeneous capacity execution, and which part of that interaction survives a stronger baseline.

## 7. P1: Tighten traffic accounting and reprofile the remaining combine

At fixed tier count, a `[T·k, d]` gather result can be independent of lane width and capacity. Increasing the number of tiers still adds gather work.

Distinguish these quantities explicitly:

| Quantity | Meaning |
|---|---|
| Useful contributions | Expert rows that mathematically contribute to the output |
| Gather-result size | Elements produced by the graph operation, potentially including masked slots |
| Physical DRAM traffic | Reads/writes observed after compiler lowering and buffering |
| Resource busy time | Total activity of the relevant execution resources |
| Exposed latency | Work on the critical path after accounting for overlap |

A 16 MiB gather result does not by itself prove 16 MiB of DRAM input reads. Masking, repeated indices, buffering, and lowering can change the relationship.

### Single gather across tiers

The proposed reduction from three gather results to one is a useful optimization candidate, but it must include:

- Any concatenation or layout-conversion cost.
- Changes to buffer lifetime and peak memory.
- Lost overlap with earlier expert stages.
- Changes to placement and cross-SoC traffic.

Reprofile after the attention fix before prioritizing this optimization. The earlier 20% busy-core share and 0.40 ms exposed tail describe different quantities and were measured before that fix.

## 8. P1: Focused mechanism sweeps

After the controlled comparisons, prioritize:

1. Local lane count with T, k, hidden width, and routed-assignment count controlled.
2. Tier count and width, reporting the additional gather work explicitly.
3. Chunk length, checking whether buffer spills explain changes in latency.

Add core/device scaling when placement and workload controls make the result interpretable. The objective is to demonstrate a predictable mechanism, not merely to accumulate more configurations.

## 9. Claim boundaries for the paper

**Supported by the updated notes, subject to verification of the underlying logs:** reductions are a major performance factor; natural graph formulations can differ substantially in placement and execution; and the attention reduction confounded an earlier attribution.

**Still to establish:** whether the proposed combine provides a distinct benefit beyond a direct top-k gather/reduce; whether a custom operation can provide that benefit more directly; and whether the graph-transformation principle holds across meaningful variations in capacities and layout.

Candidate framing:

> RankTier makes sparse MoE aggregation compatible with heterogeneous expert execution by expressing selected contributions and preserving their device locality and parallel reduction structure through graph compilation.

Use this framing only to the extent supported by the controlled experiments. Sparse aggregation and token-oriented GPU computation already have prior art; the new evidence must establish the additional graph-compilation challenge and its solution.

## 10. Immediate execution order

| Priority | Work item | Required output |
|---|---|---|
| 1 | Rebaseline with corrected attention | A consistent timing ladder and corrected causal explanation |
| 2 | Build C-topk and compare against D | Latency/traffic comparison with a trace-supported explanation |
| 3 | Test custom-op feasibility | A working integration baseline or a documented limitation |
| 4 | Run the 2×2 interaction experiment | Evidence that the reduction design enables capacity savings |
| 5 | Run focused sweeps | Scope and mechanism of the improvement |

Source: updated `RANKTIER_REDUCTION_RESEARCH_PLAN(1).md`, especially §1.1–1.5, and the accompanying discussion. See that plan's references for Tutel and existing CUDA/Triton sparse-combine implementations.

## 11. Results, October 7–8, 2026

Full 48-layer model, KV cache on device, MXFP6, four SoCs. Every head-parallel program uses the elementwise head-group sum. Latencies come from one program + session fit of log latency over every timed pass of E46–E51; every session holds the step-5 program as an anchor (largest residual 0.50%, 0.88% and 1.32% at T = 128, 256 and 512). Sources: `9.17.2026/real_model/oracle_padding/mdts_flag/full_model/e51/e46_e51_summary.txt` (from `scripts/e47_summary.py`), run logs in `full_model/e47`–`e51`, and the drivers `scripts/run_e47.sh`, `run_e48.sh`, `run_e50.sh`, `run_e51.sh`.

### 11.1 Rebaseline with the attention reduction fixed (E47; Section 2)

All ladder steps and attribution programs were re-timed. Ladder step 3 is split into routing metadata (tree prefix scans, new `build_full_stack.py --rewrite-set scan`) and the dense-combine repairs. Step 6 is attribution program D, the same algorithm built with RankTier's builder.

| Increment | T = 128 | T = 256 | T = 512 |
|---|---:|---:|---:|
| Attention: elementwise head-group sum | 1.07× | 1.13× | 1.13× |
| Routing metadata: tree prefix scans | 1.18× | 1.17× | 1.17× |
| Dense-combine repairs (no zero read, tiled lane sum) | 1.50× | 1.49× | 1.54× |
| Token-owned combine | 1.23× | 1.53× | 1.63× |
| Elementwise final sum | 1.02× | 1.03× | 1.005× |
| MoE output aggregation (repairs through final sum) | 1.88× | 2.33× | 2.51× |
| Ranked capacities (step 5 → RankTier) | 1.08× | 1.31× | 1.55× |
| Naive T/T → RankTier | 2.37× | 3.55× | 4.56× |

- Ladder, ms, T = 128 / 256 / 512: naive T/T 250.7 / 505.4 / 1033.1; step 5 113.5 / 185.9 / 351.1; RankTier 105.6 / 142.3 / 226.5.
- Attribution at tuned tiles: capacities C → E give 1.10× / 1.15× / 1.37×; RankTier is 1.08× / 1.19× / 1.28× faster than the static program A.
- At T = 256 the two-tier program (138.8 ms) is 2.5% faster than three tiers (142.3 ms); the extra stage no longer pays there.
- The scans and repairs are bit-identical to naive T/T, and A–E are bit-identical to each other. Accuracy against FP32 stays at the MXFP6 level. The next token differs only at the known T = 256 near tie.

### 11.2 Direct top-k gather/reduce against D (E48; Sections 3–4)

All variants run at capacity T with static weights, the same routing metadata and the same elementwise final sum. Each is numerically correct (same next token as D).

| Combine | T = 128 | T = 256 | T = 512 | Cross-SoC MiB per MoE layer (T = 512) | MoE layer, ms (T = 512) |
|---|---:|---:|---:|---:|---:|
| D, token-owned | 113.5 ms | 185.9 ms | 350.8 ms | 14.0 | 6.9 |
| C-lanes: gather all lanes on the lane axis, then lane sum | 1.12× | 1.11× | 1.15× | 12.1 | 7.5 |
| Repaired dense combine | 1.25× | 1.57× | 1.64× | 48.1 | 9.8 |
| C-slices: C-lanes with explicit per-SoC lane slices | 7.02× | 8.22× | 4.31× | 576.5 | 30.6 |
| C-topk: one gather of each token's k rows from all SoCs, sum over k | 4.05× | 4.70× | 4.92× | 1538 | 35.7 |

The last two columns come from instrumented two-layer profiles, layer 1. Ratios are slowdowns against D. C-topk is the same with CtxGather3D and with ONNX Gather.

**Decision rule outcome (Section 4):** C-topk fails reproducibly, and D resolves it. The profile names the cause: redistribution. C-topk ships about 110× D's cross-SoC bytes because its gather indexes expert outputs across SoCs. C-lanes keeps locality but stays 11–15% slower, because its lane sum crosses cores. This supports a distinct graph-compilation problem rather than a plain sparse-combine result.

### 11.3 Custom-op integration (E49, E51; Section 5)

`scripts/customop/tokencombine`: a TokenCombine op computes each SoC's token partials from that SoC's expert outputs, with the output tileable by SoC, token and column. It replaces D's per-SoC gather, mask and sum; the index construction and the elementwise sum across SoCs stay in the graph.

- **Compiles inside the four-SoC partition:** yes. Tiles land on all 64 cores, and each SoC's tiles read only that SoC's expert outputs. Cross-SoC bytes inside the MoE: 14.1 MiB, the same as D.
- **Correctness:** bit-identical to the graph combine on a small four-SoC test. Full model: relative L2 against FP32 0.0951 (D 0.0917), same next token.
- **Performance:**
  - The scalar kernel was 6.5–7.8× slower than D on two layers. It is compute-bound: about 26 ms of busy time per core per layer.
  - The HVX kernel with its operands in VTCM runs two layers in 14.8 ms against D's 15.7 ms.
  - On the full model at T = 512 it takes 326 ms against D's 351 (0.93×). This rests on one valid pass: the first pass hit a runtime timeout on its first execution, and the session's anchor drifted 2.6%.
- **Integration cost:** the full-model compile took 656 minutes, against 10–24 minutes for graph programs. qaic-compile inlines the kernel at every output tile and hands each SoC's network code to `hexagon-clang++ -O3`, single-threaded, about 2.7 hours per SoC. The op's VTCM preference for its data input was not met; the compiler warns about this 26,112 times without losing correctness.
- **Not measured:** T = 256 and 128 (stopped by choice), the tiered layout, and a kernel written in the C-topk style.

Interpretation: the fused kernel is a modest gain within the token-owned layout. It keeps locality because its tiles follow the explicit SoC axis of that layout, so it supports the layout principle rather than replacing it.

### 11.4 Combine × capacity with a repaired dense baseline (E50; Section 6)

All four cells rank experts at run time, gather weights from the bank, use the same elementwise final sum, and run at matched tiles. The repaired dense combine with tiers is the new `build_full_tiered.py --combine densefix`: one accumulator per tier width, no read of the zeros, read-modify-write for a width group's later tiers, and the lane sum in 16 token tiles.

| T | Repaired dense: uniform T → tiers | Token-owned: uniform T → tiers | Combine gain at uniform T | Combine gain with tiers |
|---|---|---|---:|---:|
| 128 | 146.8 → 200.9 ms (0.73×) | 115.9 → 105.6 ms (1.10×) | 1.27× | 1.90× |
| 256 | 223.5 → 232.2 ms (0.96×) | 164.1 → 142.3 ms (1.15×) | 1.36× | 1.63× |
| 512 | 411.7 → 396.8 ms (1.04×) | 309.8 → 226.5 ms (1.37×) | 1.33× | 1.75× |

The tiering penalty under a dense combine persists after the repairs at T = 128 and 256. At T = 512, tiers gain 4% under the repaired dense combine against 37% under the token-owned one.

### 11.5 What the results support (Section 9)

- **Reductions are a major performance factor:** MoE output aggregation alone gives 1.88–2.51×, and the routing scans another 1.17×.
- **A distinct graph-compilation problem:** the most direct sparse formulation is 4–5× slower than D because the compiler moves expert outputs across SoCs. D's layout prevents this.
- **A custom op as a more direct route:** about 7% faster, but only when it follows the token-owned layout, and at an 11-hour compile.
- **The reduction enables heterogeneous capacities:** supported against the repaired dense baseline at every chunk length.

### 11.6 Not done

- Section 7 traffic accounting is partial: cross-SoC bytes and MoE time are measured, DRAM reads per gather are not tabulated, and the single gather across tiers is not built.
- The Section 8 sweeps have not started.
- No second model and no GPU control.
