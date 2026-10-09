# RankTier Experiment Plan

Date: September 30, 2026

## Objective

Separate three sources of improvement in RankTier:

1. **Transferable algorithms:** runtime rank-bound capacities and compact, token-owned combination.
2. **Hardware-dependent choices:** tier widths, capacity alignment, concurrency, weight reuse, and memory traffic.
3. **Current compiler behavior:** placement restrictions, reduction serialization, weight-bank replication, and KV-cache pairing.

The central question is: **When does assigning whole experts to ranked capacity tiers outperform fixed-size block scheduling, and which benefits survive improvements to the compiler?**

This is a proposed plan, not a record of completed experiments. It builds on the supplied `ranktier.pdf` draft. Reuse existing measurements where configurations and controls already match.

## Priorities

| Priority | Experiment | Main question |
| --- | --- | --- |
| P0 | 1. Capacity attribution | Does ranking reduce latency after controlling for indirection and tuning? |
| P0 | 2. Fixed-block baseline | Does RankTier beat an alternative organization of irregular work? |
| P0 | 4a. Multi-chunk correctness and overflow recovery | Does the implementation work beyond the first chunk and recover correctly? |
| P1 | 3. Combine attribution | How much comes from eliminating dense intermediates versus repairing lowering? |
| P1 | 4b. Routing generalization | Do ranked capacities generalize across prompts, workloads, and models? |
| P2 | 5. Portability | Does the mechanism remain useful on a stack with different capabilities? |

## Common measurement protocol

- Start with the existing model, precision, batch size, and four-SoC configuration. Use chunk lengths **128, 256, and 512**.
- Within each comparison, hold context length, KV-cache state, weights, precision, inputs, and routing traces constant where applicable.
- Use identical recorded routes for kernel attribution; separately verify behavior with live routing in the full model.
- Warm up each executable. Run configurations in interleaved or randomized order within a session, repeat an anchor configuration, and record clocks, power, and thermal behavior.
- Report latency distributions or variability as well as medians. Preserve the draft's repeated-round protocol unless a concrete issue requires changing it.
- Report host-observed chunk latency and device-level component latency separately. Use uninstrumented runs for headline latency if profiling adds overhead.
- Give competing methods the same tuning budget. Show a common-setting comparison and a best-tuned comparison when tuning interacts with the method.
- Inspect compiled placement and memory behavior: equivalent graph intent does not guarantee equivalent lowering.
- Record SDK version, compiler flags, graph variant, executable identity, tile sizes, model precision, trace IDs, and device-memory footprint.

## 1. Isolate rank-bound capacities

### Configurations

Keep expert/head parallelism, token-owned combine, final reduction, precision, and KV handling identical.

| ID | Expert-to-lane assignment | Capacity | Weight access |
| --- | --- | --- | --- |
| A | Fixed identity | T for every expert | Static weights |
| B | Fixed identity | T for every expert | Runtime-indexed gather |
| C | Runtime load rank | T for every expert | Runtime-indexed gather |
| D | Runtime load rank | Existing two-tier capacities | Runtime-indexed gather |
| E | Runtime load rank | Three-tier RankTier capacities | Runtime-indexed gather |

For B, provide the identity indices as genuine runtime inputs or runtime data so constant folding does not remove the gather. Match its indexed-weight representation to C-E.

Run all configurations at T = 128, 256, and 512. First use a common maximum tile-size setting; then tune each configuration over the same feasible candidate set.

### Measurements

- Full-model chunk latency and MoE-layer latency.
- Expert GEMM windows, routing/sorting time, and weight-transfer stalls.
- Provisioned rows and useful token-expert assignments.
- Core placement/utilization, device memory, and weight traffic where measurable.

### Interpretation

| Comparison | What it measures |
| --- | --- |
| A to B | Cost of weight indirection and its compiler lowering |
| B to C | Sorting and permutation overhead, including placement interactions |
| C to D/E | Benefit of reducing capacities after paying for dynamic expert selection |
| D to E | Benefit of the three-tier configuration |

**Primary result: C versus E.** The existing draft's step 5-to-7 comparison remains useful as a system result, but also changes weight representation and tile tuning.

Changing tier count can itself change placement. Use profiler evidence to explain such effects rather than treating this as a perfectly isolated padding-only intervention.

### Deliverable

A latency decomposition and a table showing provisioned rows, weight traffic, and memory footprint for A-E.

## 2. Compare with fixed-size block scheduling

### Purpose

Determine whether RankTier's advantage comes from its work organization or from the current compiler's inability to realize efficient block execution.

Implement a **one-layer** baseline that packs assignments into blocks of B tokens, with one expert per block and runtime expert IDs for block slots. Start with **B = 16, 32, 64, 128**, subject to hardware alignment and feasible compilation.

For a global pool with T tokens, top-k routing, and E experts, a sufficient dropless block budget is:

```text
N = ceil(T * k / B) + E - 1
```

This is the block provisioning rule described in AWS's MoE guide. If blocks are partitioned by SoC, derive a safe per-SoC budget from the possible local assignment count; do not substitute the average local load for a worst-case bound.

### Variants

| Variant | Purpose |
| --- | --- |
| Fixed block budget without skipping | Compare under the same restricted execution model |
| Same-expert weight reuse, if expressible | Test whether repeated weight handling explains the draft's row-chunking regression |
| Trace-specialized active-block schedule | Diagnose the opportunity from skipping unused blocks |

The trace-specialized variant knows routing in advance and is **not deployable**. Its compilation may alter placement or scheduling, so it is not automatically a clean latency lower bound. Label it as an oracle diagnostic and inspect its lowering.

Use the same routing traces, precision, expert placement where feasible, and optimized combine as RankTier. Include packing, metadata construction, and dispatch in the full one-layer measurement; additionally report expert-compute-only timing.

### Measurements

- One-layer latency and component breakdown.
- Computed versus useful rows.
- Weight bytes transferred and dequantization work.
- Core utilization, stage count, and intermediate storage.

### Interpretation

- If RankTier wins despite comparable weight reuse and efficient block mapping, investigate padding, scheduling, and locality as the causes.
- If block execution loses because of repeated weight transfers or poor placement, attribute the result to the measured execution path.
- If skipping unused blocks removes RankTier's advantage, identify efficient dynamic block execution as an applicability boundary.
- Do not infer that repeated weight loading is inherent to block scheduling: AWS explicitly describes reuse between consecutive blocks of the same expert.

Match correctness guarantees. A guaranteed-dropless block baseline and calibrated RankTier have different guarantees; include RankTier's overflow recovery cost in the operational comparison.

### Deliverable

Latency versus block size, accompanied by weight-traffic and computed-row measurements. Include the best feasible block baseline in the main comparison.

## 3. Separate combine design from reduction lowering

### Configurations

Hold capacities at T and keep expert GEMMs unchanged.

| ID | Combine path | Purpose |
| --- | --- | --- |
| R0 | Original dense combine | Production behavior |
| R1 | Repaired dense combine | Remove unnecessary initialization and serialized work |
| R2 | Token-owned combine | Eliminate the dense expert-by-token accumulator |
| R3 | Token-owned combine with distributed final sum | Measure the additional effect of final-sum lowering |

Use identical routing and expert-output tensors in a **combine-only microbenchmark**, then confirm effects within an MoE layer. Keep unrelated routing-prefix-scan rewrites outside the combine-only comparison.

### Measurements

- Combine latency and time after the last expert GEMM.
- Intermediate allocation size and peak scratch usage.
- DRAM and cross-SoC traffic where exposed by profiling.
- Active cores, serialized regions, and collective implementation.

### Interpretation

**R1 versus R2** is the primary algorithmic comparison. A benefit after repairing dense-path serialization supports eliminating the dense representation itself. R0-to-R1 and R2-to-R3 quantify implementation/lowering effects.

Do not describe the gather work as exactly proportional to useful assignments unless the implementation skips masked entries; record the effects of stage count and masking.

### Deliverable

A latency-and-traffic table separating representation changes from compiler lowering changes. Reuse existing ablations where their controls match.

## 4. Validate correctness and capacity generalization

### 4a. Multi-chunk execution and overflow recovery — P0

- Run consecutive prefill chunks with a populated KV cache.
- Check a subsequent decode step.
- Compare against the full-capacity implementation at matched precision. Separately evaluate quality against the chosen model reference.
- Deliberately force overflow with undersized lower tiers and verify detection.
- Implement fallback to a full-capacity path, restoring or safely overwriting all affected KV state and rejecting outputs from the failed attempt.
- Check the recovered result against direct full-capacity execution, including subsequent chunks.
- Measure detection overhead, fallback frequency, replay latency, and executable residency/loading costs.

Overflow can affect downstream routing and cache updates within the attempted chunk. A valid recovery test must cover that state, not just the final logits.

### 4b. Routing generalization — P1

- Split calibration and evaluation **by source prompt/document before extracting chunks**.
- Start with at least **200 held-out chunks per chunk length** across the existing workload categories. Treat this as an initial dataset, not proof of an extremely low failure rate.
- Include leave-one-workload-out calibration.
- Add a second MoE model with a different routing configuration; routing-only analysis is sufficient initially.
- Sweep headroom, for example **1.0, 1.15, 1.28, 1.5**.
- Compare fixed-identity and ranked capacities at matched provisioning budgets or matched observed overflow rates.

### Measurements

- Provisioned rows and useful assignments.
- Sorted-load distributions and capacity utilization by tier.
- Overflowed assignment counts.
- Fraction of chunks with **any overflow in any layer or SoC**.
- Recovery-adjusted latency and model-quality checks.

Sorting guarantees decreasing loads and the bound n_(r) <= N/r for N assignments. It does not guarantee the tighter calibrated capacities used in RankTier. Report empirical coverage separately from exact correctness through fallback.

### Deliverable

A provisioned-work versus overflow-rate curve, with workload/model breakdowns, plus a multi-chunk correctness and forced-recovery report.

## 5. Test portability at the mechanism level

### Preferred experiment

Implement one MoE layer on Trainium before attempting a full-model port.

Compare:

1. Native optimized blockwise MoE.
2. Runtime-ranked full-capacity lanes.
3. Runtime-ranked tiered-capacity lanes.

Match weights, precision, inputs, and routing. Include sorting, dispatch, weight selection, expert computation, and combine in the timed boundary.

### Questions

- Does capacity reduction still save execution time?
- Does dynamic skipping of empty blocks remove the advantage?
- How does weight reuse affect the comparison?
- Does indexing weights cause copies or replication?
- Which tier configuration is preferred under the new core and memory organization?

A negative result is informative if it identifies which capabilities remove the need for ranked provisioning. Running the graph proves implementability; comparison against native execution establishes performance relevance.

### Lower-cost alternative

On AI 100, vary the cores used per SoC, for example **4, 8, and 16**, where supported. Reselect tier widths and retain matched baselines. Report how utilization and the preferred tier configuration change.

This is a resource-sensitivity study, **not cross-architecture validation**. Analytical models may supplement it, but distinguish predicted results from measurements and validate any latency model against hardware data.

## Suggested execution order

1. Add configurations B and C to the existing full-capacity and RankTier implementations.
2. Run the controlled A-E comparison at the three existing chunk lengths.
3. Build the one-layer block baseline and investigate weight reuse and placement.
4. Complete multi-chunk correctness and forced overflow recovery in parallel with the experimental implementation schedule, if personnel permit.
5. Reuse existing combine variants for controlled latency/traffic attribution.
6. Expand held-out routing analysis and add a second model.
7. Decide whether a second-platform kernel study is justified by the resulting evidence.

## How the results support the paper

| Evidence | Claim supported |
| --- | --- |
| C versus E with matched controls | Rank-bound capacities improve execution beyond sorting/indirection overhead |
| RankTier versus a competitive block baseline | The chosen work organization is advantageous under identified conditions |
| Repaired dense versus token-owned combine | Avoiding the dense representation has value beyond serialization repairs |
| Held-out and second-model routing results | Sorted-load provisioning generalizes within the tested scope |
| Multi-chunk and forced-recovery tests | Correct execution extends beyond the first chunk and handles capacity failure |
| Second-platform comparison | Mechanism portability and its performance limits |

Keep the production-baseline speedup as the complete system result. Report the isolated algorithmic gains separately, along with device-memory cost and compiler-dependent behavior.

## References

- Supplied draft: `ranktier.pdf`, especially Sections 3-6 and Tables 3-7.
- [AWS Neuron MoE inference guide, version 2.30.0](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/libraries/nxd-inference/developer_guides/moe-arch-deep-dive.html): fixed-size block provisioning, weight DMA reuse, dynamic block skipping, and core sharding. Consulted September 30, 2026.
