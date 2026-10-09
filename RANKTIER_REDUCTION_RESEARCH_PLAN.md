# RankTier: Establishing the Reduction Contribution

Working plan — October 6, 2026

**Goal:** establish what the reduction design contributes beyond existing sparse MoE combine methods, and demonstrate why it is necessary for efficient ranked-capacity execution on the evaluated AI100 compiler.

**First experiment:** compare a direct sparse gather/reduce graph with the locality-preserving implementation, keeping expert placement, capacities, weight access, and compiler settings fixed.

This document separates observations already reported in the draft from hypotheses and proposed experiments. It does not claim that the proposed novelty has already been established.

## 1. Motivation: why the reductions need their own design

An MoE layer does two kinds of work: the expert GEMMs, which the ranked capacities shrink, and the reductions around them. These are the prefix sums that give each assignment its row in an expert's input, the combine that returns expert rows to their tokens, and the sums over lanes and over SoCs. The exported graph writes each of them densely, sized by lanes × T, and the compiler runs several of them on one core or one SoC. Four observations make the reductions a design problem rather than an implementation detail; together they are the case that draft §3.2 should make.

Evidence without a draft location comes from the experiment log behind the draft, cited by experiment ID or README section (see References). Configurations differ between items (one-layer replays and full models, FP16 and MXFP6, the old and the fixed attention head-group sum), so the caution at the end of Section 2 applies here too. One-layer replays contain no attention, so the head-group sum does not affect them. Full-model numbers for head-parallel programs come from E46–E51, which re-timed every head-parallel program with the elementwise head-group sum (October 7–8; results in Section 11 of `RANKTIER_REDUCTION_FOLLOWUP_PLAN.md`). A number marked "old head-group sum" is kept as history.

### 1.1 The reductions dominate and cancel the gain from longer chunks

- With experts compiled whole on each SoC and the dense path repaired, the combine still takes 51%, 64%, and 68% of the MoE layer at T = 128, 256, and 512 (one-layer MXFP6 replays; draft Table 1, E8). The layer takes 3.2×, 5.6×, and 9.5× as long as streaming its MXFP6 weights (E8).
- Longer prefill chunks should amortize weight traffic over more tokens, and the dense combine prevents it. In one-layer replays at oracle capacities, the layer costs 20.4, 20.3, and 19.4 µs per token at T = 128, 256, and 512 with the dense combine, and 17.2, 12.5, and 11.2 µs with the token-owned combine (E15, same session). On the full model, naive T/T (the export's combine and capacity T, compiled expert-parallel) costs about 2.0 ms per token at every chunk length: 249.4, 500.8, and 1028.8 ms per chunk. The reduction steps alone bring it to 0.88, 0.72, and 0.68 ms (E46).
- Other levers gain little or nothing while the dense combine remains. On the full model at T = 128, oracle capacities recovered 0.8% and regrouping experts into hot and cold stages cost 3.6% (E10: prefill-only programs from before head-parallel attention, so without any head-group sum). Even under the repaired dense combine, RankTier's tiers give only 0.73×, 0.96×, and 1.04× at T = 128, 256, and 512 (E50).
- In the ladder re-timed with the head-group sum fixed, the reduction steps give 2.21×, 2.72×, and 2.94× (E46, steps 2→5), the largest factor in RankTier's 2.37–4.56× over naive T/T. The routing scans give 1.17–1.18× of this, and MoE output aggregation 1.88×, 2.33×, and 2.51× (E47).

**Do not motivate the reductions with the production comparison.** The draft's introduction, §3.2, and §6.4 attribute the 11–12.5% slowdown of expert-parallel compilation at T ≥ 256 to the dense combine. E46 traces it instead to attention's sum over its four KV-head groups: writing that sum elementwise saves 17, 67, and 130 ms at T = 128, 256, and 512, more than the slowdown. With that sum fixed, expert-parallel compilation is 1.98× faster than production at T = 128 and 1–2% faster at 256 and 512 (production timed in E36, a different session). The slowdown did come from a reduction, but not from the combine. Comparisons among the expert-parallel programs, as above, are the sound basis.

### 1.2 The dense representation costs time even after its lowering is repaired

The exported combine scatters each stage's weighted rows into a zero-filled `[64, T, d]` accumulator, updates it in the next stage, sums it over each SoC's 16 lanes, and then sums the four SoC partials. It is the natural static-shape formulation, like GShard's one-hot einsums: it needs no per-token index, only shapes known at compile time. Every pass scales with lanes × T whatever the routing; at T = 512 one stage's accumulator holds 128 MiB in FP16 (E42). Repairing the lowering and replacing the representation are separable steps, and both pay (full model, E47):

| Ladder step | T = 128 | T = 256 | T = 512 |
|---|---|---|---|
| 2→3a: tree prefix scans (routing metadata) | 1.18× | 1.17× | 1.17× |
| 3a→3b: repair the dense combine (drop the read of the zeroed accumulator, tile the lane sum) | 1.50× | 1.49× | 1.54× |
| 3b→4: replace it with the token-owned combine | 1.23× | 1.53× | 1.63× |

The second gain grows with T, as the removed accumulator does. At T = 512 it shortens the time after the last expert GEMM from 4.7–5.5 to 1.2–1.4 ms per layer and cuts the MoE's cross-SoC traffic from 48 to 14 MiB (draft §6.4, E42; MoE-internal figures, which the head-group sum does not enter).

Fewer rows per pass explain only part of this. Per SoC and stage, the token-owned gather produces all k slots of every token, masked slots included (`[T·k, d]`, 16 MiB at T = 512), while each SoC's slice of the dense accumulator holds 16 lane rows per token: 2× fewer rows per pass, against a tail about 4× shorter. The token-owned combine also drops whole passes (zero fill, scatter, read-modify-write, cross-core lane sum) and the serialized steps that ran them (Section 1.3). R1 should measure how the gain divides between these.

### 1.3 The obvious sparse formulations fail under graph compilation

The arithmetic of a sparse combine is established (Section 3). What is not established is how to make a graph compiler execute it locally and in parallel. A CUDA kernel fixes its indexed loads, which thread owns each token, and where partial sums live. A graph fixes only the arithmetic; the compiler chooses the partition, the placement, and the lowering of every gather and reduction. The formulations below compute the same sums, and the compiler executed them very differently:

| Formulation, in the order tried | What the compiler did | Effect | Source |
|---|---|---|---|
| Export's graph, default partitioning | Split every down projection across SoCs, with an all-gather before it and a re-shard after it | 121.6 MiB cross-SoC per layer | E1; draft App. B.1 |
| Export's dense combine, expert-parallel | Zero-filled the accumulator on one core, ran the lane sum through DRAM on one core, and summed the SoCs on four cores of SoC 0, spilling to DRAM from T = 256 | Lane sum alone: 1.72 ms of a 6.7 ms layer (FP16 replay, T = 128) | README 3.1–3.2, E6; draft App. B.1 |
| Routing prefix sums as `CumSum` | Ran each as a serial scan on one core | 0.40 ms each, two per layer (same replay) | README 3.2 |
| Sparse gather written as explicit per-SoC lane slices | Spread the tiles over SoCs and shipped expert outputs between them | Cross-SoC traffic 3 → 78 MiB; device time 4.3 → 8.6 ms (one layer). Full model 4.3–8.2× slower than token-owned, 577 MiB cross-SoC per MoE layer | E11; E48 |
| Sparse gather of all 32 lanes of each token, then a sum over lanes (token-centric) | Kept locality, but each tile's lane sum spanned all 16 cores with multicasts, so the 16 tiles ran one after another | Tail of about 5 µs per token at every T, at about 23 GB/s per card. Full model 1.11–1.15× slower than token-owned | E13; E48 |
| Each token's k rows gathered from all SoCs' expert outputs, then a sum over k (C-topk, the direct analog of a GPU combine) | Shipped expert outputs between SoCs | 1538 MiB cross-SoC per MoE layer against 14 MiB for token-owned; full model 4.05–4.92× slower | E48 |
| SoC sum as reshape–transpose–einsum, meant as a reduce-scatter | Shipped all partials to SoC 0, materialized the transpose in DRAM, and summed on one core | +34% and +43% at T = 256 and 512 | E16 |
| Attention head-group sum as `ReduceSum` over the SoC axis (outside the MoE) | Ran batched reduce-adds on four cores, then a DRAM-backed reduce on one core, on each of three SoCs | 2.39 ms, 31% of each RankTier layer at T = 512 | E44 |

The formulations that worked make the parallel structure of each reduction explicit in the graph, and the best ones leave every sum to a core that already holds its operands. Gathers kept locality only when they indexed along the axis that the compiler already splits across SoCs. Each reduction step of the ladder moves in this direction:

- **Repairs (step 3):** tree scans replaced the serial `CumSum`, and 16 token tiles replaced the one-core lane sum (1.72 → 0.41 ms in the FP16 replay). The tiles still sum across all 16 cores, one tile after another (README 3.1, E10).
- **Token-owned combine (step 4):** each core gathers at most k rows for each of its own tokens and sums them locally. At T = 256 the per-SoC combine after the last stage fell from 1.2 to under 0.1 ms, and the layer ran 8%, 14%, and 16% faster than with the token-centric combine at T = 128, 256, and 512 (E15).
- **Elementwise final sum (step 5):** written as `(P0+P1)+(P2+P3)` per token tile, the cross-SoC sum spreads over all cores as a reduce-scatter. With the head-group sum fixed it gives only 0.5–2.7% on the full model (E47), and the same rewrite took the head-group sum from 2.4 to 0.05 ms per layer (E45).

**Hypothesis for R1 and the scope sweeps:** on a graph-compiled target, a sparse combine pays only when the graph turns cross-core reductions into per-core gathers followed by local sums, keeps its gathers on the partitioned axis, and writes cross-device sums elementwise. If the failures trace to a single lowering defect of this compiler, narrow the claim to implementation evidence (Section 9).

### 1.4 The reductions decide whether ranked capacities pay

Ranked tiers save expert rows but add stages of different widths. Under a lane-dense combine each width needs its own `[lanes, T, d]` accumulator, and the accumulators absorb the saving:

- With the repaired dense combine (one accumulator per tier width, no read of the zeros, tiled lane sums), RankTier's tiers give 0.73×, 0.96×, and 1.04× over the same program at capacity T. With the token-owned combine they give 1.10×, 1.15×, and 1.37× (E50; all four cells rank, gather weights and use the same final sum). Under the export's unrepaired combine, tiers made the program 0.69–0.85× of naive T/T (E37, old head-group sum).
- At T = 512 the ranked capacities save 2.6 ms per layer: 35% of a 7.3 ms layer after the reductions, but only 12% of the 21.4 ms naive layer (E46: step 5 → RankTier, against naive T/T).
- Under the token-owned combine a tier adds one `[T·k, d]` gather per SoC, whatever its width and capacity, and the tiers pay. With ranking and weight gathers matched, the capacities give 1.10×, 1.15×, and 1.37× (E46; 1.13×, 1.13×, and 1.27× in draft Table 3 with the old head-group sum).

Sections 1.2–1.4 supply the evidence for the four properties in Section 4: sparse connectivity (1.2), device locality and distributed reduction work (1.3), and compatibility with tiers (1.4).

### 1.5 What remains open

- **The combine is not finished.** In RankTier at T = 512 the token-owned combine is the largest category of busy core time (20%; E44, before the head-group sum fix), though mostly hidden under the tiers: its exposed tail is 0.40 ms of a 7.73 ms layer. Each tier's gather reads 16 MiB per SoC, all k slots included, so three tiers read 48 MiB where one gather over all tier outputs would read 16 MiB. That variant is untested; it risks a `Concat` copy and loses the overlap of early gathers.
- **A custom kernel works, within the token-owned layout.** An HVX TokenCombine op that replaces D's per-SoC gather and sum compiles inside the four-SoC partition, stays local (14.1 MiB cross-SoC per MoE layer, as D) and is about 7% faster on the full model at T = 512 (326 vs 351 ms, one valid pass). Its full-model compile takes 11 hours (E49, E51).
- **R1 is measured.** All variants exist on the full model with the head-group sum fixed: A, B and D are ladder steps 2–4, and C was built as C-topk, C-lanes and C-slices (E47, E48; Section 10).
- **One compiler, one model.** The pattern in Section 1.3 rests on variants built for other purposes, on one compiler and one model. It needs R1's controlled comparison before it supports a general claim.

## 2. What we can claim today

The reduction optimizations are a major part of the complete system. The 1.06–1.17× improvement over the tuned static capacity-T program measures the additional benefit of tiering after that baseline has already incorporated the reduction work. It is not the speedup of the complete method.

Existing evidence in the draft includes:

| Observation | Location | What it supports |
|---|---|---|
| After expert-parallel compilation and dense-path repairs, the combine accounts for 51%, 64%, and 68% of the instrumented MoE-layer time at T = 128, 256, and 512. | Table 1, §3.2 | The remaining combine is a substantial bottleneck. |
| Replacing the repaired dense combine with the token-owned formulation reduces full-model latency by 16.9%, 27.3%, and 31.6% in the evaluation ladder. | Table 2, §6.4 | Changing the representation adds value beyond the dense-path repairs. |
| Expressing the combine through explicit per-SoC slices causes expert-output traffic to rise from 3 to 78 MiB and doubles layer device time. | Appendix B.3 | A mathematically sparse formulation can still lose device locality under compilation. |
| The ranked tiers are slower under the exported combine, reaching only 0.69–0.85× the speed of naive T/T. | §6.3 | The compute and combine layouts interact; padding optimization alone is insufficient in the tested implementation. |

These measurements come from different experimental configurations. Do not combine their traffic counts or speedups as if they were measurements of one identical setup. Recover the exact settings when presenting them together. All full-model numbers in the draft, these included, use the old head-group sum. E46–E51 re-timed every head-parallel program with the elementwise sum; Section 1 uses those numbers.

## 3. Correct the GPU comparison

The dense-combine problem has appeared on GPUs. Tutel (MLSys 2023), especially Appendix B and Figures 20–21, replaces unnecessarily dense MoE dispatch/combine operations with sparse GPU kernels and assigns token work to thread arrays or warps. Its fast decode is directly relevant prior work, beyond its adaptive parallelism contribution. [1]

Therefore, avoid claiming novelty solely for:

- Recognizing that dense MoE combine performs unnecessary work.
- Gathering only the selected expert contributions for a token.
- Assigning output tokens to independent workers.
- Performing local accumulation before a cross-device reduction.

The question to answer is: **what prevents an established sparse combine from executing efficiently through the available graph compiler, and what additional method resolves that obstacle?**

CUDA kernels can specify indexed loads, work assignment, accumulation, and stores directly. For example, vLLM's MoE sum kernel reduces a `[tokens, topk, hidden]` input into `[tokens, hidden]`. [2]

Static shapes do not inherently prevent efficient sparse execution. NVIDIA's documented vLLM-style MoE implementation uses fixed-size buffers and on-device indirection in a CUDA-graph-compatible path, and its combine accumulates local expert contributions. [3]

| Concern | CUDA/Triton kernel approach | Evaluated AI100 graph approach |
|---|---|---|
| Selecting contributing rows | Indexed loads inside the kernel | Gather/index graph operations whose lowering matters |
| Assigning output work | Programmer specifies thread/block work | Compiler partitions operations over cores |
| Keeping expert outputs local | Local buffers and communication are explicitly managed | Graph representation must preserve the intended partition |
| Accumulating contributions | Can remain inside one kernel | Separate graph operations may materialize buffers or serialize |
| Runtime routing | Device-side metadata and indexing | Runtime indices within statically shaped operations |

This comparison concerns the programming interface and compilation path. It does not establish a fundamental hardware inability of NPUs to gather or reduce. CUDA implementations can also suffer from poor layouts, communication overhead, and inefficient combine kernels.

## 4. Candidate contribution to investigate

**Hypothesis:** preserving sparse arithmetic is insufficient when graph compilation changes the placement of expert outputs or the distribution of reduction work.

The candidate contribution is a graph transformation that preserves device locality under dynamic routing and static partitioning, while distributing token reduction across cores and accommodating expert tiers with different widths and capacities.

The intended computation is:

$$
p_c[t,:] = \sum_{j:\,e_{t,j}\in E_c} w_{t,j}\,O_c[\operatorname{slot}(t,j),:],
\qquad y[t,:] = \sum_c p_c[t,:].
$$

Here, `E_c` is the set of experts owned by device c; the expert-output address includes its tier, lane, and row where necessary. Each device forms its token-indexed partial result using its own expert outputs.

The equation itself is not the novelty. The question is how to realize it through the graph interface without introducing unwanted redistribution, dense intermediates, or serialized execution.

Investigate and explain four properties of the transformation:

1. **Sparse connectivity:** routing indices identify the relevant expert rows.
2. **Device locality:** expert outputs are consumed on the device that computed them.
3. **Distributed reduction work:** token tiles are processed across cores rather than reduced through a small subset of cores.
4. **Compatibility with tiers:** changing expert lane widths and capacities does not require a new dense token accumulator for each tier shape.

Do not claim that token ownership eliminates all cross-core data movement. It can eliminate a cross-core reduction dependency while still requiring expert-output reads or writes through shared device memory.

## 5. P0 — Isolate the additional reduction contribution

### Experiment R1: representation and locality

Use uniform capacity T and static expert weights first. This isolates the reduction from rank sorting, weight indirection, calibrated capacity, and overflow recovery.

| Variant | Implementation | Question answered |
|---|---|---|
| A | Exported dense combine under expert parallelism | How costly is the original combine under a common expert placement? |
| B | Dense combine with redundant-work and serialization repairs | How much comes from fixing the dense implementation? |
| C | Direct token-indexed gather/reduce graph | Is the established sparse-combine idea sufficient? |
| D | Sparse combine preserving device locality and distributing token reduction | What does the additional graph transformation contribute? |

Use the explicit per-SoC-slice implementation from Appendix B.3 as one candidate for C. Explain why it is a natural formulation, document its operations, and give it reasonable tuning. A deliberately unfavorable graph would not establish a strong comparison.

Controls:

- [ ] Same expert placement, precision, capacity T, and weight-access mechanism.
- [ ] Same attention implementation, KV-cache residency, and context length.
- [ ] Same input routing traces and measurement protocol.
- [ ] Same tile settings for the primary attribution comparison.
- [ ] Separately report each variant's best reasonable tuning if preferred settings differ.
- [ ] Keep routing prefix-scan construction identical when attributing output-combine gains; report its optimization separately.
- [ ] Record SDK version, graph version, flags, and profiling configuration.

Measure:

| Metric | Why it matters |
|---|---|
| Absolute combine latency | Shows whether the combine itself improves, independent of its percentage of layer time |
| Full-layer and full-model latency | Establishes practical impact |
| Intermediate-buffer sizes and live ranges | Identifies dense expansion and materialization |
| DRAM reads/writes | Measures the physical cost of intermediates and gathers |
| Cross-SoC traffic | Reveals loss of device locality |
| Work distribution across cores and serialized intervals | Explains execution imbalance and synchronization |

**Primary comparison: C versus D.**

- If C performs similarly to D, describe the reduction as an effective realization of an established sparse-combine method. Investigate whether the interaction with tiering provides the stronger contribution.
- If C remains costly and D consistently eliminates identifiable redistribution or serialization, use this as evidence for the additional graph-compilation challenge.

Deliverable: one latency/traffic comparison and a small graph or execution-trace example showing exactly where C loses locality and how D preserves it.

## 6. P0 — Establish why reduction enables tiering

### Experiment R2: controlled 2×2 interaction

| Expert capacities | Repaired dense combine | Proposed combine |
|---|---|---|
| Uniform capacities | Reference | Reduction benefit at fixed capacities |
| Ranked tiers | Does combine overhead absorb the saved expert work? | Combined benefit |

For this experiment, use the same gathered-weight mechanism in all four variants if needed to avoid confounding tiering with weight access. Keep sorting present in both capacity configurations where feasible, and report any unavoidable difference. Use identical inputs and matched tile settings for attribution.

- [ ] Measure expert GEMM time separately from combine and routing time.
- [ ] Record intermediate sizes and traffic as tier widths change.
- [ ] Verify that the proposed combine supports every tested tier configuration correctly.
- [ ] Check whether the existing 0.69–0.85× result persists after these controls and dense-path repairs.

The hypothesis is that heterogeneous capacities save expert work, but a combine coupled to lane layout can introduce enough intermediate storage or communication to erase the saving. The proposed combine should decouple token aggregation from expert capacity allocation.

If this interaction disappears after reasonable baseline repairs, narrow the claim accordingly. Do not treat it as an inherent property of every dense combine or every AOT target.

Deliverable: a four-configuration stacked latency plot, with expert computation, routing metadata, and combine shown separately.

## 7. P1 — Establish a predictable cause and scope

After R1 and R2, run focused sweeps that test the mechanism rather than expanding the benchmark suite indiscriminately.

| Sweep | Hold fixed where possible | Hypothesis to test |
|---|---|---|
| Local expert/lane count | T, k, hidden width, routing-assignment count | Dense intermediate cost follows provisioned lanes; sparse useful work follows selected assignments |
| Chunk length T | Expert geometry and placement | Larger intermediates cause a measurable capacity or spill threshold |
| Tier widths/count | Routing and expert assignments | Proposed combine avoids shape-specific dense accumulators; quantify remaining per-tier overhead |
| Core/device count | Comparable local workload | Work remains distributed and expert-output locality is preserved |

Prioritize lane count and tier structure. Add device/core scaling only when placement and workload controls make the result interpretable.

For the cost analysis, distinguish:

- **Dense representation size:** a stage accumulator of shape `[L, T, D]` contains LTD elements.
- **Useful expert contributions:** across devices, top-k routing produces kT expert-output rows.
- **Partial results:** local aggregation may still produce a `[T, D]` partial per device.
- **Physical execution:** q separate tier gathers may process more than kT entries, depending on masking and lowering.

Measure actual traffic before claiming that the implementation reads only the mathematically useful rows. Report how the cost depends on tier count q.

Deliverable: one or two mechanism-focused plots and a simple explanation of the observed scaling and spill thresholds.

## 8. Optional GPU control

A small GPU microbenchmark can help explain the architectural comparison without requiring a complete second-platform port.

Compare a dense combine representation with an established sparse combine using the same dimensions and routing. If dense execution is expensive on the GPU as well, that supports the interpretation that the underlying representation problem is shared. Then explain how custom GPU kernels address it and which additional compiler constraints the AI100 implementation must handle.

This experiment is explanatory. A GPU speedup over a dense baseline would not establish novelty for RankTier, and GPU/NPU absolute latency is not the primary comparison.

## 9. Paper revision plan

- [ ] Keep reduction as a first-class contribution; distinguish its gain from the incremental tiering gain.
- [ ] Rebuild the reduction motivation in §3.2 on Section 1: the reductions dominate and cancel the gain from longer chunks, the representation costs time beyond the repairs, natural sparse formulations fail under compilation, and the reductions decide whether the capacities pay.
- [ ] Remove the claim that expert-parallel compilation is slower than production at T ≥ 256 because of the dense combine (introduction, §3.2, §6.4); E46 traces that slowdown to the head-group sum.
- [x] Put every head-parallel program on the elementwise head-group sum (E46–E51: ladder, attribution, combine variants and the 2×2). Still on the old sum: the block programs (Table 9). Production uses its own attention and stays as measured.
- [ ] Expand the Tutel discussion to cover fast encode/decode, not only adaptive parallelism.
- [ ] State the relevant execution restrictions precisely. Avoid equating all AOT compilation with an inability to execute data-dependent work.
- [ ] Move the failed explicit-SoC sparse formulation into the main text if R1 confirms its significance.
- [ ] Explain graph representation, resulting placement, and physical traffic together.
- [ ] Separate dense-path implementation repairs, sparse combine, and placement/distribution improvements in the attribution.
- [ ] Present the controlled interaction between reduction and tiering.
- [ ] Delimit portability by required compiler capabilities and failure modes.

Provisional framing, subject to the experiments: **RankTier makes sparse MoE aggregation compatible with static partitioning and heterogeneous expert capacities by preserving device locality and distributing token reduction through the graph representation.**

A strong result would establish a reusable transformation or placement principle. If the benefit is confined to one lowering quirk, present it as implementation evidence within the complete system rather than a broad architectural claim.

## 10. Results worksheet

| Experiment | Status | Key result | Implication for the claim |
|---|---|---|---|
| R1: A/B/C/D at uniform capacity | Done (E47, E48), full model, T = 128/256/512 | Against D (token-owned): repaired dense 1.25–1.64×, C-lanes 1.11–1.15×, C-slices 4.3–8.2×, C-topk 4.05–4.92× slower. Cross-SoC bytes per MoE layer at T = 512: D 14 MiB, C-topk 1538, C-slices 577. A custom op inside D's layout: 0.93× of D at T = 512 (E51) | The direct sparse formulation loses locality under compilation; D's layout keeps it. Supports a distinct graph-compilation problem |
| R2: combine × capacity interaction | Done (E50), full model, repaired dense baseline | Tiers: 0.73/0.96/1.04× under the repaired dense combine, 1.10/1.15/1.37× under token-owned at T = 128/256/512 | The reduction design is what lets ranked capacities pay; the interaction survives the dense-path repairs |
| Lane-count sweep | Pending | TBD | TBD |
| Tier-structure sweep | Pending | TBD | TBD |
| GPU explanatory control | Optional | TBD | TBD |

Record each result with the input/routing trace, model/layer, shape, SDK/flags, graph version, timing protocol, and profiler evidence. Link the exact graph transformation or code change responsible for each difference.

## References

Source manuscript: *RankTier: Efficient Mixture-of-Experts Prefill on Ahead-of-Time Compiled Inference Accelerators*, uploaded as `moe.pdf`; especially §3.2, §4.5, §6.3–6.4, Tables 1–3, and Appendix B.3. Results above are reported by the draft and have not been independently reproduced here.

Experiment log behind the draft, cited in Section 1: `9.17.2026/real_model/oracle_padding/mdts_flag/README.md` ("README" with its section numbers), whose sections 3.1–3.39 document E1–E43. E44–E46 are not yet written up; their outputs are `mdts_flag/full_model/profile_e44_T512/e44_layers.txt` (E44), `mdts_flag/full_model/e45/e45_run.txt` and `e45_T512_profile.txt` (E45), and `mdts_flag/full_model/e46/e46_summary.txt` (E46). E47–E51 (October 7–8) are fitted together in `mdts_flag/full_model/e51/e46_e51_summary.txt` (`scripts/e47_summary.py`); their drivers are `scripts/run_e47.sh`, `run_e48.sh`, `run_e50.sh` and `run_e51.sh`, and the custom op is in `scripts/customop/tokencombine/`.

1. Hwang et al., **Tutel: Adaptive Mixture-of-Experts at Scale**, MLSys 2023. See Appendix B and Figures 20–21 for sparse encode/decode and token-oriented GPU execution. [Paper PDF](https://yzygitzh.github.io/assets/papers/hwang_mlsys_2023.pdf) · [Proceedings](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)
2. vLLM, **MoE alignment and sum CUDA kernels**. Relevant source: `moe_sum_vec_kernel`, which reduces the top-k dimension. [Source](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/moe/moe_align_sum_kernels.cu)
3. NVIDIA Megatron Core, **vLLM-style fused MoE implementation**. Documents fixed-size buffers, device-side indirection, CUDA-graph compatibility, and local-expert top-k reduction. [Documentation](https://docs.nvidia.com/megatron-core/developer-guide/0.19.2/apidocs/core/core.inference.moe.vllm_fused_moe.html)

Public sources were inspected during the October 6, 2026 discussion. Pin a source commit/version for any implementation comparison included in the paper.
