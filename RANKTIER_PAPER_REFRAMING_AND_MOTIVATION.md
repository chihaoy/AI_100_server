# RankTier: Paper Reframing and Motivation

October 8, 2026

This writing plan is based on the reported E47–E54 results in `RANKTIER_E47_E51_ASSESSMENT_AND_NEXT_STEPS(1).md` and the preceding discussion. The underlying experiment logs and compiler traces have not been independently inspected.

## 1. Central framing

Frame the paper around **making runtime-varying MoE workloads efficient within a statically partitioned execution graph**. Rank-based capacities and ownership-preserving aggregation become two parts of the same solution.

The central motivation is:

> MoE execution must adapt to changing expert loads while preserving an efficient distributed execution structure. Reducing padded expert computation helps only if the resulting outputs can still be aggregated efficiently.

This is more precise than a broad claim that NPUs have inefficient padding and reductions. The organizing question is:

> How should dynamic expert workloads be represented and executed in a static distributed graph?

## 2. Start with the workload–execution mismatch

Explain the workload first: routing changes how many tokens each expert receives, across inputs and layers. Efficient execution should therefore allocate different amounts of work to different experts.

Then introduce the constraint of the target execution system: expert computations execute through compiled shapes and partitions. A uniform capacity accommodates variable routing, but may execute substantial padded work.

Distinguish two quantities:

- **Logical workload:** the token assignments produced by routing.
- **Executed workload:** the rows actually processed after tiling and compiler lowering.

Smaller allocated tensors or fewer nominal FLOPs do not automatically establish saved execution. Use the tile-size experiments to demonstrate when padded rows actually stop consuming tensor-unit work.

The first motivating question becomes:

> How can a compiled execution structure accommodate unequal, changing expert loads without paying uniform worst-case execution costs?

Rank-based capacities are the proposed answer: stable compiled capacity classes receive experts according to runtime demand. Explain precisely which parts remain static and which assignments change at runtime.

### Evidence to show

- Per-expert token counts across several representative inputs and layers.
- The relationship between uniform capacity, ranked capacities, and actual expert loads.
- Allocated rows versus rows actually executed under the tested tile settings.
- Measured expert-stage time, identifying when saved padded work translates into lower latency and when another resource becomes limiting.

One skewed routing example illustrates the opportunity; a distribution establishes its prevalence.

## 3. Show why capacity adaptation alone is insufficient

This should be the turning point in the motivation.

After expert computation, outputs must return to token order and combine across selected experts. Changing expert capacities introduces differently shaped outputs and potentially additional stages. The cost of handling those outputs can cancel the expert-compute savings.

Use E50 here. At chunk length T = 512:

| Execution | Uniform capacity | Ranked capacities | Latency reduction from tiering |
|---|---:|---:|---:|
| Repaired dense combine | 411.7 ms | 396.8 ms | 3.6% |
| Token-owned combine | 309.8 ms | 226.5 ms | 26.9% |

These are full-model latencies from the controlled experiment. Percent reductions use `(uniform latency − ranked latency) / uniform latency`; they are not speedup percentages. The experiment controls runtime ranking, weight-bank access, final summation, and tile settings across the four configurations.

At smaller chunks, the repaired dense path can become slower after tiering. The complete interaction is:

| Chunk length | Tiering with repaired dense combine | Tiering with token-owned combine |
|---|---:|---:|
| 128 | 36.9% latency increase | 8.9% latency reduction |
| 256 | 3.9% latency increase | 13.3% latency reduction |
| 512 | 3.6% latency reduction | 26.9% latency reduction |

The conclusion is:

> Capacity selection must be designed together with aggregation; minimizing padded expert work is an incomplete optimization objective.

This makes reduction central without requiring the mathematical sparse-reduction algorithm itself to be new. Avoid claiming that the proposed reduction is necessary for any capacity benefit: the repaired dense implementation already improves at T = 512.

## 4. Explain aggregation as an ownership-preservation problem

E52 provides a precise explanation.

The selected expert rows are logically local to each SoC, but equivalent graph expressions do not preserve that locality equally well. Global indices describe which rows to read; they do not necessarily preserve which device owns those rows through compilation. Even explicit slices fail in the measured configuration.

The proposed transformation makes ownership structural:

$$
[64,C,d] \longrightarrow [4,16C,d].
$$

Here C is the stage capacity and d is the hidden width. The reshape preserves the existing lane partition. The leading axis exposes the SoC partition to a batched gather with per-SoC row indices.

The E52 alternatives hold arithmetic, index values, summation order, and final summation constant:

| Ownership representation | Reported full-model outcome relative to D |
|---|---|
| Global row indices into unsplit outputs | 4.37–5.30× slower |
| Explicit per-SoC slices followed by local gathers | 3.44–4.14× slower |
| Layout-preserving SoC batch axis | D; approximately 110× less cross-SoC traffic in the T = 512 profile |

The alternatives produce bit-identical logits. This is a controlled demonstration that graph expression affects execution even when the arithmetic is unchanged.

Token-owned reduction addresses the next level of the hierarchy: communication between cores within each SoC. The motivation should distinguish two requirements:

1. **Preserve device ownership** when selecting expert outputs.
2. **Assign token reductions locally** to avoid pulling reduction operands from many cores.

E52 supports the first requirement; the C-lanes comparison supports the second. Together they explain why expressing a top-k gather alone is an incomplete graph implementation.

### Causal wording

Use:

> With arithmetic and numerical outputs held constant, the alternative graph expressions cause substantially more redistribution and much higher latency.

Avoid saying that the slowdown is entirely data movement: unchanged arithmetic does not exclude differences in scheduling, buffering, tiling, or generated instructions.

Likewise, describe the C-lanes profile as an exposed tail or post-GEMM completion window. The available phase windows are not an exact dependency-chain critical-path decomposition.

## 5. Organize motivation around three figures or panels

| Panel | Question | Evidence |
|---|---|---|
| Expert-load imbalance and executed padding | Why adapt capacities? | Per-expert token counts, allocated capacity, and measured executed work |
| Capacity × combine interaction | Why co-design aggregation? | E50's four controlled configurations |
| Equivalent arithmetic, different placement | Why is graph structure essential? | E52's ownership representations and cross-SoC traffic |

### Panel A: Expert-load imbalance and executed padding

Use routing distributions from several inputs or layers. Distinguish allocation from execution, and connect the tile configuration to the measured padded work. Avoid relying only on a single favorable routing example.

### Panel B: Capacity × combine interaction

Use one panel per chunk length, each with four bars for the two combine implementations under uniform and ranked capacities. Show absolute full-model latency and consistent colors for the combine implementations. The interaction should be visible without reconstructing it from an optimization ladder.

### Panel C: Equivalent arithmetic, different placement

Show the three ownership representations alongside traffic measurements. State that logits are bit-identical. Label traffic as coming from the instrumented two-layer profile, rather than presenting it as a full-model traffic measurement.

The large redistribution failures explain the ownership problem. Retain the locality-preserving C-lanes comparison in the mechanism evaluation so that the claim does not rest exclusively on those large failures.

## 6. Draft introduction passage

Mixture-of-experts models distribute tokens unevenly across experts, and this distribution changes across inputs and layers. On accelerators that execute statically partitioned graphs, this dynamic workload must be accommodated through compiled execution shapes. Uniform expert capacities simplify execution but can cause substantial padded computation, motivating capacity allocations that better match runtime expert demand.

However, reducing expert computation does not guarantee lower inference latency. Expert outputs must subsequently be mapped back to tokens and combined across selected experts. Heterogeneous capacities change the shapes and staging of those outputs, and aggregation overhead can absorb the saved computation. In our controlled experiment at a chunk length of 512, ranked capacities reduce full-model latency by only 3.6% with a repaired dense combine, compared with 26.9% using token-owned aggregation.

Efficient aggregation requires the graph to preserve both device ownership and local reduction structure. We find that mathematically equivalent local sparse reductions can compile into substantially different distributed executions. Representing ownership through global indices or explicit slices triggers extensive redistribution, whereas exposing the existing device partition as a batch axis preserves locality. Within each device, assigning each token's reduction to a local execution resource further avoids costly cross-core aggregation.

RankTier combines runtime assignment to compiled capacity tiers with ownership-preserving token aggregation. This design allows expert execution to adapt to unequal loads while retaining an efficient distributed aggregation structure.

## 7. Revised contributions

1. **Characterization:** demonstrate how executed padding and aggregation jointly limit MoE prefill under static partitioning.
2. **Design:** combine rank-based capacity assignment with graph transformations that preserve device ownership and local token reduction.
3. **Evaluation:** establish the capacity–aggregation interaction through controlled ablations, equivalent-arithmetic placement experiments, and end-to-end measurements.

If the tier selector becomes a validated method, include it within the design contribution. Until then, avoid presenting optimal or systematic tier selection as completed work.

## 8. Scope and claim boundaries

- Ground the ownership-propagation problem in the tested compiler and partitioned execution stack. Do not present it as a universal limitation of NPUs or all AOT compilers.
- State that local top-k aggregation is mathematically equivalent to D. The contribution concerns its graph representation, execution organization, and interaction with heterogeneous capacities.
- Treat the custom kernel as supporting implementation evidence. The repeated alternating session supports approximately 5.5% lower latency at T = 512; it is separate from the central motivation.
- Keep custom-kernel numerical differences explicit. Different aggregation orders yield deterministic output differences; matching a next token on one input is not general accuracy equivalence.
- Keep attention-reduction repairs separately attributable in the evaluation. Do not restore the earlier explanation that attributed the attention confound to MoE combine.
- Establish portability through additional evidence. A transferable principle can be articulated now; generality across models, compilers, and architectures remains an evaluation question.

## 9. Writing and experiment priorities

- [ ] Rewrite the introduction around adapting expert work while preserving distributed execution structure.
- [ ] Build the three motivation panels from matched experiments and representative routing data.
- [ ] Describe the exact static/runtime boundary in rank-based capacity assignment.
- [ ] Specify the ownership-preserving graph transformation with shapes and index semantics.
- [ ] Present E52 and the C-lanes analysis as complementary mechanism evidence.
- [ ] Validate tier selection against measured candidate structures on held-out routing workloads, including memory and overflow/recovery cost.
- [ ] Add a second model to test whether the combined benefit survives a different workload structure.

The organizing message is that reduction makes adaptive expert execution effective: capacity savings become useful only when the aggregation graph preserves locality and efficient execution.
