# RankTier: Assessment of E47–E51 and Next Steps

October 7, 2026

Source: Section 11 of `RANKTIER_REDUCTION_FOLLOWUP_PLAN(1).md`, reporting E47–E51. Measurements below are taken from that report; the underlying logs, compiler output, and traces have not been independently inspected.

## 1. Main assessment

The new results substantially strengthen the paper. The strongest contribution is the interaction between reduction layout and heterogeneous capacities. The evidence now goes beyond showing that reductions are expensive: it shows that the combine design determines whether capacity reduction produces an end-to-end benefit.

The central research question should be:

> How can heterogeneous expert capacities reduce executed work without introducing aggregation overhead that cancels those savings?

E50 should anchor the motivation and evaluation. E48 should explain the mechanism. E49/E51 should establish that the layout also supports custom-kernel integration, while keeping the preliminary kernel speedup separate from the main claim.

## 2. Make E50 the central motivation experiment

E50 controls runtime ranking, weight-bank access, final summation, and tile settings while varying capacity configuration and combine implementation.

| Chunk length | Repaired dense: uniform → tiers | Latency change | Token-owned: uniform → tiers | Latency change |
|---|---|---:|---|---:|
| 128 | 146.8 → 200.9 ms | 36.9% increase | 115.9 → 105.6 ms | 8.9% reduction |
| 256 | 223.5 → 232.2 ms | 3.9% increase | 164.1 → 142.3 ms | 13.3% reduction |
| 512 | 411.7 → 396.8 ms | 3.6% reduction | 309.8 → 226.5 ms | 26.9% reduction |

Percent changes use `(tiered latency − uniform latency) / uniform latency`. These are latency changes, not speedup percentages.

The result supports the following explanation:

> Saving expert computation does not automatically save execution time. Aggregation overhead can absorb or reverse the savings, even after repairing the dense baseline. Token-owned aggregation enables substantial and consistent capacity benefits across the tested chunk lengths.

Do not claim that the proposed reduction is necessary for any capacity benefit: the repaired dense implementation already improves at T = 512. The supported claim concerns the magnitude and consistency of the benefits.

### Suggested figure

Use one panel per chunk length, with four bars showing the two combine implementations under uniform and tiered capacities. Use absolute full-model latency and consistent colors for combine implementations. The figure should make the interaction visible without requiring readers to reconstruct it from separate optimization ladders.

## 3. Use E48 to explain the mechanism and narrow the novelty claim

The comparisons distinguish three behaviors:

| Formulation | Reported behavior | Role in the argument |
|---|---|---|
| C-topk | Selects top-k contributions, but gathers across SoCs and triggers expensive redistribution | Shows that expressing mathematical sparsity alone does not preserve locality |
| C-lanes | Preserves device locality, but retains cross-core lane reductions | Provides the more demanding locality-preserving comparison |
| D | Preserves locality and organizes token aggregation more efficiently | Proposed execution layout and graph construction |

C-topk is 4.05–4.92× slower than D in the reported full-model measurements. At T = 512, the instrumented two-layer profile reports 1,538 MiB of cross-SoC traffic per MoE layer for C-topk, compared with 14.0 MiB for D. These profiles explain a failure of that graph formulation; they do not establish that every straightforward distributed sparse combine must fail.

A reviewer may reasonably ask why a gather spanning all SoCs is the decisive distributed baseline. Therefore, foreground the 11–15% full-model advantage over locality-preserving C-lanes, and use E50 to demonstrate why the design matters when capacities become heterogeneous.

### Resolve the local top-k question

The paper should answer:

> What happens if each SoC gathers only its selected expert rows, forms local token partials, and then combines those partials across SoCs?

- [x] Determine whether this formulation is already equivalent to D. It is (Section 8.1).
- [x] If equivalent, state that directly and identify the additional graph-construction choices precisely (Section 8.1).
- [ ] (Not needed: the formulations are equivalent. Section 8.1's variants were built anyway.) If different, implement the missing comparison with the same expert placement, metadata, precision, tile controls, and complete layout-conversion costs.

The distinguishing choices may involve tensor layout, tile ownership, index construction, or avoidance of compiler-generated dependencies. Name only the choices supported by the implementation and traces. Do not imply that the mathematical local-partial reduction itself is new.

### Quantify the C-lanes mechanism

- [x] Identify which operations require cross-core lane reduction (Section 8.2).
- [x] Report the communication or synchronization they introduce (Section 8.2).
- [x] Identify serialized intervals and their exposed critical-path cost (Section 8.2).
- [x] Connect these measurements to the observed 11–15% full-model difference (Section 8.2, with its caveats).

Cross-SoC traffic alone cannot explain this comparison: C-lanes reports slightly less cross-SoC traffic than D. The explanation must account for execution within each SoC as well.

## 4. Treat the custom kernel as supporting evidence

E49/E51 establish that a custom operation can integrate into the four-SoC graph, distribute tiles across the cores, and preserve local expert-output access. The paper should no longer motivate graph transformations by assuming custom kernels are unavailable.

A supported statement is:

> The token-owned layout also supports an efficient fused custom implementation.

The stronger statement that a custom kernel is faster only when it follows this layout is not established: alternative kernel layouts were not evaluated.

### Performance status

The reported 326 ms custom-kernel result versus approximately 351 ms for D is a promising preliminary latency reduction of about 7%. It rests on one valid full-model pass, with a preceding runtime timeout and 2.6% anchor drift. It should not yet be presented as a stable quantitative improvement.

- [x] Repeat full-model timing with warmup, multiple valid passes, and a stable matched anchor if the speedup will be claimed (E54, Section 8.3).
- [x] Investigate or document the initial timeout (Section 8.3).
- [ ] Keep the roughly 11-hour compile cost separate from runtime performance.
- [ ] State that T = 128, T = 256, and tiered custom-kernel execution were not measured.

The compile cost is a meaningful development limitation in this configuration, not evidence that custom kernels are generally impractical or that the layout is unique.

### Correctness status

Matching the next token does not establish numerical equivalence. The full-model FP32-relative L2 errors of 0.0951 for the custom kernel and 0.0917 for D do not directly measure their difference.

- [x] Compare custom-kernel outputs directly with D under identical inputs (Section 8.3).
- [x] Report relative L2 and maximum absolute differences, with the accumulation precision and summation order identified (Section 8.3).
- [ ] Retain the small four-SoC bit-identical result, but distinguish it from full-model numerical agreement.

## 5. Prioritized next experiments

| Priority | Experiment or clarification | Required output | Why it matters |
|---|---|---|---|
| 1 | Resolve local top-k versus D | Explicit equivalence explanation or a controlled baseline | Establishes the exact contribution beyond ordinary distributed sparse aggregation |
| 2 | Explain C-lanes versus D | Trace-supported cross-core dependency and critical-path analysis | Explains the gain against the strongest existing locality-preserving comparison |
| 3 | Sweep tier count and capacities | Saved expert work, added aggregation cost, and end-to-end latency | Shows a predictable tradeoff rather than a single favorable configuration |
| 4 | Evaluate a second model | Matched baselines and contribution breakdown | Tests whether the result survives a different routing/workload structure |
| 5 | Repeat the custom-kernel result if claimed | Stable timing and direct numerical comparison | Makes its quantitative performance and correctness claims defensible |

The two-tier result at T = 256—138.8 ms versus 142.3 ms for three tiers—is already evidence that an extra stage need not pay for itself. Use the tier sweep to explain the balance between reduced expert work and added stage/aggregation overhead.

Further combine tuning should follow these experiments unless profiling identifies a bottleneck that materially changes the argument. The single-gather-across-tiers optimization remains a candidate, with concatenation, buffer lifetime, placement, and overlap costs included.

## 6. Revised paper framing

### Candidate contribution statement

> RankTier co-designs heterogeneous expert capacities and token-owned aggregation for a statically partitioned multi-SoC execution graph. Its aggregation layout preserves expert-output locality and avoids costly cross-core reductions, allowing capacity savings to translate into consistent end-to-end latency improvements.

The cross-core mechanism should be substantiated by the C-lanes analysis before this wording is finalized.

### Suggested narrative order

1. Dynamic routing creates unequal expert workloads, motivating heterogeneous capacities.
2. Reducing expert work alone does not reliably improve latency because aggregation can absorb the savings: show E50.
3. Mathematical sparsity, device locality, and efficient cross-core execution are separate requirements: show E48.
4. Describe exactly how the proposed layout and graph construction satisfy those requirements.
5. Present the corrected E47 ladder, keeping attention, routing scans, output aggregation, and capacity changes separately attributable.
6. Use the custom kernel to demonstrate implementation flexibility, with its preliminary performance clearly labeled.

### Claim boundaries

| Supported by the reported experiments | Still needs evidence or narrower wording |
|---|---|
| Combine design materially changes the benefit of tiered capacities | The proposed design is necessary for any tiering benefit |
| The tested global top-k graph causes heavy redistribution | All natural local sparse-combine formulations fail |
| D outperforms the locality-preserving C-lanes variant | The full 11–15% difference is causally explained without trace analysis |
| A custom op integrates successfully using the token-owned layout | Efficient custom kernels require this layout exclusively |
| The custom kernel has a promising single-pass full-model result | A stable 7% full-model latency improvement |
| The tested model benefits on this compiler and hardware configuration | Generality across models, compilers, or accelerator architectures |

## 7. Reviewer perspective

E50 directly addresses the concern that capacity reduction and reduction optimization are merely independent improvements. The paper now has a stronger systems argument: aggregation structure governs whether heterogeneous capacity execution is beneficial.

The remaining challenge is to specify the transformation precisely enough that readers can distinguish its transferable principle from the particular compiler behavior measured here. A clear local top-k comparison, a causal explanation of C-lanes, and a second model would strengthen that argument more than another small optimization on the current workload.

## 8. Results, October 8–9, 2026

Full 48-layer model, KV cache on device, MXFP6, four SoCs, with the elementwise head-group sum in every program. Latencies come from one program + session fit of log latency over every timed pass of E46–E54. All sessions share D (ladder step 5) as an anchor; the largest residual is 0.50%, 0.88% and 1.69% at T = 128, 256 and 512.

Sources, under `9.17.2026/real_model/oracle_padding/mdts_flag/`:
- fitted results: `full_model/e54/e46_e54_summary.txt`, from `scripts/e47_summary.py`;
- drivers: `scripts/run_e52.sh` and `scripts/run_e54.sh`;
- profile analysis: `scripts/combine_breakdown.py`.

### 8.1 Local top-k is D; the graph's expression of locality decides (priority 1, E52)

D computes exactly the local top-k formulation of Section 3:
1. Each SoC gathers only its own selected expert rows.
2. Each token's rows are summed into the SoC's local `[T, d]` partial.
3. The four partials are summed elementwise.

The mathematical local-partial reduction is therefore not new. What D adds is how the graph tells the compiler which SoC owns which rows. A stage's `[64 lanes, C, d]` outputs have a lane axis that the compiler already partitions across SoCs, 16 lanes each. D reshapes them without moving data to `[4, 16C, d]`, so the SoC becomes the leading batch axis of one gather with per-SoC row indices `[4, T·k]`.

Two variants keep D's arithmetic, index values, summation order and final sum, and change only this expression (`build_full_dyncard.py --combine local_global | local_slices`):

| Variant | How each SoC's rows are expressed | T = 128 | T = 256 | T = 512 | Cross-SoC MiB per MoE layer (T = 512) | Logits vs D |
|---|---|---:|---:|---:|---:|---|
| D | Leading SoC axis of a layout-preserving reshape | 113.4 ms | 185.3 ms | 349.6 ms | 14 | — |
| local_global | Global row indices into the unsplit `[1, 64C, d]` outputs; locality exists only in the index values | 4.37× | 5.14× | 5.30× | 1,539 | Bit-identical |
| local_slices | Each SoC's 16 lanes cut out by an explicit Slice, then its own gather and sum | 3.44× | 4.04× | 4.14× | 1,539 | Bit-identical |

Ratios are slowdowns against D. Cross-SoC bytes come from instrumented two-layer profiles, layer 1.

Both variants make the compiler move expert outputs between SoCs: about 110× D's bytes, like C-topk (1,538 MiB). Their logits are bit-identical to D's. The 3.4–5.3× slowdown is therefore entirely placement and data movement, chosen by the compiler from how the graph expresses ownership.

Three graph-construction choices make D local and parallel. Each is supported by a measured alternative:
1. **SoC ownership as a leading batch axis** of a layout-preserving reshape of the already partitioned lane axis. Against index values: 4.4–5.3× slower. Against explicit slices: 3.4–4.1× slower.
2. **Each token's sum over its k rows on the core that owns the token.** Against a sum over lanes across cores (C-lanes): 1.11–1.15× slower (Section 8.2).
3. **The cross-SoC sum as elementwise adds per token tile.** Against Einsum tiles: 1.005–1.03× slower (E47). Against a transpose-based reduce-scatter, which collapsed onto one SoC in a one-layer replay: 34–43% slower (E16).

### 8.2 C-lanes against D (priority 2; E48 profiles, T = 512, layer 1)

| Phase | D: ops / busy core-ms / cross-core MiB | C-lanes: ops / busy core-ms / cross-core MiB |
|---|---|---|
| Index build | 6,913 / 27.5 / 0.5 | 1,040 / 3.3 / 0.4 |
| Gather | 762 / 66.7 / 0 (898 MiB gathered) | 4,430 / 15.0 / 0 (924 MiB gathered) |
| Mask multiply | 128 / 1.4 / 0 | 3,072 / 4.6 / 0 |
| Stage add | 64 / 0.7 / 0 | 1,024 / 1.6 / 0 |
| Reduction to per-SoC partials | 304 / 2.4 / 1.9 | 2,288 / 47.1 / 241.9 |
| Final cross-SoC sum | 1,248 / 0.7 / 1.9 | 1,250 / 0.9 / 1.9 |
| Exposed tail per SoC (last expert GEMM → final sum done) | 1.22–1.69 ms | 3.10–3.68 ms |

Cross-core MiB counts the on-chip multicasts between cores.

- **The operations that need cross-core reduction:** C-lanes sums over each SoC's 16 lanes per token tile. A token's 16 lanes sit on 16 different cores (one lane per core), so every tile's sum pulls operands from all cores by multicast. That is 242 MiB of cross-core traffic per layer, about 60 MiB per SoC, and 47 core-ms. D sums a token's k rows on the core that owns the token's tile: 1.9 MiB and 2.4 core-ms.
- **Serialization:** C-lanes' 16 reduction tiles span 3.0–3.6 ms per SoC. Their windows add up to 4.6–5.6 ms with only 1.7–2.0 ms of overlap, so the tiles mostly run one after another.
- **Critical-path cost:** the instrumented tail is about 1.9 ms longer per layer. The uninstrumented full model differs by 1.1 ms per layer (402.2 against 349.6 ms over 48 layers), so instrumentation overstates the gap; the earlier phases overlap with the expert stages in both programs.
- **Cross-SoC bytes do not explain the gap:** C-lanes moves 12.1 MiB, D 14.0 MiB. D's index build is heavier (27.5 against 3.3 core-ms) but runs under the expert stages.

Caveats: this is one instrumented sample of layer 1 of a two-layer cut. The profiles' critical-path tables hold no usable rows, so the tail is attributed by phase windows, not by a traced dependency chain.

### 8.3 Custom kernel: repeated timing and direct comparison (priority 5, E54)

- **Timing:** one session, D and the custom-op program alternating in the order D, custom, custom, D, D, custom. Each pass is 3 rounds × 20 inferences after the host tool's warm-up.
  - In pass order, the custom op ran 323.6, 324.7 and 323.3 ms, and D ran 342.8, 343.8 and 341.9 ms. That is 0.945× within the session, and 0.940× in the joint fit over E46–E54, which includes E51's pass.
  - Every pass succeeded.
- **The initial timeout (E51) was most likely a collision on the shared board.** The failed pass ended 19 s after it started: the runtime cancelled the data path on SoCs 3 and 1 ("wait in kernel failed", "Operation canceled"). The next pass then waited 20 s for a board that another job had taken. A device hang would instead have reached the program's 2-hour data-path timeout. It did not recur in E54's six passes.
- **Direct comparison under identical inputs** (last-position logits at T = 512, 151,936 entries):
  - relative L2 1.68e-2, maximum absolute difference 0.273 (largest |logit| 24.7), same top five and same next token;
  - both programs are deterministic (repeated passes bit-identical), and the difference is identical in E51 and E54.
- **Summation order and precision:**
  - The custom kernel sums each token's rows in FP32 (HVX single-precision vectors), separately per expert stage. It rounds each stage's partial to FP16 once, and the graph adds the two stage partials in FP16.
  - D adds the two stages slot by slot, which is exact because each slot is non-zero in exactly one stage. It then reduces the eight slots with the compiler's ReduceSum, whose internal accumulation precision is undocumented.
  - For scale: formulations that reorder the same sums differ from D by 0.7e-2 to 3.3e-2 relative L2 (E48). Those that keep D's order (Section 8.1) are bit-identical.
- **The four-SoC test** (one stage, 128 tokens) is bit-identical to the graph combine. This does not imply full-model agreement.
- **Compile time:** 650–656 minutes for the full model (two compiles), against 10–24 minutes for the graph programs.
- **Not measured:** T = 128, T = 256 and the tiered layout. The T = 512 program is kept at `full_model/qpc_e54_512_tcc` (25 GB).

### 8.4 Status of the prioritized experiments (Section 5)

| Priority | Item | Status |
|---|---|---|
| 1 | Local top-k against D | Done: equivalent to D; the graph-construction choices are measured (Section 8.1) |
| 2 | C-lanes against D | Done from the existing E48 profiles (Section 8.2) |
| 3 | Tier count and capacity sweep | Not run: a method to choose the tiers is being designed instead. Existing data: at T = 256 two tiers beat three by 2.5% (E47); per-tier stage and gather costs come from E44; `scripts/tier_budget.py` lists the candidate structures and their rows |
| 4 | Second model | Not started; needs a model choice. Besides Qwen3-MoE, QEfficient 1.23 exports gpt-oss, Granite-MoE, Mixtral, Qwen3.5-MoE, GLM-4-MoE, DeepSeek-V3, Llama 4 and Grok-1 |
| 5 | Custom-kernel timing and correctness | Done (Section 8.3) |

### 8.5 Production re-timed against the anchor (E55, for the paper)

The paper's speedups over production previously came from E36, before the attention fix and outside the joint fit. E55 re-timed production in one session per chunk length. Production is the exported graph with the compiler's default partitioning, without `-mdts-mos=1`. Each session alternates it with D (step 5), two passes each in mirrored order. Driver: `scripts/run_e55.sh`; results: `full_model/e55/`.

Joint fit over E46–E55 (66, 66 and 75 passes; largest residual 0.50%, 0.88% and 1.69%):

| T | Production | Naive T/T with the attention fix (2′) | D (step 5) | RankTier | Production → RankTier | Production → 2′ |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 499.6 ms | 250.0 ms | 113.2 ms | 105.3 ms | 4.74× | 2.00× |
| 256 | 516.5 ms | 502.4 ms | 184.8 ms | 141.5 ms | 3.65× | 1.03× |
| 512 | 1053.5 ms | 1027.4 ms | 349.2 ms | 225.3 ms | 4.68× | 1.03× |

- Without the attention fix, naive T/T (step 2) is 10% slower than production at T = 256 and 512. With the fix (2′) it is 2.5–2.7% faster there.
- At T = 128, production takes about as long as at T = 256.
- Production's relative L2 against FP32 is 0.188, 0.089 and 0.095. Its next token matches FP32 except at T = 256, where production, D and RankTier all pick ' a' over ' the'; the two leading FP32 logits there are 0.58 apart.
- Adding the E55 sessions shifts every fitted absolute latency by at most 0.3 ms, because the fit's level is the mean over sessions. All ratios are unchanged.
- The paper (`mdts_flag/paper/`) uses this fit through `scripts/paper_numbers.py` → `paper/paper_numbers.json`.
