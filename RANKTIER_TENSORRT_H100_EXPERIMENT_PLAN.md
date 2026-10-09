# RankTier: TensorRT Portability Experiments on H100

Date: October 9, 2026

## 1. Objective and claim boundary

Test whether RankTier's dynamic assignment to compiled expert-capacity tiers and token-owned aggregation improve MoE prefill on H100 under TensorRT.

The primary question is whether the interaction observed on AI100 transfers: **does aggregation design determine whether heterogeneous expert capacities produce a net latency benefit?**

Separate three possible results:

1. **Implementation portability:** the computation can run correctly in a TensorRT engine.
2. **Graph-optimization portability:** the transformed graph is faster than matched alternative TensorRT graphs.
3. **Competitive GPU performance:** the implementation improves on a tuned native GPU MoE backend.

A result at one level does not establish the next. TensorRT is a compiler/runtime for NVIDIA hardware; an H100 result extends the study to GPU execution, not to a second NPU architecture.

## 2. Why precompilation is relevant but insufficient

TensorRT builds an optimized engine ahead of execution, but supports runtime input dimensions, data-dependent shapes, and custom plugins. A fixed graph or CUDA Graph can still invoke kernels whose useful work depends on device-resident routing metadata.

NVIDIA documents persistent MoE auxiliary kernels that loop over the valid token count on the GPU while retaining CUDA Graph compatibility. Therefore, do not motivate this experiment with the assumption that precompiled execution must compute every provisioned row. [1, 2, 3]

The hypotheses are:

- **H1 — Capacity:** rank-bound capacities reduce total MoE latency relative to uniform capacities under matched runtime ranking and weight access.
- **H2 — Aggregation:** token-owned aggregation reduces the cost of combining heterogeneous expert outputs.
- **H3 — Interaction:** the benefit of capacity reduction depends on the aggregation representation, as in AI100 E50.
- **H4 — Competitive value:** the resulting design remains useful against a tuned native MoE backend that can already exploit variable expert loads.

H1–H3 are the first-stage experiment. H4 is a required reference for claims about optimized GPU inference, but a negative H4 result does not invalidate a carefully bounded H1–H3 result.

## 3. H100 software and hardware scope

- Use a single H100 initially. Record PCIe versus SXM, memory capacity, driver, CUDA, TensorRT, GPU clocks/power limit, and whether MIG is enabled.
- Pin package/container versions and source commits. Record the actual backend and kernel names; the TensorRT-LLM name alone does not identify an execution path.
- TensorRT and TensorRT-LLM are separate targets. The main experiment must actually build and execute a TensorRT engine. A TensorRT-LLM fused operator invoked through another backend is an optimized reference, not automatically a TensorRT-engine experiment.
- H100 is Hopper/SM90. The native TensorRT `IMoELayer` documented at the time of writing lists SM100, SM103, and SM110, so **do not base the H100 implementation on that API**. Use supported primitive graph operations and/or an H100-compatible plugin. [4]
- Start with FP16, including identical weight values and routing weights in all arms. BF16 is an optional follow-up if every selected implementation supports it. Do not compare an FP16 graph to an FP8 reference and attribute the difference to layout.
- AI100's MXFP6 results are context, not numerically matched H100 baselines.

A full 30B model port is unnecessary for the first decision. Start with complete MoE blocks, including their routing metadata and combine costs.

## 4. Workload and input traces

Use the existing Qwen3-30B-A3B routed-expert dimensions:

| Parameter | Value |
|---|---:|
| Hidden width | 2048 |
| Expert intermediate width | 768 |
| Routed experts | 128 |
| Top-k | 8 |
| Initial chunk lengths | 128, 256, 512 |

Start with one representative layer. If it produces a useful result, validate on early, middle, and late MoE layers and multiple held-out prompts.

Save hidden states, selected expert IDs, routing weights, and expert token counts. For attribution experiments, replay identical selected IDs and routing weights in every arm so router numerical differences cannot change the workload. Count building, ranking, packing, index construction, expert execution, and aggregation remain inside the timed region. Router-logit computation may be excluded only if excluded consistently and stated explicitly.

Use separate traces for capacity calibration and evaluation. Do not select tier capacities using the evaluated trace.

Do not import the AI100 four-SoC tier configuration unchanged into a one-GPU experiment. A single H100 has no equivalent four-way device ownership boundary. Either recalibrate tiers over all 128 experts, or explicitly label a 32-expert replay as a local expert-partition experiment. A local replay must preserve the original top-8 routing and its local assignments; it must not reroute every token to eight local experts.

## 5. Core experiment: capacity × combine

Build the following four TensorRT variants using matched operators/plugins wherever feasible:

| ID | Expert capacities | Combine | Purpose |
|---|---|---|---|
| G00 | Uniform | Repaired dense | Controlled graph baseline |
| G10 | Rank-tiered | Repaired dense | Capacity change under dense aggregation |
| G01 | Uniform | Token-owned | Aggregation change under uniform capacity |
| G11 | Rank-tiered | Token-owned | Joint design |

For all four arms:

- Keep runtime ranking enabled, including the uniform arms.
- Keep weight selection, placement, precision, activation, and final output semantics matched.
- Use the same trace, tuning budget, workspace allowance, and timing method.
- Give the dense path reasonable fusion and reduction optimizations. Do not deliberately preserve a known bad zero-initialization or transpose implementation.
- Include every conversion and synchronization needed by that variant.
- Record the stage count and individual GEMM shapes. The capacity treatment includes stage organization, so describe it as the effect of the capacity design rather than a pure FLOP-only intervention.

Report capacity speedups separately:

`S_dense = latency(G00) / latency(G10)`

`S_token = latency(G01) / latency(G11)`

Report aggregation speedups:

`R_uniform = latency(G00) / latency(G01)`

`R_tiered = latency(G10) / latency(G11)`

An optional interaction summary is `S_token / S_dense`; interpret it alongside absolute milliseconds and measurement variability. Do not multiply unrelated speedups from different runs or configurations.

Add a tuned fixed-order/static-weight uniform baseline if runtime ranking or weight selection has a material cost. It answers whether the complete design wins after paying its own enabling overheads.

## 6. Implementation strategy

### 6.1 Start with the graph representation

Port one MoE block using TensorRT primitive operations where supported. Use runtime expert indices and fixed tier capacities. Inspect whether weight Gather materializes copied weight banks; if it does, include and report that cost.

Do not assume a graph can be ported unchanged from QAIC. In particular, AI100's leading SoC batch axis is evidence about that compiler's ownership inference, not a portable instruction to assign work to GPU SMs or devices.

After compilation, inspect engine layers, selected implementations, tensor shapes, and any inserted reformat/copy operations. Check whether TensorRT fuses away the dense intermediate or rewrites nominally different arms into equivalent execution. If it does, report this as a meaningful result rather than defeating the optimization to recreate the AI100 bottleneck.

### 6.2 Use plugins where necessary, with matched controls

A proposed GPU implementation can count/rank experts, pack token rows, execute capacity-specialized expert GEMMs, and gather selected outputs into token-owned sums. Keep metadata on device where possible. Select weights by expert ID/pointer rather than physically sorting the complete weight bank when the GEMM interface permits it.

If a custom plugin is necessary, provide matched variants through the same plugin framework. A fused proposed plugin versus an unfused primitive baseline changes both representation and implementation quality; it is not a clean attribution experiment by itself.

Separate results for:

- graph-only or graph-plus-minimal-plugin implementation;
- matched custom-kernel implementations;
- native optimized reference.

TensorRT's plugin APIs make implementation possible; they do not guarantee that the proposed path will be fast. [2]

## 7. Native optimized reference

Run an H100-compatible TensorRT-LLM fused MoE implementation, or an equivalent established grouped-GEMM MoE backend if the chosen TensorRT-LLM version is difficult to isolate.

Use the same expert weights, hidden states, selected IDs, routing weights, precision, and complete MoE boundary. Include dispatch/permutation, GEMMs, activation, and finalize/combine. Document anything that cannot be matched. Allow the reference its normal fusion and variable-work scheduling.

NVIDIA already documents token-oriented combine and FC2/finalize fusion, so token ownership alone is not a new GPU contribution. The question is whether RankTier's joint scheduling and aggregation adds value beyond those implementations. [5, 6]

If G11 beats the controlled graphs but loses to the native reference, the supported conclusion is that the representation helps general graph compilation; do not claim a state-of-the-art GPU MoE speedup.

## 8. Correctness and capacity overflow

- Compare complete MoE output tensors, not only the next token.
- Use an untiered reference that processes every routed assignment. Report relative L2 error, maximum absolute error, and determinism across repetitions.
- Establish and state numerical acceptance thresholds before the performance comparison. Explain accumulation precision and order; different reduction trees need not be bit-identical.
- Include balanced, skewed, and empty-expert cases, plus real held-out traces.
- Test an intentionally overflowing trace. Detect overflow and rerun with safe capacity; never silently drop assignments.
- Report no-overflow fast-path latency, overflow-check overhead, and recovery latency separately. An unmeasured recovery path must not be included in a deployment-ready claim.

For the initial block study there is no KV state to restore. A later full-model experiment must also validate state handling when rerunning a chunk.

## 9. Measurement protocol

1. Compile and tune before timing; record compile time separately.
2. Run one variant at a time on an otherwise idle GPU. Warm up until execution is stable.
3. Alternate variant order and repeat batches of measurements. Record batch medians and variation rather than a single best sample.
4. Use CUDA events around the complete device computation on the correct stream, with all required work ordered before the end event. Also report host-observed latency if host coordination is involved.
5. Keep launch mode consistent. If CUDA Graph capture works for all compared arms, report a matched captured comparison; otherwise label captured and uncaptured results separately.
6. Keep input/output buffers and required weights resident for steady-state timing. Report internal packing, copies, and temporary-buffer costs; do not hide them as setup.
7. Profile separately from the primary latency runs, since instrumentation changes timing.

Collect:

| Metric | Interpretation |
|---|---|
| Complete MoE latency | Primary outcome |
| Routing metadata/ranking/packing time | Cost of enabling the schedule |
| Expert GEMM time and shapes | Capacity benefit and stage overhead |
| Combine time and exposed tail | Whether aggregation absorbs GEMM savings |
| Allocated/intermediate bytes | Buffering cost |
| Observed memory traffic | Actual data movement, distinct from allocation |
| Tensor work counters or kernel-level work accounting | Executed padding, if measurable |
| Kernel launches and synchronization | Overhead of extra tiers |
| Overflow frequency and recovery cost | Correctness/performance boundary |

Provisioned rows, executed arithmetic, and elapsed time are different quantities. If hardware counters cannot isolate useful versus padded work, report the shapes and measured latency without claiming an exact arithmetic reduction.

Use Nsight Systems for execution and synchronization timelines, and targeted Nsight Compute runs for expensive kernels. Do not sum overlapping kernel durations and label the result end-to-end latency.

## 10. Figures and result tables

**Figure 1 — Port of E50:** one panel per chunk length, four bars G00/G10/G01/G11. Show absolute MoE latency and uncertainty. Add the native reference as a separately styled fifth bar.

**Figure 2 — Mechanism:** for one representative chunk, show expert-compute time, combine/tail time, intermediate bytes, and observed memory traffic. Keep timelines separate from additive breakdowns where overlap exists.

**Figure 3 — Tier tradeoff, only if warranted:** sweep a small number of tier configurations and show saved expert work versus extra packing, launch, and aggregation cost. Recalibrate each configuration without looking at held-out test results.

Suggested CSV columns:

```text
gpu,software_versions,backend,layer,trace_id,T,E,k,dtype,variant,tier_config,
launch_mode,latency_us,metadata_us,gemm_us,combine_us,provisioned_rows,
workspace_bytes,observed_dram_bytes,overflow,relative_l2,max_abs_error
```

Mark unavailable profiling fields as missing, not zero. Timing components need not add up when they overlap.

## 11. Optional multi-GPU phase

Proceed only if the single-GPU study identifies a useful question or if distributed ownership is the claim being tested.

- Partition experts explicitly across GPUs and use a communication library or kernel interface that actually supports the placement.
- Compare local token partials plus cross-GPU combination with a tuned native EP implementation, including dispatch and combine communication.
- Record NVLink/PCIe topology and actual bytes transferred.
- Do not treat an added tensor dimension as proof of device locality.
- A one-GPU result cannot validate the AI100 cross-SoC ownership transformation or its communication savings.

## 12. Decision rules and execution checklist

| Observation | Supported interpretation | Next action |
|---|---|---|
| TensorRT eliminates the dense intermediate | Its compiler already handles that representation cost | Focus on residual capacity/packing costs |
| G11 beats G00–G10 but not native MoE | Graph optimization transfers; competitive GPU benefit is unproven | Report the boundary and avoid a full-model port solely to enlarge the claim |
| Capacity benefit is much larger with token-owned combine | The capacity/aggregation interaction transfers | Validate across layers and held-out routing traces |
| G11 also beats native MoE | Potential competitive GPU contribution | Recheck numerical agreement, tuning, and complete timing boundaries; then scale up |
| Ranking/packing dominates | GPU execution does not favor the current tier design in this regime | Characterize the crossover; do not force a positive result |
| No repeatable gain | AOT alone does not explain applicability | Use the result to narrow the architecture/compiler conditions |

- [x] Pin the H100 environment and selected native reference.
- [x] Export one real MoE layer and matched routing traces.
- [x] Build G00/G01 at T=512 and inspect the compiled execution.
- [x] Build G10/G11 with calibrated capacities and overflow detection.
- [x] Verify outputs, skew/empty-expert cases, and forced overflow.
- [x] Run matched complete-block timing and the native reference.
- [x] Extend to T=128/256 and representative layers if informative (T done; layers not run, section 14.7).
- [x] Profile the mechanism behind any repeatable difference.
- [x] Decide whether multi-GPU or full-model work is justified (not justified, section 14.6).

## 13. Sources

Official documentation consulted October 9, 2026. Pin the versions actually used; latest/main pages can change.

1. [TensorRT dynamic shapes and optimization profiles](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-basics.html).
2. [TensorRT custom plugin API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html).
3. [TensorRT-LLM expert-parallelism optimizations: device-side valid-token loops and CUDA Graph compatibility](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html).
4. [TensorRT native MoE layer and hardware restrictions](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/transformers-moe.html).
5. [TensorRT-LLM token-oriented one-sided MoE combine](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.md).
6. [TensorRT-LLM MoE configuration, including finalize fusion](https://nvidia.github.io/TensorRT-LLM/llm-api/reference/MoeConfig.html).
7. [TensorRT hardware/software support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html).

Project basis: `RANKTIER_E47_E51_ASSESSMENT_AND_NEXT_STEPS(1).md`, including E50's capacity/combine interaction and E52's distinction between local top-k mathematics and compiler-visible ownership. This plan proposes H100 experiments; it contains no measured H100 results.

## 14. Results, October 10, 2026 (wolverine)

### 14.1 What was run

- **Hardware:** wolverine, one H100 PCIe 80 GB (SM90, 2.0 TB/s HBM2e, 350 W limit, maximum SM clock 1755 MHz, MIG off), driver 580.178.04.
- **Software:** TensorRT 11.4.0.106 (pip, cu12); PyTorch 2.11.0+cu128 for export and timing; vLLM 0.31.0 (PyTorch 2.13.0+cu130, separate venv) for the native reference. The GPU was idle at the start of every run.
- **Code and raw results:** `9.17.2026/real_model/oracle_padding/mdts_flag/trt_h100/` (`README.md`, `results/`, `logs/`).
- **Workload:** Qwen3-30B-A3B layer 20, FP16 weights.
  - Held-out traces at T = 128, 256 and 512 (the AI100 reference chunks). The recomputed top-8 matches the captured routing for 100% of tokens.
  - Balanced, skewed and forced-overflow stress traces.
  - The router is excluded from the timed region. Counting, ranking, packing, index construction, expert GEMMs and aggregation are inside it.
- **Tiers:** one GPU owns all 128 experts, so tiers rank all of them: 32/32/64 lanes, capacities 512/96/48, 256/48/32 and 128/32/16.
  - Calibrated on the E41 calibration split only.
  - 0 of the 1,872 E41 evaluation chunks overflow.
- **Arms:** all built from one ONNX operator set as strongly typed engines with identical builder settings.

| Arm | Capacities | Combine | Ranking and weights |
|---|---|---|---|
| S00, S01 | 128 lanes at T | dense, token-owned | none; constant weights in native order |
| G00, G01 | 128 lanes at T | dense, token-owned | run-time rank; weights gathered by rank |
| U00, U01 (added control) | 32/32/64 lanes, all at T | dense, token-owned | run-time rank; gathered |
| G10, G11 | 32/32/64 lanes at calibrated capacities | dense, token-owned | run-time rank; gathered |

- **Repaired dense combine:** each stage scatters into a run-time zero-filled `[lanes, T+1, d]` accumulator and sums over lanes. A first version used a zero constant, which TensorRT stored in the engine (269 MB) and copied in on every call; it was replaced (`results/L20_T512_run1_constinit`).
- **Protocol:** CUDA events; 6 rounds × (50 warm-up + 200 timed) iterations; alternating order; eager and CUDA-graph modes, both captured for every arm. Batch medians vary by ≤ 2%, and eager is within 1.5% of graph. All ratios compare arms from the same session; run 3 is below, and runs 2 and 3 agree to within 0.5%.

### 14.2 Latency (µs per MoE block, CUDA graph)

| T | S01 | S00 | G01 | G00 | U01 | U00 | G11 | G10 | vLLM `fused_experts` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1009 | 1250 | 3398 | 3661 | 2833 | 3229 | 2677 | 3015 | 433 |
| 256 | 1297 | 1779 | 3701 | 4245 | 3091 | 3935 | 2781 | 3488 | 509 |
| 512 | 1943 | 2961 | 4248 | 5413 | 3717 | 5408 | 3001 | 4437 | 597 |

### 14.3 The 2×2 and its decomposition

| T | S_dense = G00/G10 | S_token = G01/G11 | S_token/S_dense | R_uniform = G00/G01 | R_tiered = G10/G11 | Stages, dense G00/U00 | Stages, token G01/U01 | Capacities, dense U00/G10 | Capacities, token U01/G11 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.214 | 1.270 | 1.045 | 1.077 | 1.126 | 1.134 | 1.200 | 1.071 | 1.058 |
| 256 | 1.217 | 1.331 | 1.094 | 1.147 | 1.254 | 1.079 | 1.197 | 1.128 | 1.112 |
| 512 | 1.220 | 1.416 | 1.160 | 1.274 | 1.478 | 1.001 | 1.143 | 1.219 | 1.239 |

The capacity treatment splits into two parts:
- **Capacity cut at a fixed stage structure:** 1.06–1.24×, growing with T and about equal under both combines.
- **Stage organization (1 → 3 stages):** 1.14–1.20× with the token-owned combine but 1.00–1.13× with the dense combine.

So the combine dependence comes from the extra stages that capacity classes introduce, not from the row reduction itself.

### 14.4 Mechanism (CUPTI kernel classes, T = 512, µs of kernel time per call; parallel streams overlap)

| Arm | Weight gather | Token gather | GEMMs + activation | Combine | Metadata | Kernels |
|---|---:|---:|---:|---:|---:|---:|
| S01 | – | 357 | 1361 | 88 | 33 | 18 |
| S00 | – | 329 | 1359 | 1120 | 167 | 19 |
| G01 | 2497 | 351 | 1346 | 88 | 45 | 23 |
| G00 | 2504 | 331 | 1346 | 1124 | 180 | 24 |
| U01 | 1873 | 287 | 1891 | 88 | 79 | 33 |
| U00 | 1881 | 435 | 1904 | 1797 | 54 | 38 |
| G11 | 1868 | 116 | 1259 | 88 | 73 | 30 |
| G10 | 1882 | 190 | 1178 | 1503 | 68 | 36 |

1. **TensorRT materializes run-time weight selection.**
   - A Myelin gather kernel copies the whole 1.1 GiB bank on every call: 2.50 ms as one 128-lane gather, 1.87 ms as per-stage gathers (same bytes).
   - Either figure is more than 3× vLLM's entire latency.
   - Ranking plus weight selection makes G01 2.2–3.4× slower than S01.
   - The faster per-stage gather is most of what three stages gain.
2. **TensorRT does not remove the dense intermediate.**
   - The dense combine materializes `[lanes, T+1, d]` accumulators: 1.12 ms for one stage and 1.80 ms over three stages at T = 512.
   - The token-owned combine becomes one fused gather-sum kernel (88 µs) whatever the stage structure.
3. **Fewer rows save little GEMM time on this GPU.**
   - Within three stages, the calibrated capacities save 0.63 ms of GEMM time at T = 512.
   - Three stages cost more GEMM time than one at the same rows, and at T = 128 the tiered GEMMs exceed the uniform ones.
   - The GEMMs are bound by weight reads, and the small-M stages tile poorly.
4. **Stages overlap.** TensorRT runs the stages concurrently on several streams (signal/wait layers), unlike AI100's sequential stages.

### 14.5 Correctness and overflow

**Error levels**
- Every arm is deterministic over repeated runs.
- Against the untiered FP32 reference, TensorRT arms are at 1.07e-3 to 2.67e-3 relative L2; vLLM is at 4.79e-4 (FP32 accumulation).

**The pre-set threshold**
- The threshold of 2e-3 assumed FP32 accumulation, but TensorRT 11 picks FP16-accumulating GEMM kernels depending on the fused context and offers no builder control.
- Emulation reproduces both levels:
  - FP16 accumulation in the down GEMM alone gives 1.05e-3 (max abs 1.39e-3), matching the one-stage arms at T = 512.
  - FP16 accumulation in both GEMMs gives 2.26e-3 (max abs 2.03e-3), matching the staged arms.
- In a standalone probe, the same GEMM gets an FP32-accumulating kernel alone and an FP16-accumulating one when followed by a cast, at equal speed (260 vs 265 µs).
- So the threshold misses follow kernel selection, not the transformation. They hit different arms at different T:
  - T = 512: the staged and tiered arms.
  - T = 256: the static arms.
  - T = 128: all uniform arms.
- On CPU in FP32, the same graphs match the reference to 4e-7.

**Stress traces**
- Skewed (8 experts take every token, 120 empty) and balanced traces raise no overflow and give correct outputs.

**Forced overflow** (40 experts with 8T/40 tokens each)
- Flagged by both tiered arms at T = 256 and 512. At T = 128 the 32/32/64 capacities exceed the bound N/r, so overflow is impossible.
- Rerunning on the uniform engine returns the uniform arm's output exactly.
- The host-side check (flag copy and sync) costs 8–12 µs per chunk.
- Recovery: 7.29 ms (G11 + G01) and 9.88 ms (G10 + G00) at T = 512; 6.67 and 7.89 ms at T = 256.

### 14.6 Against the hypotheses and decision rules

- **H1, capacity: supported.** 1.21–1.22× (dense) and 1.27–1.42× (token-owned). At a fixed stage structure, 1.06–1.24×.
- **H2, aggregation: supported.** 1.08–1.27× at uniform capacity and 1.13–1.48× with tiers.
- **H3, interaction: transfers qualitatively, weaker than on AI100.**
  - S_token/S_dense is 1.05–1.16, growing with T where AI100's was largest at small T.
  - On this compiler it comes from the per-stage cost of the dense combine.
- **H4, competitive value: not supported.**
  - vLLM's Triton grouped-GEMM MoE, whose combine is itself a token-owned `moe_sum`, is 5.0–6.2× faster than G11.
  - It is also 2.3–3.3× faster than the best TensorRT graph (S01).
- **Decision table:** "G11 beats G00–G10 but not native MoE" applies. The graph-level interaction transfers, a competitive GPU benefit is unproven, and the boundary should be reported rather than extended to a full-model port.
- **Weight selection dominates.** The "ranking/packing dominates" row also applies, in the form of weight selection: TensorRT copies gathered weights, so the tier design as a graph is not favored on this GPU.

### 14.7 Limits and open items

- One layer (20) and one held-out trace per chunk length. Early and late layers were not run, per the decision rule.
- TensorRT 11.4 has its own MoE layer (`INetworkDefinition.add_moe`, `IMoELayer`, gated SiLU/SwiGLU weights), but building it on the H100 fails: "MoE requires Blackwell or Thor GPU architecture (SM 10.x or SM 11.x)" (`trt_h100/moe_native_trt.py`). On Hopper, the TensorRT family's optimized MoE is TensorRT-LLM's fused MoE, a separate library not run here.
- vLLM used its default kernel configuration (none ships for E=128, N=768 on H100 PCIe); tuning would only make it faster.
- No plugin implementation was built. A pointer-array grouped GEMM per tier would avoid the weight copy and is the experiment to run if a GPU claim is wanted.
- PCIe H100 only; SXM bandwidth would change absolute numbers.
- The multi-GPU phase is not justified by these results.
