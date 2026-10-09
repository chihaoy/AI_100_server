# TensorRT / H100 portability experiment

Implements `RANKTIER_TENSORRT_H100_EXPERIMENT_PLAN.md` (repository root) for one complete Qwen3-30B-A3B MoE block on one
H100: the capacity × combine 2×2 (G00, G10, G01, G11), static fixed-order baselines, correctness and overflow checks, and
a native fused-MoE reference. Status (2026-10-10): run on wolverine (H100 PCIe, TensorRT 11.4) at T = 128, 256 and 512.
The results are in section 14 of the plan; raw files are in `results/` (`L20_T*`: final run with all eight arms; `run2`: the
previous session without U00/U01; `L20_T512_run1_constinit`: the first dense version with a stored zero constant), with
`logs/`, `engines_meta/` (engine inspector layers, build records) and `onnx_meta/`.

## Files

| File | Role |
|---|---|
| `export_layer.py` | Writes `data/L<layer>/`: FP16 weights `w_gate`/`w_up` [128, 2048, 768], `w_down` [128, 768, 2048], `w_router`; traces `x_T*` (the layer's MoE input), `ids_T*` (top-8), `w_T*` (renormalized top-8 probabilities, FP32) for the held-out AI100 reference chunks (T = 128 prompt 41, T = 256 and 512 held-out GSM8K). The recomputed top-8 matches the captured routing for 100% of tokens at all three T (layer 20). |
| `calibrate_tiers.py` | Single-GPU tiers over all 128 experts from the E41 calibration split only (782/608/441 chunks, all layers), h = 1.28, g = 16; writes `data/tiers.json` with the overflow rate on the E41 evaluation split (805/618/449 chunks). |
| `moe_variants.py` | The MoE block as ONNX-exportable PyTorch, all arms from one operator set, and the untiered FP32 reference. |
| `export_onnx.py` | ONNX (opset 17, FP16, external weights) per (variant, T); `--selftest` checks every variant with onnxruntime on small random weights. |
| `build_engines.py` | TensorRT engines with identical settings: strongly typed, 8 GiB workspace, opt level 3, shared timing cache, detailed profiling verbosity; dumps the engine inspector's layer list. |
| `run_trt.py` | Correctness before timing, then eager and CUDA-graph latency with alternating order; host-observed overflow check and recovery. |
| `native_ref.py` | vLLM `fused_experts` (Triton grouped-GEMM MoE) on the same inputs and protocol. |
| `profile_arms.py` | CUPTI kernel traces mapped to engine layers and classified (weight gather, token gather, GEMMs, activation, combine, metadata). |
| `inspect_engine.py`, `summarize.py` | Engine layer digest; latency tables and the plan's ratios, including the stage/capacity split from U00/U01. |
| `run_queue.sh` | All chunk lengths in one queue (run_all.sh + profile per T). |
| `run_all.sh` | Driver for one T: export, build, check on four traces, time, native reference. |

## Variants

All arms replay the same `ids` and `w`; the router is excluded from the timed region in every arm, and routing metadata
(counts, prefix sums, rows, token and weight tables) is inside it. Gate and up weights are one [E, d, 2f] tensor everywhere.

| ID | Capacities | Combine | Ranking / weights |
|---|---|---|---|
| S00, S01 | uniform: 128 lanes at T | dense, token-owned | none; constant weights in native order |
| G00, G01 | uniform: 128 lanes at T, in rank order | dense, token-owned | run-time rank; weights gathered by rank |
| U00, U01 | the tiers' stage structure, every stage at T | dense, token-owned | run-time rank; weights gathered by rank (control) |
| G10, G11 | tiers by rank (default `3t_32_32_64`) | dense, token-owned | run-time rank; weights gathered by rank |

- **Dense combine (repaired).** Each tier scatters its weighted rows into a zero-filled `[lanes, T+1, d]` accumulator (row T absorbs padding) and sums over lanes. No read of the zeros, no transpose. The zeros come from `x[0,0] * 0` broadcast at run time; a constant zero tensor is folded by TensorRT into a 269 MB engine constant that is copied in on every call.
- **Token-owned combine.** All tiers' rows go into one flat buffer; each token gathers its k rows and sums them.
- **Gathered weights.** Every expert sits in some tier, so the ranked arms copy the whole 1.1 GiB bank per call unless TensorRT fuses the gather. S00/S01 measure what ranking and weight selection cost.
- **Overflow.** `ovf` flags any assignment beyond its lane's capacity. The host reruns the chunk on the uniform engine with the same combine (`run_trt.py` measures this path).

## Tiers (`data/tiers.json`, from the calibration split)

| T | 3t_32_32_64 | rows vs uniform | 5t_8_8_16_32_64 | rows vs uniform |
|---:|---|---:|---|---:|
| 128 | 128/32/16 | 37.5% | 128/96/48/32/16 | 28.1% |
| 256 | 256/48/32 | 35.9% | 256/176/80/48/32 | 25.4% |
| 512 | 512/96/48 | 34.4% | 512/352/160/96/48 | 23.8% |

No E41 evaluation chunk overflows under either structure, and the layer-20 timing traces fit. At T = 128, 3t cannot overflow at all: its capacities exceed the bound N/r at ranks 33 and 65.

## Acceptance (fixed before timing) and what TensorRT did to it

Relative L2 ≤ 2e-3 of `y` against the untiered FP32 reference. Emulating FP16 rounding of the intermediates with FP32
accumulation gives 4.8e-4 at T = 128 and 512, so the threshold leaves a 4× margin. Determinism across three runs is
recorded. Different reduction orders need not be bit-identical. On the GPU, TensorRT 11 picks FP16-accumulating GEMM
kernels in some fused contexts, with no builder control. The arms therefore land at 1.07e-3 to 2.67e-3 relative L2, which
emulation reproduces, and different arms exceed 2e-3 at different T. Every engine within 1e-2 and without overflow is
timed, and each latency row records whether it met 2e-3 (`run_trt.py`).

On CPU (eager FP32, real T = 128 trace), all arms are within 4e-7 of the reference. The onnxruntime self-test passes for
all six variants on real-like, skewed (120 empty experts) and balanced routing. A forced overflow raises the flag, and the
uniform rerun reproduces the reference.

## Running on the GPU host

```
./setup_trt_venv.sh      # .venv_trt: torch (cu128 wheels), tensorrt-cu12, onnx; used: torch 2.11.0+cu128, TensorRT 11.4.0.106
./setup_vllm_venv.sh     # .venv_vllm: vllm (brings its own torch); used: vLLM 0.31.0, torch 2.13.0+cu130
./run_queue.sh           # T = 512, 256, 128: export, build, checks, timing, vLLM, kernel profile (about 30 min)
.venv_trt/bin/python summarize.py results L20
```

Copy `data/L20` (1.2 GiB) with the scripts. Engines take about 1.13 GiB each on disk (8 per chunk length), and the ONNX
files are deleted once their engine exists. On wolverine (ssh port 425, user w2ni) the run lives in `~/workspace/trt_h100`;
its disk is shared and was 95% full, so only the T = 512 engines are kept there. Check that the GPU is otherwise idle;
`run_all.sh` logs `nvidia-smi` at the start.
