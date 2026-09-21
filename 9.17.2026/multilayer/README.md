# Different layer capacities inside one QPC

For adaptation across prefill chunks, see the [runtime design space](RUNTIME_ADAPTATION.md).
It retains capacity changes, expert regrouping, activation routing, replication
and migration across both four-card and two-card-pair layouts, and distinguishes
tested mechanisms from open probes.

This experiment joins consecutive, complete Qwen3-30B-A3B decoder layers into
one ONNX graph **before compilation**. Each variant has one QPC, one partition,
and four devices with 16 cores each. There is no per-layer program switch or
host reduction. Input and output are fp16 residual streams `[128, 2048]`.

The first experiment uses the original checkpoint, layers 0 and 1, and the
saved GSM8K prompt 41 input. It compares:

| Variant | Layer 0 stage capacities | Layer 1 stage capacities |
|---|---|---|
| Uniform | 128, 128 | 128, 128 |
| Tuned | 64, 32 | 128, 32 |

Both variants use the same expert permutation from `e2e/order_L*.json`, the
same weights, and the same input. Each stage has 64 lanes. The tuned capacities
are the elementwise maximum of the original plan's hot/cold capacities, allowing
all 64 lanes to remain in a single batched group over four cards.

## Measured result — 2026-09-21

**Different per-layer capacities compile and execute inside one QPC.** Both
precision configurations produced bit-identical tuned/uniform outputs and zero
capacity overflow on the saved input. No per-layer program loading is involved.

| Weight format | Uniform median | Tuned median | Latency reduction | Speedup |
|---|---:|---:|---:|---:|
| FP16 | 25.53965 ms | 21.31260 ms | **16.55%** | **1.1983×** |
| MXFP6 | 22.53200 ms | 20.05075 ms | **11.01%** | **1.1237×** |

These are synchronous host latencies for **two full decoder layers**, including
attention, MoE and residuals. Each median uses 300 measured inferences, split into
three rounds with 10 warm-up iterations per round. Timing graphs expose only the
final hidden state. The individual round medians were:

| Format / variant | Round 1 | Round 2 | Round 3 | P10–P90, all samples |
|---|---:|---:|---:|---:|
| FP16 uniform | 25.48850 | 25.58140 | 25.55815 | 25.26605–25.78850 ms |
| FP16 tuned | 21.36540 | 21.24855 | 21.33925 | 21.16149–21.51892 ms |
| MXFP6 uniform | 22.48425 | 22.56680 | 22.56755 | 22.11800–22.77893 ms |
| MXFP6 tuned | 20.03760 | 20.04935 | 20.06040 | 19.85828–20.26709 ms |

The actual device routing maxima (same in each variant) were:

| Format | Layer 0 stage maxima / capacities | Layer 1 stage maxima / capacities |
|---|---|---|
| FP16 | 46, 18 / 64, 32 | 101, 18 / 128, 32 |
| MXFP6 | 46, 19 / 64, 32 | 100, 18 / 128, 32 |

All 1,024 token/expert assignments per layer fit. The CPU chains also matched
exactly; the weight files have identical SHA-256 hashes between variants at both
layers. ONNX shape inference verified all six expert GEMMs per layer use the
intended row counts: layer 0 `(64,64,64,32,32,32)`, layer 1
`(128,128,128,32,32,32)`, versus all 128 for the baseline.

**Absolute accuracy remains separate from padding equivalence.** FP16 passes the
reference gate with 0.09433% relative L2 error to the independent fp32 reference
(maximum per-token relative L2: 3.0355%). MXFP6 has a shared 5.5713% relative L2
error (maximum per-token: 9.2401%) in both variants and fails the 5% reference gate.
Its latency result is explicitly an exact-equivalence comparison against the
same quantized baseline, not certification of MXFP6 accuracy. The FP16 control
isolates the larger discrepancy to the MXFP6 configuration.

Raw results:

- [FP16 validation and timing](run_2layer/fp16/run_20260921_070602/result.json)
- [MXFP6 exact-equivalence timing and accuracy failure](run_2layer/mxfp6/run_20260921_070757/result.json)
- [Original strict MXFP6 check, stopped before timing](run_2layer/mxfp6/run_20260921_070208/result.json)

Environment: SDK/runtime/firmware 1.21.6, QEfficient 1.23.0.dev0, PyTorch
2.7.0+cpu, ONNX 1.18.0, Transformers 5.5.4. All four devices were ready and idle
before the experiment and returned to zero loaded/active networks afterward.
Full 48-layer execution and held-out prompt coverage have **not** been measured
by this experiment. Halving padded expert rows did not halve total latency.

## Reproduce

From the repository root, using the original QEfficient environment:

```bash
/home/chihao/qeff-venv/bin/python tools/moe_multilayer_export.py \
  --out 9.17.2026/multilayer/run_2layer

/home/chihao/qeff-venv/bin/python tools/moe_multilayer_bench.py \
  --out 9.17.2026/multilayer/run_2layer --precision fp16 \
  --reference-npy /home/chihao/mllm/9.17.2026/e2e/ref/h_L2.npy

# MXFP6: exact padding-equivalence measurement; absolute accuracy is reported separately.
/home/chihao/qeff-venv/bin/python tools/moe_multilayer_bench.py \
  --out 9.17.2026/multilayer/run_2layer --precision mxfp6 --equivalence-only \
  --reference-npy /home/chihao/mllm/9.17.2026/e2e/ref/h_L2.npy
```

Export requires an empty output directory and never modifies the checkpoint or
original experiment files. `--model`, `--e2e`, `--input-npy`, `--start-layer`,
`--layers`, and `--plan` support other local inputs and layer intervals. A custom
plan maps layer IDs to `{"stage_widths": [C0, C1], "order": [128 expert IDs]}`.
Changing capacities requires a fresh export and compilation.

The benchmark supports `--stage compile` and `--stage run`. Each run writes a
new timestamped directory. Existing QPCs within the same export/precision directory
are reused. Use a fresh export directory when changing graph code or plans.
`--precision fp16` provides an unquantized-weight comparison; both modes use
`-convert-to-fp16` for execution.

## Validation and measurement

`moe_multilayer_export.py` reuses the existing `E2ELayer` and `MoELayerEP`, including
the compiler-recognized RMSNorm. Each layer has separate external weight files;
stitching prefixes node/tensor names, connects hidden states, and rewrites weight
paths without loading all layers' weights into memory. `onnx.checker` validates
the joined model. Per-layer `matmul_shapes.json` files record inferred GEMM shapes.

The diagnostic QPC returns the final hidden state and each layer's actual device
routing counts in permuted expert order. Overflow is the sum of
`max(count[e] - stage_capacity[e], 0)`. Nonfinite outputs, malformed counts, and
any overflow fail validation. CPU outputs of the tuned and uniform chains are
also compared during export.

Device tuned-vs-uniform acceptance thresholds are relative L2 < 0.5% and maximum
per-token relative L2 < 2%. Device-vs-exporter-CPU relative L2 must be < 5%; that
comparison includes quantization effects. Metrics are reported even when exact
equality holds. These are experiment checks, not general model-accuracy guarantees.
`--reference-npy` additionally compares both device outputs to the original,
independent fp32 decoder reference with the same 5% relative L2 threshold.

The initial MXFP6 run failed the unchanged absolute reference gate: both variants
had 5.5713% relative L2 error to the independent fp32 reference (5.6480% to the
exporter CPU reference). Their outputs were nevertheless bit-for-bit identical,
with zero overflow. The FP16 control reduced the independent-reference error to
0.0943%, tying the larger discrepancy to the MXFP6 configuration. The original
failure remains in `mxfp6/run_20260921_070208/result.json`.

`--equivalence-only` is an explicit, stricter baseline-equivalence mode for this
situation: tuned/uniform outputs must match exactly, overflow must be zero, and
every timing output must exactly match its diagnostic output. It allows timing
while preserving `reference_accuracy_pass: false` and `validation_pass: false`
when the absolute reference gate fails. It does not certify MXFP6 model accuracy.

Timing QPCs expose only the final hidden state, removing routing diagnostics
from timed inference. Their outputs are checked against the diagnostic QPCs.
The C++ runner loads and activates one QPC once, warms up for 10 iterations, then
measures 100 synchronous inferences. Timing includes input/output transfers,
`setData`, enqueue and wait; it excludes loading, activation and teardown. Three
rounds alternate variant order. Reported latency is the median of 300 samples.

Outputs are local and ignored by git: `info.json`, `mdp.json`, per-layer ONNX and
weights, QPCs, compile/run logs, actual device counts, raw latency samples, and
`result.json`. Each precision's `latest_run.txt` points to its last completed
measurement; inspect accuracy and timing scope fields before interpreting it.

## Scope

This proves fixed per-layer capacities can coexist within one executable.
It does not implement prompt-dependent capacity selection or overflow recovery.
The saved prompt is part of the earlier experiment and is not a held-out test.
The graph contains decoder layers only: no token embedding, final norm/lm-head,
or retained KV interface. Two-layer timing is not full-model time to first token.
Use separate held-out prompts and a full-model export before making deployment
or end-to-end speedup claims.

For access on this server, `tools/moe_grant_access.sh` grants the requested user's
device access and read/execute access to the specific existing assets; it requires
an administrator to run it with sudo. It prints an ACL backup/restore location.
