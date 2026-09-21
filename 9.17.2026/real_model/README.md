# Real Qwen3-30B-A3B full-model prefill baseline

Date: 2026-09-21. SDK/runtime/firmware 1.21.6; cards 0–3.

This stage establishes the **full-model baseline only**, following the user's
direction. It uses the trained Qwen3-30B-A3B checkpoint and all 48 decoder
layers, including embedding, attention, MoE, final normalization and the LM
head. Adaptive padding has not been integrated into this model run.

The subsequent [static oracle padding experiment](oracle_padding/README.md)
uses this baseline and adds per-layer capacities and expert regrouping, as
separately authorized after the baseline stage.

The graph is the existing QEfficient expert-parallel export, with one partition
across four cards, 16 cores per card, and conservative capacity 128 for every
expert. Each layer has 128 experts and top-8 routing. Batch size is 1, input
length is 128, and the compiled context length is 256. The input is the first
128 tokens of the saved GSM8K prompt 41 chat input. It is a prompt prefix,
not a complete generated answer.

## Results

**Both full-model programs completed 60 timed prefills each**, with finite,
bit-identical logits on every warmup and measured invocation. All four cards
returned to their pre-load memory and core availability after both runs.

| Weight format | Median prefill ms | P10–P90 ms | Input tokens/s | Active device GiB, four-card total |
|---|---:|---:|---:|---:|
| FP16 | 572.223 | 570.570–573.682 | 223.69 | 60.748 |
| MXFP6 | 525.103 | 522.560–528.344 | 243.76 | 27.090 |

| Weight format | Last-token logits relative L2 vs FP32 | Logit correlation | Next token matches FP32 | 5% diagnostic gate |
|---|---:|---:|---|---|
| FP16 | 5.0293% | 0.999708 | Yes | Fail |
| MXFP6 | 18.8420% | 0.991660 | Yes | Fail |

**Neither precision passes the existing 5% diagnostic reference gate.** FP16
narrowly misses it at 5.0293%; MXFP6 has a substantially larger 18.8420% error.
Both predict token 1258 (` jav`), which continues the truncated input. This is
not a generated answer or a task-accuracy score. The threshold has not been
relaxed. Remaining FP16 error and broader model accuracy require further study.

MXFP6 logits are bit-identical to the historical full-model run. Compared with
FP16, MXFP6 reduced median prefill latency by 8.23%, with the larger reference error shown
above. This comparison establishes a conservative baseline, not an adaptive
padding speedup.

| Format | Round medians, ms | Setup, s | Active GiB on cards 0/1/2/3 |
|---|---|---:|---|
| FP16 | 572.354, 572.235, 572.054 | 17.107 | 15.204/15.173/15.186/15.186 |
| MXFP6 | 524.904, 525.466, 525.014 | 7.935 | 6.778/6.776/6.760/6.776 |

The FP16 compile succeeded in 910.091 s (15.17 minutes).
Its QPC is 58.862 GiB; the existing MXFP6 QPC is 24.696 GiB.
Package file size and active device allocation are distinct measurements.

The initial MXFP6 smoke run used the older C++ driver and measured 529.13 ms
over 10 samples. It is retained as a smoke check; the tables use the subsequent
60-sample benchmark. MXFP6 timing was collected while the FP16 compiler ran on
host CPU; FP16 timing followed compilation. The medians are descriptive results
for this server/session, rather than a randomized precision comparison.

## Measurement and correctness

The benchmark keeps one complete model resident, and runs three rounds of
20 measured prefills, each preceded by two warmups. C++ host latency includes
buffer binding, enqueue and completion wait. Inputs are prepared before the
timing loop. QPC loading, program creation, activation, output validation and
file writes are outside the timed interval. Setup phase durations are retained
separately.

Every invocation checks that all 151,936 last-token logits are finite and
bit-identical to the first invocation. Input positions are reset to 0–127 for
each invocation. This exercises repeated first-chunk execution; it does not
establish multi-chunk KV continuity or decode correctness.

Saved outputs are compared with the existing explicit FP32 full-model reference
from `tools/moe_e2e_cpu_ref.py`. The report records relative L2 error, maximum
absolute error, logit correlation, next-token agreement and top-5 IDs. The 5%
relative-L2 threshold is a diagnostic reference gate inherited from the earlier
experiments, not a model-quality benchmark. MXFP6 and FP16 results are reported
separately.

Device memory is sampled before loading, after loading, after activation, after
each round and after cleanup. Active memory is total device allocation relative
to the pre-load snapshot, not just expert weights or workspace. The runner
verifies stable allocations through all rounds and release after completion.

The SDK's C API v1 buffer mapping labels this QPC's output FP16 while its
serialized IO descriptor and external buffer are FP32. The Python runner checks
the serialized descriptor; the C++ host checks the exact buffer names and sizes.
The preliminary host run that rejected the v1 type label performed no timed
inferences; its local diagnostic artifacts are retained separately.

## Reproduce

From the repository root, use `/home/chihao/qeff-venv/bin/python`. The checkpoint
is `/home/chihao/models/qwen3_30b_a3b/hf`. The default QPC is the existing MXFP6
EP prefill program; the default reference is
`/home/chihao/mllm/9.17.2026/e2e/ref`.

```bash
/home/chihao/qeff-venv/bin/python tools/moe_qwen3_baseline.py run \
  --out 9.17.2026/real_model/new_mxfp6_run --precision mxfp6

/home/chihao/qeff-venv/bin/python tools/moe_qwen3_baseline.py run \
  --out 9.17.2026/real_model/new_fp16_run --precision fp16 \
  --qpc 9.17.2026/real_model/fp16_compile/qpc
```

Each output path must be new. `analyze` instead of `run` rechecks saved results
without using the devices. The wrapper builds `moe_qwen3_baseline_host.cpp`
with `-O2 -std=c++17 -Wall -Wextra -Werror` against the installed QAic SDK.

The FP16 control compiles the **same existing ONNX and specialization** as the
MXFP6 QPC, with `-convert-to-fp16` and without `-mxfp6-matmul`:

```bash
QWEN_EP_EXPORT=/home/chihao/models/qwen3_30b_a3b/ep/qeff_cache/Qwen3MoeForCausalLM/Qwen3MoeForCausalLM-ed085dabb01c623b
QWEN_EP_CONFIG="$QWEN_EP_EXPORT/qpc-6ff7c7675b1fcae0"
/opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 \
  -m="$QWEN_EP_EXPORT/Qwen3MoeForCausalLM.onnx" \
  -retained-state -convert-to-fp16 -aic-num-cores=16 -mos=1 \
  -aic-enable-depth-first \
  -mdp-load-partition-config="$QWEN_EP_CONFIG/mdp_ts_4.json" \
  -network-specialization-config="$QWEN_EP_CONFIG/specializations_flat.json" \
  -custom-IO-list-file="$QWEN_EP_CONFIG/custom_io.yaml" \
  -compile-only -aic-binary-dir=9.17.2026/real_model/new_fp16_compile/qpc
```

Generated artifacts remain local and ignored by Git:

- `baseline_smoke/`: original host, smoke timing and reference comparison.
- `baseline_mxfp6/`: preliminary IO mapping diagnostic, not a timing result.
- `baseline_mxfp6_checked/`: repeated MXFP6 samples, logits, resources and summary.
- `fp16_compile/`: exact compile command, log, completion summary and QPC.
- `baseline_fp16/`: repeated FP16 samples, logits, resources and summary.

The next algorithm stage can use these as conservative full-model controls.
Trained-weight adaptive dispatch, multiple prompts/chunks, generation and
application-level accuracy remain separate experiments.
