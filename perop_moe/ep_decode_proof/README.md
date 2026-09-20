# EP-in-decode proof (tiny Qwen3-MoE, 2026-09-15)

Tiny random Qwen3-MoE (2 layers, 8 experts, top-2, H=256) pushed through the real QEfficient 1.23 path with
`prefill_only=False`, `batch_size=8`, `num_devices=4`, `num_cores=16`, and
`qaic_config={"moe_config": {"flavour": "expert_parallel", ...}}` passed to BOTH `from_pretrained` and `compile`.
Script: `tools/moe_tiny_ep_decode_proof.py`. QPC compiled manually with `spec_decode_B8.json` (batch 8, ctx 64, seq_len 1).

* `qefficient_specializations.json` — what QEfficient itself wrote for this build: one spec, "Decode", seq_len=1, batch 8.
* `opstats_slice00_decode_B8.summary.txt` — on-card op signature of that seq_len=1 QPC: `cumsum`, `aicbatchedreduceadd`,
  `aicscatternd`, per-lane `blockdequantize_mxfp6`; only sub-KB fp16 activation gathers, no `aicgather (uindex8)` weight gather.
* `verify_decode_B8.py` + `logits_decode_B8.bin` — 8 real tokens at position 0, batch 8: QPC logits vs PyTorch fp32,
  per-token corr 0.996-0.998, cross-token control corr ~0. The 8 tokens' top-2 experts cover all 8 experts (all 8 lanes,
  all 4 cards busy). Latency 0.59 ms/step (run_decode_B8.log).

Without `qaic_config` on `compile()` the same build silently comes out as decode_bmm (weights [E,H,I], no packing ops):
the compile-time transform (`base/modeling_qeff.py:985`) re-selects the flavour from the compile-time qaic_config and
`select_moe_flavour` (`flavours.py:310`) defaults to DECODE_BMM whenever `prefill_only=False`.

## Qwen3-30B-A3B full-scale (subdir `qwen3_30b/`, 2026-09-15)

Same three checks on the real model (export `prefill_only=False`, `batch_size=8`, EP flavour on both calls; manual decode-only
compile batch 8 / ctx 256 / seq_len 1; 24.6 GB QPC, 4 cards):
* QEfficient's own spec for the build: one entry "Decode", batch 8, seq_len 1 (`qefficient_specializations.json`).
* opstats card0 core0: 96 `cumsum`, 96x4 `aicbatchedreduceadd`, 937 MB `blockdequantize_mxfp6` (all 128 experts every step),
  no `aicgather (uindex8)` weight gather. 118.3 ms/step at batch 8 (14.8 ms/token) vs TP decode 14.3 ms/token at batch 1.
* logits, token 50744: EP QPC (batch 8, row 0) vs the 1.22 TP decode QPC (batch 1) corr 1.0000, same top-1, max|diff| 0.06.
  Both QPCs vs HF bf16 CPU: corr 0.848 each (shared MXFP6-graph deviation, not an EP effect). Rows are independent and runs deterministic.
Files: `ep_50744.npy` (EP, tokens [50744,9707,17,250,511,777,3,999]), `tp_50744.npy`, `ep_mixA.npy` (tokens [9707,17,250,511,777,3,999,64]),
`cpu_ref_50744_9707.npy`, `cpu_ref_B8.npy`, scripts `run_ep_decode.py`, `build_ep_decode.sh`, `compile_ep_decode.sh`.
* batch-1 decode-only EP QPC (`qpc_decode_B1`, same ONNX, spec batch 1): 86.4 ms/step vs TP decode 14.3 ms (6.0x slower);
  logits on token 50744 identical to the batch-8 row (max|diff| 0.000) and to TP (0.059). Files `run_decode_B1.log`, `ep_B1_50744.npy`.
