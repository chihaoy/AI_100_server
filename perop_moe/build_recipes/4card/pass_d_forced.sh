#!/bin/bash
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
D=/home/chihao/models/qwen3_30b_a3b/4card
F=/home/chihao/models/qwen3_30b_a3b/export_forced
rm -rf $D/qpc_forced
echo "=== 4-card TP compile: FORCED ROUTING (router weights = 0) ==="
time qaic-compile -aic-hw -aic-hw-version=2.0 -m="$F/Qwen3MoeForCausalLM.onnx" \
  -retained-state -convert-to-fp16 -mxfp6-matmul \
  -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
  -network-specialization-config="$D/spec_decode.json" \
  -custom-IO-list-file="$D/custom_io.yaml" \
  -mdp-load-partition-config="$D/mdp_ts_4.json" \
  -stats-level=70 -compile-only -aic-binary-dir="$D/qpc_forced"
echo "rc=$?"
[ -f "$D/qpc_forced/programqpc.bin" ] && { echo "QPC_OK"; du -h $D/qpc_forced/programqpc.bin; } || echo "QPC_MISSING"
