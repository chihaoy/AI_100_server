#!/bin/bash
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
D=/home/chihao/models/qwen3_30b_a3b/4card
E=/home/chihao/models/qwen3_30b_a3b/export-126f04c76d4a0568
rm -rf $D/qpc_fp16
echo "=== 4-card TP compile, NO mxfp6 (权重保持 fp16,便于字节比对) ==="
time qaic-compile -aic-hw -aic-hw-version=2.0 -m="$E/Qwen3MoeForCausalLM.onnx" \
  -retained-state -convert-to-fp16 \
  -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
  -network-specialization-config="$D/spec_decode.json" \
  -custom-IO-list-file="$D/custom_io.yaml" \
  -mdp-load-partition-config="$D/mdp_ts_4.json" \
  -compile-only -aic-binary-dir="$D/qpc_fp16"
echo "rc=$?"
[ -f "$D/qpc_fp16/programqpc.bin" ] && { echo "QPC_OK"; du -h $D/qpc_fp16/programqpc.bin; } || echo "QPC_MISSING"
