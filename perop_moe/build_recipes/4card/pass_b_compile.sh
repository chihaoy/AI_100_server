#!/bin/bash
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
D=/home/chihao/models/qwen3_30b_a3b/4card
E=/home/chihao/models/qwen3_30b_a3b/export-126f04c76d4a0568
REG=${1:-decode}; SPEC=$D/spec_${REG}.json
rm -rf $D/qpc_$REG
echo "=== 4-card TP compile: $REG  ($(date +%H:%M:%S)) ==="
time qaic-compile -aic-hw -aic-hw-version=2.0 -m="$E/Qwen3MoeForCausalLM.onnx" \
  -retained-state -convert-to-fp16 -mxfp6-matmul \
  -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
  -network-specialization-config="$SPEC" \
  -custom-IO-list-file="$D/custom_io.yaml" \
  -mdp-load-partition-config="$D/mdp_ts_4.json" \
  -stats-level=70 -compile-only -aic-binary-dir="$D/qpc_$REG"
rc=$?
echo "qaic-compile rc=$rc"
[ -f "$D/qpc_$REG/programqpc.bin" ] && { echo "QPC_OK"; du -h $D/qpc_$REG/programqpc.bin; } || echo "QPC_MISSING"
