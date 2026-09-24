#!/bin/bash
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
Q=/home/chihao/models/qwen3_30b_a3b/ep/qeff_cache/Qwen3MoeForCausalLM/Qwen3MoeForCausalLM-ed085dabb01c623b/qpc-6ff7c7675b1fcae0
O=/home/chihao/models/qwen3_30b_a3b/ep/qeff_cache/Qwen3MoeForCausalLM/Qwen3MoeForCausalLM-ed085dabb01c623b
rm -rf $Q/qpc
echo "=== EP 4 卡编译 (spec 已拍平以兼容 SDK 1.21.6) ==="
time qaic-compile -aic-hw -aic-hw-version=ai100 \
  -m=$O/Qwen3MoeForCausalLM.onnx \
  -retained-state -convert-to-fp16 -mxfp6-matmul \
  -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
  -mdp-load-partition-config=$Q/mdp_ts_4.json \
  -network-specialization-config=$Q/specializations_flat.json \
  -custom-IO-list-file=$Q/custom_io.yaml \
  -stats-level=70 \
  -aic-binary-dir=$Q/qpc
echo "rc=$?"
[ -f "$Q/qpc/programqpc.bin" ] && { echo QPC_OK; du -h $Q/qpc/programqpc.bin; ls $Q/qpc | head; } || echo QPC_MISSING
