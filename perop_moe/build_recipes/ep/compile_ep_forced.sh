#!/bin/bash
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
rm -rf /home/chihao/models/qwen3_30b_a3b/ep_forced/qpc
time qaic-compile -aic-hw -aic-hw-version=ai100 -m=/home/chihao/models/qwen3_30b_a3b/ep_forced/Qwen3MoeForCausalLM.onnx \
  -retained-state -convert-to-fp16 -mxfp6-matmul \
  -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
  -mdp-load-partition-config=/home/chihao/models/qwen3_30b_a3b/ep/qeff_cache/Qwen3MoeForCausalLM/Qwen3MoeForCausalLM-ed085dabb01c623b/qpc-6ff7c7675b1fcae0/mdp_ts_4.json \
  -network-specialization-config=/home/chihao/models/qwen3_30b_a3b/ep/qeff_cache/Qwen3MoeForCausalLM/Qwen3MoeForCausalLM-ed085dabb01c623b/qpc-6ff7c7675b1fcae0/specializations_flat.json \
  -custom-IO-list-file=/home/chihao/models/qwen3_30b_a3b/ep/qeff_cache/Qwen3MoeForCausalLM/Qwen3MoeForCausalLM-ed085dabb01c623b/qpc-6ff7c7675b1fcae0/custom_io.yaml \
  -stats-level=70 -aic-binary-dir=/home/chihao/models/qwen3_30b_a3b/ep_forced/qpc
echo "rc=$?"
[ -f "/home/chihao/models/qwen3_30b_a3b/ep_forced/qpc/programqpc.bin" ] && { echo QPC_OK; du -h /home/chihao/models/qwen3_30b_a3b/ep_forced/qpc/programqpc.bin; } || echo QPC_MISSING
