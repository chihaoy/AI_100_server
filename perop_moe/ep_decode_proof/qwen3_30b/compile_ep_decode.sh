#!/bin/bash
# Manual decode-only compiles (seq_len=1) of the prefill_only=False EP export, B=8 then B=1.
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
M=/home/chihao/models/qwen3_30b_a3b/ep/Qwen3MoeForCausalLM/Qwen3MoeForCausalLM-40c66dadb1e495df
Q=$M/qpc-512f2be6fd357576
[ -f $M/Qwen3MoeForCausalLM.onnx ] && [ -f $Q/custom_io.yaml ] && [ -f $Q/mdp_ts_4.json ] || { echo "ABORT: missing export artifacts"; exit 1; }
for B in 8 1; do
  echo "=== manual decode-only compile B=$B ($(date +%H:%M:%S)) ==="
  echo "{\"specializations\": [{\"batch_size\": \"$B\", \"ctx_len\": \"256\", \"seq_len\": \"1\"}]}" > $Q/spec_decode_B$B.json
  rm -rf $Q/qpc_decode_B$B
  /usr/bin/time -f "compile_wall_s=%e maxrss_kb=%M" qaic-compile -aic-hw -aic-hw-version=ai100 -m=$M/Qwen3MoeForCausalLM.onnx \
    -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
    -mdp-load-partition-config=$Q/mdp_ts_4.json -network-specialization-config=$Q/spec_decode_B$B.json \
    -custom-IO-list-file=$Q/custom_io.yaml -stats-level=70 -compile-only -aic-binary-dir=$Q/qpc_decode_B$B \
    > $Q/compile_decode_B$B.log 2>&1
  echo "compile B=$B rc=$? $( [ -f $Q/qpc_decode_B$B/programqpc.bin ] && echo QPC_OK || echo QPC_MISSING) ($(date +%H:%M:%S))"; tail -2 $Q/compile_decode_B$B.log
done
echo PIPELINE_DONE
