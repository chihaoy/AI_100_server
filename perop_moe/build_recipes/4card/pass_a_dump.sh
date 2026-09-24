#!/bin/bash
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
D=/home/chihao/models/qwen3_30b_a3b/4card
E=/home/chihao/models/qwen3_30b_a3b/export-126f04c76d4a0568
rm -f $D/moe_mdp_dump.json; rm -rf $D/throwaway_dump
# watchdog: kill the compile as soon as the dump lands (we only want the partition json)
( for i in $(seq 1 900); do
    [ -s $D/moe_mdp_dump.json ] && { sleep 3; echo "[watchdog] dump appeared -> killing compile"; pkill -f "mdp-dump-partition-config"; exit 0; }
    sleep 2
  done ) &
WD=$!
qaic-compile -aic-hw -aic-hw-version=2.0 -m="$E/Qwen3MoeForCausalLM.onnx" \
  -retained-state -convert-to-fp16 -mxfp6-matmul \
  -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
  -network-specialization-config="$D/spec_decode.json" \
  -custom-IO-list-file="$D/custom_io.yaml" \
  -mdp-dump-partition-config="$D/moe_mdp_dump.json" \
  -aic-binary-dir="$D/throwaway_dump"
echo "qaic-compile rc=$?"
kill $WD 2>/dev/null
ls -la $D/moe_mdp_dump.json 2>/dev/null || echo "NO DUMP PRODUCED"
