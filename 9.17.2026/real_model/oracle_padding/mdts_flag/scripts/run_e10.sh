#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server; O=$R/9.17.2026/real_model/oracle_padding; F=$O/mdts_flag/full_model; C=$F/configs
idle() { for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; echo "devices busy, waiting"; sleep 20; done; }
compkv() { local name=$1 graph=$2 extra=$3; local out=$F/qpc_$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -convert-to-fp16 -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -compile-only -aic-binary-dir=$out $extra > $F/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $F/compile_$name.log | sed -n 5,10p | cut -c1-160; }
runit() { local name=$1 qpc=$2 counts=$3; [ -f $qpc/programqpc.bin ] || { echo "=== run $name skipped (no qpc)"; return; }; rm -rf $F/run_$name $F/run_$name.log; idle
  cd $R && $PY tools/moe_qwen3_baseline.py run --out $F/run_$name --precision mxfp6 --qpc $qpc $counts > $F/run_$name.stdout 2>&1; echo "=== run $name rc=$?"; $PY $S/result_line.py $F/run_$name/result.json $name || tail -3 $F/run_$name.stdout | cut -c1-300; }
echo "===== E10: waiting for graph builds ====="; until grep -q STACK_BUILDS_DONE $S/build_stack.log 2>/dev/null; do grep -qiE 'Traceback|Error' $S/build_stack.log 2>/dev/null && { echo "BUILD FAILED"; tail -3 $S/build_stack.log; exit 1; }; sleep 15; done; grep -E 'stack_hc' $S/build_stack.log | cut -c1-250
echo "===== E10: full-model MXFP6 + flag, KV-free: anchor and stack variants ====="
compkv kvfree_native_flag native_c128_kvfree "-mxfp6-matmul -mdts-mos=1"; runit anchor_native_flag $F/qpc_kvfree_native_flag --routing-counts
compkv stack_native_rw    stack_native_rw    "-mxfp6-matmul -mdts-mos=1"; runit stack_native_rw $F/qpc_stack_native_rw --routing-counts
compkv stack_hc_oracle_rw stack_hc_oracle_rw "-mxfp6-matmul -mdts-mos=1"; runit stack_hc_oracle_rw $F/qpc_stack_hc_oracle_rw --routing-counts
compkv stack_hc_128_rw    stack_hc_128_rw    "-mxfp6-matmul -mdts-mos=1"; runit stack_hc_128_rw $F/qpc_stack_hc_128_rw --routing-counts
echo E10_DONE
