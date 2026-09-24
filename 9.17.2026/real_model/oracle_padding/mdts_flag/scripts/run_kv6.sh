#!/bin/bash
# KV round 6: full-model retained-state programs — wait for each compile, check bindings, run the baseline tool (3 rounds x 20)
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server; M=$R/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; echo "devices busy, waiting"; sleep 20; done; }
waitfor() { for i in $(seq 720); do [ -f $F/compile_$1.status ] && { cat $F/compile_$1.status; return; }; sleep 30; done; echo "timeout waiting for $1"; }
runit() { local name=$1 qpc=$2; [ -f $qpc/programqpc.bin ] || { echo "=== run $name skipped (no qpc)"; return; }; $PY $M/scripts/qpc_bindings.py $qpc 2>&1 | grep '###' | cut -c1-200
  rm -rf $F/run_$name $F/run_$name.log; idle; cd $R && $PY tools/moe_qwen3_baseline.py run --out $F/run_$name --precision mxfp6 --qpc $qpc --reference $F/ref --routing-counts > $F/run_$name.stdout 2>&1; echo "=== run $name rc=$?"; $PY $S/result_line.py $F/run_$name/result.json $name 2>/dev/null || tail -3 $F/run_$name.stdout | cut -c1-300; }
waitfor native_hp_flag;  runit native_hp_flag $F/qpc_native_hp_flag
waitfor native_noflag;   runit anchor3_native_noflag $F/qpc_native_noflag
waitfor stack_hp_flag;   runit stack_hp_flag $F/qpc_stack_hp_flag
echo KV6_DONE
