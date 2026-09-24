#!/bin/bash
# KV round 7: full-model profiling round on the retained-state head-parallel stack (stats70), after the baseline runs are done
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
F=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag/full_model; P=$F/profile_stack_hp_flag
for i in $(seq 720); do [ -f $F/compile_stack_hp_flag_s70.status ] && break; sleep 30; done; cat $F/compile_stack_hp_flag_s70.status
for i in $(seq 720); do grep -q KV6_DONE $S/run_kv6.out 2>/dev/null && break; sleep 30; done
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; echo "devices busy, waiting"; sleep 20; done; }
rm -rf $P; mkdir -p $P/stats $P/outputs $P/trace; $PY $S/kv/trunc_io.py $F/qpc_stack_hp_flag_s70 $P; idle
/opt/qti-aic/exec/qaic-runner -t $F/qpc_stack_hp_flag_s70 -D 0:1:2:3 --aic-batch-json-input $P/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $P/stats --write-output-start-iter 2 --write-output-num-samples 3 --write-output-dir $P/outputs > $P/runner.log 2>&1
echo "profile full stack_hp rc=$?: $(grep ExecTimeUs_Dev $P/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"
t0=$(date +%s); /opt/qti-aic/exec/qaic-opstats --qpc $F/qpc_stack_hp_flag_s70/programqpc.bin --input-dir $P/stats --output-dir $P/trace --summary --trace --merge-mq-traces true --flow-events full > $P/opstats.log 2>&1; echo "opstats rc=$? $(( $(date +%s)-t0 )) s; traces: $(ls -la $P/trace/*merged*.trace.json 2>/dev/null | awk '{print $5}' | tr '\n' ' ')"
/opt/qti-aic/tools/qaic-qpc extract --qpc $F/qpc_stack_hp_flag_s70/programqpc.bin --output-dir $P/meta -s '*opstatsdesc.bin' >/dev/null 2>&1
t=$(ls $P/trace/*merged*.trace.json | sort | head -1); t0=$(date +%s); $PY $S/detail_analyze.py $t $P/meta "full stack hp flag" $P/analysis/sample0 > $P/analysis_sample0.log 2>&1; echo "analysis rc=$? $(( $(date +%s)-t0 )) s"; head -1 $P/analysis_sample0.log
$PY $S/kv/spans.py $P/analysis/sample0 "full stack_hp_flag" | head -60
$PY $S/kv/placement.py $P/analysis/sample0 "" | head -50
echo KV7_DONE
