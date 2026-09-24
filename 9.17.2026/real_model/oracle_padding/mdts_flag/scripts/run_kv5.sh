#!/bin/bash
# KV round 5: instrumented heuristic (no flag) 2-layer program for the attention-span comparison
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
F=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag/full_model; C=$F/configs; K=$F/kvtest
out=$K/qpc_trunc2_noflag_s70; rm -rf $out; t0=$(date +%s)
/opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/trunc2_native/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -custom-IO-list-file=$F/trunc2_native/custom_io.yaml -compile-only -aic-binary-dir=$out -stats-level=70 > $K/compile_trunc2_noflag_s70.log 2>&1; echo "### noflag s70 rc=$? $(( $(date +%s)-t0 )) s"
idle() { for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; echo "devices busy, waiting"; sleep 20; done; }
name=trunc2_noflag; d=$K/prof_$name; rm -rf $d; mkdir -p $d/stats $d/outputs $d/trace; $PY $S/kv/trunc_io.py $K/qpc_${name}_s70 $d > /dev/null; idle
/opt/qti-aic/exec/qaic-runner -t $K/qpc_${name}_s70 -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats --write-output-start-iter 2 --write-output-num-samples 3 --write-output-dir $d/outputs > $d/runner.log 2>&1
echo "profile $name rc=$?: $(grep ExecTimeUs_Dev $d/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"
/opt/qti-aic/exec/qaic-opstats --qpc $K/qpc_${name}_s70/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
/opt/qti-aic/tools/qaic-qpc extract --qpc $K/qpc_${name}_s70/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' >/dev/null 2>&1
t=$(ls $d/trace/*merged*.trace.json | sort | head -1); $PY $S/detail_analyze.py $t $d/meta "trunc2 $name" $d/analysis/sample0 > $d/analysis_sample0.log 2>&1; head -1 $d/analysis_sample0.log
$PY $S/kv/spans.py $d/analysis/sample0 "trunc2 noflag (heuristic)"
echo KV5_DONE
