#!/bin/bash
# E26c: why the runtime sort loses end to end: two-layer instrumented profiles, weight-stream timing vs router
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$R/full_model; C=$F/configs; K=$F/kvtest
comp70() { local name=$1 graph=$2; local out=$K/qpc_${name}_s70; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -stats-level=70 -mdts-mos=1 > $K/compile_${name}_s70.log 2>&1; echo "### $name s70 rc=$? $(( $(date +%s)-t0 )) s"; }
comp70 trunc2_naive_to trunc2_naive & comp70 trunc2_dyncard trunc2_dyncard & wait
for name in trunc2_naive_to trunc2_dyncard; do d=$K/prof_$name; rm -rf $d; mkdir -p $d/stats $d/outputs $d/trace; $PY $S/kv/trunc_io.py $K/qpc_${name}_s70 $d > /dev/null
  for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && break; sleep 10; done
  /opt/qti-aic/exec/qaic-runner -t $K/qpc_${name}_s70 -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats > $d/runner.log 2>&1
  echo "profile $name rc=$?: $(grep ExecTimeUs_Dev $d/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"
  /opt/qti-aic/exec/qaic-opstats --qpc $K/qpc_${name}_s70/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
  /opt/qti-aic/tools/qaic-qpc extract --qpc $K/qpc_${name}_s70/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' >/dev/null 2>&1
  t=$(ls $d/trace/*merged*.trace.json | sort | head -1); $PY $S/detail_analyze.py $t $d/meta "$name" $d/analysis/sample0 > $d/analysis_sample0.log 2>&1; head -1 $d/analysis_sample0.log
  $PY $S/e26/layer_timeline.py $d/analysis/sample0 "$name"; done
echo E26C_DONE
