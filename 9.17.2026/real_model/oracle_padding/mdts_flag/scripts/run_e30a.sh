#!/bin/bash
# E30a: how large is the MoE cross-card reduction at T=512? Instrumented two-layer profile of the 512-token runtime sort
# (1024 KiB tiles); P2P per node from detail_analyze, per-layer critical path from layer_timeline
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest
name=t512_2_dyncard_ssg1024; out=$K/qpc_${name}_s70; rm -rf $out; t0=$(date +%s)
/opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/trunc2_512_dyncard/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_T512.json -custom-IO-list-file=$F/trunc2_512_dyncard/custom_io.yaml -compile-only -aic-binary-dir=$out -stats-level=70 -mdts-mos=1 -size-split-granularity=1024 > $K/compile_${name}_s70.log 2>&1
echo "### $name s70 compile rc=$? $(( $(date +%s)-t0 )) s"
d=$K/prof_$name; rm -rf $d; mkdir -p $d/stats $d/outputs $d/trace; $PY $M/scripts/trunc_io.py $out $d $F/ref512 > /dev/null
for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && break; sleep 10; done
/opt/qti-aic/exec/qaic-runner -t $out -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats > $d/runner.log 2>&1
echo "profile rc=$?: $(grep ExecTimeUs_Dev $d/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"
/opt/qti-aic/exec/qaic-opstats --qpc $out/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
/opt/qti-aic/tools/qaic-qpc extract --qpc $out/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' > /dev/null 2>&1
t=$(ls $d/trace/*merged*.trace.json | sort | head -1); $PY $M/scripts/detail_analyze.py $t $d/meta "$name" $d/analysis/sample0 > $d/analysis_sample0.log 2>&1; head -1 $d/analysis_sample0.log
$PY $M/scripts/layer_timeline.py $d/analysis/sample0 "$name"
rm -rf $out
echo E30A_DONE
