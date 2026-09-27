#!/bin/bash
# E32: end-to-end profile of the best 512-token program (48 layers, runtime sort 512/64, 1024 KiB tiles, Einsum final sum,
# KV retained), instrumented (-stats-level=70); analysis with detail_analyze, layer_timeline and e32_breakdown
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; P=$F/profile_e32_T512
q=$F/qpc512_full_dyncard_ssg1024_s70
echo "df before: $(df -h $F | tail -1 | awk '{print $4}')"
if [ ! -f $q/programqpc.bin ]; then rm -rf $q; t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/full512_dyncard_hp/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_T512.json -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$q -mdts-mos=1 -size-split-granularity=1024 -stats-level=70 > $q.log 2>&1
  echo "### compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $q | cut -f1)"; fi
[ -f $q/programqpc.bin ] || { grep -vE '^\s*$' $q.log | sed -n 3,10p; exit 1; }
rm -rf $P; mkdir -p $P/stats $P/outputs $P/trace; $PY $M/scripts/trunc_io.py $q $P $F/ref512 > /dev/null
for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && break; sleep 10; done
t0=$(date +%s); /opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $P/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $P/stats > $P/runner.log 2>&1
echo "profile rc=$? $(( $(date +%s)-t0 )) s: $(grep ExecTimeUs_Dev $P/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"
t0=$(date +%s); /opt/qti-aic/exec/qaic-opstats --qpc $q/programqpc.bin --input-dir $P/stats --output-dir $P/trace --summary --trace --merge-mq-traces true --flow-events full > $P/opstats.log 2>&1; echo "opstats rc=$? $(( $(date +%s)-t0 )) s, trace $(du -sh $P/trace | cut -f1)"
/opt/qti-aic/tools/qaic-qpc extract --qpc $q/programqpc.bin --output-dir $P/meta -s '*opstatsdesc.bin' > /dev/null 2>&1
t=$(ls $P/trace/*merged*.trace.json | sort | head -1); t0=$(date +%s); $PY $M/scripts/detail_analyze.py $t $P/meta "e32 T512 full model" $P/analysis/sample0 > $P/analysis_sample0.log 2>&1; echo "analysis rc=$? $(( $(date +%s)-t0 )) s"; head -n 1 $P/analysis_sample0.log | cut -c1-250
$PY $M/scripts/layer_timeline.py $P/analysis/sample0 "e32 T512" > $P/layer_timeline.txt 2>&1; head -n 4 $P/layer_timeline.txt
$PY $M/scripts/e32_breakdown.py $P/analysis/sample0 "e32 T512 full model" > $P/breakdown.txt 2>&1; cat $P/breakdown.txt
echo "df after: $(df -h $F | tail -1 | awk '{print $4}')"
echo E32_DONE
