#!/bin/bash
# E44: end-to-end instrumented profile of the final RankTier program at T=512 (48 layers, tiers 8x512 + 8x128 + 16x64,
# token-owned combine, elementwise final sum, head-parallel attention, KV retained, MXFP6, 1024 KiB tiles: the E38 E_1024
# program, 355.8 ms uninstrumented) compiled with -stats-level=70; one profiling run (3 samples), decoded with qaic-opstats,
# analyzed with detail_analyze.py and e44_layers.py (every layer, every SoC). The 88 GiB program and the raw traces are
# deleted after the analysis; work.csv, the summaries and the per-layer tables are kept. Usage: run_e44.sh
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs
P=$F/profile_e44_T512; q=$F/qpc512_tier3_ssg1024_s70; G=$F/full512_tier3_tileadd_hp
free=$(df -B1G $F | tail -1 | awk '{print $4}'); echo "$(date +%F_%T) df before: $free GiB"
[ -f $q/programqpc.bin ] || [ $free -ge 200 ] || { echo "stop: $free GiB free, 200 needed (88 program + ~12 traces + 100 kept free)"; exit 1; }
if [ ! -f $q/programqpc.bin ]; then rm -rf $q; t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$G/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_T512.json -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$q -mdts-mos=1 -size-split-granularity=1024 -stats-level=70 > $q.log 2>&1
  echo "$(date +%F_%T) compile rc=$? $(( ($(date +%s)-t0)/60 )) min, program $(du -sh $q | cut -f1)"; fi
[ -f $q/programqpc.bin ] || { grep -vE '^\s*$' $q.log | sed -n 3,12p; exit 1; }
rm -rf $P; mkdir -p $P/stats $P/trace; $PY $M/scripts/trunc_io.py $q $P $F/ref512 > /dev/null
for i in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && break; sleep 10; done
t0=$(date +%s); /opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $P/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $P/stats > $P/runner.log 2>&1
echo "$(date +%F_%T) profile rc=$? $(( $(date +%s)-t0 )) s: $(grep ExecTimeUs_Dev $P/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"
t0=$(date +%s); /opt/qti-aic/exec/qaic-opstats --qpc $q/programqpc.bin --input-dir $P/stats --output-dir $P/trace --summary --trace --merge-mq-traces true --flow-events full > $P/opstats.log 2>&1
echo "$(date +%F_%T) opstats rc=$? $(( $(date +%s)-t0 )) s, trace $(du -sh $P/trace | cut -f1)"
/opt/qti-aic/tools/qaic-qpc extract --qpc $q/programqpc.bin --output-dir $P/meta -s '*opstatsdesc.bin' > /dev/null 2>&1
t=$(ls $P/trace/*merged*.trace.json | sort | head -1); t0=$(date +%s)
$PY $M/scripts/detail_analyze.py $t $P/meta "e44 T512 RankTier full model" $P/analysis/sample0 > $P/analysis_sample0.log 2>&1
echo "$(date +%F_%T) analysis rc=$? $(( $(date +%s)-t0 )) s"; head -n 1 $P/analysis_sample0.log | cut -c1-250
if [ -f $P/analysis/sample0/work.csv ]; then
  $PY $M/scripts/layer_timeline.py $P/analysis/sample0 "e44 T512" > $P/layer_timeline.txt 2>&1
  $PY $M/scripts/e44_layers.py $P/analysis/sample0 $F/e38/run_T512_S1_E_1024/result.json > $P/e44_layers.txt 2>&1; echo "e44_layers rc=$?"
  cp $P/e44_layers.txt $F/e44_layers.txt
  rm -rf $q $P/stats; rm -f $P/trace/*.json; echo "$(date +%F_%T) deleted the program and the raw traces"
else echo "### no work.csv: program and traces kept for a retry"; fi
echo "$(date +%F_%T) df after: $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
echo E44_DONE
