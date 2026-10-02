#!/bin/bash
# Instrumented two-layer profiles for experiments 1 and 2 (E38/E39): compile the two-layer cut with -stats-level=70 at the
# given tile size, profile 3 samples (raw device stats), decode with qaic-opstats, analyze with detail_analyze.py and
# stage_profile.py (layer 1), then delete the program and the raw traces. Usage: run_e38p.sh <T> <label:tile> ...
#   label: the graph full_model/trunc2_<T>_<label> (e38A..e38E, blk64, ...); tile: def, 512 or 1024.
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest
T=$1; shift; cd $F || exit 1
case $T in 128) spec=specializations_flat.json; ref=$F/ref;; 256) spec=specializations_T256.json; ref=$F/ref256;; 512) spec=specializations_T512.json; ref=$F/ref512;; esac
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
for pr in "$@"; do label=${pr%%:*} ts=${pr##*:}; g=$F/trunc2_${T}_$label; q=$K/qpc_p_${T}_${label}_${ts}_s70; d=$K/prof_e38_${T}_${label}_$ts
  extra=""; [ $ts != def ] && extra="-size-split-granularity=$ts"; rm -rf $q $d; mkdir -p $d/stats $d/trace
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$g/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$g/custom_io.yaml -compile-only -aic-binary-dir=$q -mdts-mos=1 $extra -stats-level=70 > $q.log 2>&1 || { echo "### $pr compile failed"; continue; }
  $PY $M/scripts/trunc_io.py $q $d $ref > /dev/null; idle
  /opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats > $d/runner.log 2>&1
  /opt/qti-aic/exec/qaic-opstats --qpc $q/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
  /opt/qti-aic/tools/qaic-qpc extract --qpc $q/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' > /dev/null 2>&1
  t=$(ls $d/trace/*merged*.trace.json 2>/dev/null | sort | head -1)
  if [ -n "$t" ]; then $PY $M/scripts/detail_analyze.py $t $d/meta "e38p $T $label $ts" $d/analysis/sample0 > $d/analysis_sample0.log 2>&1; head -n 1 $d/analysis_sample0.log | cut -c1-200
    $PY $M/scripts/stage_profile.py $d/analysis/sample0 "T=$T $label $ts"; else echo "### $pr: no trace"; tail -3 $d/runner.log; fi
  rm -f $d/trace/*.json; rm -rf $q $d/stats
done
echo E38P_DONE
