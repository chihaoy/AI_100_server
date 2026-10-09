#!/bin/bash
# E49 instrumented two-layer profiles of given programs (compiled with -stats-level=70), scheduled into a gap of a longer experiment
# (board idle, and the other experiment only compiling or not running): raw device stats, qaic-opstats, detail_analyze.py and stage_profile.py on
# layer 1, as run_e38p.sh does. Usage: e49_profile.sh <T> <out dir> <label=qpc dir> ...
T=$1; OUT=$2; shift 2; mkdir -p $OUT
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model
case $T in 128) ref=$F/ref;; 256) ref=$F/ref256;; 512) ref=$F/ref512;; esac
idle4() { /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$'; }
gap() { for w in $(seq 2160); do if idle4 && ! pgrep -f moe_qwen3_baseline_T.py > /dev/null && ! pgrep -x qaic-runner > /dev/null && { pgrep -x qaic-compile > /dev/null || ! pgrep -f 'run_e4[78]\.sh|run_e50\.sh' > /dev/null; }; then return 0; fi; sleep 10; done; return 1; }
for pr in "$@"; do label=${pr%%=*} q=${pr#*=}; d=$OUT/prof_$label; rm -rf $d; mkdir -p $d/stats $d/trace
  gap || { echo "no gap within 6 h"; exit 1; }; echo "== $label: gap found $(date +%F_%T)"
  $PY $M/scripts/trunc_io.py $q $d $ref > /dev/null
  timeout 900 /opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats > $d/runner.log 2>&1
  /opt/qti-aic/exec/qaic-opstats --qpc $q/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
  /opt/qti-aic/tools/qaic-qpc extract --qpc $q/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' > /dev/null 2>&1
  t=$(ls $d/trace/*merged*.trace.json 2>/dev/null | sort | head -1)
  if [ -n "$t" ]; then $PY $M/scripts/detail_analyze.py $t $d/meta "e49 $T $label" $d/analysis/sample0 > $d/analysis_sample0.log 2>&1; head -n 1 $d/analysis_sample0.log | cut -c1-200
    $PY $M/scripts/stage_profile.py $d/analysis/sample0 "T=$T $label"; else echo "### $label: no trace"; tail -3 $d/runner.log; fi
  rm -f $d/trace/*.json; rm -rf $d/stats
done
echo E49P_DONE
