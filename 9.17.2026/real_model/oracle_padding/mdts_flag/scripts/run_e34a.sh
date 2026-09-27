#!/bin/bash
# E34a: tiered capacities (build_full_tiered.py) on two-layer cuts at T=256 and T=512 against the runtime sort 16xT + 16xT/8:
# same-session host timing, accuracy against the FP32 two-layer reference, and a profile of the 4-tier T=512 build
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e34; mkdir -p $OUT
comp() { local name=$1 graph=$2 spec=$3 extra=$4; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,12p | cut -c1-220; }
idle() { for i in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref > $OUT/run_$n.stdout 2>&1
  $PY -c "
import json, numpy as np; r = json.load(open('$OUT/run_$n/result.json')); t = r['timing']
y = np.fromfile('$OUT/run_$n/logits_f32.bin', np.float32).astype(np.float64); ref = np.load('$ref/trunc2_logits_last.npy').astype(np.float64)
print(f\"   $n: median {t['median_ms']:.3f} ms, rounds {[round(v, 3) for v in t['round_medians_ms'].values()]}; two-layer logits rel L2 vs FP32 {np.linalg.norm(y - ref) / np.linalg.norm(ref):.4f}, argmax {int(y.argmax())} (ref {int(ref.argmax())})\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -n 2 $OUT/run_$n.stdout | cut -c1-200; }; }
comp e34_256_base trunc2_256_dyncard specializations_T256.json "-size-split-granularity=512" &
comp e34_256_tier3 trunc2_256_tier3 specializations_T256.json "-size-split-granularity=512" &
comp e34_256_tier4 trunc2_256_tier4 specializations_T256.json "-size-split-granularity=512" &
comp e34_512_base trunc2_512_dyncard specializations_T512.json "-size-split-granularity=1024" &
comp e34_512_tier3 trunc2_512_tier3 specializations_T512.json "-size-split-granularity=1024" &
comp e34_512_tier4 trunc2_512_tier4 specializations_T512.json "-size-split-granularity=1024" &
comp e34_512_tier4_s70 trunc2_512_tier4 specializations_T512.json "-size-split-granularity=1024 -stats-level=70" &
wait
echo "===== two layers, one session (host tool, 3 rounds x 20)"
hostrun e34_256_base $K/qpc_e34_256_base $F/ref256
hostrun e34_256_tier3 $K/qpc_e34_256_tier3 $F/ref256
hostrun e34_256_tier4 $K/qpc_e34_256_tier4 $F/ref256
hostrun e34_256_base_again $K/qpc_e34_256_base $F/ref256
hostrun e34_512_base $K/qpc_e34_512_base $F/ref512
hostrun e34_512_tier3 $K/qpc_e34_512_tier3 $F/ref512
hostrun e34_512_tier4 $K/qpc_e34_512_tier4 $F/ref512
hostrun e34_512_base_again $K/qpc_e34_512_base $F/ref512
echo "===== profile, 4 tiers, T=512"
d=$K/prof_e34_512_tier4; q=$K/qpc_e34_512_tier4_s70; rm -rf $d; mkdir -p $d/stats $d/outputs $d/trace
if [ -f $q/programqpc.bin ]; then $PY $M/scripts/trunc_io.py $q $d $F/ref512 > /dev/null; idle
  /opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats > $d/runner.log 2>&1
  /opt/qti-aic/exec/qaic-opstats --qpc $q/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
  /opt/qti-aic/tools/qaic-qpc extract --qpc $q/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' > /dev/null 2>&1
  t=$(ls $d/trace/*merged*.trace.json | sort | head -1); $PY $M/scripts/detail_analyze.py $t $d/meta "e34 tier4 T512" $d/analysis/sample0 > $d/analysis_sample0.log 2>&1; head -n 1 $d/analysis_sample0.log | cut -c1-200
  rm -f $d/trace/*.json; fi
rm -rf $K/qpc_e34_256_base $K/qpc_e34_256_tier3 $K/qpc_e34_256_tier4 $K/qpc_e34_512_base $K/qpc_e34_512_tier3 $K/qpc_e34_512_tier4 $K/qpc_e34_512_tier4_s70
echo E34A_DONE
