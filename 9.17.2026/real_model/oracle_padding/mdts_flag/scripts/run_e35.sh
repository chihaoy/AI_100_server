#!/bin/bash
# E35: using the idle cores of the 8-lane tiers, T=512, two-layer cuts: (a) without -aic-enable-depth-first, do independent
# tiers overlap? (b) the top tier split over 16 lanes per card (8x512s2: each top expert on two lanes of 256 rows).
# Same-session host timing, the runtime sort and the 3-tier build repeated at the end; profiles of (a) and (b)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e35; mkdir -p $OUT
comp() { local name=$1 graph=$2 df=$3 extra=$4; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s); local dff=""; [ "$df" = df ] && dff="-aic-enable-depth-first"
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 $dff -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_T512.json -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 -size-split-granularity=1024 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $F/ref512 > $OUT/run_$n.stdout 2>&1
  $PY -c "
import json, numpy as np; r = json.load(open('$OUT/run_$n/result.json')); t = r['timing']
y = np.fromfile('$OUT/run_$n/logits_f32.bin', np.float32).astype(np.float64); ref = np.load('$F/ref512/trunc2_logits_last.npy').astype(np.float64)
print(f\"   $n: median {t['median_ms']:.3f} ms, rounds {[round(v, 3) for v in t['round_medians_ms'].values()]}; two-layer rel L2 vs FP32 {np.linalg.norm(y - ref) / np.linalg.norm(ref):.4f}, argmax {int(y.argmax())} (ref {int(ref.argmax())})\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -n 2 $OUT/run_$n.stdout | cut -c1-200; }; }
profile() { local name=$1 q=$2; local d=$K/prof_$name; rm -rf $d; mkdir -p $d/stats $d/outputs $d/trace; [ -f $q/programqpc.bin ] || { echo "   $name: no program"; return; }
  $PY $M/scripts/trunc_io.py $q $d $F/ref512 > /dev/null; idle
  /opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats > $d/runner.log 2>&1
  /opt/qti-aic/exec/qaic-opstats --qpc $q/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
  /opt/qti-aic/tools/qaic-qpc extract --qpc $q/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' > /dev/null 2>&1
  local t=$(ls $d/trace/*merged*.trace.json | sort | head -1); $PY $M/scripts/detail_analyze.py $t $d/meta "$name" $d/analysis/sample0 > $d/analysis_sample0.log 2>&1; head -n 1 $d/analysis_sample0.log | cut -c1-200
  rm -f $d/trace/*.json; }
comp e35_base_df trunc2_512_dyncard df "" &
comp e35_base_nodf trunc2_512_dyncard nodf "" &
comp e35_tier3_df trunc2_512_tier3 df "" &
comp e35_tier3_nodf trunc2_512_tier3 nodf "" &
comp e35_tier3s_df trunc2_512_tier3s df "" &
comp e35_tier3s_nodf trunc2_512_tier3s nodf "" &
comp e35_tier3_nodf_s70 trunc2_512_tier3 nodf "-stats-level=70" &
comp e35_tier3s_df_s70 trunc2_512_tier3s df "-stats-level=70" &
wait
echo "===== two layers, T=512, one session (host tool, 3 rounds x 20)"
hostrun base_df $K/qpc_e35_base_df
hostrun tier3_df $K/qpc_e35_tier3_df
hostrun tier3s_df $K/qpc_e35_tier3s_df
hostrun tier3_nodf $K/qpc_e35_tier3_nodf
hostrun tier3s_nodf $K/qpc_e35_tier3s_nodf
hostrun base_nodf $K/qpc_e35_base_nodf
hostrun tier3_df_again $K/qpc_e35_tier3_df
hostrun base_df_again $K/qpc_e35_base_df
$PY - <<EOP
import numpy as np, os
O = "$OUT"; L = lambda n: np.fromfile(f"{O}/run_{n}/logits_f32.bin", np.float32) if os.path.exists(f"{O}/run_{n}/logits_f32.bin") else None
for a, b in (("tier3_df", "tier3s_df"), ("tier3_df", "tier3_nodf"), ("base_df", "tier3_df")):
    x, y = L(a), L(b)
    if x is not None and y is not None: print(f"   {b} vs {a}: bit-identical {np.array_equal(x, y)}, rel L2 {np.linalg.norm(y.astype(float) - x) / np.linalg.norm(x):.2e}")
EOP
echo "===== profiles"
profile e35_tier3_nodf $K/qpc_e35_tier3_nodf_s70
profile e35_tier3s_df $K/qpc_e35_tier3s_df_s70
rm -rf $K/qpc_e35_*
echo E35_DONE
