#!/bin/bash
# E31: (a) does the slow-card effect follow the physical card or the logical slice? (same instrumented 2-layer T=512 program,
# device order 0:1:2:3 vs 1:2:3:0); (b) the final cross-card sum as elementwise adds (github_pack/FINAL_COMBINE_16CORE.md,
# addtree / tileadd) inside the full-model graph: 2-layer host timing at T=128 and T=512 against the Einsum tiles, and a profile
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e31; mkdir -p $OUT
comp() { local name=$1 graph=$2 spec=$3 extra=$4; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref > $OUT/run_$n.stdout 2>&1
  $PY -c "
import json, numpy as np; r = json.load(open('$OUT/run_$n/result.json')); t = r['timing']
y = np.fromfile('$OUT/run_$n/logits_f32.bin', np.float32).astype(np.float64); ref = np.load('$ref/trunc2_logits_last.npy').astype(np.float64)
print(f\"   $n: median {t['median_ms']:.3f} ms, rounds {[round(v, 3) for v in t['round_medians_ms'].values()]}; two-layer logits rel L2 vs FP32 {np.linalg.norm(y - ref) / np.linalg.norm(ref):.4f}, argmax {int(y.argmax())} (ref {int(ref.argmax())})\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -n 1 $OUT/run_$n.stdout | cut -c1-200; }; }
profile() { local name=$1 q=$2 map=$3 ref=$4; local d=$K/prof_$name; rm -rf $d; mkdir -p $d/stats $d/outputs $d/trace; $PY $M/scripts/trunc_io.py $q $d $ref > /dev/null; idle
  /opt/qti-aic/exec/qaic-runner -t $q -D $map --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats > $d/runner.log 2>&1
  echo "   profile $name (devices $map) rc=$?: $(grep ExecTimeUs_Dev $d/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"
  /opt/qti-aic/exec/qaic-opstats --qpc $q/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
  /opt/qti-aic/tools/qaic-qpc extract --qpc $q/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' > /dev/null 2>&1
  local t=$(ls $d/trace/*merged*.trace.json 2>/dev/null | sort | head -1); [ -n "$t" ] || { echo "   no trace for $name"; tail -n 3 $d/runner.log; return; }
  $PY $M/scripts/detail_analyze.py $t $d/meta "$name" $d/analysis/sample0 > $d/analysis_sample0.log 2>&1; head -n 1 $d/analysis_sample0.log | cut -c1-200
  $PY $M/scripts/e31_cards.py $d/analysis/sample0 "$name"; rm -rf $d/trace/*.json; }
echo "df before: $(df -h $F | tail -1 | awk '{print $4}')"
comp t2_128_einsum trunc2_dyncard specializations_flat.json "-size-split-granularity=512" &
comp t2_128_addtree trunc2_128_dyncard_addtree specializations_flat.json "-size-split-granularity=512" &
comp t2_128_tileadd trunc2_128_dyncard_tileadd specializations_flat.json "-size-split-granularity=512" &
comp t2_512_einsum trunc2_512_dyncard specializations_T512.json "-size-split-granularity=1024" &
comp t2_512_addtree trunc2_512_dyncard_addtree specializations_T512.json "-size-split-granularity=1024" &
comp t2_512_tileadd trunc2_512_dyncard_tileadd specializations_T512.json "-size-split-granularity=1024" &
comp t2_512_einsum_s70 trunc2_512_dyncard specializations_T512.json "-size-split-granularity=1024 -stats-level=70" &
comp t2_512_addtree_s70 trunc2_512_dyncard_addtree specializations_T512.json "-size-split-granularity=1024 -stats-level=70" &
wait
echo "===== (b) timing, two layers, one session (host tool, 3 rounds x 20)"
hostrun t2_128_einsum $K/qpc_t2_128_einsum $F/ref
hostrun t2_128_addtree $K/qpc_t2_128_addtree $F/ref
hostrun t2_128_tileadd $K/qpc_t2_128_tileadd $F/ref
hostrun t2_128_einsum_again $K/qpc_t2_128_einsum $F/ref
hostrun t2_512_einsum $K/qpc_t2_512_einsum $F/ref512
hostrun t2_512_addtree $K/qpc_t2_512_addtree $F/ref512
hostrun t2_512_tileadd $K/qpc_t2_512_tileadd $F/ref512
hostrun t2_512_einsum_again $K/qpc_t2_512_einsum $F/ref512
echo "===== (a) device order, and (b) profile of the elementwise form, T=512 instrumented"
profile e31_einsum_map0123 $K/qpc_t2_512_einsum_s70 0:1:2:3 $F/ref512
profile e31_einsum_map1230 $K/qpc_t2_512_einsum_s70 1:2:3:0 $F/ref512
profile e31_addtree_map0123 $K/qpc_t2_512_addtree_s70 0:1:2:3 $F/ref512
rm -rf $K/qpc_t2_128_einsum $K/qpc_t2_128_addtree $K/qpc_t2_128_tileadd $K/qpc_t2_512_einsum $K/qpc_t2_512_addtree $K/qpc_t2_512_tileadd $K/qpc_t2_512_einsum_s70 $K/qpc_t2_512_addtree_s70
echo "df after: $(df -h $F | tail -1 | awk '{print $4}')"
echo E31_DONE
