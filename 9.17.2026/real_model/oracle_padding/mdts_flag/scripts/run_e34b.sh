#!/bin/bash
# E34b: 48-layer T=512 programs, 3 tiers (8x512, 8x128, 16x64) against the runtime sort (16x512 + 16x64), both with 1024 KiB
# tiles, KV retained; same-session host timing with routing counts (tier capacities checked per lane); plus an instrumented
# two-layer profile of the 3-tier build (per-tier GEMM windows)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e34; mkdir -p $OUT
compile() { local out=$1 graph=$2 spec=$3 cio=$4 extra=$5; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$cio -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $out.log 2>&1
  echo "### $(basename $out) compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref --routing-counts > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; l=r['logits']; print(f\"   $n: median {t['median_ms']:.2f} ms ({t['tokens_per_second']:.0f} tok/s), rounds {[round(v,2) for v in t['round_medians_ms'].values()]}; logits rel L2 {l['relative_l2']:.4f}, next '{l['next_token']}' (ref '{l['reference_next_token']}'); memory GiB/card {[round(d['active_gib'],2) for d in r['device_memory']]}\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -n 2 $OUT/run_$n.stdout | cut -c1-200; }; }
echo "df before: $(df -h $F | tail -1 | awk '{print $4}')"
compile $F/qpc512_full_tier3_ssg1024 full512_tier3_hp specializations_T512.json $C/custom_io.yaml "-size-split-granularity=1024" &
compile $F/qpc512_full_dyncard_ssg1024 full512_dyncard_hp specializations_T512.json $C/custom_io.yaml "-size-split-granularity=1024" &
compile $K/qpc_e34_512_tier3_s70 trunc2_512_tier3 specializations_T512.json $F/trunc2_512_tier3/custom_io.yaml "-size-split-granularity=1024 -stats-level=70" &
wait
echo "===== profile, 3 tiers, two layers, T=512"
d=$K/prof_e34_512_tier3; q=$K/qpc_e34_512_tier3_s70; rm -rf $d; mkdir -p $d/stats $d/outputs $d/trace
if [ -f $q/programqpc.bin ]; then $PY $M/scripts/trunc_io.py $q $d $F/ref512 > /dev/null; idle
  /opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats > $d/runner.log 2>&1
  /opt/qti-aic/exec/qaic-opstats --qpc $q/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
  /opt/qti-aic/tools/qaic-qpc extract --qpc $q/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' > /dev/null 2>&1
  t=$(ls $d/trace/*merged*.trace.json | sort | head -1); $PY $M/scripts/detail_analyze.py $t $d/meta "e34 tier3 T512" $d/analysis/sample0 > $d/analysis_sample0.log 2>&1; head -n 1 $d/analysis_sample0.log | cut -c1-200
  rm -f $d/trace/*.json; rm -rf $q; fi
echo "===== 48 layers, T=512, one session"
hostrun T512_runtime_sort $F/qpc512_full_dyncard_ssg1024 $F/ref512
hostrun T512_tier3 $F/qpc512_full_tier3_ssg1024 $F/ref512
hostrun T512_runtime_sort_again $F/qpc512_full_dyncard_ssg1024 $F/ref512
$PY - <<EOP
import json, os, numpy as np
O = "$OUT"; p = f"{O}/run_T512_tier3/result.json"
if os.path.exists(p):
    c = np.array(json.load(open(p))['routing_counts'])        # [48, 128] lanes in tier order, card-major within a tier
    tiers = [(8, 512), (8, 128), (16, 64)]; off = 0
    for i, (l, cap) in enumerate(tiers):
        blk = c[:, off:off + 4 * l]; print(f"   tier {i} ({l} lanes/card, capacity {cap}): largest lane count {blk.max()}, lanes over capacity {int((blk > cap).sum())}"); off += 4 * l
    ok = all(c[L, 32 * 0:].size for L in range(48))
    a = np.fromfile(f"{O}/run_T512_runtime_sort/logits_f32.bin", np.float32).astype(np.float64); b = np.fromfile(f"{O}/run_T512_tier3/logits_f32.bin", np.float32).astype(np.float64)
    print(f"   logits, 3 tiers vs runtime sort: rel L2 {np.linalg.norm(b - a) / np.linalg.norm(a):.4f}, bit-identical {np.array_equal(a, b)}")
EOP
echo "df after: $(df -h $F | tail -1 | awk '{print $4}')"
echo E34B_DONE
