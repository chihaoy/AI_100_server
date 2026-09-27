#!/bin/bash
# E29d: full 48-layer runtime sort at 512 tokens with 1024 KiB tiles (best on the 12-layer cuts, E29c), back to back with naive
# T/T; plus the same 12-layer tile sweep at 256 tokens (E28 used 512 KiB tiles there)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e29
comp() { local out=$1 graph=$2 spec=$3 cio=$4 extra=$5; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$cio -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $out.log 2>&1
  echo "### $(basename $out) compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 15; done; }
hostrun() { local n=$1 q=$2 ref=$3 extra=$4; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref $extra > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; l=r['logits']; print(f\"   $n: median {t['median_ms']:.2f} ms ({t['tokens_per_second']:.0f} tok/s), rounds {[round(v,2) for v in t['round_medians_ms'].values()]}; logits rel L2 {l['relative_l2']:.4f}, next '{l['next_token']}'; memory GiB/card {[round(d['active_gib'],2) for d in r['device_memory']]}\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -1 $OUT/run_$n.log | cut -c1-200; }; }
comp $F/qpc512_full_dyncard_ssg1024 full512_dyncard_hp specializations_T512.json $C/custom_io.yaml "-size-split-granularity=1024" &
comp $K/qpc_t12_256_naive trunc12_256_naive specializations_T256.json $F/trunc12_256_naive/custom_io.yaml "" &
comp $K/qpc_t12_256_sort_default trunc12_256_dyncard specializations_T256.json $F/trunc12_256_dyncard/custom_io.yaml "" &
comp $K/qpc_t12_256_sort_ssg512 trunc12_256_dyncard specializations_T256.json $F/trunc12_256_dyncard/custom_io.yaml "-size-split-granularity=512" &
comp $K/qpc_t12_256_sort_ssg1024 trunc12_256_dyncard specializations_T256.json $F/trunc12_256_dyncard/custom_io.yaml "-size-split-granularity=1024" &
wait
echo "===== 12 layers, 256-token chunk (timing only) ====="
hostrun t12_256_naive $K/qpc_t12_256_naive $F/ref256
hostrun t12_256_sort_default $K/qpc_t12_256_sort_default $F/ref256
hostrun t12_256_sort_ssg512 $K/qpc_t12_256_sort_ssg512 $F/ref256
hostrun t12_256_sort_ssg1024 $K/qpc_t12_256_sort_ssg1024 $F/ref256
hostrun t12_256_naive_again $K/qpc_t12_256_naive $F/ref256
rm -rf $K/qpc_t12_256_naive $K/qpc_t12_256_sort_default $K/qpc_t12_256_sort_ssg512 $K/qpc_t12_256_sort_ssg1024
echo "===== full model, 512-token chunk ====="
hostrun T512_naive_tokenowned_pair1 $F/qpc512_full_naive_to_flag $F/ref512 --routing-counts
hostrun T512_runtime_sort_ssg1024 $F/qpc512_full_dyncard_ssg1024 $F/ref512 --routing-counts
[ -f $OUT/run_T512_runtime_sort_ssg1024/result.json ] || { echo "   runtime sort failed once; retrying"; mv $OUT/run_T512_runtime_sort_ssg1024 $OUT/failed1_T512_runtime_sort_ssg1024 2>/dev/null; mv $OUT/run_T512_runtime_sort_ssg1024.log $OUT/failed1_T512_runtime_sort_ssg1024.log 2>/dev/null; hostrun T512_runtime_sort_ssg1024 $F/qpc512_full_dyncard_ssg1024 $F/ref512 --routing-counts; }
hostrun T512_naive_tokenowned_pair2 $F/qpc512_full_naive_to_flag $F/ref512 --routing-counts
$PY - <<EOP
import numpy as np, json, os
O = "$OUT"; p = f"{O}/run_T512_runtime_sort_ssg1024/result.json"
if os.path.exists(p):
    c = np.array(json.load(open(p))['routing_counts'])
    print(f"   T512_runtime_sort_ssg1024: hot lanes max {c[:, :64].max()} of 512, cold lanes max {c[:, 64:].max()} of 64, layers with cold overflow {int((c[:, 64:].max(1) > 64).sum())}; per-card order holds: {all(c[l, 16*k:16*k+16].min() >= c[l, 64+16*k:64+16*k+16].max() for l in range(48) for k in range(4))}")
    a, b = (np.fromfile(f"{O}/run_{n}/logits_f32.bin", np.float32) for n in ("T512_runtime_sort", "T512_runtime_sort_ssg1024"))
    print("   logits bit-identical to the 512 KiB-tile runtime sort:", np.array_equal(a, b))
EOP
echo "df: $(df -h $F | tail -1 | awk '{print $4}')"
echo E29D_DONE
