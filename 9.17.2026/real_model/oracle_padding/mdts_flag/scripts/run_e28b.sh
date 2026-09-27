#!/bin/bash
# E28b: 256-token chunk, full 48-layer programs with retained KV: production (no flag), naive T/T token-owned (flag), runtime
# sort 256/32 with 512 KiB tiles (flag); timed in one session together with the three 128-token programs of E27c
PY=/home/chihao/qeff-venv/bin/python; R=/home/wentao/workspace/AI_100_server; M=$R/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e28; mkdir -p $OUT
compile() { local out=$1 graph=$2 extra=$3; [ -f $out/programqpc.bin ] && { echo "### $(basename $out) cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_T256.json -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$out $extra > $out.log 2>&1
  echo "### $(basename $out) compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 15; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref --routing-counts > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; l=r['logits']; print(f\"   $n: median {t['median_ms']:.2f} ms ({t['tokens_per_second']:.0f} tok/s), rounds {[round(v,2) for v in t['round_medians_ms'].values()]}; logits rel L2 {l['relative_l2']:.4f}, next '{l['next_token']}' (ref '{l['reference_next_token']}'); memory GiB/card {[round(d['active_gib'],2) for d in r['device_memory']]}\")" 2>/dev/null || tail -2 $OUT/run_$n.stdout | cut -c1-200; }
echo "df before: $(df -h $F | tail -1 | awk '{print $4}')"
compile $F/qpc256_full_naive_to_flag full256_naive_hp "-mdts-mos=1" & compile $F/qpc256_full_dyncard_ssg512 full256_dyncard_hp "-mdts-mos=1 -size-split-granularity=512" & wait
compile $F/qpc256_native_noflag native_c128 ""
echo "===== one session: 256-token chunk (held-out prompt) and 128-token chunk (prompt 41) ====="
hostrun T256_production $F/qpc256_native_noflag $F/ref256
hostrun T256_naive_tokenowned $F/qpc256_full_naive_to_flag $F/ref256
hostrun T256_runtime_sort $F/qpc256_full_dyncard_ssg512 $F/ref256
hostrun T256_naive_tokenowned_again $F/qpc256_full_naive_to_flag $F/ref256
hostrun T128_production $F/qpc_native_noflag $F/ref
hostrun T128_naive_tokenowned $F/qpc_full_naive_to_flag $F/ref
hostrun T128_runtime_sort $F/qpc_full_dyncard_ssg512 $F/ref
$PY - <<EOP
import numpy as np, json, os
O = "$OUT"
for n, C in (("T256_runtime_sort", 32), ("T128_runtime_sort", 16)):
    p = f"{O}/run_{n}/result.json"
    if not os.path.exists(p): continue
    c = np.array(json.load(open(p))['routing_counts'])
    print(f"   {n}: hot lanes max {c[:, :64].max()}, cold lanes max {c[:, 64:].max()} of {C}, layers with cold overflow {int((c[:, 64:].max(1) > C).sum())}; per-card order holds: {all(c[l, 16*k:16*k+16].min() >= c[l, 64+16*k:64+16*k+16].max() for l in range(48) for k in range(4))}")
for T in (256, 128):
    a, b = (f"{O}/run_T{T}_naive_tokenowned/logits_f32.bin", f"{O}/run_T{T}_runtime_sort/logits_f32.bin")
    if os.path.exists(a) and os.path.exists(b):
        x, y = np.fromfile(a, np.float32).astype(np.float64), np.fromfile(b, np.float32).astype(np.float64)
        print(f"   T={T} logits, runtime sort vs naive: rel L2 {np.linalg.norm(y - x) / np.linalg.norm(x):.4f}; top-5 {np.argsort(-y)[:5].tolist()} vs {np.argsort(-x)[:5].tolist()}")
EOP
echo "df after: $(df -h $F | tail -1 | awk '{print $4}')"
echo E28B_DONE
