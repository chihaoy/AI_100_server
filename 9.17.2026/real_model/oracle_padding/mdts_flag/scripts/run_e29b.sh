#!/bin/bash
# E29b: 512-token chunk, full 48-layer programs with retained KV: production (no flag), naive T/T token-owned (flag), runtime
# sort 512/64 with 512 KiB tiles (flag); timed in one session with the kept anchors (T=256 naive, T=128 production and naive)
PY=/home/chihao/qeff-venv/bin/python; R=/home/wentao/workspace/AI_100_server; M=$R/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e29; mkdir -p $OUT
compile() { local out=$1 graph=$2 extra=$3; [ -f $out/programqpc.bin ] && { echo "### $(basename $out) cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_T512.json -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$out $extra > $out.log 2>&1
  echo "### $(basename $out) compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 15; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref --routing-counts > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; l=r['logits']; print(f\"   $n: median {t['median_ms']:.2f} ms ({t['tokens_per_second']:.0f} tok/s), rounds {[round(v,2) for v in t['round_medians_ms'].values()]}; logits rel L2 {l['relative_l2']:.4f}, next '{l['next_token']}' (ref '{l['reference_next_token']}'); top5 {l['top5']} (ref {l['reference_top5']}); memory GiB/card {[round(d['active_gib'],2) for d in r['device_memory']]}; load {[round(float(x['ms'])) for x in r['setup'] if x['phase'] == 'load_program'][0]} ms\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -1 $OUT/run_$n.log | cut -c1-200; }; }
echo "df before: $(df -h $F | tail -1 | awk '{print $4}')"
compile $F/qpc512_full_naive_to_flag full512_naive_hp "-mdts-mos=1" & compile $F/qpc512_full_dyncard_ssg512 full512_dyncard_hp "-mdts-mos=1 -size-split-granularity=512" & wait
compile $F/qpc512_native_noflag native_c128 ""
echo "df after compiles: $(df -h $F | tail -1 | awk '{print $4}')"
echo "===== one session: 512-token chunk (held-out prompt) with the 256- and 128-token anchors ====="
hostrun T512_production $F/qpc512_native_noflag $F/ref512
hostrun T512_naive_tokenowned $F/qpc512_full_naive_to_flag $F/ref512
hostrun T512_runtime_sort $F/qpc512_full_dyncard_ssg512 $F/ref512
[ -f $OUT/run_T512_runtime_sort/result.json ] || { echo "   runtime sort failed once; retrying"; mv $OUT/run_T512_runtime_sort $OUT/failed1_T512_runtime_sort 2>/dev/null; mv $OUT/run_T512_runtime_sort.log $OUT/failed1_T512_runtime_sort.log 2>/dev/null; hostrun T512_runtime_sort $F/qpc512_full_dyncard_ssg512 $F/ref512; }
hostrun T512_naive_tokenowned_again $F/qpc512_full_naive_to_flag $F/ref512
hostrun T256_naive_tokenowned $F/qpc256_full_naive_to_flag $F/ref256
hostrun T128_production $F/qpc_native_noflag $F/ref
hostrun T128_naive_tokenowned $F/qpc_full_naive_to_flag $F/ref
$PY - <<EOP
import numpy as np, json, os
O = "$OUT"; p = f"{O}/run_T512_runtime_sort/result.json"
if os.path.exists(p):
    c = np.array(json.load(open(p))['routing_counts'])
    print(f"   T512_runtime_sort: hot lanes max {c[:, :64].max()} of 512, cold lanes max {c[:, 64:].max()} of 64, layers with cold overflow {int((c[:, 64:].max(1) > 64).sum())}; per-card order holds: {all(c[l, 16*k:16*k+16].min() >= c[l, 64+16*k:64+16*k+16].max() for l in range(48) for k in range(4))}")
L = lambda n: np.fromfile(f"{O}/run_{n}/logits_f32.bin", np.float32).astype(np.float64)
r = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
names = [n for n in ("T512_production", "T512_naive_tokenowned", "T512_runtime_sort") if os.path.exists(f"{O}/run_{n}/logits_f32.bin")]
if len(names) == 3:
    p, nv, rs = (L(n) for n in names)
    print(f"   T=512 logits: naive vs production {r(nv, p):.4f}; runtime sort vs production {r(rs, p):.4f}; runtime sort vs naive {r(rs, nv):.4f}")
if all(os.path.exists(f"{O}/run_{n}/logits_f32.bin") for n in ("T512_naive_tokenowned", "T512_naive_tokenowned_again")):
    print("   naive runs bit-identical:", np.array_equal(L("T512_naive_tokenowned"), L("T512_naive_tokenowned_again")))
EOP
echo "df after: $(df -h $F | tail -1 | awk '{print $4}')"
echo E29B_DONE
