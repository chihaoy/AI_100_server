#!/bin/bash
# E28c: retry of the 256-token runtime-sort run (E28b's load failed with "Device is busy" while loading constants), followed by
# naive T/T again for a back-to-back pair
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; OUT=$F/e28
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 15; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref --routing-counts > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; l=r['logits']; print(f\"   $n: median {t['median_ms']:.2f} ms ({t['tokens_per_second']:.0f} tok/s), rounds {[round(v,2) for v in t['round_medians_ms'].values()]}; logits rel L2 {l['relative_l2']:.4f}, next '{l['next_token']}' (ref '{l['reference_next_token']}'); top5 {l['top5']} (ref {l['reference_top5']}); memory GiB/card {[round(d['active_gib'],2) for d in r['device_memory']]}; load {[round(float(x['ms'])) for x in r['setup'] if x['phase'] == 'load_program'][0]} ms\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -1 $OUT/run_$n.log | cut -c1-200; }; }
mv $OUT/run_T256_runtime_sort $OUT/failed_T256_runtime_sort_load 2>/dev/null; mv $OUT/run_T256_runtime_sort.log $OUT/failed_T256_runtime_sort_load.log 2>/dev/null
hostrun T256_runtime_sort $F/qpc256_full_dyncard_ssg512 $F/ref256
hostrun T256_naive_tokenowned_pair $F/qpc256_full_naive_to_flag $F/ref256
$PY - <<EOP
import numpy as np, json, os
O = "$OUT"; p = f"{O}/run_T256_runtime_sort/result.json"
if os.path.exists(p):
    c = np.array(json.load(open(p))['routing_counts'])
    print(f"   T256_runtime_sort: hot lanes max {c[:, :64].max()} of 256, cold lanes max {c[:, 64:].max()} of 32, layers with cold overflow {int((c[:, 64:].max(1) > 32).sum())}; per-card order holds: {all(c[l, 16*k:16*k+16].min() >= c[l, 64+16*k:64+16*k+16].max() for l in range(48) for k in range(4))}")
    x, y = (np.fromfile(f"{O}/run_T256_{n}/logits_f32.bin", np.float32).astype(np.float64) for n in ("naive_tokenowned", "runtime_sort"))
    print(f"   T=256 logits, runtime sort vs naive: rel L2 {np.linalg.norm(y - x) / np.linalg.norm(x):.4f}; top-5 {np.argsort(-y)[:5].tolist()} vs {np.argsort(-x)[:5].tolist()}")
EOP
echo E28C_DONE
