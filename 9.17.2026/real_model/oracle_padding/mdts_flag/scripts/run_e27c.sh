#!/bin/bash
# E27c: last 12-layer check (naive at 1024 KiB tiles) and the full 48-layer runtime sort with 512 KiB tiles vs naive and production
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server; M=$R/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e27; mkdir -p $OUT
compile() { local out=$1 graph=$2 cio=$3 extra=$4; [ -f $out/programqpc.bin ] && { echo "### $(basename $out) cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -custom-IO-list-file=$cio -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $out.log 2>&1
  echo "### $(basename $out) compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 15; done; }
hostrun() { local n=$1 q=$2 extra=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; idle
  cd $R && $PY tools/moe_qwen3_baseline.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $F/ref $extra > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; l=r['logits']; print(f\"   $n: median {t['median_ms']:.2f} ms, rounds {[round(v,2) for v in t['round_medians_ms'].values()]}; logits rel L2 {l['relative_l2']:.4f}, next '{l['next_token']}'; memory GiB/card {[round(d['active_gib'],2) for d in r['device_memory']]}\")" 2>/dev/null || tail -2 $OUT/run_$n.stdout | cut -c1-200; }
echo "df before: $(df -h /home/wentao/workspace | tail -1 | awk '{print $4}')"
compile $F/qpc_full_dyncard_ssg512 full_dyncard_hp $C/custom_io.yaml "-size-split-granularity=512" &
compile $K/qpc_t12_naive_ssg1024 trunc12_naive $F/trunc12_naive/custom_io.yaml "-size-split-granularity=1024" &
wait
echo "===== 12 layers: naive at 1024 KiB tiles ====="
hostrun t12_naive $K/qpc_t12_naive ""; hostrun t12_naive_ssg1024 $K/qpc_t12_naive_ssg1024 ""; rm -rf $K/qpc_t12_naive_ssg1024 $K/qpc_t12_naive
echo "===== full model, 48 layers, KV retained, one session ====="
hostrun full_production $F/qpc_native_noflag "--routing-counts"
hostrun full_naive_tokenowned $F/qpc_full_naive_to_flag "--routing-counts"
hostrun full_runtime_sort_ssg512 $F/qpc_full_dyncard_ssg512 "--routing-counts"
hostrun full_naive_tokenowned_again $F/qpc_full_naive_to_flag "--routing-counts"
$PY - <<EOP
import numpy as np, json
O="$OUT"
c = np.array(json.load(open(f"{O}/run_full_runtime_sort_ssg512/result.json"))['routing_counts'])
print(f"   runtime sort: hot lanes max {c[:, :64].max()} of 128, cold lanes max {c[:, 64:].max()} of 16, layers with cold overflow {int((c[:, 64:].max(1) > 16).sum())}; per-card order holds: {all(c[l, 16*k:16*k+16].min() >= c[l, 64+16*k:64+16*k+16].max() for l in range(48) for k in range(4))}")
EOP
echo "df after: $(df -h /home/wentao/workspace | tail -1 | awk '{print $4}')"
echo E27C_DONE
