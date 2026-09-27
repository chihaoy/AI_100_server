#!/bin/bash
# E29c: why the runtime sort gains less at 512 than at 256 tokens: tile size (-size-split-granularity) on 12-layer cuts of the
# 512-token programs, host timing with the full-model tool, back to back with naive T/T (logits of a cut are not comparable to
# the 48-layer reference; timing only)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e29
comp() { local name=$1 graph=$2 extra=$3; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_T512.json -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 15; done; }
hostrun() { local n=$1 q=$2; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $F/ref512 > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; print(f\"   $n: median {t['median_ms']:.2f} ms, rounds {[round(v,2) for v in t['round_medians_ms'].values()]}; memory GiB/card {[round(d['active_gib'],2) for d in r['device_memory']]}\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -1 $OUT/run_$n.log | cut -c1-200; }; }
comp t12_512_naive trunc12_512_naive "" & comp t12_512_sort_default trunc12_512_dyncard "" & comp t12_512_sort_ssg512 trunc12_512_dyncard "-size-split-granularity=512" & comp t12_512_sort_ssg1024 trunc12_512_dyncard "-size-split-granularity=1024" & wait
echo "===== 12 layers, 512-token chunk, one session ====="
hostrun t12_512_naive $K/qpc_t12_512_naive
hostrun t12_512_sort_default $K/qpc_t12_512_sort_default
hostrun t12_512_sort_ssg512 $K/qpc_t12_512_sort_ssg512
hostrun t12_512_sort_ssg1024 $K/qpc_t12_512_sort_ssg1024
hostrun t12_512_naive_again $K/qpc_t12_512_naive
echo "df: $(df -h $F | tail -1 | awk '{print $4}')"
echo E29C_DONE
