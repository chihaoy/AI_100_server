#!/bin/bash
# E27b: tile-size sweep (-size-split-granularity) for both runtime-sort structures and naive, 12-layer cuts, paired with naive
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server; M=$R/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e27; mkdir -p $OUT
comprs() { local name=$1 graph=$2 extra=$3; local out=$K/qpc_$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 15; done; }
hostrun() { local n=$1; rm -rf $OUT/run_$n $OUT/run_$n.log; idle
  cd $R && $PY tools/moe_qwen3_baseline.py run --out $OUT/run_$n --precision mxfp6 --qpc $K/qpc_$n --reference $F/ref > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; print(f\"   $n: median {t['median_ms']:.2f} ms, rounds {[round(v,2) for v in t['round_medians_ms'].values()]}\")" 2>/dev/null || tail -2 $OUT/run_$n.stdout | cut -c1-200; }
pair() { local n=$1 graph=$2 extra=$3; comprs $n $graph "$extra"; hostrun t12_naive; hostrun $n; rm -rf $K/qpc_$n; }
echo "===== E27b: tile-size sweep, 12 layers ====="
comprs t12_naive trunc12_naive ""
pair t12_dyncard_ssg512 trunc12_dyncard "-size-split-granularity=512"
pair t12_dyncard_ssg1024 trunc12_dyncard "-size-split-granularity=1024"
pair t12_sortfirst_ssg1024 trunc12_sortfirst "-size-split-granularity=1024"
pair t12_naive_ssg1024 trunc12_naive "-size-split-granularity=1024"
hostrun t12_naive
echo E27B_DONE
