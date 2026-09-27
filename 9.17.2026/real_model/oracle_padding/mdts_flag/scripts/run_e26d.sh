#!/bin/bash
# E26d: does the runtime-sort penalty grow with depth? 2- and 12-layer cuts, host timing with the full-model tool
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server; M=$R/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest
comprs() { local name=$1 graph=$2; local out=$K/qpc_$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; }
comprs trunc12_naive_to trunc12_naive & comprs trunc12_dyncard trunc12_dyncard & wait
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 20; done; }
for n in trunc2_naive_to trunc2_dyncard trunc12_naive_to trunc12_dyncard; do rm -rf $K/run_host_$n $K/run_host_$n.log; idle
  cd $R && $PY tools/moe_qwen3_baseline.py run --out $K/run_host_$n --precision mxfp6 --qpc $K/qpc_$n --reference $F/ref > $K/run_host_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$K/run_host_$n/result.json')); t=r['timing']; print(f\"$n: median {t['median_ms']:.2f} ms, rounds {[round(v,2) for v in t['round_medians_ms'].values()]}; memory GiB/card {[round(d['active_gib'],2) for d in r['device_memory']]}\")" 2>/dev/null || tail -2 $K/run_host_$n.stdout | cut -c1-200; done
echo E26D_DONE
