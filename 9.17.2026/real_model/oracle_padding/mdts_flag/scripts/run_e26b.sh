#!/bin/bash
# E26b: full 48-layer retained-state programs: naive T/T token-owned and per-card runtime sort, vs production and current best, one session
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server; M=$R/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs
comprs() { local name=$1 graph=$2; local out=$F/qpc_$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 > $F/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $F/compile_$name.log | sed -n 3,12p | cut -c1-240; }
comprs full_naive_to_flag full_naive_hp & comprs full_dyncard_flag full_dyncard_hp & wait
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 20; done; }
runit() { local name=$1 qpc=$2; [ -f $qpc/programqpc.bin ] || { echo "=== run $name skipped (no qpc)"; return; }; $PY $M/scripts/qpc_bindings.py $qpc 2>&1 | grep '###' | cut -c1-200
  rm -rf $F/run_$name $F/run_$name.log; idle; cd $R && $PY tools/moe_qwen3_baseline.py run --out $F/run_$name --precision mxfp6 --qpc $qpc --reference $F/ref --routing-counts > $F/run_$name.stdout 2>&1; echo "=== run $name rc=$?"
  $PY $S/result_line.py $F/run_$name/result.json $name 2>/dev/null || tail -3 $F/run_$name.stdout | cut -c1-300
  $PY -c "import json; r=json.load(open('$F/run_$name/result.json')); print('   device memory GiB per card:', [round(d['active_gib'],2) for d in r['device_memory']], ' qpc GB', round(r['qpc_bytes']/1e9,1))" 2>/dev/null; }
runit e26_production $F/qpc_native_noflag
runit e26_current_best $F/qpc_stack_hp_flag
runit e26_naive_tokenowned $F/qpc_full_naive_to_flag
runit e26_runtime_sort $F/qpc_full_dyncard_flag
$PY - <<EOP
import numpy as np, json
F="$F"
def counts(r): return np.array(json.load(open(f"{F}/run_{r}/result.json"))['routing_counts'])
try:
    a = counts('e26_naive_tokenowned'); b = counts('e26_runtime_sort')
    srt = lambda c: np.concatenate([np.sort(c[:, 32*k:32*k+32], axis=1) for k in range(4)], axis=1)   # native quarters, sorted within each card
    bn = np.concatenate([np.sort(np.concatenate([b[:, 16*k:16*k+16], b[:, 64+16*k:64+16*k+16]], axis=1), axis=1) for k in range(4)], axis=1)
    same = (srt(a) == bn).all(1); print(f"per-card multiset of expert counts, runtime sort vs naive: identical in {int(same.sum())} of 48 layers; first differing layer {int(np.argmin(same)) if not same.all() else None}")
    la = np.fromfile(f"{F}/run_e26_naive_tokenowned/logits_f32.bin", np.float32).astype(np.float64); lb = np.fromfile(f"{F}/run_e26_runtime_sort/logits_f32.bin", np.float32).astype(np.float64)
    print(f"logits, runtime sort vs naive token-owned: rel L2 {np.linalg.norm(lb-la)/np.linalg.norm(la):.4f}; top-5 {np.argsort(-la)[:5].tolist()} vs {np.argsort(-lb)[:5].tolist()}")
except Exception as ex: print('comparison skipped:', ex)
EOP
echo E26B_DONE
