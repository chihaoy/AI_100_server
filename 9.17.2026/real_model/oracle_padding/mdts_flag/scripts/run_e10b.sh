#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server; O=$R/9.17.2026/real_model/oracle_padding; F=$O/mdts_flag/full_model; C=$F/configs
idle() { for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; echo "devices busy, waiting"; sleep 20; done; }
compkv() { local name=$1 graph=$2 extra=$3; local out=$F/qpc_$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -convert-to-fp16 -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -compile-only -aic-binary-dir=$out $extra > $F/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $F/compile_$name.log | sed -n 5,10p | cut -c1-160; }
runit() { local name=$1 qpc=$2 counts=$3; [ -f $qpc/programqpc.bin ] || { echo "=== run $name skipped (no qpc)"; return; }; rm -rf $F/run_$name $F/run_$name.log; idle
  cd $R && $PY tools/moe_qwen3_baseline.py run --out $F/run_$name --precision mxfp6 --qpc $qpc $counts > $F/run_$name.stdout 2>&1; echo "=== run $name rc=$?"; $PY $S/result_line.py $F/run_$name/result.json $name || tail -3 $F/run_$name.stdout | cut -c1-300; }
echo "===== E10b: waiting for E10 and for any running host job ====="; until grep -q E10_DONE $S/run_e10.log 2>/dev/null; do sleep 15; done; while pgrep -f 'moe_qwen3_baselin[e]' >/dev/null; do sleep 10; done
[ -f $F/run_stack_hc_128_rw/counts_i32.bin ] || { echo "no hc_128 counts; abort"; exit 1; }
echo "===== E10b: hot/cold oracle capacities recalibrated on the hot/cold graph's own routing ====="
rm -rf $F/stack_hc_oracle2_rw; $PY $S/build_full_stack.py $F/native_c128_kvfree $F/stack_hc_oracle2_rw $F/run_kvfree_native_flag/counts_i32.bin --regroup hc --caps oracle --rewrites 1 --banks $F/weights_hc_fp16 --poscounts $F/run_stack_hc_128_rw/counts_i32.bin 2>&1 | tail -1 | cut -c1-250
compkv stack_hc_oracle2_rw stack_hc_oracle2_rw "-mxfp6-matmul -mdts-mos=1"; runit stack_hc_oracle2_rw $F/qpc_stack_hc_oracle2_rw --routing-counts
$PY - <<EOP
import numpy as np, json
F="$F"; ref=np.fromfile(f"{F}/run_stack_hc_128_rw/counts_i32.bin",np.int32).reshape(48,128)
for name in ['stack_hc_oracle_rw','stack_hc_oracle2_rw','stack_hc_128_rw']:
    try: c=np.fromfile(f"{F}/run_{name}/counts_i32.bin",np.int32).reshape(48,128)
    except Exception: print(name,'no counts'); continue
    p=json.load(open(f"{F}/{name}/plan.json"))['layers']; ov=sum(int(np.maximum(c[L][:64]-p[str(L)]['capacities'][0],0).sum()+np.maximum(c[L][64:]-p[str(L)]['capacities'][1],0).sum()) for L in range(48))
    print(f"### overflow check {name}: dropped assignments {ov}; routing differences vs hc_128 run {int(np.abs(c-ref).sum()//2)}")
EOP
echo E10B_DONE
