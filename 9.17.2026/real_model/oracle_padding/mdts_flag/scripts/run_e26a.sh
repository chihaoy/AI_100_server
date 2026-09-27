#!/bin/bash
# E26a: two-layer check of the full-model builder: naive vs runtime-sort, retained state, flag
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$R/full_model; C=$F/configs; K=$F/kvtest
comprs() { local name=$1 graph=$2; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,12p | cut -c1-240; }
comprs trunc2_naive_to trunc2_naive & comprs trunc2_dyncard trunc2_dyncard & wait
for n in trunc2_naive_to trunc2_dyncard; do $PY $R/scripts/qpc_bindings.py $K/qpc_$n 2>&1 | grep '###' | cut -c1-200; done
for n in trunc2_naive_to trunc2_dyncard; do d=$K/run_$n; rm -rf $d; mkdir -p $d/outputs; $PY $S/kv/trunc_io.py $K/qpc_$n $d > /dev/null
  for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && break; sleep 10; done
  /opt/qti-aic/exec/qaic-runner -t $K/qpc_$n -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 30 -S 1 -T 1 --write-output-start-iter 29 --write-output-num-samples 1 --write-output-dir $d/outputs > $d/runner.log 2>&1
  echo "run $n rc=$?: $(grep -E 'Inf/Sec' $d/runner.log | tr '\n' ' ' | cut -c1-120)"; grep -iE "error|fail" $d/runner.log | head -2; done
$PY - <<EOP
import numpy as np, glob
K="$K"; ys={}
for n in ("trunc2_naive_to", "trunc2_dyncard"):
    f=glob.glob(f"{K}/run_{n}/outputs/*logits*"); ys[n]=np.fromfile(f[0], np.float32).astype(np.float64) if f else None
    print(n, "argmax", ys[n].argmax() if ys[n] is not None else None)
old=glob.glob(f"{K}/run_trunc2_hp_mos1/outputs/*logits*")
if ys["trunc2_naive_to"] is not None and ys["trunc2_dyncard"] is not None:
    a,b=ys["trunc2_naive_to"],ys["trunc2_dyncard"]; print(f"dyncard vs naive: rel L2 {np.linalg.norm(b-a)/np.linalg.norm(a):.3e}, max|diff| {np.abs(b-a).max():.4f}")
if old and ys["trunc2_naive_to"] is not None:
    o=np.fromfile(old[0], np.float32).astype(np.float64); a=ys["trunc2_naive_to"]; print(f"naive token-owned vs E17 2-layer head-parallel (token-centric-free original combine): rel L2 {np.linalg.norm(a-o)/np.linalg.norm(o):.3e}")
EOP
echo E26A_DONE
