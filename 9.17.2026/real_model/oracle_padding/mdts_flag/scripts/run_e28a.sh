#!/bin/bash
# E28a: 256-token chunk, two-layer check of the three programs (production, naive T/T token-owned, runtime sort 256/32 with
# 512 KiB tiles) against the FP32 two-layer reference of the held-out 256-token prompt
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest
comp() { local name=$1 graph=$2 extra=$3; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_T256.json -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,12p | cut -c1-240; }
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
comp t256_2_prod trunc2_native "" & comp t256_2_naive trunc2_256_naive "-mdts-mos=1" & comp t256_2_dyncard trunc2_256_dyncard "-mdts-mos=1 -size-split-granularity=512" & wait
for n in t256_2_prod t256_2_naive t256_2_dyncard; do d=$K/run_$n; rm -rf $d; mkdir -p $d/outputs; [ -f $K/qpc_$n/programqpc.bin ] || continue
  $PY $M/scripts/trunc_io.py $K/qpc_$n $d $F/ref256 > /dev/null; idle
  /opt/qti-aic/exec/qaic-runner -t $K/qpc_$n -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 30 -S 1 -T 1 --write-output-start-iter 29 --write-output-num-samples 1 --write-output-dir $d/outputs > $d/runner.log 2>&1
  echo "run $n rc=$?: $(grep -E 'Inf/Sec' $d/runner.log | tr '\n' ' ' | cut -c1-120)"; grep -iE "error|fail" $d/runner.log | head -2; done
$PY - <<EOP
import numpy as np, glob
K = "$K"; ref = np.load("$F/ref256/trunc2_logits_last.npy").astype(np.float64); ys = {}
for n in ("t256_2_prod", "t256_2_naive", "t256_2_dyncard"):
    f = glob.glob(f"{K}/run_{n}/outputs/*logits*")
    if not f: print(n, "no output"); continue
    y = np.fromfile(f[0], np.float32).astype(np.float64).reshape(-1)[-ref.size:]; ys[n] = y
    print(f"{n:16s} vs FP32 two-layer reference: rel L2 {np.linalg.norm(y - ref) / np.linalg.norm(ref):.4f}, argmax {y.argmax()} (ref {ref.argmax()})")
if "t256_2_naive" in ys and "t256_2_dyncard" in ys:
    a, b = ys["t256_2_naive"], ys["t256_2_dyncard"]; print(f"runtime sort vs naive: rel L2 {np.linalg.norm(b - a) / np.linalg.norm(a):.3e}, max|diff| {np.abs(b - a).max():.4f}")
if "t256_2_naive" in ys and "t256_2_prod" in ys:
    a, b = ys["t256_2_prod"], ys["t256_2_naive"]; print(f"naive vs production: rel L2 {np.linalg.norm(b - a) / np.linalg.norm(a):.3e}")
EOP
echo E28A_DONE
