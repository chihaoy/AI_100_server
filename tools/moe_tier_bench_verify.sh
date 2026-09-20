#!/bin/bash
# Numerics check of a tier-bench variant on the card: compile the SAME ONNX with fp16 weights (no MXFP6,
# so quantisation noise does not mask logic errors), run it on the recorded input x.bin, compare with the
# CPU capacity-drop reference.   tools/moe_tier_bench_verify.sh <variant_dir> [num_devices]
set -e
source /home/chihao/qeff-venv/bin/activate
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
D=$1; ND=${2:-4}
MDPARG=""; [ "$ND" -gt 1 ] && MDPARG="-mdp-load-partition-config=$D/mdp.json"
if [ ! -f "$D/qpc_fp16/programqpc.bin" ]; then
  qaic-compile -aic-hw -aic-hw-version=ai100 -m="$D/moe_layer.onnx" -convert-to-fp16 -aic-num-cores=16 -mos=1 \
    -aic-enable-depth-first $MDPARG -compile-only -aic-binary-dir="$D/qpc_fp16" > "$D/compile_fp16.log" 2>&1 || { tail -3 "$D/compile_fp16.log"; exit 1; }
fi
rm -rf "$D/outdir_fp16"; mkdir -p "$D/outdir_fp16"
if [ "$ND" -gt 1 ]; then DEVARG="-D $(seq -s: 0 $((ND-1)))"; else DEVARG="-d 0"; fi
qaic-runner -t "$D/qpc_fp16" $DEVARG -i "$D/x.bin" --write-output-dir "$D/outdir_fp16" --num-iter 1 > "$D/run_fp16.log" 2>&1 || { tail -3 "$D/run_fp16.log"; exit 1; }
python3 "$(dirname "$0")/moe_tier_bench_verify.py" "$D" "$D/outdir_fp16/y-activation-0-inf-0.bin" | tee "$D/verify_fp16.txt"
