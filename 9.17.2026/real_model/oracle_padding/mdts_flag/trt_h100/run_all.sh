#!/bin/bash
# TensorRT/H100 experiment driver (RANKTIER_TENSORRT_H100_EXPERIMENT_PLAN.md): export the MoE-block variants of one layer,
# build engines with identical settings, check every engine on the real, balanced, skewed and forced-overflow traces, then
# time the accepted engines (eager and CUDA graph) and the native vLLM reference. One chunk length per call, T=512 first.
# Usage: run_all.sh <T> [layer=L20] [tiers=3t_32_32_64]   (env: PY_TRT, PY_VLLM python interpreters; GPU=index)
set -u
T=$1; L=${2:-L20}; TIERS=${3:-3t_32_32_64}
H=$(cd "$(dirname "$0")" && pwd); cd "$H"
PY_TRT=${PY_TRT:-python3}; PY_VLLM=${PY_VLLM:-}; export CUDA_VISIBLE_DEVICES=${GPU:-0}
O=onnx/$L; G=engines/$L; R=results/${L}_T$T; mkdir -p $O $G $R
LOG=$R/run_all.log; exec > >(tee -a $LOG) 2>&1
echo "=== $(date +%F_%T) T=$T layer=$L tiers=$TIERS GPU=$CUDA_VISIBLE_DEVICES on $(hostname)"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
VARS=S01,S00,G00,G01,U00,U01,G10,G11
TAGS="S01_T$T,S00_T$T,G00_T$T,G01_T$T,U00_T${T}_$TIERS,U01_T${T}_$TIERS,G10_T${T}_$TIERS,G11_T${T}_$TIERS"
MISSING=""; for t in ${TAGS//,/ }; do [ -f $G/$t.engine ] || [ -f $O/$t.onnx ] || MISSING="$MISSING,${t%%_*}"; done
[ -n "$MISSING" ] && $PY_TRT export_onnx.py --data data/$L --T $T --variants ${MISSING#,} --tiers $TIERS --out $O
$PY_TRT build_engines.py $O $G ${TAGS//,/ }
for t in ${TAGS//,/ }; do [ -f $G/$t.engine ] && rm -f $O/$t.onnx $O/$t.onnx.data; done      # engines embed the weights; keep the .json
for tr in balanced skewed overflow real; do
  echo "--- check, trace $tr"
  $PY_TRT run_trt.py --engines $G --onnx $O --data data/$L --T $T --tags $TAGS --out $R --trace $tr --rounds ${ROUNDS:-6}
done
[ -n "$PY_VLLM" ] && $PY_VLLM native_ref.py --data data/$L --T $T --out $R
echo "=== $(date +%F_%T) done"
