#!/bin/bash
# export + compile the hot and cold QPCs of one layer:  build_layer.sh <layer>
set -e
source /home/chihao/qeff-venv/bin/activate; export PATH=/opt/qti-aic/exec:$PATH
L=$1; E=/home/chihao/mllm/9.17.2026/e2e; D=$E/L$L; mkdir -p $D
HW=$(python3 -c "import json;p=json.load(open('$E/plan.json'))['$L'];print(','.join(map(str,p['hot_widths'])))")
CW=$(python3 -c "import json;p=json.load(open('$E/plan.json'))['$L'];print(','.join(map(str,p['cold_widths'])))")
for G in hot cold; do
  if [ $G = hot ]; then LR=0,32; SW=$HW; RES=--with-residual; else LR=32,64; SW=$CW; RES=; fi
  if [ ! -f $D/$G/qpc/programqpc.bin ]; then
    [ -f $D/$G/moe_layer.onnx ] || python3 /home/chihao/mllm/research/tools/moe_e2e_layer_export.py --layer $L --out $D/$G --lane-range $LR --stage-widths $SW --order $E/order_L$L.json --x-npy $E/ref/h_L$L.npy $RES > $D/$G.export.log 2>&1 || { echo "L$L $G EXPORT FAILED"; exit 1; }
    rm -rf $D/$G/qpc
    qaic-compile -aic-hw -aic-hw-version=ai100 -m=$D/$G/moe_layer.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$D/$G/mdp.json -compile-only -aic-binary-dir=$D/$G/qpc > $D/$G.compile.log 2>&1 || { echo "L$L $G COMPILE FAILED"; exit 1; }
    rm -f $D/$G/moe_layer.onnx.data      # 600 MB per graph; the QPC is what we keep
  fi
  echo "L$L $G OK widths=$SW"
done
