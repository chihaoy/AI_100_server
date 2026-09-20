#!/bin/bash
# Compile + run + profile one single-layer MoE tier-bench variant on AI 100.
#   tools/moe_tier_bench.sh <name> <groups> [num_devices]
#   e.g. tools/moe_tier_bench.sh u128 128:16 1
#        tools/moe_tier_bench.sh tier_l12 128:1,64:2,32:4,16:4,8:5 1
#        tools/moe_tier_bench.sh u128_4c 128:64 4
# Output: $OUTROOT/<name>/{info.json,moe_layer.onnx,qpc/,stats/,out/,compile.log,run.log}
set -e
source /home/chihao/qeff-venv/bin/activate
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
NAME=$1; TGROUPS=$2; ND=${3:-1}
OUTROOT=${OUTROOT:-/home/chihao/mllm/perop_moe/tier_bench}
TOOLS=$(cd "$(dirname "$0")" && pwd)
D=$OUTROOT/$NAME; mkdir -p "$D"
EXTRA=${EXTRA:-}            # extra export args, e.g. "-T 256"
EXPORTER=${EXPORTER:-$TOOLS/moe_tier_bench_export.py}   # alt exporter, e.g. moe_decode_bmm_export.py
CFLAGS=${CFLAGS:-}          # extra qaic-compile flags, e.g. "-size-split-granularity=512"
ONNX_FROM=${ONNX_FROM:-}    # reuse an existing variant's ONNX (+info.json, x.bin, y_*.npy) instead of exporting
CORES=${CORES:-16}
SAMPLES=${SAMPLES:-2}; NUMITER=${NUMITER:-50}

echo "########## $NAME  groups=$TGROUPS  devices=$ND ##########"
if [ ! -f "$D/moe_layer.onnx" ]; then
  if [ -n "$ONNX_FROM" ]; then
    ln -s "$(readlink -f "$ONNX_FROM")/moe_layer.onnx" "$D/moe_layer.onnx"; cp "$ONNX_FROM"/info.json "$D/"; cp "$ONNX_FROM"/x.bin "$ONNX_FROM"/y_*.npy "$D/" 2>/dev/null || true
    echo "reusing ONNX from $ONNX_FROM"
  else
    python3 "$EXPORTER" --out "$D" --groups "$TGROUPS" --num-devices "$ND" $EXTRA 2>&1 | tail -6
  fi
fi
[ -n "$CFLAGS" ] && echo "$CFLAGS" > "$D/cflags.txt"

MDPARG=""
if [ "$ND" -gt 1 ]; then
  python3 - "$D/mdp.json" "$ND" "$CORES" <<'PY'
import json,sys
p,nd,c=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
json.dump({"connections":[{"devices":list(range(nd)),"type":"p2p"}],
           "partitions":[{"devices":[{"deviceId":i,"numCores":c} for i in range(nd)],"name":"Partition0"}]},open(p,"w"),indent=1)
PY
  MDPARG="-mdp-load-partition-config=$D/mdp.json"
fi

if [ ! -f "$D/qpc/programqpc.bin" ]; then
  echo "[1/3] qaic-compile"
  rm -rf "$D/qpc"
  /usr/bin/time -f "compile_wall_s=%e maxrss_kb=%M" qaic-compile -aic-hw -aic-hw-version=ai100 -m="$D/moe_layer.onnx" \
    -convert-to-fp16 -mxfp6-matmul -aic-num-cores=$CORES ${MOS:--mos=1} ${DFS:--aic-enable-depth-first} $CFLAGS \
    $MDPARG -stats-level=70 -compile-only -aic-binary-dir="$D/qpc" > "$D/compile.log" 2>&1 || true
  tail -3 "$D/compile.log"
  [ -f "$D/qpc/programqpc.bin" ] || { echo "COMPILE FAILED ($NAME) — see $D/compile.log"; exit 1; }
fi

echo "[2/3] qaic-runner (profiling)"
rm -rf "$D/stats"; mkdir -p "$D/stats"
if [ "$ND" -gt 1 ]; then DEVARG="-D $(seq -s: 0 $((ND-1)))"; else DEVARG="-d ${DEV:-0}"; fi
qaic-runner -t "$D/qpc" $DEVARG --aic-profiling-type raw_device_stats \
  --aic-profiling-num-samples "$SAMPLES" --aic-profiling-out-dir "$D/stats" --num-iter "$NUMITER" > "$D/run.log" 2>&1 || true
grep -E "InfPCycles|ExecTimeUs|InfPerSec|Error|error" "$D/run.log" | head -8

echo "[3/3] qaic-opstats"
rm -rf "$D/out"; mkdir -p "$D/out"
qaic-opstats --qpc "$D/qpc/programqpc.bin" --input-dir "$D/stats" --output-dir "$D/out" \
  --summary --trace --merge-mq-traces true --flow-events none > "$D/opstats.log" 2>&1 || echo "(opstats nonzero)"
ls "$D"/out/*.summary.txt 2>/dev/null | wc -l | sed 's/^/summaries: /'
