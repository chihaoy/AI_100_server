#!/bin/bash
# Expert-placement probes on the single-layer EP graph (see moe_probe_lane2device.py / moe_probe_lane2core.py).
#   tools/moe_probe_placement.sh partial   # card level: per-device partials, fp16 4-card build, lstsq readout
#   tools/moe_probe_placement.sh split     # core level: one node per lane, MXFP6 build + opstats, node-name readout
set -e
source /home/chihao/qeff-venv/bin/activate
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
TOOLS=$(cd "$(dirname "$0")" && pwd); ROOT=${OUTROOT:-/home/chihao/mllm/perop_moe/tier_bench}
case "$1" in
partial)
  D=$ROOT/probe_partial; mkdir -p "$D"
  [ -f "$D/moe_layer.onnx" ] || python3 "$TOOLS/moe_tier_bench_export.py" --out "$D" --groups 128:64 --num-devices 4 --emit-partial | tail -2
  python3 - "$D/mdp.json" <<'PY'
import json,sys
json.dump({"connections":[{"devices":[0,1,2,3],"type":"p2p"}],
           "partitions":[{"devices":[{"deviceId":i,"numCores":16} for i in range(4)],"name":"Partition0"}]},open(sys.argv[1],"w"),indent=1)
PY
  if [ ! -f "$D/qpc_fp16/programqpc.bin" ]; then
    qaic-compile -aic-hw -aic-hw-version=ai100 -m="$D/moe_layer.onnx" -convert-to-fp16 -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
      -mdp-load-partition-config="$D/mdp.json" -compile-only -aic-binary-dir="$D/qpc_fp16" > "$D/compile_fp16.log" 2>&1 || { tail -3 "$D/compile_fp16.log"; exit 1; }
  fi
  rm -rf "$D/outdir_fp16"; mkdir -p "$D/outdir_fp16"
  qaic-runner -t "$D/qpc_fp16" -D 0:1:2:3 -i "$D/x.bin" --write-output-dir "$D/outdir_fp16" --num-iter 1 > "$D/run_fp16.log" 2>&1 || { tail -3 "$D/run_fp16.log"; exit 1; }
  ls "$D/outdir_fp16"
  python3 "$TOOLS/moe_probe_lane2device.py" "$D" "$D"/outdir_fp16/partial-activation-0-inf-0.bin | tee "$D/probe_lane2device.txt"
  ;;
split)
  EXTRA="--split-lanes" "$TOOLS/moe_tier_bench.sh" c1_split 128:16 1
  python3 "$TOOLS/moe_probe_lane2core.py" "$ROOT/c1_split" | tee "$ROOT/c1_split/probe_lane2core.txt"
  EXTRA="--split-lanes" "$TOOLS/moe_tier_bench.sh" split4 128:64 4
  for c in 0 1 2 3; do python3 "$TOOLS/moe_probe_lane2core.py" "$ROOT/split4" --card $c; done | tee "$ROOT/split4/probe_lane2core.txt"
  ;;
esac
