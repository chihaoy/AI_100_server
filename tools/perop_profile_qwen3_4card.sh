#!/bin/bash
# 4-CARD AI 100 per-op profiler for Qwen3-8B: PREFILL + DECODE, tensor-parallel over 4 devices.
# Same SDK built-in chain as the 1-card script, plus MDP partitioning and multi-device merge:
#   qaic-compile -stats-level=70 -mdp-load-partition-config=<mdp.json>
#   qaic-runner  -D <devlist> --aic-profiling-type=raw_device_stats
#   qaic-opstats --summary --trace --merge-mq-traces
# Outputs per regime under mllm/perop4/<regime>/out/ :  *.qaic-opstats.summary.txt + *.trace.json (SDK).
# Optional (ARCH=1): per-projection rollup of the SDK trace (post-processing only).
set -e
source /home/chihao/qeff-venv/bin/activate 2>/dev/null || true
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH

# ---- inputs / knobs -------------------------------------------------------
ONNX=${ONNX:-/home/chihao/models/qwen3_8b_repacked/Qwen3ForCausalLM.onnx}          # repacked model for multi-card
CUSTOM_IO=${CUSTOM_IO:-/home/chihao/models/qwen3_8b_repacked/custom_io.yaml}
MDP=${MDP:-/home/chihao/models/qwen3_8b_4card/mdp_ts_4.json}                        # 4x16-core partition config
CTX=${CTX:-256}
PREFILL_SEQ=${PREFILL_SEQ:-128}
CORES=${CORES:-16}                # cores per device
DEVLIST=${DEVLIST:-0:1:2:3}       # qaic-runner -D device list (4 cards)
SAMPLES=${SAMPLES:-4}
NUMITER=${NUMITER:-100}
OUTROOT=${OUTROOT:-/home/chihao/mllm/perop4}
TOOLS=$(cd "$(dirname "$0")" && pwd)
# ---------------------------------------------------------------------------

run_regime () {   # $1=name  $2=seq_len
  local name=$1 seq=$2
  local D="$OUTROOT/$name"
  echo "############### 4-CARD $name (ctx=$CTX seq_len=$seq, devices $DEVLIST) ###############"
  mkdir -p "$D/stats" "$D/out"
  printf '{"specializations":[{"batch_size":"1","ctx_len":"%s","seq_len":"%s"}]}\n' "$CTX" "$seq" > "$D/spec.json"
  cp "$CUSTOM_IO" "$D/" 2>/dev/null || true

  if [ -f "$D/qpc/programqpc.bin" ]; then
    echo "[1/3] compile: reusing existing $D/qpc/programqpc.bin"
  else
    echo "[1/3] qaic-compile (stats-level=70, MDP 4-card) — this is a long compile"
    qaic-compile -aic-hw -aic-hw-version=2.0 -m="$ONNX" \
      -convert-to-fp16 -mxfp6-matmul -aic-num-cores=$CORES -mos=1 -aic-enable-depth-first \
      -network-specialization-config="$D/spec.json" -custom-IO-list-file="$D/custom_io.yaml" \
      -mdp-load-partition-config="$MDP" \
      -stats-level=70 -compile-only -aic-binary-dir="$D/qpc"
    [ -f "$D/qpc/programqpc.bin" ] || { echo "COMPILE FAILED ($name)"; exit 1; }
  fi

  # Always feed EXPLICIT correctly-sized bindings (MDP + qaic-runner random binding is flaky).
  # PROMPT/TOKEN_IDS if given, else a dummy token (values don't affect cycle counts).
  local -a inargs=(--token-ids "1")
  [ -n "$PROMPT" ]    && inargs=(--prompt "$PROMPT")
  [ -n "$TOKEN_IDS" ] && inargs=(--token-ids "$TOKEN_IDS")
  echo "[input] building bindings (${inargs[0]} ${inargs[1]:0:30})"
  python3 "$TOOLS/perop_make_input.py" --qpc "$D/qpc" "${inargs[@]}"

  echo "[2/3] qaic-runner -D $DEVLIST --aic-profiling-type=raw_device_stats ($SAMPLES samples)"
  rm -f "$D"/stats/*
  qaic-runner -t "$D/qpc" -D "$DEVLIST" --aic-profiling-type raw_device_stats \
    --aic-profiling-num-samples "$SAMPLES" --aic-profiling-out-dir "$D/stats" --num-iter "$NUMITER"

  echo "[3/3] qaic-opstats --summary --trace --merge-mq-traces (SDK, merge 4 devices)"
  rm -f "$D"/out/*
  # FLOW=full adds cross-op dependency arrows (incl. cross-card multicast_out->_in data flow)
  qaic-opstats --qpc "$D/qpc/programqpc.bin" --input-dir "$D/stats" --output-dir "$D/out" \
      --summary --trace --merge-mq-traces true --flow-events "${FLOW:-none}" \
    || echo "  (qaic-opstats exited nonzero — proceeding with samples it decoded)"
  ls "$D"/out/*.qaic-opstats.summary.txt >/dev/null 2>&1 \
    || { echo "FAILED: no SDK summary produced ($name)"; exit 1; }
  cp "$(ls "$D"/out/*.qaic-opstats.summary.txt | head -1)" "$D/op_level_${name}.summary.txt"
  echo "  SDK stats -> $D/out/*.qaic-opstats.summary.txt (+ .trace.json)"

  if [ "$ARCH" = 1 ]; then
    echo "[opt] ARCH=1: per-projection rollup of the SDK trace (custom post-processing)"
    local TR; TR=$(ls "$D"/out/*.qaic-opstats.trace.json | head -1)
    python3 "$TOOLS/perop_by_projection.py" "$TR" "$D/per_projection_${name}.csv" > "$D/per_projection_${name}.txt"
  fi
}

mkdir -p "$OUTROOT"
for r in ${REGIMES:-prefill decode}; do
  case "$r" in
    prefill) run_regime prefill "$PREFILL_SEQ" ;;
    decode)  run_regime decode  1 ;;
    *) echo "unknown regime: $r" ;;
  esac
done

echo
echo "===== DONE. 4-card SDK stats under $OUTROOT ====="
find "$OUTROOT" -maxdepth 3 -name '*.qaic-opstats.summary.txt' | sort | sed 's/^/  /'
