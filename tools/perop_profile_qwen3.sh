#!/bin/bash
# One-shot AI 100 per-op profiler for Qwen3-8B: PREFILL (seq_len=128) + DECODE (seq_len=1).
# Stats come ONLY from the documented SDK built-in commands (Cloud AI 1.21 Inference Profiling):
#   qaic-compile -stats-level=70  ->  qaic-runner --aic-profiling-type=raw_device_stats  ->  qaic-opstats --summary --trace
# Outputs per regime, under mllm/perop/<regime>/out/ :
#   *.qaic-opstats.summary.txt   (per-operator cycle table, per core)   <- SDK
#   *.qaic-opstats.trace.json    (Chrome/Perfetto trace)               <- SDK
# Optional (ARCH=1): a custom per-projection rollup of the SDK trace (post-processing only, not the stats).
set -e
source /home/chihao/qeff-venv/bin/activate 2>/dev/null || true
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH

# ---- inputs / knobs -------------------------------------------------------
ONNX=${ONNX:-/home/chihao/models/qeff_native_cache/Qwen3ForCausalLM/Qwen3ForCausalLM-d165ffa1b02f89bc/Qwen3ForCausalLM.onnx}
CUSTOM_IO=${CUSTOM_IO:-/home/chihao/models/perop_plain/custom_io.yaml}
CTX=${CTX:-256}          # context length (both regimes)
PREFILL_SEQ=${PREFILL_SEQ:-128}
CORES=${CORES:-16}
DEV=${DEV:-0}            # single card
SAMPLES=${SAMPLES:-4}   # profiling samples to capture/decode per regime
OUTROOT=${OUTROOT:-/home/chihao/mllm/perop}
TOOLS=$(cd "$(dirname "$0")" && pwd)
# ---------------------------------------------------------------------------

run_regime () {   # $1=name  $2=seq_len
  local name=$1 seq=$2          # NOTE: keep D on its own line — `local a=$1 b=$OUTROOT/$a`
  local D="$OUTROOT/$name"      # would expand $a before name=$1 takes effect (bash gotcha)
  echo "############### $name (ctx=$CTX seq_len=$seq) ###############"
  mkdir -p "$D/stats" "$D/out"
  printf '{"specializations":[{"batch_size":"1","ctx_len":"%s","seq_len":"%s"}]}\n' "$CTX" "$seq" > "$D/spec.json"
  cp "$CUSTOM_IO" "$D/"

  if [ -f "$D/qpc/programqpc.bin" ]; then
    echo "[1/3] compile: reusing existing $D/qpc/programqpc.bin"
  else
    echo "[1/3] qaic-compile (stats-level=70)"
    qaic-compile -aic-hw -aic-hw-version=ai100 -m="$ONNX" \
      -convert-to-fp16 -mxfp6-matmul -aic-num-cores=$CORES -mos=1 -aic-enable-depth-first \
      -network-specialization-config="$D/spec.json" -custom-IO-list-file="$D/custom_io.yaml" \
      -stats-level=70 -compile-only -aic-binary-dir="$D/qpc"
    [ -f "$D/qpc/programqpc.bin" ] || { echo "COMPILE FAILED ($name)"; exit 1; }
  fi

  # optional: drive the run with YOUR input (PROMPT / TOKEN_IDS) instead of random
  local -a inargs=()
  [ -n "$PROMPT" ]    && inargs=(--prompt "$PROMPT")           # multi-word safe
  [ -n "$TOKEN_IDS" ] && inargs=(--token-ids "$TOKEN_IDS")
  if [ ${#inargs[@]} -gt 0 ]; then
    echo "[input] building bindings from your input"
    python3 "$TOOLS/perop_make_input.py" --qpc "$D/qpc" "${inargs[@]}"
  else
    rm -rf "$D/qpc/bindings.json" "$D/qpc/bind"    # no input given -> random
  fi

  echo "[2/3] qaic-runner --aic-profiling-type=raw_device_stats ($SAMPLES samples, card $DEV)"
  rm -f "$D"/stats/*
  qaic-runner -t "$D/qpc" -d "$DEV" --aic-profiling-type raw_device_stats \
    --aic-profiling-num-samples "$SAMPLES" --aic-profiling-out-dir "$D/stats"

  echo "[3/3] qaic-opstats --summary --trace  (SDK built-in post-processing)"
  rm -f "$D"/out/*
  # qaic-opstats can crash enumerating a missing sample slot; it still writes the
  # samples it decoded first, so tolerate a nonzero exit and verify output exists.
  qaic-opstats --qpc "$D/qpc/programqpc.bin" --input-dir "$D/stats" --output-dir "$D/out" \
      --summary --trace \
    || echo "  (qaic-opstats exited nonzero — proceeding with the samples it decoded)"
  ls "$D"/out/*.qaic-opstats.summary.txt >/dev/null 2>&1 \
    || { echo "FAILED: no SDK summary produced ($name)"; exit 1; }
  cp "$(ls "$D"/out/*.qaic-opstats.summary.txt | head -1)" "$D/op_level_${name}.summary.txt"  # copy of one SDK summary
  echo "  SDK stats -> $D/out/*.qaic-opstats.summary.txt  (+ .trace.json)"

  # OPTIONAL (ARCH=1): custom per-projection rollup of the SDK trace. NOT part of the SDK stats.
  if [ "$ARCH" = 1 ]; then
    echo "[opt] ARCH=1: per-projection rollup of the SDK trace (custom post-processing)"
    local TR; TR=$(ls "$D"/out/*.qaic-opstats.trace.json | head -1)
    python3 "$TOOLS/perop_by_projection.py" "$TR" "$D/per_projection_${name}.csv" > "$D/per_projection_${name}.txt"
  fi
}

mkdir -p "$OUTROOT"
run_regime prefill "$PREFILL_SEQ"
run_regime decode  1

# ---- OPTIONAL combined side-by-side arch-level CSV (ARCH=1 only) -----------
if [ "$ARCH" = 1 ]; then
python3 - "$OUTROOT" <<'PY'
import csv, sys, os
root = sys.argv[1]
def load(p):
    d={}
    with open(p) as f:
        for r in csv.DictReader(f):
            d[r['projection']] = r
    return d
pre = load(f"{root}/prefill/per_projection_prefill.csv")
dec = load(f"{root}/decode/per_projection_decode.csv")
keys = [k for k in pre if k!='TOTAL']
out = f"{root}/prefill_vs_decode_by_projection.csv"
with open(out,"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["projection","prefill_pcycles","prefill_share%","decode_pcycles","decode_share%"])
    for k in keys:
        w.writerow([k, pre[k]['pcycles'], pre[k]['share_pct'],
                    dec.get(k,{}).get('pcycles',''), dec.get(k,{}).get('share_pct','')])
print("combined:", out)
PY
fi

# ---- correctness check: generate the model's actual response for the prompt ----
# Uses the retained-state generation QPC (real KV-cache decode loop), not the profiling QPC.
if [ -n "$PROMPT" ]; then
  echo
  echo "############### response (correctness check) ###############"
  python3 "$TOOLS/perop_generate.py" --prompt "$PROMPT" --device "$DEV" \
    --gen-len "${GEN_LEN:-20}" ${GEN_QPC:+--qpc "$GEN_QPC"} \
    --out "$OUTROOT/response.txt" 2>&1 \
    | grep -viE 'FutureWarning|param_schemas|onnxscript|resume_download|Fetching|it/s\]|unauthenticated|HF_TOKEN' \
    || echo "(generation step failed — check torch/QEfficient + GEN_QPC)"
fi

echo
echo "===== DONE. SDK stats under $OUTROOT ====="
echo "OP-LEVEL summaries (qaic-opstats --summary):"
find "$OUTROOT" -maxdepth 3 -name '*.qaic-opstats.summary.txt' | sort | sed 's/^/  /'
echo "TRACES (qaic-opstats --trace; open in https://ui.perfetto.dev or chrome://tracing):"
find "$OUTROOT" -maxdepth 3 -name '*.qaic-opstats.trace.json' | sort | sed 's/^/  /'
[ -n "$PROMPT" ] && echo "RESPONSE (correctness): $OUTROOT/response.txt"
if [ "$ARCH" = 1 ]; then
  echo "OPTIONAL per-projection rollup (ARCH=1, custom):"
  find "$OUTROOT" -maxdepth 2 \( -name 'per_projection_*.txt' -o -name 'prefill_vs_decode_*.csv' \) | sort | sed 's/^/  /'
fi
