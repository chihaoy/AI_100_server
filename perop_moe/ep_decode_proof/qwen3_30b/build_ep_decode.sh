#!/bin/bash
# EP-for-decode pipeline v2: (1) QEfficient export with prefill_only=False (its own compile fails on this SDK's
# spec JSON format, as in run_ep.py), (2) manual decode-only compiles (seq_len=1) at B=8 and B=1.
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
export QEFF_HOME=/home/chihao/models/qwen3_30b_a3b/ep
E=$QEFF_HOME; source /home/chihao/qeff-venv/bin/activate; cd $E
touch $E/.pipeline_start
( while true; do echo "$(date +%H:%M:%S) $(free -g | awk 'NR==2{print "used="$3" free="$4" avail="$7}') rss_py=$(ps -o rss= -C python3 2>/dev/null | sort -n | tail -1)"; sleep 15; done ) > mem_sampler.log 2>&1 &
SAMPLER=$!
echo "=== [1] export ($(date +%H:%M:%S)) ==="
B=8 python3 run_ep_decode.py > run_ep_decode_B8.log 2>&1
echo "export step exit=$? ($(date +%H:%M:%S))"; grep -E "COMPILE_OK|Compilation failed|Killed" run_ep_decode_B8.log | head -3
kill $SAMPLER 2>/dev/null
Q=$(find $E/qeff_cache/Qwen3MoeForCausalLM -maxdepth 2 -type d -name "qpc-*" -newer $E/.pipeline_start | head -1)
[ -z "$Q" ] && { echo "ABORT: no new qpc-* dir created after pipeline start (export did not complete)"; exit 1; }
M=$(dirname $Q); echo "Q=$Q"; ls $Q; echo "spec from QEfficient:"; cat $Q/specializations.json
[ -f $M/Qwen3MoeForCausalLM.onnx ] || { echo "ABORT: no ONNX in $M"; exit 1; }
for B in 8 1; do
  echo "=== [2] manual decode-only compile B=$B ($(date +%H:%M:%S)) ==="
  echo "{\"specializations\": [{\"batch_size\": \"$B\", \"ctx_len\": \"256\", \"seq_len\": \"1\"}]}" > $Q/spec_decode_B$B.json
  rm -rf $Q/qpc_decode_B$B
  /usr/bin/time -f "compile_wall_s=%e maxrss_kb=%M" qaic-compile -aic-hw -aic-hw-version=ai100 -m=$M/Qwen3MoeForCausalLM.onnx \
    -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
    -mdp-load-partition-config=$Q/mdp_ts_4.json -network-specialization-config=$Q/spec_decode_B$B.json \
    -custom-IO-list-file=$Q/custom_io.yaml -stats-level=70 -compile-only -aic-binary-dir=$Q/qpc_decode_B$B \
    > $Q/compile_decode_B$B.log 2>&1
  echo "compile B=$B rc=$? $( [ -f $Q/qpc_decode_B$B/programqpc.bin ] && echo QPC_OK || echo QPC_MISSING) ($(date +%H:%M:%S))"; tail -2 $Q/compile_decode_B$B.log
done
echo PIPELINE_DONE
