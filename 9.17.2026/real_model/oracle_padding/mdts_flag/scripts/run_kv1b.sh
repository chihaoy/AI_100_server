#!/bin/bash
# Companion: T256 token-owned replay compiled with and without -mdts-mos=1
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; E=$S/kv; mkdir -p $E/qpc
comp() { local name=$1 model=$2 extra=$3; local out=$E/qpc/$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=0 $extra > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 4,12p | cut -c1-220; }
comp T256_tokenowned_flag   $S/e15/T256_tokenowned/model.onnx "-mdts-mos=1" &
comp T256_tokenowned_noflag $S/e15/T256_tokenowned/model.onnx "" &
wait; echo KV1B_DONE
