#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad
comp70() { local v=$1 extra=$2; local d=$S/kv/T256_tokenowned_$v; rm -rf $d/qpc_s70; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$S/e15/T256_tokenowned/model.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$d/qpc_s70 -stats-level=70 $extra > $d/compile_s70.log 2>&1; echo "### s70 $v rc=$? $(( $(date +%s)-t0 )) s"; }
comp70 flag "-mdts-mos=1" & comp70 noflag "" & wait
# wait for the timing run to release the devices
for i in $(seq 120); do pgrep -f 'e7_timing.py .*spec_T256' >/dev/null || break; sleep 10; done
for v in flag noflag; do bash $S/profile_case.sh $S/kv/T256_tokenowned_$v 256 "T256 token-owned $v"; done
echo KV1C_DONE
