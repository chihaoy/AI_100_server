#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad
prof() { local d=$1 T=$2 label=$3; rm -rf $d/qpc_s70 $d/stats $d/trace $d/analysis $d/meta $d/outputs; t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$d/model.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$d/qpc_s70 -stats-level=70 -mdts-mos=1 > $d/compile_s70.log 2>&1; echo "### $label stats70 rc=$? $(( $(date +%s)-t0 )) s"
  bash $S/profile_case.sh $d $T "$label"; }
prof $S/e11/T512_tokencentric 512 "T512 token-centric MXFP6"
prof $S/e13/T128_anchor 128 "T128 anchor hc 92/2 MXFP6"
prof $S/e13/T512_anchor 512 "T512 anchor hc 296/32 MXFP6"
echo E13_DONE
