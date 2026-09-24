#!/bin/bash
# KV round 1: does -mdts-mos break retained-state pairing on a 2-layer graph, and which values do it?
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server; M=$R/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; mkdir -p $K
comprs() { local name=$1 graph=$2 extra=$3; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,9p | cut -c1-200; }
comprs trunc2_noflag trunc2_native "" &
comprs trunc2_mos1   trunc2_native "-mdts-mos=1" &
comprs trunc2_mos2   trunc2_native "-mdts-mos=2" &
comprs trunc2_mos4   trunc2_native "-mdts-mos=4" &
wait
for n in trunc2_noflag trunc2_mos1 trunc2_mos2 trunc2_mos4; do $PY $M/scripts/qpc_bindings.py $K/qpc_$n 2>&1 | cut -c1-400; done
echo KV1_DONE
