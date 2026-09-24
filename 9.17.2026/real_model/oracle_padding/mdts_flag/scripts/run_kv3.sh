#!/bin/bash
# KV round 3: head-parallel attention on the 2-layer graph, with the flag (retained state) - bindings + instrumented build
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest
comprs() { local name=$1 graph=$2 extra=$3; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,12p | cut -c1-240; }
comprs trunc2_hp_mos1 trunc2_headpar "-mdts-mos=1" &
comprs trunc2_hp_noflag trunc2_headpar "" &
comprs trunc2_hp_mos1_s70 trunc2_headpar "-mdts-mos=1 -stats-level=70" &
wait
for n in trunc2_hp_mos1 trunc2_hp_noflag; do $PY $M/scripts/qpc_bindings.py $K/qpc_$n 2>&1 | grep '###' | cut -c1-300; done
echo KV3_DONE
