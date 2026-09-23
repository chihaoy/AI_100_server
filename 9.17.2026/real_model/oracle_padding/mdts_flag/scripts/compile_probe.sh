#!/bin/bash
# usage: compile_probe.sh <name> <model.onnx> [extra qaic-compile flags...]
# Compiles with the project's standard flags (stats-level 0), extracts descriptors, prints inventory, deletes the QPC.
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad
name=$1; model=$2; shift 2
out=$S/probe/$name; mkdir -p $out
CFG=/home/chihao/models/qwen3_30b_a3b/ep/qeff_cache/Qwen3MoeForCausalLM/Qwen3MoeForCausalLM-ed085dabb01c623b/qpc-6ff7c7675b1fcae0
start=$(date +%s)
/opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
  -mdp-load-partition-config=$CFG/mdp_ts_4.json -compile-only -aic-binary-dir=$out/qpc -stats-level=70 "$@" > $out/compile.log 2>&1
rc=$?; end=$(date +%s)
echo "### $name: compile rc=$rc in $((end-start)) s, flags: $*"
if [ -f $out/qpc/programqpc.bin ]; then
  ls -l $out/qpc/programqpc.bin | awk '{print "    qpc bytes", $5}'
  /opt/qti-aic/tools/qaic-qpc extract --qpc $out/qpc/programqpc.bin --output-dir $out/meta -s '*opstatsdesc.bin' >/dev/null 2>&1
  /home/chihao/qeff-venv/bin/python $S/qpc_inventory.py $out/meta
  rm -rf $out/qpc
else
  tail -15 $out/compile.log
fi
