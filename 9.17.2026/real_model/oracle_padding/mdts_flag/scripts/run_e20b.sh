#!/bin/bash
# E20b: runtime expert selection cost — hot banks gathered by index (fp16) vs static MXFP6 / static fp16, T=128 real routing
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e20; mkdir -p $E/qpc
comp() { local name=$1 model=$2 extra=$3 stats=${4:-0} out=${5:-$E/qpc/$1}; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 $extra -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 3,12p | cut -c1-220; }
comp T128_static_mx $S/e8/T128_hc_92_2/model.onnx "-mxfp6-matmul" & comp T128_static_fp16 $S/e8/T128_hc_92_2/model.onnx "" & comp T128_dyngather_mx $E/T128_dyngather/model.onnx "-mxfp6-matmul" & comp T128_dyngather_fp16 $E/T128_dyngather/model.onnx "" & wait
comp T128_dyngather_mx_s70 $E/T128_dyngather/model.onnx "-mxfp6-matmul" 70 $E/T128_dyngather/qpc_s70
$PY - <<EOP
import json, os
S="$S"; E="$E"; anc=f"{S}/e8/T128_hc_92_2"; refy=f"{S}/e8/timing_D_runs/T128_hc_92_2__mx_r0/y.bin"
dirs={"T128_static_mx":anc,"T128_static_fp16":anc,"T128_dyngather_mx":f"{E}/T128_dyngather","T128_dyngather_fp16":f"{E}/T128_dyngather"}
cases=[dict(label=n, qpc=f"{E}/qpc/{n}", input=f"{d}/input_f16.bin", counts=f"{d}/expected_counts_i32.bin", ref_y=refy) for n,d in dirs.items() if os.path.isfile(f"{E}/qpc/{n}/programqpc.bin")]
json.dump(dict(cases=cases), open(f"{E}/spec_b.json","w"), indent=1); print(len(cases), "cases")
EOP
$PY $S/e7_timing.py $E/spec_b.json $E/timing_b.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|round 0"
d=$E/T128_dyngather; bash $S/profile_case2.sh $d 128 "T128 dyngather MXFP6" | head -1; $PY $S/tail_breakdown.py $d/analysis/sample0 128 "T128_dyngather" 2>&1 | sed -n 1,7p; $PY $S/e18/e18_hot.py $d/analysis/sample0 "dyngather hot/cold engines"
$PY $S/kv/placement.py $d/analysis/sample0 "" 2>/dev/null | head -14
echo E20B_DONE
