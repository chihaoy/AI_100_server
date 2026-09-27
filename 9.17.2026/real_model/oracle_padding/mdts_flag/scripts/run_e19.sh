#!/bin/bash
# E19: deployable lane split — all 128 experts kept, cold stage widened to 64+k lanes; vs anchors and the E18 drop-empties splits
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e18; mkdir -p $E/qpc
comp() { local name=$1 model=$2 stats=${3:-0} out=${4:-$E/qpc/$1}; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 4,12p | cut -c1-220; }
echo "===== E19 COMPILE ====="
comp T512_wide104 $E/T512_wide104_to/model.onnx & comp T512_wide88 $E/T512_wide88_to/model.onnx & comp T256_wide58 $E/T256_wide58_to/model.onnx & comp T256_wide84 $E/T256_wide84_to/model.onnx & wait
$PY - <<EOP
import json, os
S="$S"; E="$E"
anc={"T256":f"{S}/e7/T256_hc_158_4","T512":f"{S}/e7/T512_hc_296_32"}; refy={"T256":f"{S}/e8/timing_D_runs/T256_hc_158_4__mx_r0/y.bin","T512":f"{S}/e8/timing_D_runs/T512_hc_296_32__mx_r0/y.bin"}
dirs={"T256_anchor":None,"T256_split58":f"{E}/T256_split58_to","T256_wide84":f"{E}/T256_wide84_to","T256_wide58":f"{E}/T256_wide58_to",
      "T512_anchor":None,"T512_split104":f"{E}/T512_split104_to","T512_split88":f"{E}/T512_split88_to","T512_wide104":f"{E}/T512_wide104_to","T512_wide88":f"{E}/T512_wide88_to"}
cases=[]
for name,d in dirs.items():
    T=name[:4]
    if not os.path.isfile(f"{E}/qpc/{name}/programqpc.bin"): print("missing", name); continue
    src = d or anc[T]
    cases.append(dict(label=name, qpc=f"{E}/qpc/{name}", input=f"{src}/input_f16.bin", counts=f"{src}/expected_counts_i32.bin", ref_y=refy[T]))
json.dump(dict(cases=cases), open(f"{E}/spec19.json","w"), indent=1); print(len(cases), "cases")
EOP
echo "===== E19 TIMING (uninstrumented, 3 alternating rounds x 100) ====="
$PY $S/e7_timing.py $E/spec19.json $E/timing19.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|WARN|waiting|round 0"
echo "===== E19 PROFILE: T512 wide104 (stats70) ====="
d=$E/T512_wide104_to; comp T512_wide104_s70 $d/model.onnx 70 $d/qpc_s70; bash $S/profile_case2.sh $d 512 "T512_wide104 MXFP6"; $PY $S/tail_breakdown.py $d/analysis/sample0 512 "T512_wide104" 2>&1 | head -12; $PY $E/e18_hot.py $d/analysis/sample0 "T512_wide104 hot/cold engines"
echo E19_DONE
