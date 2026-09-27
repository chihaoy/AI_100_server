#!/bin/bash
# E18: exact-shape hot stage — lane-split hot experts (virtual lanes) and row-chunked hot GEMMs vs the token-owned anchors
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e18; mkdir -p $E/qpc
comp() { local name=$1 model=$2 stats=${3:-0} out=${4:-$E/qpc/$1}; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 4,12p | cut -c1-220; }
declare -A DIR=([T256_anchor]=$S/e15/T256_tokenowned [T512_anchor]=$S/e15/T512_tokenowned [T256_split84]=$E/T256_split84_to [T256_split58]=$E/T256_split58_to [T512_split148]=$E/T512_split148_to [T512_split104]=$E/T512_split104_to [T512_split88]=$E/T512_split88_to [T512_rowchunk2]=$E/T512_rowchunk2 [T512_rowchunk4]=$E/T512_rowchunk4)
declare -A ANC=([T256]=$S/e7/T256_hc_158_4 [T512]=$S/e7/T512_hc_296_32)
echo "===== E18 COMPILE ====="
comp T256_anchor ${DIR[T256_anchor]}/model.onnx & comp T512_anchor ${DIR[T512_anchor]}/model.onnx & comp T256_split84 ${DIR[T256_split84]}/model.onnx & comp T256_split58 ${DIR[T256_split58]}/model.onnx & wait
comp T512_split148 ${DIR[T512_split148]}/model.onnx & comp T512_split104 ${DIR[T512_split104]}/model.onnx & comp T512_split88 ${DIR[T512_split88]}/model.onnx & wait
comp T512_rowchunk2 ${DIR[T512_rowchunk2]}/model.onnx & comp T512_rowchunk4 ${DIR[T512_rowchunk4]}/model.onnx & wait
$PY - <<EOP
import json, os
S="$S"; E="$E"
anc={"T256":f"{S}/e7/T256_hc_158_4","T512":f"{S}/e7/T512_hc_296_32"}; refy={"T256":f"{S}/e8/timing_D_runs/T256_hc_158_4__mx_r0/y.bin","T512":f"{S}/e8/timing_D_runs/T512_hc_296_32__mx_r0/y.bin"}
dirs={"T256_anchor":None,"T256_split84":f"{E}/T256_split84_to","T256_split58":f"{E}/T256_split58_to","T512_anchor":None,"T512_rowchunk2":None,"T512_rowchunk4":None,"T512_split148":f"{E}/T512_split148_to","T512_split104":f"{E}/T512_split104_to","T512_split88":f"{E}/T512_split88_to"}
cases=[]
for name,d in dirs.items():
    T=name[:4]
    if not os.path.isfile(f"{E}/qpc/{name}/programqpc.bin"): continue
    src = d or anc[T]
    cases.append(dict(label=name, qpc=f"{E}/qpc/{name}", input=f"{src}/input_f16.bin", counts=f"{src}/expected_counts_i32.bin", ref_y=refy[T]))
json.dump(dict(cases=cases), open(f"{E}/spec.json","w"), indent=1); print(len(cases), "cases")
EOP
echo "===== E18 TIMING (uninstrumented, 3 alternating rounds x 100) ====="
$PY $S/e7_timing.py $E/spec.json $E/timing.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|WARN|waiting|round 0"
echo "===== E18 PROFILE: T512 anchor and T512 split104 (stats70) ====="
for v in T512_anchor T512_split104; do d=${DIR[$v]}; for f in input_f16.bin expected_counts_i32.bin; do [ -e $d/$f ] || ln -s ${ANC[T512]}/$f $d/$f; done
  comp ${v}_s70 $d/model.onnx 70 $d/qpc_s70; bash $S/profile_case.sh $d 512 "$v MXFP6"; $PY $S/tail_breakdown.py $d/analysis/sample0 512 "$v" 2>&1 | head -12; $PY $E/e18_hot.py $d/analysis/sample0 "$v hot/cold engines"; done
echo E18_DONE
