#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e15; mkdir -p $E/qpc
comp() { local name=$1 model=$2 stats=${3:-0}; local out=$E/qpc/$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 4,12p | cut -c1-220; }
declare -A ANC=([T128]=$S/e8/T128_hc_92_2 [T256]=$S/e7/T256_hc_158_4 [T512]=$S/e7/T512_hc_296_32)
echo "===== E15 COMPILE (MXFP6, flag) ====="
for T in T128 T256 T512; do comp ${T}_anchor ${ANC[$T]}/model.onnx; comp ${T}_tokencentric $S/e11/${T}_tokencentric/model.onnx; comp ${T}_tokenowned $E/${T}_tokenowned/model.onnx; done
$PY - <<EOP
import json, os
S="$S"; E="$E"; anc={"T128":f"{S}/e8/T128_hc_92_2","T256":f"{S}/e7/T256_hc_158_4","T512":f"{S}/e7/T512_hc_296_32"}; ref={"T128":"T128_hc_92_2","T256":"T256_hc_158_4","T512":"T512_hc_296_32"}
cases=[]
for T in ("T128","T256","T512"):
    for name in (f"{T}_anchor", f"{T}_tokencentric", f"{T}_tokenowned"):
        if not os.path.isfile(f"{E}/qpc/{name}/programqpc.bin"): continue
        cases.append(dict(label=name, qpc=f"{E}/qpc/{name}", input=f"{anc[T]}/input_f16.bin", counts=f"{anc[T]}/expected_counts_i32.bin", ref_y=f"{S}/e8/timing_D_runs/{ref[T]}__mx_r0/y.bin"))
json.dump(dict(cases=cases), open(f"{E}/spec.json","w"), indent=1); print(len(cases), "cases")
EOP
echo "===== E15 TIMING (uninstrumented, 3 alternating rounds x 100) ====="
$PY $S/e7_timing.py $E/spec.json $E/timing.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|WARN|waiting|round 0"
echo "===== E15 PROFILE: T256 token-owned (stats70) ====="
d=$E/T256_tokenowned; rm -rf $d/qpc_s70; /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$d/model.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$d/qpc_s70 -stats-level=70 -mdts-mos=1 > $d/compile_s70.log 2>&1; echo "### T256_tokenowned stats70 rc=$?"; bash $S/profile_case.sh $d 256 "T256 token-owned MXFP6"
echo E15_DONE
