#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e14; mkdir -p $E/qpc
comp() { local name=$1 model=$2 stats=${3:-0}; local out=$E/qpc/$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 4,9p | cut -c1-200; }
declare -A NAIVE=([T128]=$S/e8/T128_hc_128_128 [T256]=$S/e7/T256_hc_256_256 [T512]=$S/e7/T512_hc_512_512) ORACLE=([T128]=$S/e8/T128_hc_92_2 [T256]=$S/e7/T256_hc_158_4 [T512]=$S/e7/T512_hc_296_32)
echo "===== E14 COMPILE ====="
for T in T128 T256 T512; do comp ${T}_naive_dense ${NAIVE[$T]}/model.onnx; comp ${T}_naive_tc $E/${T}_naive_tc/model.onnx; comp ${T}_oracle_dense ${ORACLE[$T]}/model.onnx; comp ${T}_oracle_tc $S/e11/${T}_tokencentric/model.onnx; done
$PY - <<EOP
import json
S="$S"; E="$E"
naive={"T128":f"{S}/e8/T128_hc_128_128","T256":f"{S}/e7/T256_hc_256_256","T512":f"{S}/e7/T512_hc_512_512"}; oracle={"T128":f"{S}/e8/T128_hc_92_2","T256":f"{S}/e7/T256_hc_158_4","T512":f"{S}/e7/T512_hc_296_32"}
refn={"T128":"T128_hc_128_128","T256":"T256_hc_256_256","T512":"T512_hc_512_512"}; refo={"T128":"T128_hc_92_2","T256":"T256_hc_158_4","T512":"T512_hc_296_32"}
cases=[]
for T in ("T128","T256","T512"):
    for name,d,ref in ((f"{T}_naive_dense",naive[T],refn[T]),(f"{T}_naive_tc",naive[T],refn[T]),(f"{T}_oracle_dense",oracle[T],refo[T]),(f"{T}_oracle_tc",oracle[T],refo[T])):
        cases.append(dict(label=name, qpc=f"{E}/qpc/{name}", input=f"{d}/input_f16.bin", counts=f"{d}/expected_counts_i32.bin", ref_y=f"{S}/e8/timing_D_runs/{ref}__mx_r0/y.bin"))
json.dump(dict(cases=cases), open(f"{E}/spec.json","w"), indent=1); print(len(cases), "cases")
EOP
echo "===== waiting for figure renders to finish ====="; until grep -q RENDER13_DONE $S/render_e13.log 2>/dev/null; do sleep 20; done
echo "===== E14 TIMING (uninstrumented, 3 alternating rounds x 100) ====="
$PY $S/e7_timing.py $E/spec.json $E/timing.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|WARN|waiting"
echo "===== E14 PROFILE: T256 naive token-centric (stats70) ====="
d=$E/T256_naive_tc; rm -rf $d/qpc_s70; /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$d/model.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$d/qpc_s70 -stats-level=70 -mdts-mos=1 > $d/compile_s70.log 2>&1; echo "### T256_naive_tc stats70 rc=$?"; bash $S/profile_case.sh $d 256 "T256 naive 256/256 token-centric MXFP6"
echo E14_DONE
