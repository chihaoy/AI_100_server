#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e11; mkdir -p $E/qpc
comp() { local name=$1 model=$2 stats=${3:-0}; local out=$E/qpc/$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s qpc=$(stat -c %s $out/programqpc.bin 2>/dev/null)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 4,9p | cut -c1-200; }
echo "===== E11 COMPILE (MXFP6, flag) ====="
comp T128_anchor $S/e8/T128_hc_92_2/model.onnx; comp T256_anchor $S/e7/T256_hc_158_4/model.onnx; comp T512_anchor $S/e7/T512_hc_296_32/model.onnx
for T in T128 T256 T512; do for mode in tiledfinal tokencentric; do comp ${T}_$mode $E/${T}_$mode/model.onnx; done; done
$PY - <<EOP
import json
S="$S"; E="$E"; src={"T128":f"{S}/e8/T128_hc_92_2","T256":f"{S}/e7/T256_hc_158_4","T512":f"{S}/e7/T512_hc_296_32"}; ref={"T128":"T128_hc_92_2","T256":"T256_hc_158_4","T512":"T512_hc_296_32"}
cases=[]
for T in ("T128","T256","T512"):
    for name in (f"{T}_anchor", f"{T}_tiledfinal", f"{T}_tokencentric"):
        d = src[T] if name.endswith("anchor") else f"{E}/{name}"
        cases.append(dict(label=name, qpc=f"{E}/qpc/{name}", input=f"{d}/input_f16.bin", counts=f"{d}/expected_counts_i32.bin", ref_y=f"{S}/e8/timing_D_runs/{ref[T]}__mx_r0/y.bin"))
json.dump(dict(cases=cases), open(f"{E}/spec.json","w"), indent=1); print(len(cases), "cases")
EOP
echo "===== E11 TIMING (uninstrumented, 3 alternating rounds x 100) ====="
$PY $S/e7_timing.py $E/spec.json $E/timing.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|WARN|waiting|round 0"
echo "===== E11 PROFILE: T256 tokencentric + T128 tokencentric (stats70) ====="
for v in T256_tokencentric T128_tokencentric; do d=$E/$v; T=${v:1:3}; rm -rf $d/qpc_s70; /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$d/model.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$d/qpc_s70 -stats-level=70 -mdts-mos=1 > $d/compile_s70.log 2>&1; echo "### $v stats70 rc=$?"; bash $S/profile_case.sh $d $T "$v MXFP6"; done
echo E11_DONE
