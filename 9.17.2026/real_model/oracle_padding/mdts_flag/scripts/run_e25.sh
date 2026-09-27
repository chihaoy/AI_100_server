#!/bin/bash
# E25: per-card runtime sort vs naive T/T (native order, both stages at capacity T) vs static oracle, same session
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e25; mkdir -p $E/qpc
comp() { local name=$1 model=$2; local out=$E/qpc/$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=0 -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 3,10p | cut -c1-200; }
for T in 128 256 512; do comp T${T}_naive_dense $E/T${T}_naive/model.onnx & comp T${T}_naive_to $E/T${T}_naive_to/model.onnx & done; wait
$PY - <<EOP
import json, os
S="$S"; E="$E"
P={128:"T128_nat41", 256:"T256_cat0", 512:"T512_cat0"}
dyn={128:f"{S}/e22/qpc/T128_dyncard4", 256:f"{S}/e22/qpc/T256_dyncard4t", 512:f"{S}/e22/qpc/T512_dyncard4t"}
ora={128:(f"{S}/e20/qpc/T128_static_mx", f"{S}/e8/T128_hc_92_2", f"{S}/e8/timing_D_runs/T128_hc_92_2__mx_r0/y.bin"),
     256:(f"{S}/e18/qpc/T256_anchor", f"{S}/e7/T256_hc_158_4", f"{S}/e8/timing_D_runs/T256_hc_158_4__mx_r0/y.bin"),
     512:(f"{S}/e18/qpc/T512_anchor", f"{S}/e7/T512_hc_296_32", f"{S}/e8/timing_D_runs/T512_hc_296_32__mx_r0/y.bin")}
cases=[]
for T in (128, 256, 512):
    I=f"{S}/e20/inputs{T}"; p=P[T]; inp=f"{I}/{p}_input_f16.bin"; ref=f"{I}/{p}_y_ref_f16.bin"
    cases += [dict(label=f"T{T}_naive_dense", qpc=f"{E}/qpc/T{T}_naive_dense", input=inp, counts=f"{I}/{p}_counts_native_i32.bin", ref_y=ref),
              dict(label=f"T{T}_naive_tokenowned", qpc=f"{E}/qpc/T{T}_naive_to", input=inp, counts=f"{I}/{p}_counts_native_i32.bin", ref_y=ref),
              dict(label=f"T{T}_runtime_sort", qpc=dyn[T], input=inp, counts=f"{I}/{p}_counts_percard_i32.bin", ref_y=ref),
              dict(label=f"T{T}_static_oracle", qpc=ora[T][0], input=f"{ora[T][1]}/input_f16.bin", counts=f"{ora[T][1]}/expected_counts_i32.bin", ref_y=ora[T][2])]
cases=[c for c in cases if os.path.isfile(f"{c['qpc']}/programqpc.bin")]; json.dump(dict(cases=cases), open(f"{E}/spec.json","w"), indent=1); print(len(cases), "cases")
EOP
$PY $S/e7_timing.py $E/spec.json $E/timing.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|round 0"
echo E25_DONE
