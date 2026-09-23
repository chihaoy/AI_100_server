#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e7; mkdir -p $E
declare -A OH=([256]=158 [512]=296) OC=([256]=4 [512]=8)
build() { local T=$1 mode=$2 ch=$3 cc=$4; local d=$E/T${T}_${mode}_${ch}_${cc}; [ -f $d/model.onnx ] || { echo "build $(basename $d): $($PY $S/make_cap.py $T $S/e6/counts_T${T}_grow.npy $d $ch $cc $mode 2>&1 | tail -1)"; }; }
comp() { local out=$1/qpc_s$2; [ -f $out/programqpc.bin ] && { echo "### $(basename $1) stats$2 cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$1/model.onnx -convert-to-fp16 -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$2 -mdts-mos=1 > $1/compile_s$2.log 2>&1
  echo "### $(basename $1) stats$2 rc=$? $(( $(date +%s)-t0 )) s qpc=$(stat -c %s $out/programqpc.bin 2>/dev/null)"; [ -f $out/programqpc.bin ] || tail -3 $1/compile_s$2.log; }
echo "===== E7 BUILD (grown counts, T=256 and 512) ====="
for T in 256 512; do build $T lpt $T $T; for ch in $T ${OH[$T]}; do for cc in $T 32 ${OC[$T]}; do build $T hc $ch $cc; done; done; done
echo "===== E7 COMPILE ====="
for T in 256 512; do for d in $E/T${T}_lpt_* $E/T${T}_hc_*; do comp $d 0; done; done
comp $E/T256_hc_158_4 70; comp $E/T512_hc_512_512 70; comp $E/T512_hc_296_8 70
echo "===== E7 TIMING SPEC ====="
$PY - <<EOP
import json
S="$S"; E="$E"; OH={256:158,512:296}; OC={256:4,512:8}; cases=[]
for T in (256,512):
    names=[f"T{T}_lpt_{T}_{T}"]+[f"T{T}_hc_{ch}_{cc}" for ch in (T,OH[T]) for cc in (T,32,OC[T])]
    for n in names: cases.append(dict(label=n, qpc=f"{E}/{n}/qpc_s0", input=f"{E}/{n}/input_f16.bin", counts=f"{E}/{n}/expected_counts_i32.bin", ref_y=f"{S}/e6/timing_runs/T{T}_grow_r0/y.bin"))
json.dump(dict(cases=cases), open(f"{E}/spec_timing.json","w"), indent=1); print(len(cases), "cases")
EOP
echo "===== E7 TIMING (uninstrumented, 3 alternating rounds x 100 iterations) ====="
$PY $S/e7_timing.py $E/spec_timing.json $E/timing.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|WARN|waiting|round"
echo "===== E7 PROFILES (stats70) ====="
bash $S/profile_case.sh $E/T256_hc_158_4 256 "T256 hc 158/4"
bash $S/profile_case.sh $E/T512_hc_512_512 512 "T512 hc 512/512"
bash $S/profile_case.sh $E/T512_hc_296_8 512 "T512 hc 296/8"
echo E7_DONE
