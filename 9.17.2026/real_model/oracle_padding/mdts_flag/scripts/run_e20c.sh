#!/bin/bash
# E20c: one dynamic-split QPC per T serves every real prompt: timing vs the static anchor (T=128) and correctness sweeps
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e20; mkdir -p $E/qpc
comp() { local name=$1 model=$2 extra=$3 stats=${4:-0} out=${5:-$E/qpc/$1}; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 $extra -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 3,12p | cut -c1-220; }
comp T128_dynsplit $E/T128_dynsplit_to/model.onnx "-mxfp6-matmul" & comp T256_dynsplit $E/T256_dynsplit_to/model.onnx "-mxfp6-matmul" & comp T512_dynsplit $E/T512_dynsplit_to/model.onnx "-mxfp6-matmul" & comp T256_anchor $S/e15/T256_tokenowned/model.onnx "-mxfp6-matmul" & wait
comp T512_anchor $S/e15/T512_tokenowned/model.onnx "-mxfp6-matmul"
$PY - <<EOP
import json, os, glob
S="$S"; E="$E"
# A: timing, dynamic vs static anchors (same session); the dynamic graphs get the CPU-routing input of the natural prompt 41 (T=128) or the first concatenated prompt
A=[dict(label="T128_static_mx", qpc=f"{E}/qpc/T128_static_mx", input=f"{S}/e8/T128_hc_92_2/input_f16.bin", counts=f"{S}/e8/T128_hc_92_2/expected_counts_i32.bin", ref_y=f"{S}/e8/timing_D_runs/T128_hc_92_2__mx_r0/y.bin"),
   dict(label="T128_dynsplit", qpc=f"{E}/qpc/T128_dynsplit", input=f"{E}/inputs128/T128_nat41_input_f16.bin", counts=f"{E}/inputs128/T128_nat41_counts_sorted_i32.bin", ref_y=f"{E}/inputs128/T128_nat41_y_ref_f16.bin"),
   dict(label="T256_anchor", qpc=f"{E}/qpc/T256_anchor", input=f"{S}/e7/T256_hc_158_4/input_f16.bin", counts=f"{S}/e7/T256_hc_158_4/expected_counts_i32.bin", ref_y=f"{S}/e8/timing_D_runs/T256_hc_158_4__mx_r0/y.bin"),
   dict(label="T256_dynsplit", qpc=f"{E}/qpc/T256_dynsplit", input=f"{E}/inputs256/T256_cat0_input_f16.bin", counts=f"{E}/inputs256/T256_cat0_counts_sorted_i32.bin", ref_y=f"{E}/inputs256/T256_cat0_y_ref_f16.bin"),
   dict(label="T512_anchor", qpc=f"{E}/qpc/T512_anchor", input=f"{S}/e7/T512_hc_296_32/input_f16.bin", counts=f"{S}/e7/T512_hc_296_32/expected_counts_i32.bin", ref_y=f"{S}/e8/timing_D_runs/T512_hc_296_32__mx_r0/y.bin"),
   dict(label="T512_dynsplit", qpc=f"{E}/qpc/T512_dynsplit", input=f"{E}/inputs512/T512_cat0_input_f16.bin", counts=f"{E}/inputs512/T512_cat0_counts_sorted_i32.bin", ref_y=f"{E}/inputs512/T512_cat0_y_ref_f16.bin")]
A=[c for c in A if os.path.isfile(f"{c['qpc']}/programqpc.bin")]; json.dump(dict(cases=A), open(f"{E}/spec_c.json","w"), indent=1); print(len(A), "timing cases")
for T in (128, 256, 512):
    names=sorted(os.path.basename(f)[:-len('_input_f16.bin')] for f in glob.glob(f"{E}/inputs{T}/T{T}_*_input_f16.bin"))
    B=[dict(label=n, qpc=f"{E}/qpc/T{T}_dynsplit", input=f"{E}/inputs{T}/{n}_input_f16.bin", counts=f"{E}/inputs{T}/{n}_counts_sorted_i32.bin", ref_y=f"{E}/inputs{T}/{n}_y_ref_f16.bin") for n in names if os.path.isfile(f"{E}/inputs{T}/{n}_y_ref_f16.bin")]
    json.dump(dict(cases=B), open(f"{E}/spec_sweep{T}.json","w"), indent=1); print(T, len(B), "sweep cases")
EOP
echo "===== timing: dynamic split vs static anchors ====="
$PY $S/e7_timing.py $E/spec_c.json $E/timing_c.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|round 0"
for T in 128 256 512; do for i in $(seq 40); do [ -f $E/inputs$T/.done ] && break; n=$(ls $E/inputs$T/*_y_ref_f16.bin 2>/dev/null | wc -l); [ "$n" -ge "$( [ $T = 128 ] && echo 50 || ( [ $T = 256 ] && echo 16 || echo 8 ) )" ] && break; sleep 15; done; done
echo "===== sweeps: every prompt through the one dynamic QPC (1 round) ====="
for T in 128 256 512; do $PY $S/e7_timing.py $E/spec_sweep$T.json $E/sweep$T.json 1 2>&1 | grep -E "round 0|FAILED|Traceback" | awk '{print $3, $5, $NF}' | sort > $E/sweep${T}.txt; echo "T=$T: $(wc -l < $E/sweep${T}.txt) prompts; rel L2 vs FP32 ref: $(awk '{print $3}' $E/sweep${T}.txt | sort -g | awk 'NR==1{mn=$1} {v[NR]=$1} END{print "min", mn, "median", v[int((NR+1)/2)], "max", v[NR]}'); latency median $(awk '{print $2}' $E/sweep${T}.txt | sort -g | awk '{v[NR]=$1} END{print v[int((NR+1)/2)], "min", v[1], "max", v[NR]}'); counts mismatches: $(grep -c False $E/sweep${T}.txt)"; done
echo E20C_DONE
