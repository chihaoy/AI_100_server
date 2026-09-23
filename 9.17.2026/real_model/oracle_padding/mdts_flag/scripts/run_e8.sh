#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding; E=$S/e8; CC=$R/cold_capacity_control; mkdir -p $E/qpc
comp() { local name=$1 model=$2 prec=$3 flag=$4 stats=${5:-0}; local out=$E/qpc/$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }
  [ -f $model ] || { echo "### $name MISSING MODEL $model"; return; }; rm -rf $out; local t0=$(date +%s); local extra=""; [ $prec = mx ] && extra="$extra -mxfp6-matmul"; [ $flag = 1 ] && extra="$extra -mdts-mos=1"
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats $extra > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s qpc=$(stat -c %s $out/programqpc.bin 2>/dev/null)"; [ -f $out/programqpc.bin ] || tail -3 $E/qpc/$name.log; }
echo "===== E8 COMPILE block A (T=128 sorted layout, capacity sweep) ====="
for c in 128 64 32 16 8 4 2; do comp cc_c${c}__mx $CC/c$c/model.onnx mx 1; done
comp hot96_cold2__mx $S/e2/hot96_cold2/model.onnx mx 1; comp hot92_cold2__mx $S/e2/hot92_cold2/model.onnx mx 1
comp cc_c128__mx_noflag $CC/c128/model.onnx mx 0; comp cc_c2__mx_noflag $CC/c2/model.onnx mx 0
comp cc_c128__fp16 $CC/c128/model.onnx fp16 1; comp cc_c2__fp16 $CC/c2/model.onnx fp16 1
echo "===== E8 COMPILE block B (T=128 placement) ====="
comp c2tree_sorted__fp16 $R/routing_retune/c2_tree/model.onnx fp16 1; comp c2tree_sorted__mx $R/routing_retune/c2_tree/model.onnx mx 1; comp c2tree_sorted__mx_noflag $R/routing_retune/c2_tree/model.onnx mx 0
comp c2tree_native__fp16 $S/e5/c2tree_native/model.onnx fp16 1; comp c2tree_native__mx $S/e5/c2tree_native/model.onnx mx 1
comp c2tree_bal__fp16 $S/e5/c2tree_bal/model.onnx fp16 1; comp c2tree_bal__mx $S/e5/c2tree_bal/model.onnx mx 1
comp e4native_128_128__mx $S/e4/native_c128_128/model.onnx mx 1; comp e4native_92_34__mx $S/e4/native_min/model.onnx mx 1
echo "===== E8 COMPILE block D (chunk size x capacity) ====="
D128="T128_lpt_128_128 T128_hc_128_128 T128_hc_92_2"; D256="T256_lpt_256_256 T256_hc_256_256 T256_hc_256_4 T256_hc_158_256 T256_hc_158_32 T256_hc_158_4"; D512="T512_lpt_512_512 T512_hc_512_512 T512_hc_512_8 T512_hc_296_512 T512_hc_296_32 T512_hc_296_8"
dirof() { case $1 in T128_*) echo $E/$1;; *) echo $S/e7/$1;; esac; }
for v in $D128 $D256 $D512; do comp ${v}__mx $(dirof $v)/model.onnx mx 1; done
for v in T128_hc_128_128 T128_hc_92_2 T256_hc_256_256 T256_hc_158_4 T512_hc_512_512 T512_hc_296_32; do comp ${v}__fp16 $(dirof $v)/model.onnx fp16 1; done
echo "===== E8 SPECS ====="
$PY - <<EOP
import json, os
S="$S"; E="$E"; R="$R"; CC="$CC"
def case(label, inp, cnt, ref): return dict(label=label, qpc=f"{E}/qpc/{label}", input=inp, counts=cnt, ref_y=ref)
A_in, A_cnt, A_ref = f"{CC}/input_f16.bin", f"{CC}/expected_counts_i32.bin", f"{CC}/c128/timing_r0/y.bin"
A=[case(l, A_in, A_cnt, A_ref) for l in ["cc_c128__fp16","cc_c128__mx","cc_c128__mx_noflag"]+[f"cc_c{c}__mx" for c in (64,32,16,8,4)]+["cc_c2__fp16","cc_c2__mx","cc_c2__mx_noflag","hot96_cold2__mx","hot92_cold2__mx"]]
B=[case(l, A_in, A_cnt, A_ref) for l in ["c2tree_sorted__fp16","c2tree_sorted__mx","c2tree_sorted__mx_noflag"]]
B+=[case(l, f"{S}/e5/c2tree_native/input_f16.bin", f"{S}/e5/c2tree_native/expected_counts_i32.bin", A_ref) for l in ["c2tree_native__fp16","c2tree_native__mx"]]
B+=[case(l, f"{S}/e5/c2tree_bal/input_f16.bin", f"{S}/e5/c2tree_bal/expected_counts_i32.bin", A_ref) for l in ["c2tree_bal__fp16","c2tree_bal__mx"]]
B+=[case(l, f"{S}/e4/input_native_f16.bin", f"{S}/e4/expected_counts_native_i32.bin", A_ref) for l in ["e4native_128_128__mx","e4native_92_34__mx"]]
B=[c for c in B if os.path.isfile(c['qpc']+'/programqpc.bin')]
D=[]
for v in "$D128 $D256 $D512".split():
    d = f"{E}/{v}" if v.startswith("T128") else f"{S}/e7/{v}"; T=v.split('_')[0]
    ref = f"{S}/e6/timing_runs/T128_synth_r0/y.bin" if T=="T128" else f"{S}/e7/timing_runs/{v}_r0/y.bin"
    labels=[f"{v}__fp16", f"{v}__mx"] if os.path.isfile(f"{E}/qpc/{v}__fp16/programqpc.bin") else [f"{v}__mx"]
    D+=[case(l, f"{d}/input_f16.bin", f"{d}/expected_counts_i32.bin", ref) for l in labels]
for name, cases in (("A",A),("B",B),("D",D)): json.dump(dict(cases=cases), open(f"{E}/spec_{name}.json","w"), indent=1); print(name, len(cases), "cases")
EOP
for blk in A B D; do echo "===== E8 TIMING block $blk (uninstrumented, 3 alternating rounds x 100 iterations) ====="; $PY $S/e7_timing.py $E/spec_$blk.json $E/timing_$blk.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|WARN|waiting|round 0"; done
echo "===== E8 block C: all-active plans, FP16 flag vs MXFP6 flag ====="
$PY $S/e8_allactive.py aggregate_c32_32,aggregate_c32_4,aggregate_c8_8,identity_c32_32,identity_c8_8,sorted_a_c32_2,sorted_a_c32_32,sorted_a_c32_4,sorted_b_c32_32,sorted_c_c32_32,stripe_c8_8 2 2>&1 | grep --line-buffered -v -i warning
echo "===== E8 block E: MXFP6 profiles (stats70) ====="
prof() { local name=$1 src=$2 T=$3; local d=$E/prof/$name; mkdir -p $d; ln -sf $src/input_f16.bin $d/input_f16.bin; ln -sf $src/expected_counts_i32.bin $d/expected_counts_i32.bin
  [ -f $d/qpc_s70/programqpc.bin ] || { rm -rf $d/qpc_s70; /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$src/model.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$d/qpc_s70 -stats-level=70 -mdts-mos=1 > $d/compile_s70.log 2>&1; echo "### prof $name stats70 rc=$? qpc=$(stat -c %s $d/qpc_s70/programqpc.bin 2>/dev/null)"; }
  bash $S/profile_case.sh $d $T "$name MXFP6"; }
prof c2tree_bal $S/e5/c2tree_bal 128; prof T256_hc_158_4 $S/e7/T256_hc_158_4 256; prof T512_hc_296_8 $S/e7/T512_hc_296_8 512
echo E8_DONE
