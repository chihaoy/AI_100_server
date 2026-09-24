#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; PLOT=/tmp/qwen3_profile_plot_wentao_20260921/bin/python; E=$S/e16; mkdir -p $E/qpc $E/figs
comp() { local name=$1 model=$2 stats=${3:-0}; local out=$E/qpc/$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 4,12p | cut -c1-220; }
declare -A ANC=([T128]=$S/e8/T128_hc_92_2 [T256]=$S/e7/T256_hc_158_4 [T512]=$S/e7/T512_hc_296_32)
echo "===== E16 COMPILE (MXFP6, flag) ====="
for T in T128 T256 T512; do comp ${T}_anchor ${ANC[$T]}/model.onnx; comp ${T}_tokenowned $S/e15/${T}_tokenowned/model.onnx; for m in to_fp16 to_idx to_rs; do comp ${T}_$m $E/${T}_$m/model.onnx; done; done
$PY - <<EOP
import json, os
S="$S"; E="$E"; anc={"T128":f"{S}/e8/T128_hc_92_2","T256":f"{S}/e7/T256_hc_158_4","T512":f"{S}/e7/T512_hc_296_32"}; ref={"T128":"T128_hc_92_2","T256":"T256_hc_158_4","T512":"T512_hc_296_32"}
cases=[]
for T in ("T128","T256","T512"):
    for name in (f"{T}_anchor", f"{T}_tokenowned", f"{T}_to_fp16", f"{T}_to_idx", f"{T}_to_rs"):
        if os.path.isfile(f"{E}/qpc/{name}/programqpc.bin"): cases.append(dict(label=name, qpc=f"{E}/qpc/{name}", input=f"{anc[T]}/input_f16.bin", counts=f"{anc[T]}/expected_counts_i32.bin", ref_y=f"{S}/e8/timing_D_runs/{ref[T]}__mx_r0/y.bin"))
json.dump(dict(cases=cases), open(f"{E}/spec.json","w"), indent=1); print(len(cases), "cases")
EOP
echo "===== E16 TIMING (uninstrumented, 3 alternating rounds x 100) ====="
$PY $S/e7_timing.py $E/spec.json $E/timing.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|WARN|waiting|round 0"
echo "===== E16 PROFILES: T256 to_rs, T512 to_rs (stats70) ====="
for v in T256_to_rs T512_to_rs; do d=$E/$v; T=${v:1:3}; rm -rf $d/qpc_s70; /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$d/model.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$d/qpc_s70 -stats-level=70 -mdts-mos=1 > $d/compile_s70.log 2>&1; echo "### $v stats70 rc=$?"; bash $S/profile_case.sh $d $T "$v MXFP6"; done
echo "===== E16 FIGURES ====="
$PLOT $S/detail_plot2.py $S/e15/T256_tokenowned/analysis/sample0 'T=256, hot/cold 158/4, MXFP6, token-owned combine (E15), instrumented sample 0' $E/figs/T256_tokenowned_cores.png 0,1,2,3
$PLOT $S/detail_plot2.py $E/T256_to_rs/analysis/sample0 'T=256, hot/cold 158/4, MXFP6, token-owned + fp16 + trimmed index + reduce-scatter, instrumented sample 0' $E/figs/T256_to_rs_cores.png 0,1,2,3
$PLOT $S/detail_plot2.py $E/T512_to_rs/analysis/sample0 'T=512, hot/cold 296/32, MXFP6, token-owned + fp16 + trimmed index + reduce-scatter, instrumented sample 0' $E/figs/T512_to_rs_cores.png 0,1,2,3
echo E16_DONE
