#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PLOT=/tmp/qwen3_profile_plot_wentao_20260921/bin/python; R=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag
until grep -q E16_DONE $S/run_e16.log 2>/dev/null; do sleep 15; done
echo "===== E16b: token-owned profiles at T=128 and T=512 ====="
for v in T128_tokenowned T512_tokenowned; do d=$S/e15/$v; T=${v:1:3}; rm -rf $d/qpc_s70; /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$d/model.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$d/qpc_s70 -stats-level=70 -mdts-mos=1 > $d/compile_s70.log 2>&1; echo "### $v stats70 rc=$?"; bash $S/profile_case.sh $d $T "$v MXFP6"; rm -rf $d/qpc_s70; done
echo "===== E16b: figures ====="
$PLOT $S/detail_plot2.py $S/e15/T128_tokenowned/analysis/sample0 'T=128, hot/cold 92/2, MXFP6, token-owned combine, instrumented sample 0' $S/e16/figs/T128_tokenowned_cores.png 0,1,2,3
$PLOT $S/detail_plot2.py $S/e15/T512_tokenowned/analysis/sample0 'T=512, hot/cold 296/32, MXFP6, token-owned combine, instrumented sample 0' $S/e16/figs/T512_tokenowned_cores.png 0,1,2,3
cp $S/e16/figs/T128_tokenowned_cores.png $S/e16/figs/T256_tokenowned_cores.png $S/e16/figs/T512_tokenowned_cores.png $R/combine/ 2>/dev/null; echo E16B_DONE
