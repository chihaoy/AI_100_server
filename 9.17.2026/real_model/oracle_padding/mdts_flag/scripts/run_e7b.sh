#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; E=$S/e7
for v in T512_hc_512_32 T512_hc_296_512; do out=$E/$v/qpc_s70; [ -f $out/programqpc.bin ] || { rm -rf $out; t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$E/$v/model.onnx -convert-to-fp16 -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=70 -mdts-mos=1 > $E/$v/compile_s70.log 2>&1
  echo "### $v stats70 rc=$? $(( $(date +%s)-t0 )) s qpc=$(stat -c %s $out/programqpc.bin 2>/dev/null)"; }; done
bash $S/profile_case.sh $E/T512_hc_512_32 512 "T512 hc 512/32"
bash $S/profile_case.sh $E/T512_hc_296_512 512 "T512 hc 296/512"
echo E7B_DONE
