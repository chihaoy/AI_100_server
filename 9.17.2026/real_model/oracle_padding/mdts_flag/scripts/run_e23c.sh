#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e23; d=$E/toy_indep
FL="-aic-hw -aic-hw-version=ai100 -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -compile-only"
rm -rf $d/qpc_dump; /opt/qti-aic/exec/qaic-compile $FL -m=$d/model.onnx -aic-binary-dir=$d/qpc_dump -mdp-dump-partition-config=$d/dump.json > $d/dump.log 2>&1
$PY $E/make_pconfig.py $d/model.onnx $d/dump.json $d/pconfig.json '/toy/c(\d)/'
/opt/qti-aic/exec/qaic-compile $FL -stats-level=70 -m=$d/model.onnx -aic-binary-dir=$d/qpc_pp_s70 -mdp-load-partition-config=$d/pconfig.json > $d/compile_pp_s70.log 2>&1; echo "compile rc=$?"; grep -o "Error message: .*" $d/compile_pp_s70.log | cut -c1-300
w=$d/prof_pp; rm -rf $w; mkdir -p $w/stats $w/trace; for c in 0 1 2 3; do head -c $((128*768*2)) /dev/zero > $w/r$c.bin; done
printf '{"IO-files": [[{"path": "%s/x_f16.bin", "dims": [128, 2048], "elem-size": 2, "io-direction": "in", "map-to": "x"}, {"path": "%s/idx_i64.bin", "dims": [64], "elem-size": 8, "io-direction": "in", "map-to": "idx_in"}, {"path": "%s/r0.bin", "dims": [128, 768], "elem-size": 2, "io-direction": "out", "map-to": "r_c0"}, {"path": "%s/r1.bin", "dims": [128, 768], "elem-size": 2, "io-direction": "out", "map-to": "r_c1"}, {"path": "%s/r2.bin", "dims": [128, 768], "elem-size": 2, "io-direction": "out", "map-to": "r_c2"}, {"path": "%s/r3.bin", "dims": [128, 768], "elem-size": 2, "io-direction": "out", "map-to": "r_c3"}]]}' $d $d $w $w $w $w > $w/io.json
/opt/qti-aic/exec/qaic-runner -t $d/qpc_pp_s70 -D 0:1:2:3 --aic-batch-json-input $w/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $w/stats > $w/runner.log 2>&1
echo "runner rc=$?: $(grep ExecTimeUs_Dev $w/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"; grep -iE "error" $w/runner.log | head -3
/opt/qti-aic/exec/qaic-opstats --qpc $d/qpc_pp_s70/programqpc.bin --input-dir $w/stats --output-dir $w/trace --summary --trace --merge-mq-traces true --flow-events full > $w/opstats.log 2>&1
$PY $E/trace_windows.py $w/trace 2>&1 | grep -E "windows|profiling" | head -6
echo E23C_DONE
