#!/bin/bash
# E23b: do the four partitions run concurrently? stats70 profiles of the TS and per-card-partition probes
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e23
FL="-aic-hw -aic-hw-version=ai100 -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -compile-only -stats-level=70"
d=$E/toy_percard
/opt/qti-aic/exec/qaic-compile $FL -m=$d/model.onnx -aic-binary-dir=$d/qpc_ts_s70 -mdp-load-partition-config=$S/mdp_ts_4.json -mdts-mos=1 > $d/compile_ts_s70.log 2>&1 & 
/opt/qti-aic/exec/qaic-compile $FL -m=$d/model.onnx -aic-binary-dir=$d/qpc_pp_s70 -mdp-load-partition-config=$d/pconfig.json > $d/compile_pp_s70.log 2>&1 & wait
for v in ts pp; do q=$d/qpc_${v}_s70; w=$d/prof_$v; rm -rf $w; mkdir -p $w/stats $w/out $w/trace; head -c $((128*768*2)) /dev/zero > $w/y.bin
  printf '{"IO-files": [[{"path": "%s/x_f16.bin", "dims": [128, 2048], "elem-size": 2, "io-direction": "in", "map-to": "x"}, {"path": "%s/idx_i64.bin", "dims": [64], "elem-size": 8, "io-direction": "in", "map-to": "idx_in"}, {"path": "%s/y.bin", "dims": [128, 768], "elem-size": 2, "io-direction": "out", "map-to": "y"}]]}' $d $d $w > $w/io.json
  for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && break; sleep 10; done
  /opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $w/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $w/stats > $w/runner.log 2>&1
  echo "=== $v runner rc=$?: $(grep ExecTimeUs_Dev $w/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"
  /opt/qti-aic/exec/qaic-opstats --qpc $q/programqpc.bin --input-dir $w/stats --output-dir $w/trace --summary --trace --merge-mq-traces true --flow-events full > $w/opstats.log 2>&1; echo "opstats rc=$? traces $(ls $w/trace/*merged*.trace.json 2>/dev/null | wc -l)"; ls $w/trace | head -8
  $PY $E/trace_windows.py $w/trace 2>&1 | head -40; done
echo E23B_DONE
