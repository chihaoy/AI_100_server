#!/bin/bash
# E23a: one partition per card, probe graph: compile per-card constants under TS (one partition, 4 cards) and under 4 partitions
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e23
FL="-aic-hw -aic-hw-version=ai100 -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -compile-only"
d=$E/toy_percard; rm -rf $d/qpc_dump; /opt/qti-aic/exec/qaic-compile $FL -m=$d/model.onnx -aic-binary-dir=$d/qpc_dump -mdp-dump-partition-config=$d/dump.json > $d/dump.log 2>&1; echo "dump rc=$?"; tail -2 $d/dump.log | cut -c1-160
$PY $E/make_pconfig.py $d/model.onnx $d/dump.json $d/pconfig.json '/toy/c(\d)/' '/toy/sum/'
comp() { local name=$1 dir=$2 cfg=$3 extra=$4; local out=$dir/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile $FL -m=$dir/model.onnx -aic-binary-dir=$out -mdp-load-partition-config=$cfg $extra > $dir/compile_$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $dir/compile_$name.log | sed -n 3,14p | cut -c1-240; }
comp ts $E/toy_percard $S/mdp_ts_4.json "-mdts-mos=1" &
comp pp $E/toy_percard $E/toy_percard/pconfig.json "" &
comp ts $E/toy_single $S/mdp_ts_4.json "-mdts-mos=1" &
wait
idle() { for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
for v in toy_percard:ts toy_percard:pp toy_single:ts; do dir=$E/${v%%:*}; q=$dir/qpc_${v#*:}; [ -f $q/programqpc.bin ] || { echo "=== $v: no qpc"; continue; }; w=$dir/run_${v#*:}; rm -rf $w; mkdir -p $w/out; head -c $((128*768*2)) /dev/zero > $w/y.bin
  printf '{"IO-files": [[{"path": "%s/x_f16.bin", "dims": [128, 2048], "elem-size": 2, "io-direction": "in", "map-to": "x"}, {"path": "%s/idx_i64.bin", "dims": [64], "elem-size": 8, "io-direction": "in", "map-to": "idx_in"}, {"path": "%s/y.bin", "dims": [128, 768], "elem-size": 2, "io-direction": "out", "map-to": "y"}]]}' $dir $dir $w > $w/io.json
  idle; base=$(/opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep 'Dram Free' | grep -oE '[0-9]+' | tr '\n' ' ')
  (/opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $w/io.json -n 3000 -S 1 -T 1 > $w/mem_runner.log 2>&1 &); sleep 6; used=$(/opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep 'Dram Free' | grep -oE '[0-9]+' | tr '\n' ' ')
  mem=$($PY -c "b='$base'.split(); u='$used'.split(); print(' '.join(f'c{i}:{(int(b[i])-int(u[i]))//1024}' for i in range(4)))"); for i in $(seq 60); do pgrep -f "[q]aic-runner -t $q " >/dev/null || break; sleep 2; done
  idle; /opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $w/io.json -n 1000 -S 1 -T 1 --write-output-start-iter 999 --write-output-num-samples 1 --write-output-dir $w/out > $w/runner.log 2>&1; rc=$?
  ips=$(grep -oE 'Inf/Sec [0-9.]+' $w/runner.log | awk '{print $2}')
  err=$($PY -c "
import numpy as np, glob
f=glob.glob('$w/out/*y*'); r=np.fromfile('$dir/y_ref_f32.bin',np.float32).astype(np.float64)
y=np.fromfile(f[0],np.float16).astype(np.float64) if f else None
print(f'{np.linalg.norm(y-r)/np.linalg.norm(r):.2e}' if y is not None else 'no output')")
  echo "=== $v: rc=$rc  inf/s ${ips}  (~$($PY -c "print(f'{1000/float(\"${ips:-0}\" or 1):.3f}')") ms/inf)  DRAM MiB in use ${mem}  rel L2 vs fp32 ref ${err}"; done
echo E23A_DONE
