#!/bin/bash
source /home/chihao/qeff-venv/bin/activate; export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
P=/home/chihao/mllm/9.17.2026/mdp2_g128_64; cd /home/chihao/mllm
ln -sf $(readlink -f 9.17.2026/g128_64/moe_layer.onnx) $P/moe_layer.onnx; cp 9.17.2026/g128_64/info.json 9.17.2026/g128_64/x.bin 9.17.2026/g128_64/y_*.npy $P/ 2>/dev/null
echo "[1/3] compile with 2 partitions"; rm -rf $P/qpc
qaic-compile -aic-hw -aic-hw-version=ai100 -m=$P/moe_layer.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
  -mdp-load-partition-config=$P/mdp2.json -stats-level=70 -aic-binary-dir=$P/qpc > $P/compile.log 2>&1; echo "compile rc=$?"; tail -3 $P/compile.log
[ -f $P/qpc/programqpc.bin ] || { echo "COMPILE FAILED"; exit 1; }
echo "[2/3] run"; rm -rf $P/stats; mkdir -p $P/stats
qaic-runner -t $P/qpc -D 0:1:2:3 --aic-profiling-type raw_device_stats --aic-profiling-num-samples 2 --aic-profiling-out-dir $P/stats --num-iter 50 > $P/run.log 2>&1; echo "run rc=$?"; grep -E "ExecTimeUs|Error|error" $P/run.log | head -6
echo "[3/3] opstats"; rm -rf $P/out; mkdir -p $P/out
qaic-opstats --qpc $P/qpc/programqpc.bin --input-dir $P/stats --output-dir $P/out --summary --trace --merge-mq-traces true --flow-events none > $P/opstats.log 2>&1; ls $P/out | wc -l
echo "=== MDP2 DONE $(date +%T)"
