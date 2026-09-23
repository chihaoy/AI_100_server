#!/bin/bash
# usage: profile_case.sh <variant dir with qpc_s70> <T> <label>
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
d=$1; T=$2; L=$3; rm -rf $d/stats $d/outputs $d/trace $d/meta $d/analysis; mkdir -p $d/stats $d/outputs $d/trace
cat > $d/io.json <<EOJ
{"IO-files": [[{"path": "$d/input_f16.bin", "dims": [$T, 2176], "elem-size": 2, "io-direction": "in", "map-to": "probe_input"},
 {"path": "$d/y_placeholder.bin", "dims": [$T, 2048], "elem-size": 2, "io-direction": "out", "map-to": "y"},
 {"path": "$d/expected_counts_i32.bin", "dims": [128], "elem-size": 4, "io-direction": "out", "map-to": "counts"}]]}
EOJ
head -c $((T*2048*2)) /dev/zero > $d/y_placeholder.bin
for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && break; echo "devices busy, waiting"; sleep 20; done
/opt/qti-aic/exec/qaic-runner -t $d/qpc_s70 -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats --write-output-start-iter 2 --write-output-num-samples 3 --write-output-dir $d/outputs > $d/runner.log 2>&1
echo "runner $L: $(grep ExecTimeUs_Dev $d/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"
/opt/qti-aic/exec/qaic-opstats --qpc $d/qpc_s70/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
/opt/qti-aic/tools/qaic-qpc extract --qpc $d/qpc_s70/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' >/dev/null 2>&1
i=0; for t in $(ls $d/trace/*merged*.trace.json | sort); do $PY $S/detail_analyze.py $t $d/meta "$L sample$i" $d/analysis/sample$i > $d/analysis_sample$i.log 2>&1; head -1 $d/analysis_sample$i.log; i=$((i+1)); done
