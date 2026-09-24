#!/bin/bash
# KV round 4: 2-layer programs on the prompt-41 inputs: logits agreement + device time; instrumented profiles (hp_mos1, mos1, mos4)
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
F=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag/full_model; K=$F/kvtest
idle() { for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; echo "devices busy, waiting"; sleep 20; done; }
runit() { local name=$1; local d=$K/run_$name; rm -rf $d; mkdir -p $d/outputs; $PY $S/kv/trunc_io.py $K/qpc_$name $d > /dev/null; idle
  /opt/qti-aic/exec/qaic-runner -t $K/qpc_$name -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 30 -S 1 -T 1 -c --write-output-start-iter 29 --write-output-num-samples 1 --write-output-dir $d/outputs > $d/runner.log 2>&1
  echo "run $name rc=$?: $(grep -E 'ExecTimeUs_Dev|Inf/Sec|Latency' $d/runner.log | tr '\n' ' ' | cut -c1-260)"; }
for n in trunc2_noflag trunc2_mos4 trunc2_hp_mos1 trunc2_hp_noflag; do runit $n; done
$PY - <<EOP
import numpy as np, glob
K="$K"; ref=None
for n in ("trunc2_noflag","trunc2_mos4","trunc2_hp_mos1","trunc2_hp_noflag"):
    fs=glob.glob(f"{K}/run_{n}/outputs/*logits*"); 
    if not fs: print(n, "no logits"); continue
    y=np.fromfile(fs[0], np.float32).astype(np.float64)
    if ref is None: ref=y; print(f"{n:>18s}: argmax {y.argmax()} |y|max {np.abs(y).max():.3f} (reference)")
    else: print(f"{n:>18s}: argmax {y.argmax()} rel L2 vs noflag {np.linalg.norm(y-ref)/np.linalg.norm(ref):.3e} max|diff| {np.abs(y-ref).max():.4f} bit-exact {np.array_equal(y,ref)}")
EOP
prof() { local name=$1; local d=$K/prof_$name; rm -rf $d; mkdir -p $d/stats $d/outputs $d/trace; $PY $S/kv/trunc_io.py $K/qpc_${name}_s70 $d > /dev/null; idle
  /opt/qti-aic/exec/qaic-runner -t $K/qpc_${name}_s70 -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 5 -S 1 -T 1 -c --aic-profiling-type raw_device_stats --aic-profiling-start-iter 2 --aic-profiling-num-samples 3 --aic-profiling-out-dir $d/stats --write-output-start-iter 2 --write-output-num-samples 3 --write-output-dir $d/outputs > $d/runner.log 2>&1
  echo "profile $name rc=$?: $(grep ExecTimeUs_Dev $d/runner.log | awk '{printf "%s %s ", substr($1,12,5), $2}')"
  /opt/qti-aic/exec/qaic-opstats --qpc $K/qpc_${name}_s70/programqpc.bin --input-dir $d/stats --output-dir $d/trace --summary --trace --merge-mq-traces true --flow-events full > $d/opstats.log 2>&1
  /opt/qti-aic/tools/qaic-qpc extract --qpc $K/qpc_${name}_s70/programqpc.bin --output-dir $d/meta -s '*opstatsdesc.bin' >/dev/null 2>&1
  t=$(ls $d/trace/*merged*.trace.json | sort | head -1); $PY $S/detail_analyze.py $t $d/meta "trunc2 $name" $d/analysis/sample0 > $d/analysis_sample0.log 2>&1; head -1 $d/analysis_sample0.log; }
prof trunc2_hp_mos1; prof trunc2_mos1; prof trunc2_mos4
echo KV4_DONE
