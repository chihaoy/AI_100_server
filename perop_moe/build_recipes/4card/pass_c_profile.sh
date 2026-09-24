#!/bin/bash
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
source /home/chihao/qeff-venv/bin/activate 2>/dev/null || true
D=/home/chihao/models/qwen3_30b_a3b/4card
REG=${1:-decode}; Q=$D/qpc_$REG
OUT=/home/chihao/mllm/perop_moe/$REG; mkdir -p $OUT/stats $OUT/out
echo "=== [1/3] bindings (48 layers, 4 kv-heads) ==="
python3 /home/chihao/mllm/tools/perop_make_input.py --qpc "$Q" --token-ids "1" \
  --layers 48 --kv-heads 4 --head-dim 128 || exit 1
echo "=== [2/3] qaic-runner -D 0:1:2:3 raw_device_stats ==="
rm -f $OUT/stats/*
qaic-runner -t "$Q" -D 0:1:2:3 --aic-profiling-type raw_device_stats \
  --aic-profiling-num-samples 4 --aic-profiling-out-dir "$OUT/stats" --num-iter 100
echo "runner rc=$?"; ls $OUT/stats | head -5; ls $OUT/stats | wc -l
echo "=== [3/3] qaic-opstats --summary --trace --merge-mq-traces ==="
rm -f $OUT/out/*
qaic-opstats --qpc "$Q/programqpc.bin" --input-dir "$OUT/stats" --output-dir "$OUT/out" \
  --summary --trace --merge-mq-traces true --flow-events none \
  || echo "(opstats nonzero — continuing)"
ls -la $OUT/out | head -12
