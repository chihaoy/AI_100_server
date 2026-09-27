#!/bin/bash
# E27: optimizations 1 (sort first, then the two original stage chains) and 2 (DMA issue / tile-size compiler options)
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python
R=/home/wentao/workspace/AI_100_server; M=$R/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e27; mkdir -p $OUT
comprs() { local name=$1 graph=$2 extra=$3; local out=$K/qpc_$name; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_flat.json -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 180); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 15; done; }
hostrun() { local n=$1; rm -rf $OUT/run_$n $OUT/run_$n.log; idle
  cd $R && $PY tools/moe_qwen3_baseline.py run --out $OUT/run_$n --precision mxfp6 --qpc $K/qpc_$n --reference $F/ref > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; print(f\"   $n: median {t['median_ms']:.2f} ms, rounds {[round(v,2) for v in t['round_medians_ms'].values()]}\")" 2>/dev/null || tail -2 $OUT/run_$n.stdout | cut -c1-200; }
echo "===== 2-layer correctness: sort-first vs naive ====="
comprs t2_naive trunc2_naive "" & comprs t2_sortfirst trunc2_sortfirst "" & wait
for n in t2_naive t2_sortfirst; do d=$OUT/io_$n; rm -rf $d; mkdir -p $d/outputs; $PY $S/kv/trunc_io.py $K/qpc_$n $d > /dev/null; idle
  /opt/qti-aic/exec/qaic-runner -t $K/qpc_$n -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 10 -S 1 -T 1 --write-output-start-iter 9 --write-output-num-samples 1 --write-output-dir $d/outputs > $d/runner.log 2>&1; done
$PY - <<EOP
import numpy as np, glob
O="$OUT"; y={}
for n in ("t2_naive", "t2_sortfirst"):
    f=glob.glob(f"{O}/io_{n}/outputs/*logits*"); y[n]=np.fromfile(f[0], np.float32).astype(np.float64) if f else None
a,b=y["t2_naive"],y["t2_sortfirst"]; print(f"   sort-first vs naive, 2 layers: rel L2 {np.linalg.norm(b-a)/np.linalg.norm(a):.3e}, argmax {a.argmax()} / {b.argmax()}")
EOP
rm -rf $K/qpc_t2_naive $K/qpc_t2_sortfirst
echo "===== 12-layer timing, each variant back to back with naive ====="
comprs t12_naive trunc12_naive ""
pair() { local n=$1 graph=$2 extra=$3; comprs $n $graph "$extra"; hostrun t12_naive; hostrun $n; rm -rf $K/qpc_$n; }
pair t12_dyncard trunc12_dyncard ""
pair t12_sortfirst trunc12_sortfirst ""
pair t12_sortfirst_pdma trunc12_sortfirst "-use-producer-dma"
pair t12_sortfirst_ssg512 trunc12_sortfirst "-size-split-granularity=512"
pair t12_naive_pdma trunc12_naive "-use-producer-dma"
pair t12_naive_ssg512 trunc12_naive "-size-split-granularity=512"
hostrun t12_naive
echo E27_DONE
