#!/bin/bash
# E33: attention's sum over the four head groups after o_proj (ReduceSum_o, 2.4 ms per layer at T=512 in the 2-layer profile)
# as elementwise adds (headpar_graph.py --osum addtree|tileadd): two-layer host timing at T=128 and T=512 against the
# ReduceSum form, accuracy against the FP32 two-layer reference, and an instrumented profile of the T=512 elementwise build
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e33; mkdir -p $OUT
comp() { local name=$1 graph=$2 spec=$3 extra=$4; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref > $OUT/run_$n.stdout 2>&1
  $PY -c "
import json, numpy as np; r = json.load(open('$OUT/run_$n/result.json')); t = r['timing']
y = np.fromfile('$OUT/run_$n/logits_f32.bin', np.float32).astype(np.float64); ref = np.load('$ref/trunc2_logits_last.npy').astype(np.float64)
print(f\"   $n: median {t['median_ms']:.3f} ms, rounds {[round(v, 3) for v in t['round_medians_ms'].values()]}; two-layer logits rel L2 vs FP32 {np.linalg.norm(y - ref) / np.linalg.norm(ref):.4f}, argmax {int(y.argmax())} (ref {int(ref.argmax())})\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -n 1 $OUT/run_$n.stdout | cut -c1-200; }; }
comp e33_128_base trunc2_dyncard specializations_flat.json "-size-split-granularity=512" &
comp e33_128_osaddtree trunc2_128_dyncard_osaddtree specializations_flat.json "-size-split-granularity=512" &
comp e33_128_ostileadd trunc2_128_dyncard_ostileadd specializations_flat.json "-size-split-granularity=512" &
comp e33_512_base trunc2_512_dyncard specializations_T512.json "-size-split-granularity=1024" &
comp e33_512_osaddtree trunc2_512_dyncard_osaddtree specializations_T512.json "-size-split-granularity=1024" &
comp e33_512_ostileadd trunc2_512_dyncard_ostileadd specializations_T512.json "-size-split-granularity=1024" &
wait
echo "waiting for the E32 profile to release the devices"; for i in $(seq 720); do grep -q E32_DONE $M/e2e_profile/e32_T512.txt 2>/dev/null && break; sleep 10; done
echo "===== two layers, one session (host tool, 3 rounds x 20)"
hostrun e33_128_base $K/qpc_e33_128_base $F/ref
hostrun e33_128_osaddtree $K/qpc_e33_128_osaddtree $F/ref
hostrun e33_128_ostileadd $K/qpc_e33_128_ostileadd $F/ref
hostrun e33_128_base_again $K/qpc_e33_128_base $F/ref
hostrun e33_512_base $K/qpc_e33_512_base $F/ref512
hostrun e33_512_osaddtree $K/qpc_e33_512_osaddtree $F/ref512
hostrun e33_512_ostileadd $K/qpc_e33_512_ostileadd $F/ref512
hostrun e33_512_base_again $K/qpc_e33_512_base $F/ref512
rm -rf $K/qpc_e33_128_base $K/qpc_e33_128_osaddtree $K/qpc_e33_128_ostileadd $K/qpc_e33_512_base $K/qpc_e33_512_osaddtree $K/qpc_e33_512_ostileadd
echo E33_DONE
