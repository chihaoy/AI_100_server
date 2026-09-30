#!/bin/bash
# E37a: 2x2 ablation of the two optimization series (reduction x padding) on two-layer cuts at T=512, one session.
#   N  naive reduction, naive padding : native graph + head-parallel attention (step 2 of E36)
#   R  our reduction, naive padding   : rewrites + token-owned combine, capacity T (naive T/T of E36, Einsum final sum)
#   P2 naive reduction, rank sort     : export combine, per-card runtime sort 16xT + 16xT/8   (build_full_tiered.py --combine dense)
#   P  naive reduction, three tiers   : export combine, tiers 8xT, 8x128, 16x64                (build_full_tiered.py --combine dense)
#   D  our reduction, three tiers     : token-owned combine and tiers (E34, Einsum final sum)
# Accuracy against the FP32 two-layer reference; cross-program logits differences; programs deleted at the end.
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e37; mkdir -p $OUT
cd $F || exit 1
hp() { rm -rf ${1}_hp; $PY $M/scripts/headpar_graph.py $1 ${1}_hp > /dev/null 2>&1 || echo "headpar failed $1"; rm -rf ${1}_hp/weights_hp; ln -s ../native_c128_hp/weights_hp ${1}_hp/weights_hp; ln -sf $(realpath configs/custom_io.yaml) ${1}_hp/custom_io.yaml; }
for g in full512_sort_dense full512_tier3_dense; do
  [ -d ${g}_hp ] || hp $g
  t=trunc2_512_${g#full512_}; rm -rf $t; $PY $M/scripts/truncate_layers.py ${g}_hp $t 2 configs/custom_io.yaml 2>&1 | tail -n 1 | cut -c1-120
done
comp() { local name=$1 graph=$2 spec=$3 extra=$4; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,12p | cut -c1-220; }
idle() { for i in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref > $OUT/run_$n.stdout 2>&1
  $PY -c "
import json, numpy as np; r = json.load(open('$OUT/run_$n/result.json')); t = r['timing']
y = np.fromfile('$OUT/run_$n/logits_f32.bin', np.float32).astype(np.float64); ref = np.load('$ref/trunc2_logits_last.npy').astype(np.float64)
print(f\"   $n: median {t['median_ms']:.3f} ms, rounds {[round(v, 3) for v in t['round_medians_ms'].values()]}; two-layer logits rel L2 vs FP32 {np.linalg.norm(y - ref) / np.linalg.norm(ref):.4f}, argmax {int(y.argmax())} (ref {int(ref.argmax())})\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -n 2 $OUT/run_$n.stdout | cut -c1-200; }; }
S5=specializations_T512.json
comp e37_N trunc2_headpar $S5 "" &
comp e37_R trunc2_512_naive $S5 "" &
comp e37_P2 trunc2_512_sort_dense $S5 "-size-split-granularity=1024" &
comp e37_P trunc2_512_tier3_dense $S5 "-size-split-granularity=1024" &
comp e37_D trunc2_512_tier3 $S5 "-size-split-granularity=1024" &
wait
echo "===== two layers, T=512, one session (host tool, 3 rounds x 20), order N R P2 P D, then reversed"
for n in N R P2 P D; do hostrun ${n} $K/qpc_e37_$n $F/ref512; done
for n in D P P2 R N; do hostrun ${n}_again $K/qpc_e37_$n $F/ref512; done
$PY - <<EOP
import numpy as np, os
O = "$OUT"; L = lambda n: np.fromfile(f"{O}/run_{n}/logits_f32.bin", np.float32).astype(np.float64) if os.path.exists(f"{O}/run_{n}/logits_f32.bin") else None
for a, b in (("N", "P2"), ("N", "P"), ("P2", "P"), ("R", "D"), ("N", "R"), ("P", "D"), ("N", "N_again"), ("P", "P_again")):
    x, y = L(a), L(b)
    if x is not None and y is not None: print(f"   {b} vs {a}: bit-identical {np.array_equal(x, y)}, rel L2 {np.linalg.norm(y - x) / np.linalg.norm(x):.2e}, same argmax {x.argmax() == y.argmax()}")
EOP
rm -rf $K/qpc_e37_N $K/qpc_e37_R $K/qpc_e37_P2 $K/qpc_e37_P $K/qpc_e37_D
echo E37A_DONE
