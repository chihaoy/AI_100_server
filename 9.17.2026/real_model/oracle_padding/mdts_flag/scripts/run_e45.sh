#!/bin/bash
# E45: the sum over the four attention head groups after o_proj written as elementwise adds (headpar_graph.py --osum
# addtree, E33) in the programs we report. (1) Graphs: the *_hpos twins of the ladder's step 2 (native_c128), step 5
# (full*_naive_tileadd), step 7 (full*_tier3_tileadd) and E38's program C (full*_e38C) at T = 128/256/512; their regrouped
# Wq/Wk/Wv files are shared with native_c128_hp after an exact comparison. (2) Two-layer check per chunk length, one session
# each: step 2 (naive T/T, default tiles) and step 7 (RankTier, its tuned tile from E38) with the ReduceSum and with the
# elementwise sum, each program twice in mirrored order, accuracy against the two-layer FP32 reference, base against fixed
# logits. (3) Instrumented two-layer profile of the fixed RankTier at T=512 (run_e38p.sh). Usage: run_e45.sh [T ...]
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e45; mkdir -p $OUT
cd $F || exit 1
echo "$(date +%F_%T) df $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
for src in native_c128 full_naive_tileadd full256_naive_tileadd full512_naive_tileadd full_tier3_tileadd full256_tier3_tileadd full512_tier3_tileadd full128_e38C full256_e38C full512_e38C; do
  out=${src}_hpos; [ -f $out/model.onnx ] && continue
  $PY $M/scripts/headpar_graph.py $src $out --osum addtree > $OUT/build_$out.log 2>&1 || { echo "### build failed: $out"; tail -3 $OUT/build_$out.log; exit 1; }
  if diff -rq $out/weights_hp native_c128_hp/weights_hp > /dev/null; then rm -rf $out/weights_hp; ln -s $F/native_c128_hp/weights_hp $out/weights_hp; else echo "### $out: weights_hp differ from native_c128_hp, kept"; fi
  echo "built $out: $(tail -1 $OUT/build_$out.log | cut -c1-160)"
done
trunc() { [ -f $2/model.onnx ] && [ $2/model.onnx -nt $1/model.onnx ] && return; rm -rf $2; $PY $M/scripts/truncate_layers.py $1 $2 2 configs/custom_io.yaml > /dev/null 2>&1 || echo "### truncate failed: $2"; }
trunc native_c128_hp trunc2_s2hp; trunc native_c128_hpos trunc2_s2hpos
comp() { local name=$1 graph=$2 spec=$3 extra=$4; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,12p | cut -c1-220; }
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref > $OUT/run_$n.stdout 2>&1
  $PY -c "
import json, numpy as np; r = json.load(open('$OUT/run_$n/result.json')); t = r['timing']
y = np.fromfile('$OUT/run_$n/logits_f32.bin', np.float32).astype(np.float64); ref = np.load('$ref/trunc2_logits_last.npy').astype(np.float64)
print(f\"   $n: median {t['median_ms']:.3f} ms, rounds {[round(v, 3) for v in t['round_medians_ms'].values()]}; two-layer rel L2 vs FP32 {np.linalg.norm(y - ref) / np.linalg.norm(ref):.4f}, argmax {int(y.argmax())} (ref {int(ref.argmax())})\")" 2>/dev/null || { echo "   $n: run failed"; tail -2 $OUT/run_$n.stdout; }; }
for T in ${@:-512 256 128}; do
  case $T in 128) spec=specializations_flat.json; ref=$F/ref; src=full_tier3_tileadd; tile=512;; 256) spec=specializations_T256.json; ref=$F/ref256; src=full256_tier3_tileadd; tile=1024;;
             512) spec=specializations_T512.json; ref=$F/ref512; src=full512_tier3_tileadd; tile=1024;; esac
  trunc ${src}_hp trunc2_${T}_e38E; trunc ${src}_hpos trunc2_${T}_e45Eos
  echo "================ T=$T: step 2 (naive T/T, default tiles) and step 7 (RankTier, ${tile} KiB), head-group sum ReduceSum vs elementwise; $(date +%F_%T)"
  comp e45_${T}_s2 trunc2_s2hp $spec "" & comp e45_${T}_s2os trunc2_s2hpos $spec "" &
  comp e45_${T}_E trunc2_${T}_e38E $spec "-size-split-granularity=$tile" & comp e45_${T}_Eos trunc2_${T}_e45Eos $spec "-size-split-granularity=$tile" & wait
  names=(${T}_s2 ${T}_s2os ${T}_E ${T}_Eos)
  for n in "${names[@]}"; do hostrun $n $K/qpc_e45_$n $ref; done
  for (( j=${#names[@]}-1; j>=0; j-- )); do n=${names[$j]}; hostrun ${n}_again $K/qpc_e45_$n $ref; done
  $PY - "$OUT" "$T" <<'EOP'
import sys, json, os, numpy as np
O, T = sys.argv[1], sys.argv[2]
L = lambda n: np.fromfile(f'{O}/run_{n}/logits_f32.bin', np.float32).astype(np.float64) if os.path.exists(f'{O}/run_{n}/logits_f32.bin') else None
med = lambda n: json.load(open(f'{O}/run_{n}/result.json'))['timing']['median_ms'] if os.path.exists(f'{O}/run_{n}/result.json') else float('nan')
for a, b in (('s2', 's2os'), ('E', 'Eos')):
    x, y = L(f'{T}_{a}'), L(f'{T}_{b}')
    if x is not None and y is not None: print(f'   {b} vs {a}: bit-identical {np.array_equal(x, y)}, rel L2 {np.linalg.norm(y - x) / np.linalg.norm(x):.2e}, same argmax {int(x.argmax()) == int(y.argmax())}')
m = {k: np.mean([med(f'{T}_{k}'), med(f'{T}_{k}_again')]) for k in ('s2', 's2os', 'E', 'Eos')}
print(f'| T={T}, two layers, mean of both passes | head-group ReduceSum | elementwise | change |\n|---|---:|---:|---:|')
for k, nm in (('s2', 'step 2, naive T/T'), ('E', 'step 7, RankTier')):
    print(f'| {nm} | {m[k]:.3f} ms | {m[k + "os"]:.3f} ms | {100 * (m[k + "os"] / m[k] - 1):+.1f}% |')
print(f'| step 2 / step 7 | {m["s2"] / m["E"]:.3f}x | {m["s2os"] / m["Eos"]:.3f}x | |')
EOP
  rm -rf $K/qpc_e45_${T}_*
done
echo "================ instrumented two-layer profile of the fixed RankTier at T=512; $(date +%F_%T)"
bash $M/scripts/run_e38p.sh 512 e45Eos:1024 > $OUT/e45_T512_profile.txt 2>&1; grep -E '^==|weights from DDR|E38P_DONE' $OUT/e45_T512_profile.txt | cut -c1-200 | head -8
echo "$(date +%F_%T) df $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
echo E45_DONE
