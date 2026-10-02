#!/bin/bash
# E39: fixed-size block scheduling against RankTier on two-layer cuts (RANKTIER_EXPERIMENT_PLAN.md, experiment 2), one session.
# Usage: run_e39.sh <T> <session> <label:tile> [<label:tile> ...]
#   label: the graph full_model/trunc2_<T>_<label> (e38A static naive T/T, e38C rank sort at capacity T, e38E RankTier,
#   blk<B> dropless block budget, blk<B>cal calibrated budget, blk<B>orc trace-specialized budget); tile: def, 512 or 1024 KiB.
# Compiles the programs (at most 6 at a time; an existing program is reused), times each twice in mirrored order with the
# host tool (3 rounds x 20), reports accuracy against the FP32 two-layer reference and bit-identity against the first
# program, then deletes the programs unless E39_KEEP is set.
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e39; mkdir -p $OUT
T=$1; S=$2; shift 2; cd $F || exit 1
case $T in 128) spec=specializations_flat.json; ref=$F/ref;; 256) spec=specializations_T256.json; ref=$F/ref256;; 512) spec=specializations_T512.json; ref=$F/ref512;; esac
qp() { echo $K/qpc_e39_${T}_$1; }
comp() { local label=${1%%:*} ts=${1##*:}; local out=$(qp ${label}_$ts) g=$F/trunc2_${T}_$label; [ -f $out/programqpc.bin ] && return; rm -rf $out; local t0=$(date +%s) extra=""
  [ $ts != def ] && extra="-size-split-granularity=$ts"
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$g/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$g/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $out.log 2>&1
  echo "### ${label}_$ts compile rc=$? $(( $(date +%s)-t0 )) s, $(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref > $OUT/run_$n.stdout 2>&1
  $PY -c "
import json, numpy as np; r = json.load(open('$OUT/run_$n/result.json')); t = r['timing']
y = np.fromfile('$OUT/run_$n/logits_f32.bin', np.float32).astype(np.float64); ref = np.load('$ref/trunc2_logits_last.npy').astype(np.float64)
print(f\"   $n: median {t['median_ms']:.3f} ms, p10-p90 {t['p10_ms']:.3f}-{t['p90_ms']:.3f}, rounds {[round(v, 3) for v in t['round_medians_ms'].values()]}; rel L2 vs FP32 {np.linalg.norm(y - ref) / np.linalg.norm(ref):.4f}, argmax {int(y.argmax())} (ref {int(ref.argmax())}); GiB/SoC {[round(d['active_gib'], 2) for d in r['device_memory']]}\")" 2>/dev/null || { grep -m1 -iE "error" $OUT/run_$n.log | cut -c1-200; tail -n 2 $OUT/run_$n.stdout | cut -c1-200; }; }
progs=("$@")
echo "================ E39 T=$T session $S: ${progs[*]}; df $(df -h $F | tail -1 | awk '{print $4}')"
for pr in "${progs[@]}"; do comp $pr & while [ $(jobs -rp | wc -l) -ge 6 ]; do wait -n; done; done; wait
echo "   df after compiles $(df -h $F | tail -1 | awk '{print $4}')"
for pr in "${progs[@]}"; do hostrun T${T}_${S}_${pr/:/_} $(qp ${pr/:/_}); done
for (( j=${#progs[@]}-1; j>=0; j-- )); do pr=${progs[$j]}; hostrun T${T}_${S}_${pr/:/_}_again $(qp ${pr/:/_}); done
$PY - "$OUT" "$T" "$S" "${progs[@]}" <<'EOP'
import sys, os, json, numpy as np
O, T, S = sys.argv[1:4]; progs = [p.replace(':', '_') for p in sys.argv[4:]]
L = lambda n: np.fromfile(f"{O}/run_{n}/logits_f32.bin", np.float32) if os.path.exists(f"{O}/run_{n}/logits_f32.bin") else None
med = lambda n: json.load(open(f"{O}/run_{n}/result.json"))['timing']['median_ms'] if os.path.exists(f"{O}/run_{n}/result.json") else float('nan')
x0 = L(f"T{T}_{S}_{progs[0]}")
for p in progs[1:]:
    y = L(f"T{T}_{S}_{p}")
    if x0 is not None and y is not None: print(f"   logits {p} vs {progs[0]}: bit-identical {np.array_equal(x0, y)}, rel L2 {np.linalg.norm(y.astype(np.float64) - x0) / np.linalg.norm(x0.astype(np.float64)):.2e}")
print("   session means (both passes): " + ', '.join(f"{p} {np.nanmean([med(f'T{T}_{S}_{p}'), med(f'T{T}_{S}_{p}_again')]):.3f} ms" for p in progs))
EOP
[ -z "$E39_KEEP" ] && for pr in "${progs[@]}"; do rm -rf $(qp ${pr/:/_}); done
echo "   df after cleanup $(df -h $F | tail -1 | awk '{print $4}')"
echo E39_${S}_DONE
