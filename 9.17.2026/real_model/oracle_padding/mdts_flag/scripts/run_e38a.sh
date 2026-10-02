#!/bin/bash
# E38a: capacity attribution A-E (RANKTIER_EXPERIMENT_PLAN.md, experiment 1) on two-layer cuts, one session:
#   A fixed identity, capacity T, static weights (card-major layout: card c computes experts 32c..32c+31)
#   B fixed identity, capacity T, runtime-indexed gather (identity index as runtime data)
#   C runtime rank, capacity T, gather        D runtime rank, 16xT + 16xT/8        E runtime rank, three tiers (RankTier)
# All with token-owned combine, elementwise final sum, head-parallel attention, flag, MXFP6; each at three maximum tile
# sizes (default, 512, 1024 KiB), timed twice in mirrored order. Accuracy vs the FP32 two-layer reference and pairwise
# bit-identity. E38_T selects the chunk length (default 512); E38_KEEP=1 keeps the programs.
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e38; mkdir -p $OUT
cd $F || exit 1
T=${E38_T:-512}; case $T in 128) spec=specializations_flat.json; ref=$F/ref; E=full_tier3_tileadd_hp; p=full128;; 256) spec=specializations_T256.json; ref=$F/ref256; E=full256_tier3_tileadd_hp; p=full256;; 512) spec=specializations_T512.json; ref=$F/ref512; E=full512_tier3_tileadd_hp; p=full512;; esac
declare -A G=([A]=${p}_e38A_hp [B]=${p}_e38B_hp [C]=${p}_e38C_hp [D]=${p}_e38D_hp [E]=$E)
comp() { local name=$1 graph=$2 extra=$3; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,12p | cut -c1-220; }
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref > $OUT/run_$n.stdout 2>&1
  $PY -c "
import json, numpy as np; r = json.load(open('$OUT/run_$n/result.json')); t = r['timing']
y = np.fromfile('$OUT/run_$n/logits_f32.bin', np.float32).astype(np.float64); ref = np.load('$ref/trunc2_logits_last.npy').astype(np.float64)
print(f\"   $n: median {t['median_ms']:.3f} ms, rounds {[round(v, 3) for v in t['round_medians_ms'].values()]}; two-layer rel L2 vs FP32 {np.linalg.norm(y - ref) / np.linalg.norm(ref):.4f}, argmax {int(y.argmax())} (ref {int(ref.argmax())})\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -n 2 $OUT/run_$n.stdout | cut -c1-200; }; }
for x in A B C D E; do t=trunc2_${T}_e38$x; [ -d $t ] && [ $t/model.onnx -nt ${G[$x]}/model.onnx ] && continue; rm -rf $t; $PY $M/scripts/truncate_layers.py ${G[$x]} $t 2 configs/custom_io.yaml > /dev/null 2>&1 || echo "truncate failed $x"; done
names=(); for x in A B C D E; do for ts in def 512 1024; do names+=(${T}_${x}_$ts); done; done
for x in A B C D E; do for ts in def 512 1024; do extra=""; [ $ts != def ] && extra="-size-split-granularity=$ts"; comp e38_${T}_${x}_$ts trunc2_${T}_e38$x "$extra" & done; wait; done
echo "===== two layers, T=$T, one session (host tool, 3 rounds x 20), each program twice in mirrored order"
for n in "${names[@]}"; do hostrun $n $K/qpc_e38_$n; done
for (( j=${#names[@]}-1; j>=0; j-- )); do n=${names[$j]}; hostrun ${n}_again $K/qpc_e38_$n; done
$PY - <<EOP
import json, os, numpy as np
O = "$OUT"; T = "$T"
L = lambda n: np.fromfile(f"{O}/run_{n}/logits_f32.bin", np.float32).astype(np.float64) if os.path.exists(f"{O}/run_{n}/logits_f32.bin") else None
med = lambda n: json.load(open(f"{O}/run_{n}/result.json"))['timing']['median_ms'] if os.path.exists(f"{O}/run_{n}/result.json") else float('nan')
for a, b in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("A", "E")):
    x, y = L(f"{T}_{a}_def"), L(f"{T}_{b}_def")
    if x is not None and y is not None: print(f"   {b} vs {a} (default tiles): bit-identical {np.array_equal(x, y)}, rel L2 {np.linalg.norm(y - x) / np.linalg.norm(x):.2e}")
print("\n| Two layers, T=" + T + ", mean of both passes | default | 512 KiB | 1024 KiB |\n|---|---:|---:|---:|")
for c in "ABCDE":
    v = [np.mean([med(f"{T}_{c}_{t}"), med(f"{T}_{c}_{t}_again")]) for t in ("def", "512", "1024")]
    print(f"| {c} | " + " | ".join(f"{x:.2f} ms" for x in v) + " |")
EOP
[ -z "$E38_KEEP" ] && rm -rf $K/qpc_e38_${T}_*
echo E38A_DONE
