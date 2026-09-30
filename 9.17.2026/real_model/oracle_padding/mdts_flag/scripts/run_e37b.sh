#!/bin/bash
# E37b: 2x2 ablation of the two optimization series on the 48-layer model (KV retained, MXFP6) at T = 512, 256, 128.
#   N  naive reduction + naive padding : native graph + head-parallel attention, flag (E36 step 2)
#   R  our reduction + naive padding   : rewrites + token-owned combine + elementwise final sum, capacity T (E36 step 5)
#   P  naive reduction + our padding   : export combine + per-card runtime sort + three tiers (build_full_tiered.py --combine dense)
#   D  our reduction + our padding     : RankTier (E36 step 7)
#   P2 naive reduction + rank sort only: export combine + runtime sort 16xT + 16xT/8 (--combine dense)
# Session 1 per T: N R P D, then D P R N, plus the kept E36 naive T/T program as a cross-session anchor. Session 2 per T:
# N P2 P, then P P2 N (P2 chained through N and P). Gathered-weight programs use the E36 tile sizes. Programs are deleted
# after their session (disk: session 1 needs 226 GB, session 2 201 GB).
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e37; mkdir -p $OUT
compile() { local out=$1 graph=$2 spec=$3 flags=$4; [ -f $out/programqpc.bin ] && { echo "### $(basename $out) exists"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$out $flags > $out.log 2>&1
  echo "### $(basename $out) compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref --routing-counts > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; l=r['logits']; print(f\"   $n: median {t['median_ms']:.2f} ms ({t['tokens_per_second']:.0f} tok/s), rounds {[round(v,2) for v in t['round_medians_ms'].values()]}; logits rel L2 {l['relative_l2']:.4f}, next '{l['next_token']}' (ref '{l['reference_next_token']}'); memory GiB/card {[round(d['active_gib'],2) for d in r['device_memory']]}\")" 2>/dev/null || { grep -m1 -iE "error" $OUT/run_$n.log | cut -c1-200; tail -n 2 $OUT/run_$n.stdout | cut -c1-200; }; }
check_caps() { local n=$1 spec=$2; $PY -c "
import json, numpy as np; c = np.array(json.load(open('$OUT/run_$n/result.json'))['routing_counts']); off = 0; out = []
for part in '$spec'.split(','):
    l, cap = (int(v) for v in part.split('x')); blk = c[:, off:off + 4 * l]; out.append(f'{l}x{cap}: max {blk.max()}, over {int((blk > cap).sum())}'); off += 4 * l
print('   capacity check $n: ' + '; '.join(out) + f'; assignments per layer {sorted(set(c.sum(1).tolist()))}')" 2>/dev/null; }
same() { $PY -c "
import numpy as np, os
x, y = ('$OUT/run_$1/logits_f32.bin', '$OUT/run_$2/logits_f32.bin')
if os.path.exists(x) and os.path.exists(y):
    a, b = np.fromfile(x, np.float32).astype(np.float64), np.fromfile(y, np.float32).astype(np.float64)
    print(f'   logits $2 vs $1: bit-identical {np.array_equal(a, b)}, rel L2 {np.linalg.norm(b - a) / np.linalg.norm(a):.2e}')" 2>/dev/null; }
for T in ${E37_TS:-512 256 128}; do
  case $T in
    128) spec=specializations_flat.json; ref=$F/ref; ssg=512; e36=$F/qpc_full_naive_to_flag; R=full_naive_tileadd_hp; D=full_tier3_tileadd_hp; P=full128_tier3_dense_hp; P2=full128_sort_dense_hp; tr="8x128,8x48,16x16"; so="16x128,16x16";;
    256) spec=specializations_T256.json; ref=$F/ref256; ssg=512; e36=$F/qpc256_full_naive_to_flag; R=full256_naive_tileadd_hp; D=full256_tier3_tileadd_hp; P=full256_tier3_dense_hp; P2=full256_sort_dense_hp; tr="8x256,8x64,16x32"; so="16x256,16x32";;
    512) spec=specializations_T512.json; ref=$F/ref512; ssg=1024; e36=$F/qpc512_full_naive_to_flag; R=full512_naive_tileadd_hp; D=full512_tier3_tileadd_hp; P=full512_tier3_dense_hp; P2=full512_sort_dense_hp; tr="8x512,8x128,16x64"; so="16x512,16x64";;
  esac
  qN=$F/qpc_e37_${T}_N; qR=$F/qpc_e37_${T}_R; qP=$F/qpc_e37_${T}_P; qD=$F/qpc_e37_${T}_D; qP2=$F/qpc_e37_${T}_P2
  echo "================ T=$T session 1 (N R P D); df $(df -h $F | tail -1 | awk '{print $4}')"
  compile $qN native_c128_hp $spec "-mdts-mos=1" &
  compile $qR $R $spec "-mdts-mos=1" &
  compile $qP $P $spec "-mdts-mos=1 -size-split-granularity=$ssg" &
  compile $qD $D $spec "-mdts-mos=1 -size-split-granularity=$ssg" &
  wait
  echo "   df after compiles $(df -h $F | tail -1 | awk '{print $4}')"
  hostrun T${T}_E36naive $e36 $ref
  for n in N R P D; do q=q$n; hostrun T${T}_$n ${!q} $ref; done
  for n in D P R N; do q=q$n; hostrun T${T}_${n}_again ${!q} $ref; done
  check_caps T${T}_P $tr; check_caps T${T}_D $tr
  same T${T}_P T${T}_P_again; same T${T}_N T${T}_P; same T${T}_R T${T}_D
  rm -rf $qR $qD
  echo "================ T=$T session 2 (N P2 P); df $(df -h $F | tail -1 | awk '{print $4}')"
  compile $qP2 $P2 $spec "-mdts-mos=1 -size-split-granularity=$ssg"
  for n in N P2 P; do q=q$n; hostrun T${T}_S2_$n ${!q} $ref; done
  for n in P P2 N; do q=q$n; hostrun T${T}_S2_${n}_again ${!q} $ref; done
  check_caps T${T}_S2_P2 $so; same T${T}_S2_P2 T${T}_S2_P
  rm -rf $qN $qP $qP2
done
echo "df after: $(df -h $F | tail -1 | awk '{print $4}')"
echo E37B_DONE
