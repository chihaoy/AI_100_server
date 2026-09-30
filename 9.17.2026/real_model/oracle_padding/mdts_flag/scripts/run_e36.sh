#!/bin/bash
# E36: the full ladder on the 48-layer model (KV retained, MXFP6) at T = 128, 256, 512, one step added at a time:
#   1 production (no flag)  2 + flag and head-parallel attention (native graph, dense combine)  3 + stack rewrites (dense
#   combine)  4 + token-owned combine (= naive T/T)  5 + elementwise final sum (tileadd)  6 + runtime sort hot/cold (best tile
#   size)  7 + 3 tiers. Batch A (steps 1-5, static programs) and batch B (steps 6-7) per chunk length; naive T/T and naive +
#   tileadd run in both batches as anchors. Programs are deleted after their batch; the per-T naive and the 128-token
#   production programs are kept.
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e36; mkdir -p $OUT
compile() { local out=$1 graph=$2 spec=$3 flags=$4; [ -f $out/programqpc.bin ] && { echo "### $(basename $out) exists"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$out $flags > $out.log 2>&1
  echo "### $(basename $out) compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { for i in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
hostrun() { local n=$1 q=$2 ref=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref --routing-counts > $OUT/run_$n.stdout 2>&1
  $PY -c "import json; r=json.load(open('$OUT/run_$n/result.json')); t=r['timing']; l=r['logits']; print(f\"   $n: median {t['median_ms']:.2f} ms ({t['tokens_per_second']:.0f} tok/s), rounds {[round(v,2) for v in t['round_medians_ms'].values()]}; logits rel L2 {l['relative_l2']:.4f}, next '{l['next_token']}' (ref '{l['reference_next_token']}'); memory GiB/card {[round(d['active_gib'],2) for d in r['device_memory']]}\")" 2>/dev/null || { grep -m1 -E "error" $OUT/run_$n.log | cut -c1-200; tail -n 2 $OUT/run_$n.stdout | cut -c1-200; }; }
check_caps() { local n=$1 spec=$2; $PY -c "
import json, numpy as np; c = np.array(json.load(open('$OUT/run_$n/result.json'))['routing_counts']); off = 0; out = []
for part in '$spec'.split(','):
    l, cap = (int(v) for v in part.split('x')); blk = c[:, off:off + 4 * l]; out.append(f'{l}x{cap}: max {blk.max()}, over {int((blk > cap).sum())}'); off += 4 * l
print('   capacity check $n: ' + '; '.join(out))" 2>/dev/null; }
for T in 128 256 512; do
  case $T in
    128) spec=specializations_flat.json; ref=$F/ref; ssg=512; naive=$F/qpc_full_naive_to_flag; prod=$F/qpc_native_noflag; rw=stack_native_rw_ret_dense_hp; nt=full_naive_tileadd_hp; rs=full_dyncard_tileadd_hp; tr=full_tier3_tileadd_hp; rsspec="16x128,16x16"; trspec="8x128,8x48,16x16";;
    256) spec=specializations_T256.json; ref=$F/ref256; ssg=512; naive=$F/qpc256_full_naive_to_flag; prod=$F/qpc256_native_noflag; rw=stack256_native_rw_ret_dense_hp; nt=full256_naive_tileadd_hp; rs=full256_dyncard_tileadd_hp; tr=full256_tier3_tileadd_hp; rsspec="16x256,16x32"; trspec="8x256,8x64,16x32";;
    512) spec=specializations_T512.json; ref=$F/ref512; ssg=1024; naive=$F/qpc512_full_naive_to_flag; prod=$F/qpc512_native_noflag; rw=stack512_native_rw_ret_dense_hp; nt=full512_naive_tileadd_hp; rs=full512_dyncard_tileadd_hp; tr=full512_tier3_tileadd_hp; rsspec="16x512,16x64"; trspec="8x512,8x128,16x64";;
  esac
  q_flag=$F/qpc_e36_${T}_flaghp; q_rw=$F/qpc_e36_${T}_rewrites; q_nt=$F/qpc_e36_${T}_naive_tileadd; q_rs=$F/qpc_e36_${T}_runtime_sort; q_tr=$F/qpc_e36_${T}_tiers
  echo "================ T=$T: batch A (steps 1-5); df $(df -h $F | tail -1 | awk '{print $4}')"
  compile $prod native_c128 $spec "" &
  compile $q_flag native_c128_hp $spec "-mdts-mos=1" &
  compile $q_rw $rw $spec "-mdts-mos=1" &
  compile $q_nt $nt $spec "-mdts-mos=1" &
  wait
  hostrun T${T}_4_naive $naive $ref
  hostrun T${T}_1_production $prod $ref
  hostrun T${T}_2_flag_headpar $q_flag $ref
  hostrun T${T}_3_rewrites $q_rw $ref
  hostrun T${T}_5_tileadd $q_nt $ref
  hostrun T${T}_4_naive_again $naive $ref
  rm -rf $q_flag $q_rw; [ $T != 128 ] && rm -rf $prod
  echo "================ T=$T: batch B (steps 6-7); df $(df -h $F | tail -1 | awk '{print $4}')"
  compile $q_rs $rs $spec "-mdts-mos=1 -size-split-granularity=$ssg" &
  compile $q_tr $tr $spec "-mdts-mos=1 -size-split-granularity=$ssg" &
  wait
  hostrun T${T}_5_tileadd_B $q_nt $ref
  hostrun T${T}_6_runtime_sort $q_rs $ref
  hostrun T${T}_7_tiers $q_tr $ref
  hostrun T${T}_5_tileadd_B_again $q_nt $ref
  hostrun T${T}_4_naive_B $naive $ref
  check_caps T${T}_6_runtime_sort $rsspec; check_caps T${T}_7_tiers $trspec
  rm -rf $q_rs $q_tr $q_nt
done
echo "================ summary"
$PY $M/scripts/e36_summary.py $OUT
echo "df after: $(df -h $F | tail -1 | awk '{print $4}')"
echo E36_DONE
