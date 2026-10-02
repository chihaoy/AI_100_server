#!/bin/bash
# E38b: full-model sessions for the capacity attribution A-E (RANKTIER_EXPERIMENT_PLAN.md, experiment 1).
# Usage: run_e38b.sh <T> <session> <ID:tile> [<ID:tile> ...]   ID in A B C D E, tile in def 512 1024
#   A card-major static capacity T  B identity gather capacity T  C rank sort capacity T  D rank 16xT+16xT/8  E RankTier
# Compiles the session's programs in parallel (an existing program with the same name is reused), records the program
# hash, times each program twice in mirrored order (3 rounds x 20) while sampling SoC clocks, power and temperature every
# 2 s, checks lane capacities from the routing counts, compares logits pairwise, then deletes the programs not listed in
# E38_KEEP (space-separated program names such as "A_512").
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e38; mkdir -p $OUT
T=$1; S=$2; shift 2; cd $F || exit 1
case $T in
  128) spec=specializations_flat.json; ref=$F/ref; p=full128; E=full_tier3_tileadd_hp; tr="8x128,8x48,16x16";;
  256) spec=specializations_T256.json; ref=$F/ref256; p=full256; E=full256_tier3_tileadd_hp; tr="8x256,8x64,16x32";;
  512) spec=specializations_T512.json; ref=$F/ref512; p=full512; E=full512_tier3_tileadd_hp; tr="8x512,8x128,16x64";;
esac
declare -A G=([A]=${p}_e38A_hp [B]=${p}_e38B_hp [C]=${p}_e38C_hp [D]=${p}_e38D_hp [E]=$E)
declare -A CAP=([A]="16x$T,16x$T" [B]="16x$T,16x$T" [C]="16x$T,16x$T" [D]="16x$T,16x$((T/8))" [E]=$tr)
qp() { echo $F/qpc_e38_${T}_$1; }
compile() { local id=${1%%:*} ts=${1##*:}; local out=$(qp ${id}_$ts); [ -f $out/programqpc.bin ] && { echo "### ${id}_$ts exists"; return; }
  rm -rf $out; local t0=$(date +%s); local extra=""; [ $ts != def ] && extra="-size-split-granularity=$ts"
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/${G[$id]}/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $out.log 2>&1
  echo "### ${id}_$ts compile rc=$? $(( ($(date +%s)-t0)/60 )) min qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
telemetry() { local f=$1; while true; do echo "@ $(date +%s.%N)"; /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -E "NSP Frequency|SOC power|Board power|SoC temperature"; sleep 2; done > $f; }
hostrun() { local n=$1 q=$2 id=$3; rm -rf $OUT/run_$n $OUT/run_$n.log; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }; idle
  telemetry $OUT/telemetry_$n.txt & local tp=$!
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref --routing-counts > $OUT/run_$n.stdout 2>&1
  kill $tp 2>/dev/null; wait $tp 2>/dev/null
  $PY - "$OUT" "$n" "${CAP[$id]}" <<'EOP' 2>/dev/null || { grep -m1 -iE "error" $OUT/run_$n.log | cut -c1-200; tail -n 2 $OUT/run_$n.stdout | cut -c1-200; }
import sys, json, re, numpy as np
O, n, spec = sys.argv[1:4]; r = json.load(open(f'{O}/run_{n}/result.json')); t = r['timing']; l = r['logits']
import csv; samples = np.array([float(x['host_ms']) for x in csv.DictReader(open(f'{O}/run_{n}/samples.csv'))])
spread = f", p10-p90 {np.percentile(samples, 10):.2f}-{np.percentile(samples, 90):.2f}" if samples.size else ''
c = np.array(r['routing_counts']); off = 0; caps = []
for part in spec.split(','):
    L, cap = (int(v) for v in part.split('x')); blk = c[:, off:off + 4 * L]; caps.append(f'{L}x{cap} max {blk.max()} over {int((blk > cap).sum())}'); off += 4 * L
txt = open(f'{O}/telemetry_{n}.txt').read().split('@ ')[1:]; fr, pw, tp = [], [], []
for blk in txt:
    fr.append([float(x) for x in re.findall(r'NSP Frequency\(Mhz\):([\d.]+)', blk)]); pw += [float(x) for x in re.findall(r'Board power\(Watts\):([\d.]+)', blk)]; tp += [float(x) for x in re.findall(r'temperature\(degree C\):([\d.]+)', blk)]
fr = np.array([f for f in fr if len(f) == 4]) if fr else np.zeros((0, 4))
tele = f"; NSP MHz per SoC median {[int(v) for v in np.median(fr, 0)]} min {[int(v) for v in fr.min(0)]}; board W peak {max(pw):.0f}; SoC temp max {max(tp):.0f} C" if len(fr) else ''
print(f"   {n}: median {t['median_ms']:.2f} ms ({t['tokens_per_second']:.0f} tok/s), rounds {[round(v, 2) for v in t['round_medians_ms'].values()]}{spread}; logits rel L2 {l['relative_l2']:.4f}, next '{l['next_token']}' (ref '{l['reference_next_token']}'); memory GiB/SoC {[round(d['active_gib'], 2) for d in r['device_memory']]}; caps: {'; '.join(caps)}{tele}")
EOP
}
progs=("$@")
echo "================ E38b T=$T session $S: ${progs[*]}; df $(df -h $F | tail -1 | awk '{print $4}')"
for pr in "${progs[@]}"; do compile $pr & done; wait
for pr in "${progs[@]}"; do id=${pr%%:*} ts=${pr##*:}; q=$(qp ${id}_$ts); [ -f $q/programqpc.bin ] && echo "   program ${id}_$ts: $(stat -c %s $q/programqpc.bin) bytes, sha256 $(sha256sum $q/programqpc.bin | cut -c1-16), graph ${G[$id]}, tile $ts"; done
echo "   df after compiles $(df -h $F | tail -1 | awk '{print $4}')"
for pr in "${progs[@]}"; do id=${pr%%:*} ts=${pr##*:}; hostrun T${T}_${S}_${id}_$ts $(qp ${id}_$ts) $id; done
for (( j=${#progs[@]}-1; j>=0; j-- )); do pr=${progs[$j]}; id=${pr%%:*} ts=${pr##*:}; hostrun T${T}_${S}_${id}_${ts}_again $(qp ${id}_$ts) $id; done
$PY - "$OUT" "$T" "$S" "${progs[@]}" <<'EOP'
import sys, os, json, itertools, numpy as np
O, T, S = sys.argv[1:4]; progs = [p.replace(':', '_') for p in sys.argv[4:]]
L = lambda n: np.fromfile(f"{O}/run_{n}/logits_f32.bin", np.float32).astype(np.float64) if os.path.exists(f"{O}/run_{n}/logits_f32.bin") else None
med = lambda n: json.load(open(f"{O}/run_{n}/result.json"))['timing']['median_ms'] if os.path.exists(f"{O}/run_{n}/result.json") else float('nan')
for a, b in itertools.combinations(progs, 2):
    x, y = L(f"T{T}_{S}_{a}"), L(f"T{T}_{S}_{b}")
    if x is not None and y is not None: print(f"   logits {b} vs {a}: bit-identical {np.array_equal(x, y)}, rel L2 {np.linalg.norm(y - x) / np.linalg.norm(x):.2e}")
print("   session means (both passes): " + ', '.join(f"{p} {np.mean([med(f'T{T}_{S}_{p}'), med(f'T{T}_{S}_{p}_again')]):.2f} ms" for p in progs))
EOP
for pr in "${progs[@]}"; do id=${pr%%:*} ts=${pr##*:}; case " $E38_KEEP " in *" ${id}_$ts "*) ;; *) rm -rf $(qp ${id}_$ts);; esac; done
echo "   df after cleanup $(df -h $F | tail -1 | awk '{print $4}')"
echo E38B_${S}_DONE
