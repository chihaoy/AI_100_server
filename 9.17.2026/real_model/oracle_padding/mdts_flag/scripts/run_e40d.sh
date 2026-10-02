#!/bin/bash
# E40d: where the 18% of E40b comes from. The same capacity-input RankTier graph (T=128, ctx_len 512, 512 KiB tiles) compiled
# with only the tiered specialization, timed over the same four chunks as E40b's schedule R. If it runs like the
# single-profile RankTier program (122.5 ms), the cost is the extra specializations; if like E40b's tiered profile
# (144.8 ms), it is the capacity inputs themselves. Compiles at once; times after the E39 session at T=256 (the queue is
# held with its pause file in between, so the device is idle).
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e40
cd $F || exit 1; q=$F/qpc_e40_128_tier3cap1
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
t0=$(date +%s); /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/full128_tier3cap_hp/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_T128_cap1_ctx512.json -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$q -mdts-mos=1 -size-split-granularity=512 > $q.log 2>&1
echo "### compile rc=$? $(( ($(date +%s)-t0)/60 )) min, programqpc.bin $(stat -c %s $q/programqpc.bin 2>/dev/null) bytes"
until grep -q E39_P1dg_DONE $F/e39/e39_T256_P1dg.txt 2>/dev/null; do sleep 20; done
$PY $M/scripts/e40_plan.py $OUT/plan_d_R.txt --tiers 8x128,8x48,16x16 --profile tiered:48,16 --primary tiered
rm -rf $OUT/d_R; idle; $OUT/multichunk_host $q $F/ref512/input_ids_i64.bin $OUT/d_R 128 5 $OUT/plan_d_R.txt > $OUT/d_R.log 2>&1
echo "=== single-profile capacity-input program: $(tail -1 $OUT/d_R.log); setup $(tr '\n' ' ' < $OUT/d_R/setup.csv)"
$PY - "$OUT" <<'EOP'
import sys, csv, numpy as np, collections
O = sys.argv[1]; by = collections.defaultdict(list)
for r in csv.DictReader(open(f'{O}/d_R/attempts.csv')): by[int(r['chunk'])].append(float(r['host_ms']))
print('per chunk median ms: ' + ', '.join(f'{c}: {np.median(v):.2f}' for c, v in sorted(by.items())))
for c in range(4):
    a = np.fromfile(f'{O}/d_R/logits_chunk{c}.bin', np.float32); b = np.fromfile(f'{O}/b_R/logits_chunk{c}.bin', np.float32)
    print(f'chunk {c}: bit-identical to the three-profile program (E40b schedule R): {np.array_equal(a, b)}')
EOP
rm -rf $q; rm -f $F/e38/queue.pause; echo "$(date +%F_%T) E40D_DONE, queue released"
