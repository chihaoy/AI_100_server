#!/bin/bash
# E40a: overflow detection and fallback on a two-layer cut (RANKTIER_EXPERIMENT_PLAN.md, experiment 4a, mechanism check).
# One program, RankTier at T=128 with capacity inputs (build_full_tiered.py --capacity-inputs) and three specializations:
# tiered 8x128+8x48+16x16, full 8x128+8x128+16x128, and tight 8x128+8x48+16x4 (undersized, to force overflow); ctx_len 512.
# Four consecutive 128-token chunks (the 512-token held-out prompt) through one retained KV cache, three schedules:
#   R: every chunk tiered (fallback full)   F: chunk 1 forced tight, detected, rerun full   U: every chunk full
# Every schedule is repeated 5 times (the host checks the accepted logits repeat exactly); accepted logits must agree.
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; K=$F/kvtest; OUT=$F/e40; mkdir -p $OUT
cd $F || exit 1; q=$K/qpc_e40_128_tier3cap2L
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
[ -f $q/programqpc.bin ] || { t0=$(date +%s); /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/trunc2_128_tier3capC/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/specializations_T128_cap_ctx512.json -custom-IO-list-file=$F/trunc2_128_tier3capC/custom_io.yaml -compile-only -aic-binary-dir=$q -mdts-mos=1 -size-split-granularity=512 > $q.log 2>&1; echo "### compile rc=$? $(( $(date +%s)-t0 )) s, $(du -sh $q | cut -f1)"; }
g++ -O2 -std=c++17 -Wall -Wextra -Werror -I/opt/qti-aic/dev/inc $M/scripts/moe_qwen3_multichunk_host.cpp -o $OUT/multichunk_host -L/opt/qti-aic/dev/lib/x86_64 -lQAic -Wl,-rpath,/opt/qti-aic/dev/lib/x86_64 || exit 1
P="--tiers 8x128,8x48,16x16 --profile tiered:48,16 --profile full:128,128 --profile tight:48,4"
$PY $M/scripts/e40_plan.py $OUT/plan_a_R.txt $P --primary tiered --fallback full
$PY $M/scripts/e40_plan.py $OUT/plan_a_F.txt $P --primary tiered --fallback full --force 1:tight
$PY $M/scripts/e40_plan.py $OUT/plan_a_U.txt $P --primary full
for s in R F U; do rm -rf $OUT/a_$s; idle; $OUT/multichunk_host $q $F/ref512/input_ids_i64.bin $OUT/a_$s 128 5 $OUT/plan_a_$s.txt > $OUT/a_$s.log 2>&1; echo "=== schedule $s: $(tail -1 $OUT/a_$s.log)"; done
$PY - "$OUT" <<'EOP'
import sys, csv, numpy as np, collections
O = sys.argv[1]
for s in 'RFU':
    rows = list(csv.DictReader(open(f'{O}/a_{s}/attempts.csv')))
    by = collections.defaultdict(list)
    for r in rows: by[(r['chunk'], r['attempt'], r['profile'], r['accepted'], r['overflow_lanes'], r['first_overflow_layer'], r['max_excess'])].append(float(r['host_ms']))
    print(f'schedule {s}: ' + '; '.join(f'chunk {k[0]} attempt {k[1]} {k[2]} accepted {k[3]} overflow lanes {k[4]} (first layer {k[5]}, max excess {k[6]}) median {np.median(v):.3f} ms' for k, v in sorted(by.items())))
L = lambda s, c: np.fromfile(f'{O}/a_{s}/logits_chunk{c}.bin', np.float32)
for c in range(4):
    print(f'chunk {c}: F vs R bit-identical {np.array_equal(L("F", c), L("R", c))}, U vs R bit-identical {np.array_equal(L("U", c), L("R", c))}')
bad = np.fromfile(f'{O}/a_F/logits_chunk1_attempt0.bin', np.float32); ok = L('R', 1)
print(f'rejected tight attempt of chunk 1 vs accepted: rel L2 {np.linalg.norm(bad - ok) / np.linalg.norm(ok):.3e}')
EOP
echo E40A_DONE
