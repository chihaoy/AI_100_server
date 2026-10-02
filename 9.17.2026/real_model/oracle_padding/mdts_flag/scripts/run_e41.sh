#!/bin/bash
# E41: device routing capture for experiment 4b. Runs the held-out chunk sets of e41_make_chunks.py (every workload, both
# halves, at most 200 chunks per half) through a full-capacity program (static naive T/T, program A of E38) as independent
# prefills, and stores its per-layer routing counts as full_model/e41/counts_T<T>_<workload>_<split>.npy [n, 48, 128], in the
# program's card-major lane order (lane i holds expert 32c+j as listed in e41_analysis.py ORDER), not in expert order.
# Usage: run_e41.sh <T> <program dir>
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; O=$F/e41; E=$F/e40
T=$1; q=$2; cd $F || exit 1
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
$PY - "$O" "$T" <<'EOP'
import sys, glob, os, json, numpy as np
O, T = sys.argv[1], int(sys.argv[2]); parts = []; cat = []
for f in sorted(glob.glob(f'{O}/chunks_T{T}_*_*.bin')):
    w, split = os.path.basename(f)[len(f'chunks_T{T}_'):-4].rsplit('_', 1); a = np.fromfile(f, np.int64).reshape(-1, T)[:200]
    parts.append([w, split, len(a)]); cat.append(a)
np.concatenate(cat).tofile(f'{O}/capture_T{T}_ids.bin'); json.dump(parts, open(f'{O}/capture_T{T}_manifest.json', 'w'))
print(f'T={T}: {sum(p[2] for p in parts)} chunks: ' + ', '.join(f'{w}/{s} {n}' for w, s, n in parts))
EOP
$PY $M/scripts/e40_plan.py $O/plan_capture_T$T.txt --tiers 16x$T,16x$T --profile own: --no-inputs --primary own --capture
rm -rf $O/capture_T$T; idle; t0=$(date +%s)
$E/multichunk_host $q $O/capture_T${T}_ids.bin $O/capture_T$T $T 1 $O/plan_capture_T$T.txt > $O/capture_T$T.log 2>&1; echo "$(tail -1 $O/capture_T$T.log) in $(( $(date +%s) - t0 )) s"
$PY - "$O" "$T" <<'EOP'
import sys, json, numpy as np
O, T = sys.argv[1], int(sys.argv[2]); parts = json.load(open(f'{O}/capture_T{T}_manifest.json'))
c = np.fromfile(f'{O}/capture_T{T}/counts_all.bin', np.int32).reshape(-1, 48, 128); off = 0
assert len(c) == sum(p[2] for p in parts), (len(c), sum(p[2] for p in parts)); assert (c.sum(-1) == 8 * T).all()
for w, s, n in parts: np.save(f'{O}/counts_T{T}_{w}_{s}.npy', c[off:off + n]); off += n
print(f'saved {len(parts)} count sets, {len(c)} chunks; largest expert count {c.max()}')
EOP
echo E41_DONE
