#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e18
comp() { local name=$1 model=$2 out=$3; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=70 -mdts-mos=1 > $E/qpc/$name.log 2>&1; echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; }
comp probe16_s70 $E/T512_probe_cold16_to/model.onnx $E/T512_probe_cold16_to/qpc_s70 & comp probe8_s70 $E/T512_probe_cold8_to/model.onnx $E/T512_probe_cold8_to/qpc_s70 & wait
for L in 16 8; do d=$E/T512_probe_cold${L}_to; bash $S/profile_case2.sh $d 512 "probe cold$L" | head -1; $PY $S/tail_breakdown.py $d/analysis/sample0 512 "probe cold$L" 2>&1 | sed -n 3,4p
$PY - <<EOP
import csv, collections
d="$d/analysis/sample0"; rows=list(csv.DictReader(open(f"{d}/work.csv"))); COLD={'MatMul_3','MatMul_4','MatMul_5'}
def short(n): return n.split('/mlp/')[-1]
ev=collections.Counter(); wd=collections.Counter()
for r in rows:
    if short(r['node']) in COLD and int(r['card'])>=0:
        if r['kind']=='aicconvolutiond32': ev[(int(r['card']),int(r['core']))]+=1
        if r['kind']=='aiccopytovtcm': wd[(int(r['card']),int(r['core']))]+=int(float(r['bytes'] or 0))
for c in range(2): print(f"  card {c} cold HMX events/core:", [ev.get((c,k),0) for k in range(16)], " weight MiB/core:", [round(wd.get((c,k),0)/2**20,1) for k in range(16)])
EOP
done; echo E19F_DONE
