#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e18
comp() { local name=$1 model=$2 stats=${3:-0} out=${4:-$E/qpc/$1}; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; }
comp T512_wide88_s70 $E/T512_wide88_to/model.onnx 70 $E/T512_wide88_to/qpc_s70
for v in T512_wide104 T512_wide88; do d=$E/${v}_to; bash $S/profile_case2.sh $d 512 "$v MXFP6"; $PY $S/tail_breakdown.py $d/analysis/sample0 512 "$v" 2>&1 | head -12; $PY $E/e18_hot.py $d/analysis/sample0 "$v hot/cold engines"
  $PY - <<EOP
import csv, collections
d="$d/analysis/sample0"; rows=list(csv.DictReader(open(f"{d}/work.csv")))
COLD={'MatMul_3','MatMul_4','MatMul_5'}
def short(n): return n.split('/mlp/')[-1]
ev=collections.Counter(); dur=collections.Counter()
for r in rows:
    if short(r['node']) in COLD and r['kind']=='aicconvolutiond32' and int(r['card'])>=0: ev[(int(r['card']),int(r['core']))]+=1; dur[(int(r['card']),int(r['core']))]+=float(r['dur'])
for c in range(4):
    print(f"  card {c} cold HMX events per core:", [ev.get((c,k),0) for k in range(16)], " busy us:", [round(dur.get((c,k),0)) for k in range(16)])
# per-core cold-stage weight DMA bytes
wd=collections.Counter()
for r in rows:
    if short(r['node']) in COLD and r['kind']=='aiccopytovtcm' and int(r['card'])>=0: wd[(int(r['card']),int(r['core']))]+=int(float(r['bytes'] or 0))
print("  cold weight DMA MiB per core, card 0:", [round(wd.get((0,k),0)/2**20,1) for k in range(16)])
EOP
done
echo E19B_DONE
