#!/bin/bash
# E19c: 128-lane cold stage (2 lanes per core) vs anchor, drop-empties split and 76-lane wide
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e18
comp() { local name=$1 model=$2 stats=${3:-0} out=${4:-$E/qpc/$1}; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 4,12p | cut -c1-220; }
for i in 1 2 3 4; do [ -f $E/T512_wide104_L128_to/model.onnx ] && break; sleep 15; done
comp T512_wide104_L128 $E/T512_wide104_L128_to/model.onnx & comp T512_wide88_L128 $E/T512_wide88_L128_to/model.onnx & comp T512_wide104_L128_s70 $E/T512_wide104_L128_to/model.onnx 70 $E/T512_wide104_L128_to/qpc_s70 & wait
$PY - <<EOP
import json, os
S="$S"; E="$E"; anc=f"{S}/e7/T512_hc_296_32"; refy=f"{S}/e8/timing_D_runs/T512_hc_296_32__mx_r0/y.bin"
dirs={"T512_anchor":None,"T512_split104":f"{E}/T512_split104_to","T512_wide104":f"{E}/T512_wide104_to","T512_wide104_L128":f"{E}/T512_wide104_L128_to","T512_wide88_L128":f"{E}/T512_wide88_L128_to"}
cases=[dict(label=n, qpc=f"{E}/qpc/{n}", input=f"{(d or anc)}/input_f16.bin", counts=f"{(d or anc)}/expected_counts_i32.bin", ref_y=refy) for n,d in dirs.items() if os.path.isfile(f"{E}/qpc/{n}/programqpc.bin")]
json.dump(dict(cases=cases), open(f"{E}/spec19c.json","w"), indent=1); print(len(cases), "cases")
EOP
$PY $S/e7_timing.py $E/spec19c.json $E/timing19c.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|round 0"
d=$E/T512_wide104_L128_to; bash $S/profile_case2.sh $d 512 "T512_wide104_L128 MXFP6"; $PY $S/tail_breakdown.py $d/analysis/sample0 512 "T512_wide104_L128" 2>&1 | head -8; $PY $E/e18_hot.py $d/analysis/sample0 "T512_wide104_L128 hot/cold engines"
$PY - <<EOP
import csv, collections
d="$d/analysis/sample0"; rows=list(csv.DictReader(open(f"{d}/work.csv"))); COLD={'MatMul_3','MatMul_4','MatMul_5'}
def short(n): return n.split('/mlp/')[-1]
ev=collections.Counter(); wd=collections.Counter()
for r in rows:
    if short(r['node']) in COLD and int(r['card'])>=0:
        if r['kind']=='aicconvolutiond32': ev[(int(r['card']),int(r['core']))]+=1
        if r['kind']=='aiccopytovtcm': wd[(int(r['card']),int(r['core']))]+=int(float(r['bytes'] or 0))
for c in range(4): print(f"  card {c} cold HMX events per core:", [ev.get((c,k),0) for k in range(16)], " weight MiB:", [round(wd.get((c,k),0)/2**20,1) for k in range(16)])
EOP
echo E19C_DONE
