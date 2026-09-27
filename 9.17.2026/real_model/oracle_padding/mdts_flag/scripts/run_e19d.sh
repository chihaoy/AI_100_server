#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e18
comp() { local name=$1 model=$2 stats=${3:-0} out=${4:-$E/qpc/$1} extra=$5; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 $extra > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 4,10p | cut -c1-200; }
comp T512_wide88_ols2 $E/T512_wide88_to/model.onnx 0 $E/qpc/T512_wide88_ols2 "-ols=2" & comp T512_wide88_ols4 $E/T512_wide88_to/model.onnx 0 $E/qpc/T512_wide88_ols4 "-ols=4" & comp T512_wide104_L128_ols2 $E/T512_wide104_L128_to/model.onnx 0 $E/qpc/T512_wide104_L128_ols2 "-ols=2" & comp T256_wide58_s70 $E/T256_wide58_to/model.onnx 70 $E/T256_wide58_to/qpc_s70 & wait
$PY - <<EOP
import json, os
S="$S"; E="$E"; anc=f"{S}/e7/T512_hc_296_32"; refy=f"{S}/e8/timing_D_runs/T512_hc_296_32__mx_r0/y.bin"
dirs={"T512_anchor":None,"T512_wide88":f"{E}/T512_wide88_to","T512_wide88_ols2":f"{E}/T512_wide88_to","T512_wide88_ols4":f"{E}/T512_wide88_to","T512_wide104_L128_ols2":f"{E}/T512_wide104_L128_to"}
cases=[dict(label=n, qpc=f"{E}/qpc/{n}", input=f"{(d or anc)}/input_f16.bin", counts=f"{(d or anc)}/expected_counts_i32.bin", ref_y=refy) for n,d in dirs.items() if os.path.isfile(f"{E}/qpc/{n}/programqpc.bin")]
json.dump(dict(cases=cases), open(f"{E}/spec19d.json","w"), indent=1); print(len(cases), "cases")
EOP
$PY $S/e7_timing.py $E/spec19d.json $E/timing19d.json 2 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error"
d=$E/T256_wide58_to; bash $S/profile_case2.sh $d 256 "T256_wide58 MXFP6"; $PY $S/tail_breakdown.py $d/analysis/sample0 256 "T256_wide58" 2>&1 | sed -n 1,7p
$PY - <<EOP
import csv, collections
d="$d/analysis/sample0"; rows=list(csv.DictReader(open(f"{d}/work.csv"))); COLD={'MatMul_3','MatMul_4','MatMul_5'}
def short(n): return n.split('/mlp/')[-1]
ev=collections.Counter(); wd=collections.Counter()
for r in rows:
    if short(r['node']) in COLD and int(r['card'])>=0:
        if r['kind']=='aicconvolutiond32': ev[(int(r['card']),int(r['core']))]+=1
        if r['kind']=='aiccopytovtcm': wd[(int(r['card']),int(r['core']))]+=int(float(r['bytes'] or 0))
for c in range(2): print(f"  card {c} cold HMX events per core:", [ev.get((c,k),0) for k in range(16)], " weight MiB:", [round(wd.get((c,k),0)/2**20,1) for k in range(16)])
EOP
echo E19D_DONE
