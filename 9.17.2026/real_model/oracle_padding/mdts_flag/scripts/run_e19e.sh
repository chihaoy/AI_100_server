#!/bin/bash
# E19e: can the core mapping of the 80-lane cold stage be steered? -mos=2/4, larger cold capacity
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e18
compm() { local name=$1 model=$2 mos=$3 stats=${4:-0} out=${5:-$E/qpc/$1}; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=$mos -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 4,10p | cut -c1-200; }
compm T512_wide88_mos2 $E/T512_wide88_to/model.onnx 2 & compm T512_wide88_mos4 $E/T512_wide88_to/model.onnx 4 & compm T512_wide88_cc64 $E/T512_wide88_cc64_to/model.onnx 1 & compm T512_anchor_mos4 $S/e15/T512_tokenowned/model.onnx 4 & wait
compm T512_wide88_mos4_s70 $E/T512_wide88_to/model.onnx 4 70 $E/T512_wide88_to/qpc_s70_mos4 & compm T512_wide88_cc64_s70 $E/T512_wide88_cc64_to/model.onnx 1 70 $E/T512_wide88_cc64_to/qpc_s70 & wait
$PY - <<EOP
import json, os
S="$S"; E="$E"; anc=f"{S}/e7/T512_hc_296_32"; refy=f"{S}/e8/timing_D_runs/T512_hc_296_32__mx_r0/y.bin"
dirs={"T512_anchor":None,"T512_anchor_mos4":None,"T512_wide88":f"{E}/T512_wide88_to","T512_wide88_mos2":f"{E}/T512_wide88_to","T512_wide88_mos4":f"{E}/T512_wide88_to","T512_wide88_cc64":f"{E}/T512_wide88_cc64_to"}
cases=[dict(label=n, qpc=f"{E}/qpc/{n}", input=f"{(d or anc)}/input_f16.bin", counts=f"{(d or anc)}/expected_counts_i32.bin", ref_y=refy) for n,d in dirs.items() if os.path.isfile(f"{E}/qpc/{n}/programqpc.bin")]
json.dump(dict(cases=cases), open(f"{E}/spec19e.json","w"), indent=1); print(len(cases), "cases")
EOP
$PY $S/e7_timing.py $E/spec19e.json $E/timing19e.json 2 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error"
cores() { $PY - <<EOP
import csv, collections
d="$1"; rows=list(csv.DictReader(open(f"{d}/work.csv"))); COLD={'MatMul_3','MatMul_4','MatMul_5'}; HOT={'MatMul','MatMul_1','MatMul_2'}
def short(n): return n.split('/mlp/')[-1]
for stg, S_ in (('cold', COLD), ('hot', HOT)):
    ev=collections.Counter(); wd=collections.Counter()
    for r in rows:
        if short(r['node']) in S_ and int(r['card'])>=0:
            if r['kind']=='aicconvolutiond32': ev[(int(r['card']),int(r['core']))]+=1
            if r['kind']=='aiccopytovtcm': wd[(int(r['card']),int(r['core']))]+=int(float(r['bytes'] or 0))
    print(f"  {stg} card0 HMX events/core:", [ev.get((0,k),0) for k in range(16)], " weight MiB/core:", [round(wd.get((0,k),0)/2**20,1) for k in range(16)])
EOP
}
for v in wide88_mos4:qpc_s70_mos4:T512_wide88_to wide88_cc64:qpc_s70:T512_wide88_cc64_to; do n=${v%%:*}; q=$(echo $v | cut -d: -f2); dd=$E/$(echo $v | cut -d: -f3); rm -rf $dd/qpc_s70_use; ln -s $dd/$q $dd/qpc_s70_use
  sed "s#\$d/qpc_s70#\$d/qpc_s70_use#g" $S/profile_case2.sh > $S/profile_case3.sh; bash $S/profile_case3.sh $dd 512 "$n" | head -2; $PY $S/tail_breakdown.py $dd/analysis/sample0 512 "$n" 2>&1 | sed -n 1,7p; cores $dd/analysis/sample0; done
echo E19E_DONE
