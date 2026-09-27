#!/bin/bash
# E20d: where does the dynamic split spend its extra time? stats70 profile of the T=256 dynamic graph vs the anchor
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e20
for i in $(seq 60); do grep -q E20C_DONE $S/run_e20c.out && break; sleep 15; done
d=$E/T256_dynsplit_to; rm -rf $d/qpc_s70; t0=$(date +%s)
/opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$d/model.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$d/qpc_s70 -stats-level=70 -mdts-mos=1 > $d/compile_s70.log 2>&1; echo "### T256_dynsplit s70 rc=$? $(( $(date +%s)-t0 )) s"
for f in input_f16.bin expected_counts_i32.bin; do rm -f $d/$f; done; ln -s $E/inputs256/T256_cat0_input_f16.bin $d/input_f16.bin; ln -s $E/inputs256/T256_cat0_counts_sorted_i32.bin $d/expected_counts_i32.bin
bash $S/profile_case2.sh $d 256 "T256 dynsplit MXFP6" | head -1; $PY $S/tail_breakdown.py $d/analysis/sample0 256 "T256_dynsplit" 2>&1 | sed -n 1,12p
$PY - <<EOP
import csv, collections
d="$d/analysis/sample0"; rows=list(csv.DictReader(open(f"{d}/work.csv"))); t0=min(float(r['start']) for r in rows)
def short(n): return n.split('/mlp/')[-1] if '/mlp/' in n else n
fam=collections.defaultdict(lambda: [1e18,-1e18,0.0,set()])
for r in rows:
    n=short(r['node']); key=n if n.startswith('ds_') or n.startswith('probe') else None
    if key is None:
        b=n.split('_rc')[0]
        if b in ('MatMul','MatMul_1','MatMul_2'): key='hot GEMM/DMA'
        elif b in ('MatMul_3','MatMul_4','MatMul_5'): key='cold GEMM/DMA'
        elif b.startswith('CumSum'): key='scan chains'
        elif b.startswith('CtxGather3D'): key='token gathers'
        elif b.startswith('to_'): key='token-owned combine'
        else: continue
    if int(r['card'])<0: continue
    f=fam[key]; f[0]=min(f[0],float(r['start'])); f[1]=max(f[1],float(r['end'])); f[2]+=float(r['dur']); f[3].add(r['kind'])
print("  family                    first start  last end  busy(ms)  kinds")
for k,(s,e,b,kinds) in sorted(fam.items(), key=lambda kv: kv[1][0]): print(f"  {k:24s} {(s-t0)/1e3:8.3f} {(e-t0)/1e3:9.3f} {b/1e3:8.2f}  {sorted(kinds)[:4]}")
EOP
echo E20D_DONE
