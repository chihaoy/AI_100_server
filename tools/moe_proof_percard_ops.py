#!/usr/bin/env python3
"""Emit the per-(op, card) evidence table that proves how expert weights are placed.

For each named op in the graph it reports, per card:
   events   -- number of profiled op events
   KB       -- bytes the op moved, derived as opPCycle / PCyclesPerKB (both from the SDK trace)
   pcycles  -- hardware cycles

The discriminator: under expert parallelism (card c owns experts 32c..32c+31) a Gather of
8 experts would land unevenly across cards -- and if the 8 are forced to be experts 0..7,
entirely on card 0. Under tensor parallelism every card holds 1/4 of every expert, so all
four cards move exactly the same bytes no matter which experts the router picks.

Usage:
  python3 moe_proof_percard_ops.py --label natural <trace.json> \
                                   --label forced  <trace.json> --csv out.csv
"""
import sys, re, argparse, collections, csv

TN=re.compile(r'"name"\s*:\s*"thread_name"'); TID=re.compile(r'"tid"\s*:\s*(\d+)')
SL=re.compile(r'QAicGraph_slice(\d+)_Core_(\d+)')
NAME=re.compile(r'"opName"\s*:\s*"([^"]*)"'); KIND=re.compile(r'"opKind"\s*:\s*"([^"]*)"')
PC=re.compile(r'"opPCycle"\s*:\s*"?([\d.]+)"?'); PKB=re.compile(r'"PCyclesPerKB"\s*:\s*"?([\d.]+)"?')

def canon(op): return re.sub(r"/__\d+$", "", op.strip().split()[-1])

def scan(path, sites):
    t2c={}
    with open(path, errors="ignore") as f:
        for line in f:
            if TN.search(line):
                t=TID.search(line); s=SL.search(line)
                if t and s: t2c[int(t.group(1))]=int(s.group(1))
    ev=collections.defaultdict(collections.Counter)
    kb=collections.defaultdict(lambda: collections.defaultdict(float))
    cy=collections.defaultdict(collections.Counter)
    with open(path, errors="ignore") as f:
        for line in f:
            n=NAME.search(line)
            if not n: continue
            nm=canon(n.group(1))
            if nm not in sites: continue
            t=TID.search(line)
            if not t: continue
            c=t2c.get(int(t.group(1)))
            if c is None: continue
            ev[nm][c]+=1
            p=PC.search(line); q=PKB.search(line)
            if p: cy[nm][c]+=int(float(p.group(1)))
            if p and q and float(q.group(1))>0:
                kb[nm][c]+=float(p.group(1))/float(q.group(1))
    return sorted({c for c in t2c.values()}), ev, kb, cy

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("trace", nargs="+")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--csv")
    a=ap.parse_args()
    assert len(a.label)==len(a.trace), "每个 --label 要配一个 trace"
    L=a.layer
    SITES=[f"/model/layers.{L}/mlp/gate/MatMul",
           f"/model/layers.{L}/mlp/Gather_3", f"/model/layers.{L}/mlp/Gather_4",
           f"/model/layers.{L}/mlp/Gather_5", f"/model/layers.{L}/mlp/MatMul",
           f"/model/layers.{L}/mlp/MatMul_1", f"/model/layers.{L}/mlp/MatMul_2",
           f"/model/layers.{L}/mlp/Einsum",
           f"/model/layers.{L}/self_attn/q_proj/MatMul"]
    SS=set(SITES)
    rows=[]
    for lab, tr in zip(a.label, a.trace):
        cards, ev, kb, cy = scan(tr, SS)
        print("="*94)
        print(f"### {lab}    {tr.split('/')[-1][:70]}")
        print(f"{'op (layer %d)'%L:26s} " + " ".join(f"{'card'+str(c):>11s}" for c in cards)
              + f" {'合计':>11s} {'极差%':>7s}")
        for nm in SITES:
            if nm not in kb: continue
            v=[kb[nm].get(c,0.0) for c in cards]
            sp=(max(v)/min(v)-1)*100 if min(v)>0 else float("inf")
            short=nm.replace(f"/model/layers.{L}/","")
            print(f"{short:26s} " + " ".join(f"{x:11.1f}" for x in v)
                  + f" {sum(v):11.1f} {sp:6.2f}%")
            for c in cards:
                rows.append([lab, short, c, ev[nm].get(c,0),
                             round(kb[nm].get(c,0.0),2), cy[nm].get(c,0)])
        print("  (数值单位 KB,由 SDK trace 的 opPCycle / PCyclesPerKB 推出)")
    if a.csv:
        with open(a.csv,"w",newline="") as fh:
            w=csv.writer(fh); w.writerow(["run","op","card","events","kb","pcycles"]); w.writerows(rows)
        print(f"\nwrote {a.csv}  ({len(rows)} rows)")

if __name__=="__main__": main()
