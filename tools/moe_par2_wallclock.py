#!/usr/bin/env python3
"""Path 2 of naive_vs_64_16: two independent QPCs (hot lane group on cards 0,1; cold lane group on cards 2,3) launched from
the host at the same time, partial sums added on the host.  Host wall clock per inference via the qaicrt Python API:
  each QPC solo, hot->cold serial in one thread, hot||cold concurrent (two threads released by a barrier) + host add,
  plus the 4-card naive / 2-partition 64/16 QPCs for reference.  Median / p10 / p90 over N iterations after warm-up.
A QPC may output "y" [T,H] (reduced on card) or "partial" [D,T,H] (per-device partials; the host sums over D).

  python3 tools/moe_par2_wallclock.py --pair A=DIR/hot_d4/qpc,DIR/cold_d4/qpc --pair B=... --x DIR/hot/x.bin [--naive QPC] [--mdp2 QPC] [-n 50]
"""
import argparse, statistics, threading, time, numpy as np
from QEfficient.generation.cloud_infer import QAICInferenceSession

def stats(ts):
    ts = sorted(ts); n = len(ts)
    return f"median {statistics.median(ts)*1e3:6.2f} ms  p10 {ts[n//10]*1e3:6.2f}  p90 {ts[(9*n)//10]*1e3:6.2f}  min {ts[0]*1e3:6.2f}"

def timed(fn, n, warm=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return ts

def out_sum(o):
    """[T,H] from a session output dict: 'y' as is, 'partial' [D,T,H] summed over D (fp32)."""
    a = o["y"] if "y" in o else o["partial"]
    a = a.astype(np.float32)
    return a.sum(0) if a.ndim == 3 else a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="NAME=hot_qpc,cold_qpc")
    ap.add_argument("--x", required=True); ap.add_argument("--naive", default=None); ap.add_argument("--mdp2", default=None)
    ap.add_argument("-n", type=int, default=50)
    a = ap.parse_args()
    x = np.fromfile(a.x, np.float16).reshape(128, 2048); inp = {"x": x}
    res = {}; sums = {}
    for spec in a.pair:
        name, qpcs = spec.split("="); hq, cq = qpcs.split(",")
        hot = QAICInferenceSession(hq, [0, 1]); cold = QAICInferenceSession(cq, [2, 3])
        out = {}
        def run_hot(): out["hot"] = out_sum(hot.run(inp))
        def run_cold(): out["cold"] = out_sum(cold.run(inp))
        def run_serial(): run_hot(); run_cold(); out["sum"] = out["hot"] + out["cold"]
        def run_conc():
            b = threading.Barrier(3)
            def w(f): b.wait(); f()
            th = [threading.Thread(target=w, args=(run_hot,)), threading.Thread(target=w, args=(run_cold,))]
            for t in th: t.start()
            b.wait()
            for t in th: t.join()
            out["sum"] = out["hot"] + out["cold"]
        res[f"[{name}] hot solo   ({hot.output_names[0]}, {hq.split('/')[-2]})"] = timed(run_hot, a.n)
        res[f"[{name}] cold solo  ({cold.output_names[0]}, {cq.split('/')[-2]})"] = timed(run_cold, a.n)
        res[f"[{name}] hot -> cold serial, 1 thread"] = timed(run_serial, a.n)
        res[f"[{name}] hot || cold 2 threads + host add"] = timed(run_conc, a.n)
        sums[name] = out["sum"].copy()
        hot.deactivate(); cold.deactivate(); del hot, cold
    h = np.zeros((128, 2048), np.float32); c = np.ones((128, 2048), np.float32)
    res["host add only, 2 x [128,2048] fp32"] = timed(lambda: h + c, a.n)
    ynaive = None
    if a.naive:
        s = QAICInferenceSession(a.naive, [0, 1, 2, 3]); res["naive 4-card QPC (1 partition)"] = timed(lambda: s.run(inp)["y"], a.n)
        ynaive = s.run(inp)["y"].astype(np.float32); s.deactivate(); del s
    if a.mdp2:
        s = QAICInferenceSession(a.mdp2, [0, 1, 2, 3]); res["64/16 2-partition QPC (partitions serial)"] = timed(lambda: s.run(inp)["y"], a.n); s.deactivate(); del s
    print(f"\nhost wall clock per inference, N={a.n} after 5 warm-up (MXFP6 QPCs):")
    for k, v in res.items(): print(f"  {k:58s} {stats(v)}")
    if ynaive is not None:
        print("\nnumerics (all MXFP6 on card): host-added hot+cold vs naive 4-card output")
        for name, s in sums.items():
            d = np.abs(s - ynaive).max(1)
            print(f"  [{name}] max|diff| {d.max():.4g}  tokens diff<1e-3: {(d < 1e-3).sum()}/128  tokens >1% of max|y|={np.abs(ynaive).max():.3g}: {(d > 0.01*np.abs(ynaive).max()).sum()}/128")

if __name__ == "__main__":
    main()
