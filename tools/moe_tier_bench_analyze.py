#!/usr/bin/env python3
"""Summarise moe_tier_bench variants: per-variant inference cycles + per-op/per-core breakdown.

Reads <root>/<variant>/{info.json,run.log,out/*.summary.txt}. All numbers are hardware measurements
from the SDK's qaic-opstats per-core tables (pcycles where the SDK gives them, else ucycles) and
qaic-runner's ExecTimeUs / InfPCycles. Samples (profiled inferences) are averaged.

  python3 moe_tier_bench_analyze.py <root> [variant ...] [--ops] [--cores]
"""
import argparse, glob, json, os, re, statistics, sys
from collections import defaultdict

ROW = re.compile(r"^(c\d+)\s+(\d+)\s+([\d.]+)%\s+(N/A|\d+)\s+(N/A|[\d.]+%)\s+(\d+)\s+(N/A|[\d.e+]+)\s+\S+\s+\S+\s+\S+\s+(.*)$")
TOT = re.compile(r"^(c\d+) total:\s+(\d+) ucycles\s+(\d+) pcycles")
INF = re.compile(r"^(c\d+): Total inference pcycle is (\d+)")

# op families we care about (substring match on opdetails)
FAM = [
    ("gemm", "aicconvolutiond32"),
    ("dequant", "blockdequantize_mxfp6"),
    ("cumsum", "cumsum"),
    ("reduce_ddr", "aicbatchedreduceadd HVX DDR"),
    ("reduce_tcm", "aicbatchedreduceadd HVX TCM"),
    ("copy2vtcm", "aiccopytovtcm"),
    ("copyddr", "aiccopyddr"),
    ("scatter", "aicscatter"),
    ("gather", "aicgather"),
    ("sync_hmx", "sync HMX"),
    ("sync_hvx", "sync HVX"),
    ("barrier", "barrier"),
]


def parse_summary(path):
    """-> {core: {"total_u":..,"total_p":..,"inf_p":.., "ops": {opdetails: (u, p, count, outkb)}}}"""
    cores = defaultdict(lambda: {"ops": {}})
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = TOT.match(line)
        if m:
            cores[m.group(1)]["total_u"] = int(m.group(2)); cores[m.group(1)]["total_p"] = int(m.group(3)); continue
        m = INF.match(line)
        if m:
            cores[m.group(1)]["inf_p"] = int(m.group(2)); continue
        m = ROW.match(line)
        if m:
            c, u, _, p, _, cnt, outkb, det = m.groups()
            p = int(p) if p != "N/A" else None
            outkb = float(outkb) if outkb != "N/A" else 0.0
            d = cores[c]["ops"]
            det = det.strip()
            if det in d:
                u0, p0, c0, k0 = d[det]
                d[det] = (u0 + int(u), (p0 or 0) + (p or 0) if (p0 is not None or p is not None) else None, c0 + int(cnt), k0 + outkb)
            else:
                d[det] = (int(u), p, int(cnt), outkb)
    return dict(cores)


def parse_runlog(path):
    r = {}
    if not os.path.exists(path):
        return r
    for line in open(path, errors="replace"):
        for key in ("ExecTimeUs", "InfPCycles"):
            if line.startswith(key):
                parts = [x.strip() for x in line.split(",") if x.strip()]
                try:
                    r[key] = dict(mean=float(parts[1]), min=float(parts[2]), max=float(parts[3]), std=float(parts[4]))
                except Exception:
                    pass
    return r


def load_variant(d):
    info = json.load(open(os.path.join(d, "info.json")))
    sums = sorted(glob.glob(os.path.join(d, "out", "*.summary.txt")))
    if not sums:
        return info, None, parse_runlog(os.path.join(d, "run.log"))
    # group by slice (card); average over samples
    by_slice = defaultdict(list)
    for s in sums:
        m = re.search(r"slice(\d+)", os.path.basename(s))
        by_slice[int(m.group(1)) if m else 0].append(parse_summary(s))
    return info, by_slice, parse_runlog(os.path.join(d, "run.log"))


def avg_cores(samples):
    """average per-core totals and per-family cycles over samples -> {core: {...}}"""
    out = {}
    cores = sorted(set().union(*[s.keys() for s in samples]), key=lambda c: int(c[1:]))
    for c in cores:
        recs = [s[c] for s in samples if c in s]
        tot_u = statistics.mean(r.get("total_u", 0) for r in recs)
        tot_p = statistics.mean(r.get("total_p", 0) for r in recs)
        fam = {}
        for name, sub in FAM:
            vals_u, vals_p, cnts = [], [], []
            for r in recs:
                u = p = cnt = 0
                for det, (uu, pp, cc, kk) in r["ops"].items():
                    if sub in det:
                        u += uu; p += (pp or 0); cnt += cc
                vals_u.append(u); vals_p.append(p); cnts.append(cnt)
            fam[name] = (statistics.mean(vals_u), statistics.mean(vals_p), statistics.mean(cnts))
        # busy = everything that's not sync/barrier (ucycles), per core
        busy = []
        for r in recs:
            b = 0
            for det, (uu, pp, cc, kk) in r["ops"].items():
                if det.startswith("sync") or det.startswith("barrier"):
                    continue
                b += uu
            busy.append(b)
        out[c] = dict(total_u=tot_u, total_p=tot_p, busy_u=statistics.mean(busy), fam=fam)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--ops", action="store_true", help="print per-op family table")
    ap.add_argument("--cores", action="store_true", help="print per-core totals")
    ap.add_argument("--csv", help="write summary csv")
    ap.add_argument("--md", action="store_true", help="print a markdown report (needs merged traces)")
    ap.add_argument("--baseline", default="u128")
    a = ap.parse_args()
    if a.md:
        variants = a.variants or sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.root, "*")) if os.path.isdir(p))
        text, _ = md_report(a.root, variants, a.baseline)
        print(text)
        return
    variants = a.variants or sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.root, "*")) if os.path.isdir(p))
    rows = []
    print(f"{'variant':<14}{'groups':<34}{'rows':>6}{'ratio':>7}{'ExecUs':>9}{'InfPcyc(runner)':>17}{'max core pcyc':>14}{'mean core pcyc':>15}{'busy cv':>9}{'gemm':>9}{'dequant':>9}{'cumsum':>8}{'reduce':>9}{'cp2vtcm':>9}")
    for v in variants:
        d = os.path.join(a.root, v)
        if not os.path.exists(os.path.join(d, "info.json")):
            continue
        info, by_slice, run = load_variant(d)
        groups = ",".join(f"{t}:{n}" for t, n in info["groups"])
        if by_slice is None:
            print(f"{v:<14}{groups:<34}{info['padded_rows_total']:>6}{info['row_ratio']:>7.3f}   (no profiling output)")
            continue
        # aggregate over cards: for each card, avg over samples; then report max over all (card,core)
        percard = {sl: avg_cores(samps) for sl, samps in by_slice.items()}
        allcores = [(sl, c, r) for sl, cs in percard.items() for c, r in cs.items()]
        max_p = max(r["total_p"] for _, _, r in allcores)
        mean_p = statistics.mean(r["total_p"] for _, _, r in allcores)
        busy = [r["busy_u"] for _, _, r in allcores]
        cv = statistics.pstdev(busy) / statistics.mean(busy) if statistics.mean(busy) else 0
        # family sums (pcycles for compute families, ucycles for DMA/sync) over all cores
        def fam_sum(name, use="p"):
            idx = 1 if use == "p" else 0
            return sum(r["fam"][name][idx] for _, _, r in allcores)
        gemm = fam_sum("gemm"); deq = fam_sum("dequant"); cums = fam_sum("cumsum")
        red = fam_sum("reduce_ddr") + fam_sum("reduce_tcm"); cp = fam_sum("copy2vtcm", "u")
        ex = run.get("ExecTimeUs", {}).get("mean", float("nan")); ip = run.get("InfPCycles", {}).get("min", float("nan"))
        print(f"{v:<14}{groups:<34}{info['padded_rows_total']:>6}{info['row_ratio']:>7.3f}{ex:>9.0f}{ip:>17.0f}{max_p:>14.0f}{mean_p:>15.0f}{cv:>9.3f}{gemm/1e3:>8.0f}k{deq/1e3:>8.0f}k{cums/1e3:>7.0f}k{red/1e3:>8.0f}k{cp/1e3:>8.1f}k")
        rows.append(dict(variant=v, groups=groups, rows=info["padded_rows_total"], ratio=info["row_ratio"], exec_us=ex,
                         inf_pcycles_min=ip, max_core_pcycles=max_p, mean_core_pcycles=mean_p, busy_cv=cv,
                         gemm_pcycles=gemm, dequant_pcycles=deq, cumsum_pcycles=cums, reduce_pcycles=red, copy2vtcm_ucycles=cp))
        if a.cores:
            for sl, cs in sorted(percard.items()):
                line = " ".join(f"{c}:{r['total_p']/1e3:.0f}k/{r['busy_u']/1e3:.1f}k" for c, r in cs.items())
                print(f"    card{sl} total_pcyc/busy_ucyc per core: {line}")
                g = [r['fam']['gemm'] for c, r in cs.items()]
                d = [r['fam']['dequant'] for c, r in cs.items()]
                print(f"    card{sl} gemm  pcyc/core: " + " ".join(f"{x[1]/1e3:.0f}k" for x in g) + f"   (calls {sum(x[2] for x in g):.0f}, card total {sum(x[1] for x in g)/1e3:.0f}k)")
                print(f"    card{sl} deq   pcyc/core: " + " ".join(f"{x[1]/1e3:.0f}k" for x in d) + f"   (calls {sum(x[2] for x in d):.0f}, card total {sum(x[1] for x in d)/1e3:.0f}k)")
        if a.ops:
            for name, _ in FAM:
                us = [r["fam"][name][0] for _, _, r in allcores]; ps = [r["fam"][name][1] for _, _, r in allcores]; cn = [r["fam"][name][2] for _, _, r in allcores]
                print(f"    {name:<11} u={sum(us):>10.0f}  p={sum(ps):>11.0f}  count={sum(cn):>6.0f}   per-core u: min {min(us):.0f} max {max(us):.0f}")
    if a.csv and rows:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------------------------
# Markdown report (phases from the merged trace + per-op families from the SDK summaries)
# ---------------------------------------------------------------------------------------------
def md_report(root, variants, baseline="u128"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from moe_tier_bench_phases import analyse as ph_analyse, load as ph_load
    rows = []
    for v in variants:
        d = os.path.join(root, v)
        if not os.path.exists(os.path.join(d, "info.json")):
            continue
        info, by_slice, run = load_variant(d)
        if by_slice is None:
            continue
        # runner: per-device ExecTimeUs mean -> take max over devices (latency is gated by slowest card)
        ex = {}
        for line in open(os.path.join(d, "run.log"), errors="replace"):
            m = re.match(r"ExecTimeUs_Dev_(\d+)_Func_0,\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", line)
            if m:
                ex[int(m.group(1))] = dict(mean=float(m.group(2)), min=float(m.group(3)), max=float(m.group(4)), std=float(m.group(5)))
        ex_max = max(x["mean"] for x in ex.values()) if ex else float("nan")
        ex_std = max(x["std"] for x in ex.values()) if ex else float("nan")
        percard = {sl: avg_cores(samps) for sl, samps in by_slice.items()}
        allcores = [(sl, c, r) for sl, cs in percard.items() for c, r in cs.items()]
        def fam_sum(name, use="p"):
            idx = 1 if use == "p" else 0
            return sum(r["fam"][name][idx] for _, _, r in allcores)
        def fam_cnt(name):
            return sum(r["fam"][name][2] for _, _, r in allcores)
        # phases (sample 1 trace)
        tr = sorted(glob.glob(os.path.join(d, "out", "*inf-1-*merged*.trace.json"))) or sorted(glob.glob(os.path.join(d, "out", "*merged*.trace.json")))
        exp_ms = red_ms = float("nan"); cards_exec = []
        gemm_core_min = gemm_core_max = float("nan")
        if tr:
            cards = ph_analyse(ph_load(tr[0]))
            ee, gl = [], []
            for card, cores in sorted(cards.items()):
                e = max(x.get("exec_end", 0) for x in cores.values()); g = max(x.get("gemm_last", 0) for x in cores.values() if "gemm_last" in x)
                ee.append(e); gl.append(g); cards_exec.append(e / 1e3)
            exp_ms = max(gl) / 1e3; red_ms = (max(ee) - max(gl)) / 1e3
        gp = [r["fam"]["gemm"][1] for _, _, r in allcores]
        gemm_core_min, gemm_core_max = min(gp), max(gp)
        busy = [r["busy_u"] for _, _, r in allcores]
        cv = statistics.pstdev(busy) / statistics.mean(busy)
        rows.append(dict(variant=v, groups=",".join(f"{t}:{n}" for t, n in info["groups"]), rows=info["padded_rows_total"],
                         ratio=info["row_ratio"], exec_ms=ex_max / 1e3, exec_std_ms=ex_std / 1e3, exp_ms=exp_ms, red_ms=red_ms,
                         cards_exec=cards_exec, gemm_calls=fam_cnt("gemm"), gemm_M=fam_sum("gemm") / 1e6, deq_M=fam_sum("dequant") / 1e6,
                         cumsum_M=fam_sum("cumsum") / 1e6, cumsum_n=fam_cnt("cumsum"), reduce_M=(fam_sum("reduce_ddr") + fam_sum("reduce_tcm")) / 1e6,
                         cp_M=fam_sum("copy2vtcm", "u") / 1e6, synchvx_M=fam_sum("sync_hvx", "u") / 1e6, busy_cv=cv,
                         gemm_core_min=gemm_core_min / 1e3, gemm_core_max=gemm_core_max / 1e3))
    base = next((r for r in rows if r["variant"] == baseline), None)
    out = []
    out.append("| variant | lane groups (tier:lanes) | padded rows | rows vs u128 | **latency ms** (max card, mean±std) | vs u128 | expert phase ms | post-GEMM+reduce ms | GEMM calls | GEMM Mpcyc | dequant Mpcyc | cumsum Mpcyc (calls) | copy2vtcm Mucyc | sync HVX Mucyc | busy cv |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        rel = r["exec_ms"] / base["exec_ms"] if base else float("nan")
        out.append(f"| {r['variant']} | `{r['groups']}` | {r['rows']} | {r['ratio']:.3f} | **{r['exec_ms']:.2f}** ±{r['exec_std_ms']:.2f} | {rel:.2f}x | {r['exp_ms']:.2f} | {r['red_ms']:.2f} | {r['gemm_calls']:.0f} | {r['gemm_M']:.1f} | {r['deq_M']:.1f} | {r['cumsum_M']:.2f} ({r['cumsum_n']:.0f}) | {r['cp_M']:.2f} | {r['synchvx_M']:.1f} | {r['busy_cv']:.3f} |")
    out.append("")
    out.append("| variant | per-card latency ms (trace, card0..3) | GEMM pcycles per core min / max (k) |")
    out.append("|---|---|---|")
    for r in rows:
        out.append(f"| {r['variant']} | {' / '.join(f'{x:.2f}' for x in r['cards_exec'])} | {r['gemm_core_min']:.0f} / {r['gemm_core_max']:.0f} |")
    return "\n".join(out), rows


if __name__ == "__main__":
    main()
