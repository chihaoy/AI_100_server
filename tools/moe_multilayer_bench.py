#!/usr/bin/env python3
"""Compile and validate a stitched MoE graph, then benchmark one resident QPC.

Diagnostics expose counts; separate timing graphs expose only the final hidden
state. Both variants use the same compiler flags. Results stay beside the graphs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np


def command(argv, log):
    print("RUN", " ".join(map(str, argv)), flush=True)
    with log.open("w") as stream:
        stream.write("COMMAND " + repr(list(map(str, argv))) + "\n")
        stream.flush()
        subprocess.run(list(map(str, argv)), stdout=stream, stderr=subprocess.STDOUT, check=True)


def compare(actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Invalid output shape or nonfinite values")
    err = a - b
    return dict(exact=bool(np.array_equal(a, b)), max_abs=float(np.abs(err).max()),
                relative_l2=float(np.linalg.norm(err) / max(np.linalg.norm(b), 1e-12)),
                max_token_relative_l2=float(np.max(np.linalg.norm(err, axis=-1) /
                                                   np.maximum(np.linalg.norm(b, axis=-1), 1e-8))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Exporter output directory")
    ap.add_argument("--stage", choices=["all", "compile", "run"], default="all")
    ap.add_argument("--precision", choices=["fp16", "mxfp6"], default="mxfp6")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reference-npy", type=Path, help="Independent fp32 residual reference after the last layer")
    ap.add_argument("--equivalence-only", action="store_true",
                    help="Measure an EXACT baseline match despite a shared reference error; keep accuracy failure recorded")
    ap.add_argument("--sdk", type=Path, default=Path("/opt/qti-aic"))
    a = ap.parse_args()
    if a.iterations < 1 or a.rounds < 1 or a.warmup < 0:
        ap.error("Invalid iteration/round/warmup count")
    root = a.out.resolve()
    info = json.loads((root / "info.json").read_text())
    compiled = root / a.precision
    compiled.mkdir(exist_ok=True)
    if a.stage != "run":
        for variant in ("uniform", "tuned"):
            for kind, filename in (("diagnostic", "model.onnx"), ("timing", "timing.onnx")):
                directory = compiled / variant / kind
                directory.mkdir(parents=True, exist_ok=True)
                qpc = directory / "qpc"
                if (qpc / "programqpc.bin").exists():
                    print(f"Reusing {qpc}", flush=True)
                    continue
                cmd = [a.sdk / "exec/qaic-compile", "-aic-hw", "-aic-hw-version=ai100",
                       f"-m={root / variant / filename}", "-convert-to-fp16", "-aic-num-cores=16",
                       "-mos=1", "-aic-enable-depth-first", f"-mdp-load-partition-config={root / 'mdp.json'}",
                       "-stats-level=70", "-compile-only", f"-aic-binary-dir={qpc}"]
                if a.precision == "mxfp6":
                    cmd.append("-mxfp6-matmul")
                command(cmd, directory / "compile.log")
                if not (qpc / "programqpc.bin").exists():
                    raise RuntimeError(f"Compiler did not produce {qpc}")
    if a.stage == "compile":
        return

    run = compiled / ("run_" + time.strftime("%Y%m%d_%H%M%S"))
    run.mkdir(exist_ok=False)
    host = run / "host"
    command(["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", f"-I{a.sdk / 'dev/inc'}",
             Path(__file__).with_name("moe_single_qpc_host.cpp"), "-o", host,
             f"-L{a.sdk / 'dev/lib/x86_64'}", "-lQAic", f"-Wl,-rpath,{a.sdk / 'dev/lib/x86_64'}",
             "-lpthread", "-ldl"], run / "host_build.log")
    result = dict(layers=info["layers"], precision=a.precision, iterations=a.iterations,
                  warmup=a.warmup, rounds=a.rounds, scope=info["reference_scope"], diagnostics={}, timing={})
    shape = (info["T"], info["H"])
    outputs = {}
    reference_pass = True
    capacity_pass = True
    for variant in ("uniform", "tuned"):
        destination = run / f"{variant}_diagnostic"
        command([host, compiled / variant / "diagnostic/qpc", root / "x.bin", destination, 1, 2],
                run / f"{variant}_diagnostic.log")
        y = np.fromfile(destination / "y.bin", np.float16).reshape(shape)
        outputs[variant] = y
        cpu = np.load(root / variant / "y_cpu.npy")
        report = dict(vs_cpu=compare(y, cpu), routing={})
        if a.reference_npy:
            report["vs_independent_reference"] = compare(y, np.load(a.reference_npy))
            reference_pass &= report["vs_independent_reference"]["relative_l2"] < 0.05
        # This comparison includes quantization and serves as a gross error check.
        reference_pass &= report["vs_cpu"]["relative_l2"] < 0.05
        for layer in info["layers"]:
            counts = np.fromfile(destination / f"L{layer}_counts.bin", np.int32)
            if counts.shape != (128,) or np.any(counts < 0) or np.any(counts > info["T"]):
                raise RuntimeError(f"Invalid device routing counts at layer {layer}")
            caps = [info["T"]] * 2 if variant == "uniform" else info["plan"][str(layer)]["stage_widths"]
            overflow = int(np.maximum(counts - np.repeat(caps, 64), 0).sum())
            report["routing"][str(layer)] = dict(capacities=caps,
                stage_max=counts.reshape(2,64).max(1).tolist(), assignments=int(counts.sum()), overflow=overflow)
            capacity_pass &= overflow == 0
        result["diagnostics"][variant] = report
    result["tuned_vs_uniform"] = compare(outputs["tuned"], outputs["uniform"])
    if a.reference_npy:
        result["independent_reference"] = str(a.reference_npy.resolve())
    capacity_pass &= result["tuned_vs_uniform"]["relative_l2"] < 0.005
    capacity_pass &= result["tuned_vs_uniform"]["max_token_relative_l2"] < 0.02
    result["reference_accuracy_pass"] = bool(reference_pass)
    result["capacity_equivalence_pass"] = bool(capacity_pass)
    result["validation_pass"] = bool(reference_pass and capacity_pass)
    result["equivalence_only"] = a.equivalence_only
    exact_equivalence = capacity_pass and result["tuned_vs_uniform"]["exact"]
    (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    accepted = exact_equivalence if a.equivalence_only else result["validation_pass"]
    if not accepted:
        raise RuntimeError(f"Validation failed; latency is not accepted. See {run / 'result.json'}")

    samples = {v: [] for v in outputs}
    for round_id in range(a.rounds):
        # Reverse execution order each round to reduce order/temperature bias.
        order = ("uniform", "tuned") if round_id % 2 == 0 else ("tuned", "uniform")
        for variant in order:
            destination = run / f"{variant}_timing_{round_id}"
            command([host, compiled / variant / "timing/qpc", root / "x.bin", destination,
                     a.iterations, a.warmup], run / f"{variant}_timing_{round_id}.log")
            y = np.fromfile(destination / "y.bin", np.float16).reshape(shape)
            diff = compare(y, outputs[variant])
            if a.equivalence_only and not diff["exact"]:
                raise RuntimeError("Equivalence-only timing requires exact diagnostic/timing output equality")
            if diff["relative_l2"] >= 0.005 or diff["max_token_relative_l2"] >= 0.02:
                raise RuntimeError(f"Timing graph differs from diagnostic graph: {diff}")
            timing = json.loads((destination / "timing.json").read_text())
            samples[variant].append(timing["samples_ms"])
    for variant, rounds in samples.items():
        ts = np.array(rounds)
        result["timing"][variant] = dict(median_ms=float(np.median(ts)), p10_ms=float(np.percentile(ts,10)),
            p90_ms=float(np.percentile(ts,90)), round_medians_ms=np.median(ts, axis=1).tolist())
    base, tuned = (result["timing"][v]["median_ms"] for v in ("uniform", "tuned"))
    result["speedup"] = base / tuned
    result["latency_reduction_percent"] = 100 * (1 - tuned / base)
    result["timing_scope"] = ("exact padding equivalence; absolute reference accuracy gate FAILED"
                              if not reference_pass else "padding equivalence and reference accuracy gates passed")
    (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (compiled / "latest_run.txt").write_text(str(run) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
