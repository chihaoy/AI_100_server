# Constraint 1: runtime capacity selection at fixed token count

Measured 2026-09-21, SDK 1.21.6. **One loaded QPC can select different expert
capacities on successive invocations while the token count remains 128.** This
works for a four-card program and a two-card program mapped to either card pair.
The host creates, loads and activates one program and creates one ExecObj per run.

This is a mechanism probe: synthetic weights, 128 experts, top-8, 64 lanes and two
stages, H=256 and intermediate width=128. The graph includes routing, token packing,
all three expert projections, weighting and reduction. It excludes attention, KV
state and actual Qwen checkpoint weights. Repeated invocations use the same normal
input to isolate switching; they are not a stateful prefill-chunk experiment.

## Available freedom and actual constraints

| Question | Result |
|---|---|
| Select capacities without changing token count? | Yes: `(128,128)`, `(64,32)`, `(32,64)`, `(32,32)` in one program |
| Choose asymmetric stage capacities? | Yes; both orientations were executed and validated |
| Change between calls without host program reload/reactivation? | Yes; every measured run uses one load and activation |
| Use a two-card QPC? | Yes; separately tested on cards 0/1 and 2/3 |
| Arbitrary uncompiled capacity? | No in this path: `(48,48)` fails runtime shape matching |
| Select using a scalar profile value alone? | Not implemented; selection here uses input tensor shapes |
| Multiple symbolic capacities without a unique identifying input? | The tested four-profile configuration fails compilation |
| Avoid a large joint selector tensor? | Yes: separate row vectors plus a small profile tag |
| Four full copies of the weights? | Packed constant storage grew only 1.60% in FP16, not fourfold |
| Free/zero-cost switching at every scale? | Not established; small measured differences include noise/cache effects |
| Runtime expert regrouping, migration, or within-call dispatch? | Not tested by this probe |

The two-card runs execute the same complete toy MoE on two cards. They do not
measure the original hot/cold split, concurrent pair execution, or host merging.
Latency comparisons across card counts do not predict the best full-model layout.

## Selector design

The successful compact graph accepts:

- `x`: FP16 `[128,256]`, identical shape and contents across profiles.
- `rows0`: int32 `[C0]`, values `0..C0-1`.
- `rows1`: int32 `[C1]`, values `0..C1-1`.
- `tag`: int32 `[P]`, all zero. Its length uniquely identifies the profile.

The row-vector lengths determine packing canvases and GEMM rows. Row values
participate in valid-row masks. The tag participates in the first stage's mask as
a zero offset, preserving it as a real graph input. Profile configurations are:

| P | C0 | C1 | Extra input bytes, excluding x |
|---|---|---|---|
| 1 | 128 | 128 | 1,028 |
| 2 | 64 | 32 | 392 |
| 3 | 32 | 64 | 396 |
| 4 | 32 | 32 | 272 |

Calling `qaicExecObjSetDataExt` with matching input/output dimensions selects the
variant. All dimensions must match a compiled combination. In the deliberate
48/48 negative test, binding and enqueue returned success, but completion returned
status 300 and runtime logs said `No matching specialization found`. The production
host must validate profiles before enqueueing, rather than rely on SetData alone.
This agrees with the SDK's [specialized-buffer shape rules](https://quic.github.io/cloud-ai-sdk-pages/1.21/Getting-Started/SDK-Tools/shared/qaic_runner/aic_batch_json_input.html).

## FP16 correctness and timing

Each cell below is **repeated-profile / cycling-profiles** median host latency in
milliseconds. Each median contains 300 samples across three rounds. Order is
reversed between rounds. Timing includes buffer binding, enqueue and completion;
output validation is outside the timed region. Routing counts are returned in all
timed executions. Compilation uses stats level 70, so this is an instrumented
mechanism benchmark, not an optimized application latency claim.

| Capacities | Cards 0-3 | Cards 0/1 | Cards 2/3 |
|---|---|---|---|
| 128 / 128 | 3.4747 / 3.4026 | 2.7896 / 2.7121 | 2.8150 / 2.7992 |
| 64 / 32 | 2.5403 / 2.5803 | 2.2724 / 2.2413 | 2.1935 / 2.1978 |
| 32 / 64 | 2.5214 / 2.5316 | 2.1907 / 2.2042 | 2.1937 / 2.1953 |
| 32 / 32 | 2.4254 / 2.4165 | 2.1883 / 2.1896 | 2.1530 / 2.1702 |

All 2,400 timed outputs and count vectors per run matched that run's 128/128
output exactly. CPU and device counts matched, with 1,024 assignments and stage
maxima 15/14; no normal input overflowed. Relative L2 error against the independent
FP32 expert loop was 0.1231% on four cards and about 0.1233% on two cards. The
reference uses the CPU FP16 router, whose decisions match this controlled input.

Four-card 32/32 reduced repeated-profile latency by 30.20%. Cycling minus repeated
medians ranged from -72 to +40 microseconds on four cards. Negative differences
illustrate timing variability; these data do not establish exactly zero overhead.

The ONNX packing subgraph was independently executed for every profile. Its two
expert activation tensors were `[64,C0,256]` and `[64,C1,256]`, giving logical
expert GEMM row counts `[C0,C0,C0,C1,C1,C1]`. Separate device traces for every
profile were decoded successfully, and runner outputs matched the host benchmark.
The shape audit establishes logical dimensions; it does not claim exact physical
HMX tile shapes or prove identical expert-to-core mapping across specializations.

A separate stress input routes all 128 tokens to the same eight experts. At 32/32,
the pre-truncation counters report 768 excess assignments and the output differs
from full capacity. This proves the overflow detector is necessary and exercised;
it is not an overflow-recovery implementation.

## Weight storage and MXFP6

| Four-card package | One profile | Four profiles |
|---|---|---|
| FP16 packed static constants | 32.2891 MiB | 32.8066 MiB |
| FP16 entire QPC | 35.7323 MiB | 44.0060 MiB |
| MXFP6 packed static constants | 17.6250 MiB | 18.1436 MiB |
| MXFP6 entire QPC | 21.1457 MiB | 29.5410 MiB |

Constants are shared substantially, but code/metadata growth is significant.
Four-card FP16 device free-memory snapshots were unchanged across profile switches.
These checks plus the single-load host path do not constitute a PCIe DMA trace:
on-device weight reads, specialization-specific copies and VTCM reloads may occur.
The top-level MDP wrapper reports zero constants/required-memory fields; those are
not valid memory measurements. The audit extracts the actual per-card segments.

MXFP6 also produced bit-identical outputs across all profiles and matching routing
counts. Repeated-profile latency was 3.3695 ms at 128/128 and 2.3792 ms at 32/32.
However, **all MXFP6 profiles have a shared 9.9676% CPU-reference error**, failing
the 5% absolute reference gate. Its result records `validation_pass: false`,
`reference_accuracy_pass: false`, and `capacity_equivalence_pass: true`. MXFP6
timing is accepted only in explicit equivalence-only mode. This is not certification
of model accuracy. Two-card MXFP6 was not tested.

## Failed and intermediate probes retained

- `capacity_4card`: separate row inputs, no unique profile tag; compiler rejects
  the four profiles with `No input that uniquely identifies specialization`.
- `capacity_joint_4card`: a uint8 joint selector gets past that issue but the
  uint8-to-int32 conversion is rejected as `Unsupported cast from quantized type`.
  This only establishes a failure of that specific cast path.
- `capacity_joint_i32_4card`: int32 `[C0,C1]` selector works. The initial random
  router had CPU/device top-k differences and 3.71% shared output error, so it was
  replaced by a router with explicit 8th/9th logit margins for the main validation.
- `capacity_margin_4card` / `capacity_margin_2card`: joint int32 selector with
  controlled routing passes. The compact tagged design is the final implementation.
- Direct ExecObj trace retrieval returned empty data in the attempted path;
  final device traces use the SDK runner and `qaic-opstats` instead.

## Reproduce and inspect

Use fresh output directories. Artifacts are ignored by git.

```bash
PY=/home/chihao/qeff-venv/bin/python
$PY tools/moe_capacity_probe.py --out /tmp/capacity4 --devices 4
$PY tools/moe_capacity_audit.py --out /tmp/capacity4 --profile
$PY tools/moe_capacity_probe.py --out /tmp/capacity2 --devices 2
$PY tools/moe_capacity_probe.py --out /tmp/capacity2 --stage run \
  --device-ids 2,3 --run-name run_pair23
$PY tools/moe_capacity_probe.py --out /tmp/capacity4 --stage compile --precision mxfp6
$PY tools/moe_capacity_probe.py --out /tmp/capacity4 --stage run --precision mxfp6 --equivalence-only
```

Raw final results:

- [Four-card FP16](capacity_tagged_4card/fp16/run/result.json), [samples](capacity_tagged_4card/fp16/run/samples.csv), [shape/storage audit](capacity_tagged_4card/fp16/package_audit.json)
- [Cards 0/1 FP16](capacity_tagged_2card/fp16/run/result.json), [cards 2/3 FP16](capacity_tagged_2card/fp16/run_pair23/result.json)
- [Four-card MXFP6, equivalence only](capacity_tagged_4card/mxfp6/run/result.json)
- [Device traces](capacity_tagged_4card/fp16/profiling/) and [memory snapshots](capacity_tagged_4card/fp16/run/resources.csv)

## Consequence for algorithm design

A finite library of capacity vectors is a feasible runtime decision space. It
can include asymmetric capacities and use a small identifying input. It must
include a conservative option and enforce exact compiled combinations. Scaling
profile count, Qwen-sized matrices, many layers, correct KV continuation and
state-safe overflow recovery remain open.

The second independent constraint, [runtime selection/regrouping of resident
expert weights](RESIDENT_SELECTION.md), has now been tested separately on one
card at small and Qwen matrix dimensions. Runtime IDs work, and the tested
weight-gather graph preserves the MXFP6 path. The [combined-adaptation probe](COMBINED_ADAPTATION.md)
now verifies membership and capacity selection together at group width 4,
including stateless overflow replay and the cost of profile-library storage.
The third probe, [group count/width](GROUP_WIDTH.md), tests widths 1–32 at equal
padded work and measures the resulting latency and HMX core participation.
The broader [design space](../multilayer/RUNTIME_ADAPTATION.md) remains open.
