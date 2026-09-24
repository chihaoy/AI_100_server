# Joint runtime controls and single-card scaling

Date: 2026-09-21. SDK/firmware: 1.21.6. Device: card 0.

This implements the user's requested steps 1 and 2 before the new multi-card
experiments. All weights and inputs are synthetic. Matrix dimensions match
Qwen3-30B-A3B experts: hidden size 2048, intermediate size 768, global 128 experts,
top-8 routing. The 32-expert bank computes a partial global contribution; the
128-expert bank computes the complete synthetic MoE contribution.

<!-- MEASUREMENTS_BEGIN -->
## Measured results

**Steps 1 and 2 passed their mechanism checks across 44,400 timed transactions.**
Both expert-bank sizes and all three layer counts ran on card 0. Counts were exact,
safe profiles preserved the conservative result within the equivalence gate,
and device allocations stayed constant while inputs and profiles changed.
Four layers is a tested point, not the maximum supported depth.

Maximum error against the independent FP32 reference:

| Resident experts / layers | FP16 relative L2 | MXFP6 relative L2 |
|---|---:|---:|
| 32 / 1 | 0.0645% | 6.0345% |
| 128 / 1 | 0.0671% | 6.0187% |
| 128 / 2 | 0.0668% | 6.0069% |
| 128 / 4 | 0.0694% | 5.9984% |

**MXFP6 remains above the 5% accuracy target.** Its dispatch/equivalence results
are valid mechanism measurements, not a resolved model-accuracy result.

Balanced workloads, held median end-to-end latency in ms. Adaptive includes the
current one-core router, host policy/handoff and 15-core experts. Conservative
is the fused 15-core `full4` profile on the same input and sorted IDs.

| Experts | Tokens | FP16 adaptive | FP16 conservative | MXFP6 adaptive | MXFP6 conservative |
|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 5.290 | 5.222 | 3.789 | 3.713 |
| 32 | 128 | 6.571 | 16.442 | 5.016 | 6.342 |
| 32 | 256 | 9.061 | 19.139 | 7.633 | 9.294 |
| 128 | 64 | 18.145 | 18.283 | 12.347 | 12.457 |
| 128 | 128 | 21.431 | 63.481 | 15.913 | 22.689 |
| 128 | 256 | 26.893 | 70.545 | 21.859 | 31.088 |

At 128 experts and T=128, the count-driven policy selected these profiles.
The following latencies use the cycling schedule, which changes workloads, IDs,
chunk shapes and profiles within one activation.

| Load | Selected profile | FP16 ms | MXFP6 ms |
|---|---|---:|---:|
| balanced | small4 | 21.916 | 16.267 |
| warm | mixed | 24.288 | 14.214 |
| hot | mixed | 24.859 | 14.283 |
| concentrated | mixed | 12.192 | 7.941 |

**Padding volume is not a sufficient latency model.** For the 128-expert warm
workload in FP16, sorted-ID `medium4` used 4096 padded rows and took 21.49 ms;
`mixed` used only 3072 rows but took 24.24 ms (same split path, held medians).
A practical selector should use measured profile costs and routing-load features.

Single-layer storage and device allocation. Constant sizes below are actual
`constants.bin` lengths including metadata. The single-profile control uses 16
cores, while the adaptive library uses 15 plus a one-core router; packing can
also depend on core count, so these ratios do not isolate profile-count cost.

| Experts | Precision | Single C32 constants GiB | 18-profile expert constants GiB | Router + experts device GiB | Expert activation increment MiB |
|---:|---|---:|---:|---:|---:|
| 32 | fp16 | 0.282 | 1.698 | 2.196 | 152.9 |
| 32 | mxfp6 | 0.111 | 0.341 | 0.758 | 68.2 |
| 128 | fp16 | 1.128 | 6.800 | 7.380 | 191.5 |
| 128 | mxfp6 | 0.442 | 1.354 | 1.823 | 79.8 |

The 128-expert FP16 expert package is **7,301,812,692 bytes**. The SDK size
field reports 3,006,845,420 bytes, exactly 2^32 fewer. Device allocation and
actual extracted segment lengths are therefore used for the memory conclusions.

Dependent layer chains, T=128 balanced workload, all groups width 4, two
resident capacity profiles and 16 cores. These are fused oracle controls.

| Layers | Precision | C32 ms | C128 ms | Constants GiB | Device GiB | Activation increment MiB |
|---:|---|---:|---:|---:|---:|---:|
| 1 | fp16 | 21.845 | 60.257 | 2.255 | 2.679 | 122.2 |
| 1 | mxfp6 | 15.180 | 21.819 | 0.885 | 1.223 | 32.0 |
| 2 | fp16 | 43.038 | 121.301 | 4.526 | 4.966 | 133.3 |
| 2 | mxfp6 | 29.731 | 43.569 | 1.770 | 2.131 | 48.9 |
| 4 | fp16 | 86.050 | 243.181 | 9.036 | 9.516 | 151.6 |
| 4 | mxfp6 | 59.331 | 87.587 | 3.541 | 3.959 | 80.0 |

This establishes a bounded single-card operating region: 128 resident experts,
18 joint profiles for one layer, and four dependent layers with a two-profile
library. More profiles, trained-model depth, KV state and other group shapes
require their own admission and correctness measurements. The current router
boundary also uses host copies of hidden states/routing weights; it does not
measure a device-buffer-sharing implementation.

With these two steps complete, the next measurements use four stationary
32-expert banks with independent per-card profile choices; see
[the multi-card probe](MULTICARD_JOINT.md).

<!-- MEASUREMENTS_END -->

## Runtime interface

One resident expert program combines runtime local expert IDs, different group
widths/counts, and different capacities. For each 64/128/256-token invocation,
the host selects one of six compiled layouts. IDs are ordinary tensor values;
the profile and layout tag **shapes** select ahead-of-time specializations.
Both tag tensors contain zeros. No compilation, activation or weight transfer
is performed by the host when changing a profile.

| Layout | Group widths | Capacity per expert in each group |
|---|---|---|
| `full4` | All 4 | T |
| `small4` | All 4 | 16 |
| `medium4` | All 4 | 32 |
| `small8` | All 8 | 16 |
| `medium8` | All 8 | 32 |
| `mixed` | 4, 4, 8, then 16 for the remaining experts | T, T, T/4, then 16 |

Each bank has 18 profiles: three token lengths times six layouts. For T=256,
the fully conservative control uses capacity 256, which also covers a routing
load in which all tokens select the same expert. Capacity 128 would not cover
that case.

The first selector combined three shape comparisons with `Or`; this compiler
path rejected its `If` condition as non-constant. Direct comparisons of the
layout-tag shape compiled. A separate, unique profile-tag length distinguishes
all 18 specializations. This is a limitation of the tested folding path, not a
claim that every use of `Or` is unsupported. The previously established
single-output `If` restriction is respected by keeping counts outside branches.

A one-core router and 15-core expert program remain active together. The host
reads **current** router counts, sorts local IDs by descending count, and picks
the safe profile with the fewest padded expert rows. It then passes the current
dense global routing weights to the experts. This intentionally simple policy
demonstrates the interface; it is not a latency-optimal algorithm.

The host also sweeps every safe profile with identity and count-sorted IDs.
Fused controls use precomputed counts to remove the extra routing invocation.
Held and cycling schedules run in three rounds with reversed case ordering.
Every execution checks counts, capacity safety and repeatability. Independent
sparse FP32 references and conservative device controls check the saved outputs.
Different group widths can change accumulation order, so equivalence is tested
with a 0.5% relative-L2 threshold; absolute reference accuracy has a separate 5%
threshold. MXFP6 equivalence does not waive its reference-accuracy failure.

## Layer-scaling scope

The 1/2/4-layer controls use 128 experts per layer, independent synthetic weights,
T=128 and two profiles: all groups have capacity 32 or all have capacity 128.
All groups have width 4. There are independent expert-ID inputs for each layer,
but one common capacity profile for the whole chain; this does not measure a
Cartesian library of independent capacity choices for every layer.

The next layer depends on the previous MoE output through `x + 0.1*y` in the
last 1920 features. The first 128 routing features are preserved so that routing
loads remain controlled and counts are exactly verifiable. The returned tensor
is the last MoE contribution, rather than a residual-dominated output. These are
dependent synthetic MoE chains without attention or KV state. Their profile
selection is an oracle control; current-count dispatch inside a general fused
multi-layer model remains a separate integration question.

Memory snapshots distinguish pre-load, loaded and activated states. Loading can
allocate constants and program/runtime state; activation adds further memory.
The activation increment alone is **not** a measurement of all workspace.
The tables report total device allocation and the additional activation cost.
SDK `reqMem` is retained as total program-required memory, not labeled workspace.

`QAicQpcConstantsInfo_t.size` is a uint32 field and wraps for packages above
4 GiB. The storage audit extracts `constants.bin` and records its actual file
length. That segment includes constant metadata and static/dynamic sections;
it is not labeled pure packed-weight bytes. The QPC, descriptors, extraction
logs and measured sizes are retained; duplicate extracted binary payloads are
removed after measurement.

## Reproduction and artifacts

Use `/home/chihao/qeff-venv/bin/python` and `/opt/qti-aic` on this server.

```bash
python tools/moe_joint_scale.py export --out <bank-root> --experts 128
python tools/moe_joint_scale.py compile --out <bank-root> --precision fp16
python tools/moe_joint_scale.py run --out <bank-root> --precision fp16
python tools/moe_joint_scale.py analyze --out <bank-root> --precision fp16
python tools/moe_joint_scale.py audit --out <bank-root> --precision fp16
python tools/moe_layer_scale.py --source <128-bank-root> --out <layer-root>
python tools/moe_joint_scale.py compile --out <layer-root>/l4 --precision fp16 --variants fused16
python tools/moe_joint_scale.py run --out <layer-root>/l4 --precision fp16 --modes fused16 --iterations 5
python tools/moe_joint_scale.py analyze --out <layer-root>/l4 --precision fp16 --modes fused16
python tools/moe_joint_scale.py audit --out <layer-root>/l4 --precision fp16 --variants fused16
```

Repeat for MXFP6, the 32-expert bank, and layer counts 1 and 2. Export uses
temporary weight graph inputs that become parent-scope external initializers;
this avoids repeatedly serializing large parameter banks during Torch shape
inference. Export-only MatMul stubs emit ordinary ONNX MatMul nodes; numerical
references execute the independent sparse FP32 implementation.

Local artifacts are `joint_scale_e32/`, `joint_scale_e128/`, and
`joint_layer_scale/l{1,2,4}/`. Each contains source graphs, specialization lists,
input/reference tensors, compilation logs, QPCs, raw timing/resource CSVs,
per-case validation and latency in `<precision>/run/result.json`, and extracted
shape/storage audits in `<precision>/audit.json`. Generated binaries and result
files remain local under the repository's existing tracking policy.
