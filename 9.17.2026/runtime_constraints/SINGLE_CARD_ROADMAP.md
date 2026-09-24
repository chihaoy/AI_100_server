# Single-card exploration before new multi-card work

**Steps 1 and 2 completed on 2026-09-21.** The [joint/scaling probe](JOINT_SCALING.md)
passed 44,400 timed mechanism checks over 32/128 experts, 64/128/256 tokens,
uniform/mixed layouts and 1/2/4 dependent synthetic layers. The four-layer
two-profile setup occupied 9.516 GiB in FP16 and 3.959 GiB in MXFP6. MXFP6's
approximately 6% reference error remains separate from mechanism equivalence.
The [new multi-card probe](MULTICARD_JOINT.md) followed these measurements and
passed 8,640 timed mechanism checks with independent per-card padding choices.

Updated user direction on 2026-09-21: complete **steps 1 and 2 below**, then move
to multi-card experiments. These two steps use **card 0**. Trained-weight
accuracy, alternate overflow/observation implementations, KV-safe chunking and
chronological application traces remain future work; they are no longer gates
before starting multi-card experiments. This supersedes the earlier ordering
that put all single-card questions before multi-card work.

1. Combine expert IDs, group topology (including mixed widths) and per-group
   capacities, including selection from current router counts.
2. Measure chunk-length/capacity/routing-load costs; scale the resident bank from
   32 to 128 experts and then measure multiple layers within card memory. Include
   compiled constants, workspace/device allocation, and complete host latency.

The initial bounded matrix is token lengths 64/128/256, resident banks 32/128,
uniform and mixed-width layouts, and balanced/hot routing. Layer scaling uses
1/2/4 dependent synthetic MoE layers at a fixed representative token length.
Unsupported configurations and measured memory limits are valid outcomes.
After these measurements, start multi-card placement and orchestration tests
without waiting for the deferred single-card topics.

“Finished” here means that each bounded question below has measured support and
costs, or a reproducible limitation. It does not mean an exhaustive search over
every possible MoE algorithm or compiler transformation.

| Single-card question | Evidence so far | Remaining experiment |
|---|---|---|
| Can resident expert membership change? | [Resident selection](RESIDENT_SELECTION.md) and [joint controls](JOINT_SCALING.md): runtime permutations at 32/128 resident experts; FP16/MXFP6 agree with conservative controls | Integrate trained weights and realistic routing distributions |
| How should groups use cores? | [Group width](GROUP_WIDTH.md), [topology switching](TOPOLOGY_SWITCHING.md) and [joint scaling](JOINT_SCALING.md): widths 1–32 and joint uniform/mixed profiles execute; group width does not directly determine core participation | Cost tables on trained-model inputs and additional profile candidates |
| Can membership and padding change together? | [Joint controls](JOINT_SCALING.md): IDs, uniform/mixed widths and capacities in one 18-profile expert program, selected from current counts | Extend profile candidates only when measured costs justify their memory |
| Can current-layer counts select the current execution? | [Current-count dispatch](CURRENT_COUNT_DISPATCH.md): coactive 1-core router + 15-core experts; exact outputs; 0.30–0.61 ms net boundary cost across tested precisions/workloads; both ProgramGroup protocols cost more | Sparse routing IO, device-buffer handoff and CPU-router alternatives; integrate realistic layers |
| Can overflow be processed exactly? | Stateless reject/replay works in one resident program | Compare token tiling/overflow work with replay; establish state-safe behavior when KV state and dependent layers are present |
| Does the mechanism survive realistic scale and chunking? | [Controlled scaling](JOINT_SCALING.md): full 128-expert bank, T=64/128/256 and 1/2/4 dependent synthetic layers, with memory and latency measured | Trained weights, full-model depth, chronological routing traces and KV continuity |

For each experiment retain a conservative control, check every accepted expert
assignment, report absolute numerical accuracy separately from mechanism
equivalence, and include setup/memory costs alongside steady-state latency.
MXFP6's existing approximately 6% synthetic reference error remains an open
accuracy issue; equivalent padding variants do not resolve it.

After steps 1 and 2 are measured, carry the supported mechanisms and their cost
tables into the [multi-card design space](../multilayer/RUNTIME_ADAPTATION.md).
