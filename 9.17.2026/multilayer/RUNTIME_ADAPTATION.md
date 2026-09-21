# Runtime adaptation of MoE padding and expert groups

Design scope, 2026-09-21: retain all grouping and placement options. Hardware and
compiler measurements should constrain the algorithm; fixed expert membership is
an experimental baseline, not a requirement. The first [capacity-switching probe](../runtime_constraints/README.md)
now validates four profiles at fixed T=128 on four-card and two-card programs,
using synthetic weights. The [resident-selection probe](../runtime_constraints/RESIDENT_SELECTION.md)
also validates runtime expert regrouping on one card, including Qwen-sized
matrices and the MXFP6 weight path. The [group-width sweep](../runtime_constraints/GROUP_WIDTH.md)
tests widths 1–32 at equal padded work; width 4 is fastest in that case, and
group width does not directly determine core participation. Other candidate freedoms remain open.
The [combined-adaptation probe](../runtime_constraints/COMBINED_ADAPTATION.md)
now validates eight groups of four with simultaneous membership/capacity changes
and stateless replay. It also measures substantial capacity-dependent constant
storage growth in some profile libraries.

Updated experiment order: complete joint-control integration and cost/scaling
measurements (steps 1 and 2 in the [single-card roadmap](../runtime_constraints/SINGLE_CARD_ROADMAP.md))
on card 0, then start multi-card experiments. Other single-card topics are
deferred and do not block this transition. Both card layouts below remain
candidates; this ordering does not remove any grouping or placement option.
The [current-count probe](../runtime_constraints/CURRENT_COUNT_DISPATCH.md) now
validates a one-core router and 15-core expert program kept active together on
card 0. The host chooses current-layer IDs/capacity after routing, with exact
same-precision fused outputs. Net split cost is 0.30–0.61 ms across the tested
precisions/workloads; both tested ProgramGroup protocols are substantially slower.
The [topology-switching probe](../runtime_constraints/TOPOLOGY_SWITCHING.md)
now validates widths 1–32 selected through shape-resolved `If` branches in one
resident program, with runtime expert IDs and exact same-width controls. Only
the selected branch executes. Six layouts cost about 4.8× the packed constants
of width 4 alone, with a measurable recurring latency cost. Input-value-dependent
conditions and two-output branches are rejected by the tested compiler path.

Steps 1 and 2 are now complete in the [joint/scaling probe](../runtime_constraints/JOINT_SCALING.md):
current-count selection combines runtime IDs, mixed group widths and capacities,
with 18 profiles at T=64/128/256, banks of 32/128 experts, and 1/2/4 dependent
synthetic-layer controls. All 44,400 timed mechanism checks passed. The four-layer
two-profile program occupied 9.516 GiB in FP16 and 3.959 GiB in MXFP6. Fewer
padded rows did not always give lower latency. The new
[four-card probe](../runtime_constraints/MULTICARD_JOINT.md) uses independent
card-local programs as a baseline; the single four-card and two-pair QPC layouts
remain candidates, as do replication and ownership migration.

## Independent decisions

For layer `l`, chunk `t`, and expert `e`, record the actual routed token count
`n[l,t,e]` before any capacity truncation. The algorithm may choose:

1. Padding capacities for each group/stage.
2. Expert membership and the number/size of groups.
3. Where expert weights reside, including replicas.
4. Where computation runs and how token activations reach the selected expert.
5. When to update these choices: per request, chunk, layer, or after routing.

Logical regrouping does not necessarily require inter-card weight migration.
Local weight selection, alternative compiled schedules, or activation routing
are separate possibilities. Each still needs an executable implementation and a
measured cost on this SDK.

## Evidence and limits

Each row describes that experiment's scope; later probes resolve some earlier
limits. The latest joint/scaling and four-card rows describe this round's work.

| Observation | What it establishes | Limits of that experiment |
|---|---|---|
| [Two-layer experiment](README.md): different capacities inside one four-card QPC; exact uniform/tuned outputs on one saved input | Static layer-specific capacities work | Runtime capacity switching, prompt generalization, full-model chunking |
| [Capacity switching probe](../runtime_constraints/README.md): four fixed-T profiles, exact outputs across profiles on four cards and either card pair | Finite capacity selection between invocations; compact unique shape tag; limited constant duplication | Actual Qwen-sized weights, many layers, more profiles, KV continuity, regrouping |
| [Resident selection](../runtime_constraints/RESIDENT_SELECTION.md): runtime IDs permute a resident 32-expert bank between two fixed-capacity groups; same-precision outputs match fixed layouts exactly | Local membership changes without reloading the QPC; FP16 and MXFP6 compile and execute at H=2048/I=768 | Combined capacity/membership selection, different group counts, trained-model accuracy, multi-card placement and current-router decisions |
| [Group width](../runtime_constraints/GROUP_WIDTH.md): widths 1–32 at capacity 32, same bank and padded work | Every tested topology executes with runtime IDs and MXFP6; width 4 has lowest measured latency; a single expert can use 16 HMX cores | Runtime topology switching, other capacities/workloads, combined capacity and membership selection, multi-card placement |
| [Combined adaptation](../runtime_constraints/COMBINED_ADAPTATION.md): five eight-group capacity vectors × three ID permutations, one resident program | Exact same-ID outputs across profiles; stateless overflow rejection/replay; quantified switching and storage costs in FP16/MXFP6 | Current-router decisions, KV-safe replay, many-layer/profile scale, physical ownership across cards |
| [Current-count dispatch](../runtime_constraints/CURRENT_COUNT_DISPATCH.md): split router/expert programs, four varying inputs, host policy uses current router outputs | Coactive 1+15 cores, exact outputs, no overflow; measured observation and ProgramGroup costs; core allocation can change constant storage | Trained weights, alternative handoffs, full-bank/many-layer scale and KV-safe chunking |
| [Topology switching](../runtime_constraints/TOPOLOGY_SWITCHING.md): finite width library selected by tag shape, with runtime IDs; fixed T=128/C=32 | All six widths switch in one resident program; only selected expert kernels execute; same-width outputs exact; MXFP6 path preserved | Joint width/capacity/current-count integration, useful workloads, library/bank/layer scaling |
| [Joint/scaling probe](../runtime_constraints/JOINT_SCALING.md): IDs, widths/counts and capacities selected together from current counts; 32/128 experts, T=64/128/256, 1/2/4 synthetic layers | Joint control works in one resident expert program; full-bank and bounded layer scaling fit card 0; measured constants, device allocation and latency | Trained models/KV, depth beyond four layers, independent per-layer profile libraries, alternative handoffs and cross-card ownership |
| [Four independent card programs](../runtime_constraints/MULTICARD_JOINT.md): stationary 32-expert owners, current global routing, independent/common/conservative profiles, parallel/serial controls | Per-card padding can differ within a chunk; 8,640 timed mechanism checks pass; balanced T=128 parallel latency is 7.928 ms FP16 and 6.480 ms MXFP6 | Changing ownership or replica selection, per-card completion traces, pair/MDP layouts, alternative communication paths and model integration |
| Earlier [local specialization notes](</home/chihao/mllm/research/9.17.2026/结论总结/一个program能否换图_探索.md>) report sequence-length 128/16 switching in one program; [host code](../e2e/pg/spec_run.cpp) uses `qaicExecObjSetDataExt` | Shape-selected execution has a local precedent | That earlier experiment alone did not establish fixed-T capacity selection; see the new probe above |
| [SDK network specialization](https://quic.github.io/cloud-ai-sdk-pages/1.21/Getting-Started/SDK-Tools/network-compilation/qaic-compile/apps_sdk_network_specialization.html) selects compiled symbolic-shape variants using input shapes | Finite, ahead-of-time shape variants are an available mechanism | Arbitrary graph dispatch or changing internal integer constants by passing a profile ID |
| New minimal probes reject input-value-dependent ONNX `If` and branches with two outputs; shape-resolved one-output branches compile and execute | Finite shape-specialized topology selection is supported in the tested path; a runtime condition tensor is not | Other supported/custom mechanisms; state and multiple-output graph organization |
| Earlier local ProgramGroup tests report switching overhead consistent with weight transfer, and failure to add a two-card MDP QPC | Program switching is not established as a cheap substitute for specialization | Other configurations/APIs, residency arrangements and newer SDK behavior |
| Original multilayer exporter slices constant weights using fixed expert order and Python capacity values | Those multilayer QPCs do not implement runtime membership changes | Integrating the separately demonstrated resident-selection mechanism into the full model |

The old conclusion that specialization cannot choose different shapes *within*
one invocation does not rule out selecting a whole capacity profile *between*
invocations. Conversely, sequence-length specialization alone does not prove
that internal MoE capacities can vary at a fixed token count.

## Candidate mechanisms to retain

| Candidate | Algorithm freedom | Main implementation question / cost |
|---|---|---|
| Fixed groups, shape-specialized capacities | Change each group's padding through a finite set of profiles | Make capacity a shape-dependent quantity used by packing and GEMMs; confirm compiler support and placement |
| Finite library of expert layouts | Choose among representative group memberships and capacities | How to select layouts without expensive program swaps or duplicated weights; shape specialization alone does not automatically rewrite constant expert order |
| Runtime regrouping among experts resident on one card | Change membership without transferring weights between cards | Dynamic gather/indexing, temporary weight traffic, core placement and batched GEMM efficiency |
| Route activations to stationary experts | Change work distribution while retaining weight ownership | Runtime dispatch and return path, communication and synchronization; owner-card load remains a constraint without replicas |
| Selective expert replication | Execute a replicated expert on different cards/pairs | Memory/workspace budget, dispatch mechanism and replica selection |
| Migrate expert weights | Change physical ownership across cards/pairs | Transfer/repacking cost, destination space, synchronization and binding the new weights into executable code |
| Small token tiles / overflow work | Change how many token blocks an expert processes | Efficient repeated dispatch or a custom kernel; a statically unrolled masked maximum still pays for all scheduled GEMMs |

Runtime-selected weights may lose optimizations available to constant weights.
In particular, the documented `-mxfp6-matmul` path compresses constant MatMul
weights. The resident-selection probe demonstrates that this SDK preserves
compression through the tested constant-bank/runtime-Gather pattern: packed
constants shrink, expert dequantization kernels execute, and same-precision
outputs match the static controls. This does not establish support for arbitrary
runtime weight tensors or graph transformations.
See the [compiler options](https://quic.github.io/cloud-ai-sdk-pages/1.21/Getting-Started/SDK-Tools/network-compilation/qaic-compile/index.html).

## Keep both card layouts

**One QPC across cards 0-3:** provides the demonstrated static multi-layer path.
A specialization would select an entire per-layer capacity vector before a
chunk starts. Independent per-layer choices can produce a combinatorial number
of whole-model variants, so measure how well a small profile library covers
traces. A single QPC does not by itself provide runtime expert dispatch.

**Separate QPCs on cards 0-1 and 2-3:** retain as the explicit host-orchestration
candidate. Each pair could select a capacity profile if specialization works for
that graph. The existing split-expert implementation must merge partial outputs
before the next layer. Program residency across layers and switching costs remain
separate issues. Changing membership across pairs needs a migration, replication,
or activation-routing mechanism; two QPCs alone do not supply it.

Both layouts should first be tested with the same routing traces and correctness
contract. Compare complete chunk latency, including merges, switching and copies.

## Observation timing and exactness

- Counts emitted after chunk `t` can predict choices for chunk `t+1`. They cannot
  determine the current chunk's earlier layer choices retroactively.
- Using exact current-layer counts requires an execution boundary after routing,
  or a supported device mechanism that dispatches the expert work. Later-layer
  routing depends on earlier-layer outputs, so all layer counts cannot generally
  be collected in advance with one independent router pass.
- Collect counts before truncation. Check assignment totals against valid token
  count times top-k, and validate the router counters against selected expert IDs.
- A predicted capacity can overflow. Preserve exact routing semantics with either
  correct overflow processing before dependent work, or discard/replay using a
  conservative profile. Never treat dropping assignments as a valid speedup.
- If overflow is detected only at the end of a full-model chunk, downstream
  results and their counts may already be invalid; do not use them to train the
  predictor. Replayed results must supply the accepted trace.
- Chunked prefill needs position inputs, causal masking and KV state. Replay must
  restore or safely overwrite affected state. The current two-layer experiment
  is not yet a stateful chunked-prefill implementation.

## Cost model for algorithm design

Minimize measured end-to-end time, including:

`compute + packing/regrouping + selection/switching + communication + observation + recovery`

This is an accounting model; overlapping components must be measured on the
critical path rather than blindly added. Also constrain memory, compiled-profile
count, compilation time and numerical correctness.
For combined capacity/ID adaptation, profile count alone does not predict
storage: three 16/32-capacity profiles nearly share one bank, while adding
128-capacity and the tested mixed-capacity profile substantially increases
packed constants. The effect differs between FP16 and MXFP6; audit each library.

A first-order padding proxy is `sum_g |g| * C_g` per layer, but fewer padded rows
do not guarantee lower latency: splitting groups can alter core mapping,
serialization, reductions and weight traffic.
The equal-work group-width sweep demonstrates this directly: with 1,024 padded
expert rows, FP16 medians span 5.287–8.306 ms and MXFP6 3.571–7.225 ms. Use
measured cost tables for candidate topologies; do not infer core use from
expert count or choose the smallest possible group unconditionally.

For scale, one Qwen3-30B-A3B expert has `3 * 2048 * 768` matrix parameters:
9 MiB at FP16, or nominally 3.375 MiB at six bits before format overhead. A single
FP16 hidden vector is 4 KiB; sending an input and returning one output is 8 KiB
per expert-token assignment before protocol overhead. These byte counts motivate
testing activation routing alongside migration; they do not predict measured
latency or establish that either dispatch path is supported.

Migration/replica updates should use a measured break-even rule: predicted
cumulative future latency savings must exceed update cost. Compare per-chunk
updates with less frequent updates and hysteresis to avoid repeatedly reversing
decisions when expert counts fluctuate.

## Experiments that determine the algorithm's constraints

Capacity specialization has passed the initial synthetic probe linked above;
resident selection has also passed its local mechanism and same-precision
equivalence checks. MXFP6 absolute reference accuracy is reported separately.
The remaining scale, combination, placement and state questions still require measurement.

| Probe | Keep fixed / vary | Required evidence | Algorithm implication |
|---|---|---|---|
| Activation traces | Conservative execution; vary requests and chunk positions with correct KV state | Per-layer expert counts, temporal persistence, group maxima, oracle padding benefit | Whether prediction and regrouping have enough potential value |
| Capacity specialization | Fixed 128-token input and weights; vary capacities | Distinct smaller GEMMs, correct outputs, low switch cost, memory use; test four-card and two-card layouts | Feasible profile granularity and selection frequency |
| Resident expert selection | Same local weight bank; fixed vs runtime-selected expert IDs, first FP16 then MXFP6 | Correct token/expert association, compiler support, placement, gather cost and GEMM throughput | Whether arbitrary local regrouping is practical |
| Group count/width | Same bank, capacity and total padded work; vary sequential group width | Initial sweep complete: numerical equivalence, latency, core participation and quantized path; expand to other capacities/workloads | No one-expert-per-core requirement; topology needs a measured cost model |
| Combined capacity/membership adaptation | Fixed width 4; vary compiled capacity vector and runtime IDs | Initial local probe complete: exact profile equivalence, switch timing, storage subsets and stateless replay; stateful/multi-card scale remains open | The freedoms compose, but profile memory and recovery cost must enter the policy |
| Alternative layouts | Same weights and workload; two representative compiled layouts | Selection mechanism, duplication/residency and switch overhead | Feasibility of a finite membership library |
| Communication and placement | Same selected experts; stationary ownership, replicas and migration | Supported executable data path, full latency, bytes and memory budget | Whether and how often ownership/replica choices should change |
| Current-count dispatch / overflow | Vary routed counts around capacity boundaries | Exact outputs, state-safe recovery and dispatch overhead | Predictive chunk adaptation versus reactive per-layer scheduling |

Use chronological held-out evaluation. Future counts may define an oracle upper
bound but must not leak into an online predictor. Report counter collection and
overflow/replay costs in adaptive latency. Retain the conservative baseline and
the separate absolute-accuracy and padding-equivalence checks.
