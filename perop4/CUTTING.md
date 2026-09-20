# Cutting the graph — MDP partition configs for Qwen3-8B on AI 100

How to decide **which operator runs on which card**, why the choice matters, and the exact
workflow. Everything here was verified against the SDK on this machine (4 × AI 100 "Pcie Ultra").

Companion doc: [COMMUNICATION.md](COMMUNICATION.md) describes what the cards actually *send*
once a cut is chosen.

## What "the graph" is

The compiler turns the ONNX export into **1852 nodes** — not the 8906 nodes in the ONNX file,
because it folds away every `Constant` and shape op. Each of the 36 layers is ~50 nodes. Cutting
the graph means assigning each of those 1852 nodes to a card.

Custom ops fan out to sub-nodes: ONNX `/…/CtxScatter` becomes `/…/CtxScatter/n21`…`/n24`.
**This is why you cannot write the node list from raw ONNX names** — see the workflow below.

## The two cuts

Same 4 cards, opposite direction, wildly different cost.

### Vertical — tensor parallel (what `mdp_ts_4.json` produces today)

Every op is split across all 4 cards. For `Y = X · W`:

| | weight split | each card holds | comm |
|---|---|---|---|
| **column-parallel** `q/k/v_proj`, `gate/up_proj` | `W = [W₁\|W₂\|W₃\|W₄]` | a **slice of the final answer** | none |
| **row-parallel** `o_proj`, `down_proj` | `W = [W₁;W₂;W₃;W₄]`, `X` split too | a **partial sum of the whole answer** | **AllReduce** |

Transformers must alternate the two (column feeds row), so **every layer needs 2 AllReduce**.
That is not a compiler weakness — it is what tensor parallelism costs. It shows up in the trace
as `o_proj` and `Add_1` (post-`down_proj`) firing cross-card, while `q/k/v/gate/up_proj` never do.

```
        card0   card1   card2   card3
layer0  [1/4]   [1/4]   [1/4]   [1/4]     every matrix cut into 4 pieces
          └──── AllReduce ────┘           4 partial sums must be added
layer1  [1/4]   [1/4]   [1/4]   [1/4]
          └──── AllReduce ────┘
 ⋮      × 36 layers
```

Each card holds **¼ of every weight matrix**. For `o_proj` and `down_proj` the split is row-wise,
so each card produces a **partial sum of the whole answer** — the 4 pieces must be added together
before the layer can continue. That is **2 AllReduce per layer, 72 in total**, and all four cards
are working on **the same** inference throughout.

### Horizontal — pipeline (what `mdp_pp4_bal.json` produces)

Contiguous layer ranges are pinned per card. No matrix is ever split, so **no AllReduce at all** —
only the hidden state crosses at each seam.

```
card0 │ embed + RoPE + layers 0..8         ─→ hidden state ─→
      │        ✂ /model/layers.8/Add_1_output_0     [1, seq, 4096]
card1 │ layers 9..18                        ─→ hidden state ─→
      │        ✂ /model/layers.18/Add_1_output_0    [1, seq, 4096]
card2 │ layers 19..28                       ─→ hidden state ─→
      │        ✂ /model/layers.28/Add_1_output_0    [1, seq, 4096]
card3 │ layers 29..35 + norm + lm_head      ─→ logits
```

**Cut on the residual stream.** `Add_1` is the MLP residual add — the narrowest edge in a layer:

| edge inside a layer | width |
|---|---|
| `gate/up_proj` output | `[1, seq, 12288]` — 3× wider |
| attention scores | `[1, 32, seq, seq]` — wider still |
| **`Add_1` output (residual stream)** | **`[1, seq, 4096]`** ← narrowest |

Cutting anywhere else moves more bytes.

Each card holds **complete** layers, so nothing it produces is a partial sum — every card's output
is already correct and can move straight on. Only one card works on a given inference at a time.

### Side by side

| | vertical (tensor parallel) | horizontal (pipeline) |
|---|---|---|
| what is split | **each matrix**, 4 ways | **the layer stack**, 4 ways |
| each card holds | ¼ of every weight | **all** weights of 9 layers |
| a card's output is | a **partial sum** (row-parallel ops) | **already correct** |
| cards on one inference | **all 4** | **1** (other 3 wait) |
| parallelism is | **spatial** | **temporal** |
| crosses the PCIe link | partial sums, 2× per layer | hidden state, 3× per inference |
| bytes per prefill | **736 MB** | **3.19 MB** |
| wins on | **latency** (52.9 ms) | **throughput** (28.27 Inf/Sec, +88%) |

Tensor parallelism makes *one* inference finish fast by throwing 4 cards at it. Pipelining makes
*many* inferences finish fast by keeping each card busy on a different one. They optimise different
things — which is why neither dominates.

## What each cut costs

Measured (TP) and computed (pipeline, via `tools/perop_cut_cost.py`), per inference:

| | tensor-parallel `mdp_ts_4.json` | pipeline `mdp_pp4_bal.json` | |
|---|---|---|---|
| prefill (seq=128) | 736 MB | **3.19 MB** | 231× less |
| decode (seq=1) | 6.06 MB | **25.5 KB** | 243× less |
| prefill comm share of runtime | **38.9%** | — | |
| cards busy on one inference | **4** | **1** (others wait) | |
| latency | 52.9 ms | ~160 ms (4 stages serial) | |
| **throughput** | **15.04 Inf/Sec** | **28.27 Inf/Sec** | **+88%** |

Pipeline trades *spatial* parallelism for *temporal*: it only pays off when several inferences
are queued so the stages stay full. **Optimise for latency → tensor-parallel. For throughput →
pipeline** (subject to the runtime actually overlapping inferences across partitions).

**The KV cache does *not* stay put — this cost is not in the table above.** A partition config
assigns the 1852 *compute* nodes; the `past_key.N_RetainedState` / `past_value.N_RetainedState`
graph outputs are not among them, so the compiler places those buffers itself, and it does not put
them next to the layer that writes them. Compiling `mdp_pp4_bal.json` warns:

```
Pipeline partitioner: Node "/model/layers.0/self_attn/CtxScatter/n24" is producing a value that is
consumed by node "past_key.0_RetainedState." These nodes reside in non-adjacent partitions.
As a result, the value will be copied through intermediate partitions, which may have
performance implications.
```

Sizing the exposure at `ctx_len=256`: `past_key.N` is `[1, 8, 256, 128]` fp16 = 512 KB, so 1 MB per
layer for key+value and **36 MB if every layer's KV has to reach one partition** — 11× the 3.19 MB
of hidden-state traffic the cut was designed around. The compile log was truncated when captured,
so the exact number of affected layers is not yet known.

Measured throughput (below) still favours the pipeline cut by a wide margin, so whatever this costs
it does not negate the win — but it is the first thing to attack when tuning. `-split-model-io` and
the `splitRetainedStateIO` field in `QAicGraphApi.h` are the candidate levers; neither is tested yet.

Wart: the RoPE tables (`/model/Unsqueeze_2`, `/model/Unsqueeze_3`) are computed on card 0 and
consumed by every layer, so card 0 broadcasts 64 KB to each other card. A node can belong to only
one partition, so they cannot be replicated. 192 KB — 6% of the pipeline's traffic, not worth chasing.

## Measured: the pipeline cut is 88% faster on throughput

### What was compared

**Baseline — `mdp_ts_4.json`, the config in production today.** One partition, no `nodeList`, so the
compiler tensor-slices every one of the 1852 nodes across all 4 cards. `type: "p2p"`, 16 cores per
card (64 NSPs). Two AllReduce per layer → **736 MB cross-card per prefill, 38.9% of runtime**.
QPC 11 GB. (The 15.04 Inf/Sec below also matches the 15.124 recorded in `run_4card.log` from July,
so the machine is in the same state.)

**Result — `mdp_pp4_bal.json`.** Same 4 cards, same 16 cores each, same `type: "p2p"`, same
specialization, same runner command — *only the cut changed*. `nodeList` pins layer ranges
(card0 = 0–8, card1 = 9–18, card2 = 19–28, card3 = 29–35), so no matrix is ever split and there is
**zero AllReduce**; only the residual stream crosses, **3.19 MB per prefill**. QPC 7.25 GB.

Both built at `ctx_len=256, seq_len=128`; both run with `qaic-runner -D 0:1:2:3 --num-iter 200`
and no profiling (profiling adds overhead that would pollute the number). Two repeats each.

| | baseline `mdp_ts_4.json` | result `mdp_pp4_bal.json` | |
|---|---:|---:|---|
| **throughput** | **15.04 Inf/Sec** | **28.27 Inf/Sec** | **+88%** |
| repeats | 15.036 / 15.075 / 15.044 / 14.983 | 28.238 / 28.306 / 28.258 / 28.287 | < 0.6% spread |
| latency | 52.9 ms | ~160 ms | 3× worse |
| cross-card bytes / prefill | 736 MB | 3.19 MB | 231× less |
| QPC size | 11 GB | 7.25 GB | |
| best `--set-size` | any (flat) | 8–10 (cliff after) | |

### Which "baseline"? Two things could mean that, and only one was measured

| | what it is | measured |
|---|---|---|
| `mdp_ts_4.json` | 4-card tensor-parallel — **our override**, not the compiler's own choice | ✅ 15.04 Inf/Sec |
| compiler's true default | 2-card pipeline over **host DRAM** — what `-mdp-dump-partition-config` emits when left alone (see below) | ❌ never run |

So the defensible claim is: **the pipeline cut is 88% faster than the 4-card tensor-parallel config
we have been running.** "Faster than what the compiler picks on its own" is a different claim and
still needs a run — the config for it already exists as `qwen3_mdp_dump.json`.

### The runtime does overlap inferences across partitions

This was the one unknown the whole plan hung on, and the queue-depth sweep answers it outright:

| `--set-size` | pipeline Inf/Sec | tensor-parallel Inf/Sec |
|---|---:|---:|
| 1 | 6.25 | — |
| 8 | **28.24** | 15.04 |
| 10 | **28.26** | 15.01 |
| 12 | 24.99 | — |
| 16 | 24.96 | — |
| 32 | 24.45 | 14.71 |
| 64 | 12.52 | — |

- **S=1 → S=8 is a 4.5× jump.** With no queue the pipeline runs strictly serially (6.25 Inf/Sec =
  160 ms/inference, i.e. all four stages back to back). Queue it and the stages fill.
- **Tensor-parallel is flat** across the same range (15.04 → 14.71). It has no pipeline structure
  to fill, and it was never starved — the earlier `InputWaitTimeUs 91 ms` reading was the mean
  being dragged by a 4.6 s first-inference outlier, not a real feeding gap.
- **Steady-state throughput implies a 35.4 ms slowest stage** (1 / 28.27). The a-priori estimate
  from the tensor-parallel breakdown — total compute 2067 core-ms ÷ 4 stages ÷ 16 cores — was
  32.3 ms, so the model of where the time goes is sound; the ~3 ms gap plus the serial-latency gap
  (160 ms measured vs 129 ms predicted) is the unaccounted KV-cache traffic described above.

### Do not go deeper than S≈10

There is a cliff between S=10 (28.26) and S=12 (24.99) — a 12% drop that then plateaus, and
collapses to 12.52 at S=64, *below* tensor-parallel. Deeper queues do not help and eventually
hurt badly. The cause is not yet diagnosed.

### What this costs you

Throughput is up 88%; **single-inference latency is ~3× worse** (160 ms vs 52.9 ms). This is the
expected trade — the pipeline converts spatial parallelism into temporal. Pick the cut from the
metric that matters:

- **latency-bound / interactive** → tensor-parallel `mdp_ts_4.json`
- **throughput-bound / serving** → pipeline `mdp_pp4_bal.json` at `--set-size 8..10`

## Switching to pipeline: what changes operationally

Three things behave differently once the cut changes, and two of them will bite you.

**1. `--set-size` suddenly matters.** Tensor-parallel is flat across queue depth (15.04 at S=8,
14.71 at S=32) — there is no pipeline to fill. The pipeline cut swings from **6.25** (S=1) to
**28.26** (S=10) to **12.52** (S=64). Under-queue it and it is *worse than tensor-parallel*;
over-queue it and it collapses below tensor-parallel. Keep 8–10 inferences in flight. Any harness
that submits one inference at a time will make this cut look like a regression.

**2. The KV cache stops being free.** `past_key.N_RetainedState` is a graph output, not one of the
1852 nodes you assign, so the compiler places it — and not beside the layer that writes it. Up to
36 MB per inference of extra traffic, 11× the hidden state the cut was designed around. Tensor
parallelism has no equivalent problem. It is already costing measurable time: serial latency came
out at 160 ms against a 129 ms prediction. See the warning quoted above.

**3. The QPC shrinks**, 11 GB → 7.25 GB, since there is no per-card slicing of every weight.

Everything else — cards, cores per card, `connections.type`, the specialization, the runner
command — stays identical. The *only* difference between the two QPCs measured here is the
`nodeList` in the partition config.

## Do not assume the default is tensor-parallel

Left alone (`-mdp-dump-partition-config` with no load config), the compiler picks:

```
Partition0 → card 0 : embed + layers 0..17            (926 nodes)
Partition1 → card 1 : layers 17..35 + norm + lm_head  (926 nodes)
connections: [{ devices: [0,1], type: "host" }]
```

**Two cards, a pipeline cut, over host memory.** It splits into the fewest partitions that fit and
balances by exact node count (which straddles layer 17) — it is solving *fit*, not *speed*.

The 4-way tensor-parallel setup is **our** override: `mdp_ts_4.json` has no `nodeList`, which tells
the compiler "all 1852 nodes belong to these 4 cards, slice them yourself." Any "faster than the
compiler" claim has to say which baseline it means.

## The config format

The only structural difference is whether `nodeList` is present.

```jsonc
// vertical: no nodeList -> compiler tensor-slices everything across the 4 devices
{ "connections": [{ "devices": [0,1,2,3], "type": "p2p" }],
  "partitions": [ { "name": "Partition0",
      "devices": [{"deviceId":0,"numCores":16}, …, {"deviceId":3,"numCores":16}] } ] }

// horizontal: explicit nodeList per partition
{ "connections": [{ "devices": [0,1,2,3], "type": "p2p" }],
  "partitions": [
    { "name": "Partition0", "nodeList": ["/model/embed_tokens/Gather", …],
      "devices": [{"deviceId":0,"numCores":16}] },
    { "name": "Partition1", "nodeList": ["/model/layers.9/input_layernorm/CustomRMSNorm", …],
      "devices": [{"deviceId":1,"numCores":16}] }, … ] }
```

Hybrid works too — a partition may hold a `nodeList` **and** several devices, giving pipeline
stages that are themselves tensor-sliced (verified: emits `QAicGraph_P0_slice00/01`, `P1_slice00/01`).

### `connections.type` — `"p2p"` or `"host"`, nothing else

The compiler rejects anything else: `Invalid connection type "…". Expected either "host" or "p2p".`

All four cards hang off one PCIe switch:

```
                CPU / host DRAM
                      │  Gen4 x16 ≈ 32 GB/s   ← shared by all 4 cards
             ┌──[ PCIe switch 0000:02:00.0 ]──┐
          x8 │       x8 │       x8 │       x8 │     ≈ 16 GB/s each
           card0     card1     card2      card3     (30 GB DDR each)
```

- **`p2p`** — card → switch → card. One hop, never touches host RAM, and different card pairs
  traverse the switch concurrently.
- **`host`** — card → switch → **up the shared x16** → host DRAM → back down → switch → card.
  Two crossings of the one link every card shares.

For prefill's 736 MB: p2p ≈ 184 MB per card over its own x8 ≈ **11.5 ms** (measured 20.6 ms, 56% of
peak). Routing the same traffic through host DRAM pushes 1472 MB over the shared x16 ≈ **46 ms** —
roughly **4× worse**. On this topology the bottleneck is link count and sharing, not the medium;
per-KB costs are near-identical (p2p 2.3 vs DDR→VTCM 2.36 ucycles/KB).

## Workflow

### 1. Dump the compiler's namespace

You cannot hand-write the node list. Node names must be the compiler's, and the file lands ~12 s
into the compile — kill it once it appears.

```bash
qaic-compile -aic-hw -aic-hw-version=2.0 -m=$ONNX -convert-to-fp16 -mxfp6-matmul \
  -aic-num-cores=16 -mos=1 -aic-enable-depth-first \
  -network-specialization-config=spec.json -custom-IO-list-file=$CUSTOM_IO \
  -mdp-dump-partition-config=qwen3_mdp_dump.json -aic-binary-dir=/tmp/throwaway
```

Saved: `/home/chihao/models/qwen3_8b_4card/qwen3_mdp_dump.json`.

### 2. Re-cut

```bash
python3 tools/perop_make_mdp.py --dump qwen3_mdp_dump.json --onnx $ONNX \
        --mode pipeline --stages 4 --boundaries 9,19,29 --conn p2p -o mdp_pp4_bal.json
```

`--mode tensor|pipeline|hybrid`, `--tp`, `--conn p2p|host`, `--cores`, `--devices`.

**`--boundaries` matters.** Pipeline throughput is `1/max(stage_time)`, and equal layer counts do
not give equal times — the last stage also carries `lm_head/MatMul` (4096 × 151936 = 622 M params,
the largest single matrix in the model) and the first carries the embedding. `9,19,29` gives the
tail stage 7 layers instead of 9. It is a guess until per-stage times are measured.

### 3. Check what the cut costs

```bash
python3 tools/perop_cut_cost.py --mdp mdp_pp4_bal.json --onnx $ONNX --seq 128
```

Lists every tensor crossing a card boundary with its size, and the per-inference total.

### 4. Compile and profile with it

```bash
MDP=/home/chihao/models/qwen3_8b_4card/mdp_pp4_bal.json \
OUTROOT=/home/chihao/mllm/perop4_pp4 \
  bash tools/perop_profile_qwen3_4card.sh
```

## Three constraints that will bite you

| symptom | cause | fix |
|---|---|---|
| `Invalid node name: /model/layers.0/self_attn/CtxScatter` | raw ONNX names; the compiler folds constants (8906 → 1852) and expands custom ops to `/n21` | start from the dump (step 1) |
| `Consumer "/model/layers.0/Add" appears in partition before producer "/model/embed_tokens/Gather"` | nodes must be **topologically ordered within a partition**; the dump concatenates partitions, so its raw order is not globally topological | `perop_make_mdp.py` reorders against the ONNX graph, which is stored topologically |
| `Invalid connection type` | only `"host"` and `"p2p"` exist | — |

## Runtime limits worth knowing

- `--set-size` 10 → 32 and `--threads-per-queue` 4 → 8 change tensor-parallel throughput by
  **0.1%** (14.688 → 14.707 Inf/Sec). The TP baseline is saturated, not starved.
- `--aic-num-activations 2` **fails**: `NSPs: 32 — None of the devices can run network`. The
  network already claims all 16 NSPs per card, so multiple activations are impossible for this
  model at this core count. A pipeline cut therefore has to earn its throughput from the runtime
  overlapping inferences across partitions — there is no second lever.

## One-line summary

- **Cutting the graph** = assigning each of 1852 compiler nodes to a card, via `nodeList`.
- **Vertical (TP)** splits every matrix → 2 AllReduce per layer → 736 MB/prefill, all 4 cards on
  one inference. **15.04 Inf/Sec, 52.9 ms latency** — the latency winner.
- **Horizontal (pipeline)** splits by layer range on the residual stream → 3.19 MB/prefill, one
  card per inference. **28.27 Inf/Sec (+88%), ~160 ms latency** — the throughput winner, at
  `--set-size 8..10`.
- The runtime **does** overlap inferences across pipeline partitions (S=1 → S=8 is a 4.5× jump),
  which is what makes the pipeline cut pay.
- The op-placement knob is the **compile-time partition config**, not the `QAicGraph::addNode`
  graph-building API — that one genuinely has no device field.
