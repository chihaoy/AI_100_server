# AI 100 per-op profiling — Qwen3-8B on **4 cards** (tensor-parallel)

Per-operator hardware-cycle profiling of Qwen3-8B run **tensor-parallel across 4 AI 100 cards
(4 × 16 = 64 cores)**, for both inference regimes, using the SDK built-in commands — plus the
**card-to-card data flow** (AllGather/AllReduce-style combine).

- **Regimes:** `prefill` (`seq_len=128`) and `decode` (`seq_len=1`), both `ctx_len=256`.
- **Stats are 100% SDK built-in:** `qaic-compile -stats-level=70 -mdp-load-partition-config` → `qaic-runner -D 0:1:2:3 --aic-profiling-type=raw_device_stats` → `qaic-opstats --summary --trace --merge-mq-traces true`.
- **Cross-card flow:** `--flow-events full` + `perop_crosscard_flow.py` decode the P2P transfers between cards.

> **Companion docs**
> - [CUTTING.md](CUTTING.md) — how to choose *which op runs on which card* (MDP partition configs,
>   tensor-parallel vs pipeline cuts, what each costs, the three constraints that reject a config).
> - [COMMUNICATION.md](COMMUNICATION.md) — what the cards actually send once a cut is chosen.

## How to run

**Default — both regimes, 4 cards** (this recompiles two ~11 GB MDP QPCs; long the first time):

```bash
bash /home/chihao/mllm/research/tools/perop_profile_qwen3_4card.sh
```

**Just one regime** (reuses the other's QPC/result):

```bash
REGIMES=decode  bash /home/chihao/mllm/research/tools/perop_profile_qwen3_4card.sh
REGIMES=prefill bash /home/chihao/mllm/research/tools/perop_profile_qwen3_4card.sh
```

**With cross-card data-flow arrows** (adds dependency flows to the merged trace):

```bash
FLOW=full bash /home/chihao/mllm/research/tools/perop_profile_qwen3_4card.sh
```

**Drive with your own prompt / add the per-projection rollup:**

```bash
PROMPT="what is the capital of china" ARCH=1 bash /home/chihao/mllm/research/tools/perop_profile_qwen3_4card.sh
```

### Knobs (env vars, defaults shown)

| Var | Default | Meaning |
|---|---|---|
| `REGIMES` | `prefill decode` | which phases to run |
| `FLOW` | `none` | `full` = add cross-op / cross-card dependency arrows to the trace |
| `DEVLIST` | `0:1:2:3` | the 4 cards (`qaic-runner -D`) |
| `CTX` / `PREFILL_SEQ` | `256` / `128` | shape — **forces a recompile** when changed |
| `SAMPLES` | `4` | profiling samples per regime |
| `CORES` | `16` | cores **per card** |
| `NUMITER` | `100` | total inferences the runner executes |
| `PROMPT` / `TOKEN_IDS` | *(dummy token)* | your input (values don't change cycle counts) |
| `ARCH` | *(off)* | `ARCH=1` adds the per-projection rollup of the SDK trace |
| `MDP` | `qwen3_8b_4card/mdp_ts_4.json` | 4×16-core partition config |
| `ONNX` / `CUSTOM_IO` | `qwen3_8b_repacked/…` | repacked model for multi-card |

## Files needed (inputs)

| What | Path (default) |
|---|---|
| SDK binaries | `/opt/qti-aic/exec/{qaic-compile,qaic-runner,qaic-opstats}` |
| **MDP partition config** | `/home/chihao/models/qwen3_8b_4card/mdp_ts_4.json` (4 devices × 16 cores, p2p) |
| **Repacked ONNX** (multi-card) | `/home/chihao/models/qwen3_8b_repacked/Qwen3ForCausalLM.onnx` (+ external weights, ~30 GB) |
| custom-IO | `/home/chihao/models/qwen3_8b_repacked/custom_io.yaml` |
| scripts | `tools/perop_profile_qwen3_4card.sh`, `perop_make_input.py`, `perop_crosscard_flow.py`, `perop_by_projection.py` |
| hardware | **4 free AI 100 cards** (QID 0–3 Ready) |

## What it generates

```
perop4/
├── prefill/                                (seq_len=128, ctx=256, 4 cards)
│   ├── out/*slice00…03*.qaic-opstats.summary.txt   ← per-DEVICE per-op tables (slice0N = card N)
│   ├── out/*-merged*.qaic-opstats.trace.json       ← all 4 cards merged into one Perfetto trace
│   ├── op_level_prefill.summary.txt                ← copy of one slice summary
│   └── crosscard_flow_prefill.txt                  ← card→card transfer table (if extracted)
└── decode/                                 (seq_len=1)  — same layout
```

- Per regime: `SAMPLES × 4` summaries (one per device slice × sample) + `SAMPLES` merged traces.
- Each `*_slice0N_*.summary.txt` is **card N's** per-op table; it also has a **"PCI DMA Op Kind Summary"** for the inter-card traffic.

## Which op ran on which card
- **By file:** `slice00`→card 0, `slice01`→card 1, `slice02`→card 2, `slice03`→card 3.
- **In the merged trace:** thread names are `QAicGraph_slice0N_Core_M_<HVX|HMX|DMAIssue>` — so every op sits under its exact **card → core → engine** in Perfetto (<https://ui.perfetto.dev>).

## Cross-card data flow (AllGather / AllReduce)

Extract the card-to-card transfers from a merged trace:

```bash
python3 /home/chihao/mllm/research/tools/perop_crosscard_flow.py \
  /home/chihao/mllm/perop4/decode/out/*inf-1*-merged*.qaic-opstats.trace.json
```

This reads the `aicmulticastvtcm_out/_in` events (`PortDescription:"src->dst"`, `PortType:P2P`) and prints:
- the **card→card transfer matrix** (all-to-all: every card ↔ every card),
- **which ops trigger a cross-card send** (`o_proj`, `down_proj`/`Add_1` → AllReduce; `Transpose_2` → AllGather),
- transfers per layer.

### Two levels of tensor parallelism

| | Within a card (16 cores) | Across cards (4 cards) |
|---|---|---|
| Op | `aicmulticastvtcm` | `aicmulticastvtcm_out` / `_in` |
| Path | on-chip NOC → VTCM (`opMemory:TCM`) | P2P fabric (`PortType:P2P`) |
| Frequency (decode) | ~57,000 (frequent, cheap) | ~2,600 (rare, at TP seams) |

Reduction is hierarchical: the 16 cores combine **on-chip first**, then the 4 cards combine over
**P2P** — row-parallel ops (`o_proj`, `down_proj`) produce partial sums that are AllReduced across cards.

## Notes
- **All 4 cards must be free** (`qaic-util -q` → QID 0–3 Ready).
- Single-spec QPCs are compiled per regime (the combined multi-spec QPC can't be `qaic-runner`-profiled).
- `--merge-mq-traces` requires an explicit value on this SDK: **`--merge-mq-traces true`** (already handled).
- The heavy `out/`, `stats/`, `qpc/`, `*.bin`, `*.trace.json` are git-ignored — regenerate with the script.
- 1-card equivalent: `tools/perop_profile_qwen3.sh` (see `perop/README.md`).
