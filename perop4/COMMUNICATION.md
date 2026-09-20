# How the cores & cards communicate — Qwen3-8B tensor-parallel on AI 100

Qwen3-8B is sharded across **64 NSP cores = 4 cards × 16 cores**, so combining partial results
happens at **two levels**, using **different operators over different links**. This doc lists the
exact functions/operators used and what each does. All names/counts below are read from the
`qaic-opstats` merged trace of a 4-card decode run (`perop4/decode/out/*-merged*.trace.json`).

## The two communication levels

| | **Within a card** (16 cores) | **Across cards** (4 cards) |
|---|---|---|
| Link | on-chip **NOC** (Network-on-Chip) | **P2P** fabric (card-to-card) |
| Destination memory | **VTCM** (`opMemory: TCM`) | remote card (`PortType: P2P`) |
| Primary operator | **`aicmulticastvtcm`** | **`aicmulticastvtcm_out` / `_in`** |
| Trace marker | `opMemory:TCM`, *no* port fields | `PortType:P2P`, `PortDescription:"src->dst"`, `TransactionId` |
| Count (decode run) | **~57,000** (frequent, cheap) | **~2,600** (rare, at TP seams) |
| Speed | fast (on-chip) | slower (off-chip) |

Reduction is **hierarchical**: the 16 cores of a card combine on-chip first, then the 4 cards
combine over P2P — roughly 22× more on-chip transfers than cross-card ones.

---

## 1. Within a card — across the 16 cores (on-chip)

Each op sharded across a card's 16 cores shares/combines data over the on-chip NOC into VTCM.

| Operator | Count | What it does |
|---|---:|---|
| **`aicmulticastvtcm`** | 57,177 | broadcast/share a tensor across the 16 cores' VTCM (intra-card AllGather/AllReduce) |
| `aiccopysamevtcm` | 14,031 | copy within VTCM (local data movement) |
| `aicgather` / `aicscatternd` | ~130 / ~570 | gather/scatter (indexing, KV, layout) |
| `sync HVX` / `sync HMX` / `sync DMAIssue` | ~thousands | engines wait for each other before/after exchange |
| `barrier HVX` / `barrier HMX` / `barrier DMAIssue` | ~thousands | all threads/cores rendezvous at a boundary |

Example event (note: on-chip, no `PortType`):
```json
{"opKind":"aicmulticastvtcm", "opMemory":"TCM",
 "opName":"/model/layers.0/input_layernorm/CustomRMSNorm", "opPCycle":"956"}
```
`sync`/`barrier` are not data transfers — they are the **coordination** that makes the exchange
safe (producers finish before consumers read). On the 1-card summaries these dominate the
`ucycles` (~80%), i.e. cores spend most time *waiting to synchronize*, not computing.

---

## 2. Across the cards — 4-card P2P (off-chip)

When a shard's result must be combined across cards, each card sends its partial over the P2P
fabric and every card receives the others'.

| Operator | Count | What it does |
|---|---:|---|
| **`aicmulticastvtcm_out`** | 2,607 | this card **sends** its partial **out** to another card |
| **`aicmulticastvtcm_in`** | 2,604 | this card **receives** a partial **in** from another card |

Example event (note `PortType:P2P` and the direction):
```json
{"opKind":"aicmulticastvtcm_out", "PortType":"P2P", "PortDescription":"0->1",
 "TransactionId":"1", "opName":"/model/layers.0/self_attn/Transpose_2", "opOutputSize":"512"}
```
- **`PortDescription`** gives the exact direction: `"0->1"` = card 0 → card 1 (`_out`), `"0<-1"` = card 0 ← card 1 (`_in`).
- The transfers form a **full all-to-all mesh** — 12 directed edges (every card ↔ every other). Card 0 receives slightly more (it's the gather point for the final logits).

---

## Which model operators trigger a cross-card combine

The P2P exchange fires exactly where tensor-parallel math requires a global combine:

| Model op (`opName`) | TP role | Collective type |
|---|---|---|
| **`self_attn/o_proj`** | attention output — **row-parallel** (each card = partial sum) | **AllReduce** (sum) |
| **`Add_1`** (MLP residual ← **down_proj**) | MLP output — **row-parallel** | **AllReduce** (sum) |
| **`Transpose_2`** | attention heads sharded per card, redistributed | **AllGather** |
| `Add` (attn residual), `Mul` | finish the combine after the reduce | — |

By contrast, **column-parallel** ops — `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` —
need **no** cross-card comm: each card computes its own output-shard locally.

### The per-layer pattern
Each of the 36 layers does ~6 cross-card combine rounds (uniform), so cross-card comm is small in
count but each is an expensive P2P round-trip. The sequence per layer is roughly:

```
q/k/v_proj (local) → attention → Transpose_2 (AllGather heads) → o_proj (AllReduce)
                   → gate/up_proj (local) → SiLU → down_proj/Add_1 (AllReduce)
```

---

## How to observe it yourself

```bash
# card->card transfer matrix + which ops combine/reduce, from a merged trace:
python3 /home/chihao/mllm/research/tools/perop_crosscard_flow.py \
  /home/chihao/mllm/perop4/decode/out/*inf-1*-merged*.qaic-opstats.trace.json

# visual timeline with cross-card dependency arrows (compile trace with FLOW=full first):
#   open the *-merged*.trace.json in https://ui.perfetto.dev
#   threads are named  QAicGraph_slice0N_Core_M_<HVX|HMX|DMAIssue>  (slice0N = card N)
```

## One-line summary
- **Within a card:** `aicmulticastvtcm` over the on-chip NOC into VTCM (frequent, cheap), guarded by `sync`/`barrier`.
- **Across cards:** `aicmulticastvtcm_out`/`_in` over the P2P fabric (rare, expensive), in an all-to-all mesh.
- **Combine happens at** row-parallel `o_proj` and `down_proj` (**AllReduce**) and attention `Transpose_2` (**AllGather**); column-parallel projections need no comm.
