# 本目录数据的出处
> **2026-09-22:** `/home/chihao/models/qwen3_30b_a3b/{ep,4card}` 与三个 `qeff*_cache` 已删除以腾磁盘(共 518 GB)。原始权重 `hf/` 保留;重建脚本、spec、custom_io 和日志见 [`build_recipes/`](../build_recipes/README.md)。

**没有跑 benchmark。** 输入是**单个 token `9707`**。

| | |
|---|---|
| 模型 | Qwen3-30B-A3B (128E / top-8 / 48L),MXFP6 |
| 并行 | 4 卡 TP,`mdp_ts_4.json`(单 partition,4 卡 x 16 核,p2p) |
| 规格 | `seq_len=1, ctx_len=256`(**decode**) |
| 输入 | **token id 9707**,单个;KV 缓存留给设备驻留 |
| QPC | `/home/chihao/models/qwen3_30b_a3b/4card/qpc_decode/`(25 GB,编译 3m51s) |
| 采样 | `--aic-profiling-num-samples 2 --num-iter 30` |

## 复现

```bash
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH
Q=/home/chihao/models/qwen3_30b_a3b/4card/qpc_decode
O=/home/chihao/mllm/perop_moe/decode_realtok

python3 tools/perop_make_input.py --qpc $Q --token-ids 9707 --layers 0 --seqlen 1 --ctxlen 256
qaic-runner -t $Q -D 0:1:2:3 --aic-profiling-type raw_device_stats \
    --aic-profiling-num-samples 2 --num-iter 30 --aic-profiling-out-dir $O/stats
qaic-opstats --qpc $Q/programqpc.bin --input-dir $O/stats --output-dir $O/out \
    --summary --trace --merge-mq-traces true --flow-events none
python3 tools/perop_percore.py $(ls $O/out/*inf-1*-merged*.trace.json|head -1) --csv $O/percore_realtok.csv
python3 tools/perop_percard_table.py $O/percore_realtok.csv --metric events  > $O/per_card_op_events.txt
python3 tools/perop_percard_table.py $O/percore_realtok.csv --metric pcycles > $O/per_card_op_pcycles.txt
```

## 为什么卡上不跑 benchmark

Phase 1 实测:**cycle 数与输入无关**。

| 对照 | 差异 |
|---|---|
| 随机输入 vs 真实 token 9707 | MoE 各 op <= 0.64% |
| 5 个 benchmark token x 交错重复 3 轮 | 排序在轮次间不稳定 -> 无可检测效应 |

原因是 lowering 索引式 + 形状固定:`Gather -> [T*8, 2048, 768]`,TopK 选谁都不改变形状。
**卡上跑 GSM8K 拿不到额外信息。**

benchmark(GSM8K / HumanEval 各 100 题)是在 **CPU** 上跑的,只采 router 的 top-8 索引,
产物在 `../routing/`,见 `../README.md` 的 Phase 2 一节。

## 本目录文件

| 文件 | 内容 |
|---|---|
| `per_card_op_events.txt` | 53 个模型 op x 4 卡的**事件数**(静态量,证明放置用这个) |
| `per_card_op_pcycles.txt` | 同上,**pcycles**(含 +-5% 计时抖动,不要拿来判不均衡) |
| `percore_realtok.csv` | 最细粒度:`(op-site, card, core)` x `events, pcycles`,8602 行 |
| `out/*.summary.txt` | SDK `qaic-opstats --summary` 原样输出,每卡一份 |
| `out/*-merged*.trace.json` | merged Perfetto trace(343 MB,git 忽略) |
| `stats/*.bin` | 原始 cycle 计数缓冲(git 忽略) |
