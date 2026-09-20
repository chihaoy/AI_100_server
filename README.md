# AI_100_server — Qwen3-30B-A3B MoE 在 4× Cloud AI 100 上的实验

> 本仓库只含脚本和文档。实验产物(QPC、ONNX、trace、日志、路由数据,约 232 GB)在服务器 `/home/chihao/mllm/research/` 本地,不入库。
> 脚本里的绝对路径(`/home/chihao/mllm/...`、`/home/chihao/models/...`、`/opt/qti-aic/...`)指向那台 AI 100 服务器。

AI 100 上 MoE expert 放置与 padding 研究的脚本和结论文档,从 mllm fork 的 `research/` 目录抽出。

## 目录

| 目录 | 内容 | 大小 |
|---|---|---:|
| [`9.17.2026/`](9.17.2026/) | **主实验区**(9 月 17 日起):单层 MoE 微基准的各变体、真实权重/输入实测、路一/路二、端到端。结论全在 [`9.17.2026/结论总结/`](9.17.2026/结论总结/) | 118 G |
| [`perop_moe/`](perop_moe/) | 更早的 MoE 观测:整模型 EP prefill/decode 的 per-op profiling、100/300 题路由采集(`routing/`)、tier_bench 单层分档基准(`tier_bench/`,含 README) | 77 G |
| [`perop/`](perop/)、[`perop4/`](perop4/)、[`perop4_pp4/`](perop4_pp4/) | Qwen3-8B 的 per-op profiling(单卡 / 4 卡 TP / 4 卡 PP),各有 README | 37 G |
| [`tools/`](tools/) | 全部脚本,见 [`tools/README.md`](tools/README.md) | |
| [`phone_perf/`](phone_perf/) | 更早的手机端(QNN/OpenCL/CPU)profiling 原始文件和 adb 脚本,与 AI 100 无关 | |
| [`MOE_EXPERT_LOAD_PLAN.md`](MOE_EXPERT_LOAD_PLAN.md) | 最初的观测计划(9 月上旬) | |

重型产物(QPC、ONNX、trace、npy、bin)由 `research/.gitignore` 排除,git 里只保留脚本、配置、日志、小结果。

## 结论文档阅读顺序(`9.17.2026/结论总结/`)

按问题链排列;每篇都是卡上实测,数字可追溯到对应目录。

**一、机制:AI 100 上 EP MoE 是怎么算的**
1. [`怎么EP的.md`](9.17.2026/结论总结/怎么EP的.md) — QEfficient EP 图的结构:64 lane × 2 轮,一核一个 expert
2. [`STAGE_BATCHING.md`](9.17.2026/结论总结/STAGE_BATCHING.md) — 一核两个 expert 是两轮串行不是 batched GEMM;两轮画布可以不同(128/32 = 8.2 ms vs 9.9)
3. [`一个core两个expert之间的计算顺序.txt`](9.17.2026/结论总结/一个core两个expert之间的计算顺序.txt)、[`CORE_SYNC.md`](9.17.2026/结论总结/CORE_SYNC.md)、[`核级放置能否手控.md`](9.17.2026/结论总结/核级放置能否手控.md) — 核间同步、lane→核号不可控
4. [`padding的限制.txt`](9.17.2026/结论总结/padding的限制.txt)、[`python或者compiler改padding size.md`](9.17.2026/结论总结/python或者compiler改padding%20size.md) — padding 在图里的位置和改法

**二、分档:不同 expert 不同 padding**
5. [`两组padding实验.md`](9.17.2026/结论总结/两组padding实验.md) — 两组画布同一分区会串行;分区放置
6. [`初始测一层layer_GSM8K.md`](9.17.2026/结论总结/初始测一层layer_GSM8K.md)、[`卡上采集layer0路由.md`](9.17.2026/结论总结/卡上采集layer0路由.md) — 100 题路由统计,每 expert 单题最大值
7. [`微基准输入缺陷.md`](9.17.2026/结论总结/微基准输入缺陷.md) — 9/19 前微基准输入退化(每 expert 恰好 1 token)的发现与修复
8. [`热冷分组32_16按卡对实测.md`](9.17.2026/结论总结/热冷分组32_16按卡对实测.md)、[`32加32与64的区别.md`](9.17.2026/结论总结/32加32与64的区别.md)、[`实验记录_同输入同层两种padding.md`](9.17.2026/结论总结/实验记录_同输入同层两种padding.md) — 热冷两组按卡对放,两分区串行
9. [`真实权重真实输入一层实测.md`](9.17.2026/结论总结/真实权重真实输入一层实测.md) — 真实 layer 0 权重 + GSM8K 真实输入:时间只由形状决定
10. [`不丢token的分档设计_64_16.md`](9.17.2026/结论总结/不丢token的分档设计_64_16.md) — 按单题最大值分 64/16,100 题零丢弃,8.87 ms
11. [`naive_vs_64_16_讨论整理_继续做这个.md`](9.17.2026/结论总结/naive_vs_64_16_讨论整理_继续做这个.md) — **阶段总结**:naive 9.77 vs 64/16 8.87,两分区为何串行,路一/路二计划

**三、并行:让两对卡同时算**
12. [`路二_两个QPC并行实测.md`](9.17.2026/结论总结/路二_两个QPC并行实测.md) — 两个 QPC 主机同时启动:并行可行,一层 5.00 ms,输出与 naive 相同;lane 归约切法的坑
13. [`路一_一个QPC内并行实测.md`](9.17.2026/结论总结/路一_一个QPC内并行实测.md) — 一个 QPC 内分区严格按列表顺序串行(人造图零边验证);一卡不能在两分区
14. [`解决往返主机办法.md`](9.17.2026/结论总结/解决往返主机办法.md) — 每层回主机的成本(Python 2.9 → C++ 0.3–0.6 ms)、QCCL 排除、为何每层必须交换一次
15. [`目前最好的sol.md`](9.17.2026/结论总结/目前最好的sol.md) — **方案整理**:数据切分、步骤、C++ 主机循环、数据、限制
16. [`端到端prefill_naive_vs_两QPC.md`](9.17.2026/结论总结/端到端prefill_naive_vs_两QPC.md) — **48 层端到端**:正确性与 naive 一样;但每层换 program 不可行,若能常驻 388 vs 520 ms;fp16 RMSNorm 溢出与画布余量两个坑
17. [`一个program能否换图_探索.md`](9.17.2026/结论总结/一个program能否换图_探索.md) — specialization / `If` / ProgramGroup / 多 program QPC 逐一验证;换 program 的真实成本是重搬权重(6.9 ms / 171 MB)

**当前结论(9 月 21 日)**:整模型只能是一个 QPC、一个分区、四卡;可用的自由度是每层的 expert 顺序和两轮各自的画布(128 / 32),估一层 8.2 ms;lane 分档估 6.5 ms(未测)。下一步是把 48 层按此编成一个 QPC 跑端到端。

## 复现入口

- 单层微基准:`tools/moe_tier_bench.sh <name> <groups> [num_devices]`,导出器 `tools/moe_tier_bench_export.py`
- 两 QPC 并行 + C++ 主机循环:`目前最好的sol.md` 第二节
- 48 层端到端:`9.17.2026/e2e/build_layer.sh <layer>`(导出 + 编译)、`9.17.2026/e2e/run_final.sh`、驱动 `tools/moe_e2e_host.cpp`
- 环境:`source /home/chihao/qeff-venv/bin/activate; export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH`;SDK 1.21.6,QEfficient 1.23;模型 `/home/chihao/models/qwen3_30b_a3b/hf`
