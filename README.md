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

**静态路径的阶段结论(9 月 21 日，运行时探针之前)**:当时采用的整模型路径是一个 QPC、一个分区、四卡;可用的自由度是每层的 expert 顺序和两轮各自的画布(128 / 32),估一层 8.2 ms;lane 分档估 6.5 ms(未测)。48 层单 QPC 端到端仍待验证；后续运行时实验见下文。

**新增两层验证(9 月 21 日)**:[不同层 padding 编进同一 QPC](9.17.2026/multilayer/README.md) 已在四卡实测:layer 0 用 64/32、layer 1 用 128/32,两层输出与 uniform 128/128 逐位相同、零溢出。FP16 主机延迟 25.54 → 21.31 ms(降 16.55%),MXFP6 22.53 → 20.05 ms(降 11.01%);MXFP6 两个变体均有 5.57% 的 fp32 参考误差,详见报告。仅一条已用过的 prompt,48 层和独立测试集尚未验证。

**运行时自适应的研究范围**:上述静态路径结论不排除其他运行时设计。[跨 prefill chunk 的设计空间](9.17.2026/multilayer/RUNTIME_ADAPTATION.md) 保留单四卡 QPC 与两对卡 QPC、动态容量、expert 重分组、activation 路由、权重复制和迁移，按实测支持程度与成本决定算法约束。[首个约束探针](9.17.2026/runtime_constraints/README.md) 已验证固定 128 token、单个常驻 QPC 的四种容量切换，支持四卡和任一两卡对。[第二个探针](9.17.2026/runtime_constraints/RESIDENT_SELECTION.md) 已验证单卡常驻 expert 的运行时 ID 重分组及 MXFP6 路径，含 Qwen 矩阵尺寸；目前仍为合成权重，完整模型及 KV 状态尚待验证。

[第三个约束探针](9.17.2026/runtime_constraints/GROUP_WIDTH.md) 固定总 padding 工作量扫描每组 1–32 expert，均可执行；本配置下每组 4 expert 相比 16 expert 降低 FP16/MXFP6 延迟约 20%/26%。设备 trace 显示单 expert 也可使用全部 16 个 HMX 核，分组宽度不能直接当作核数。该探针中不同组数为分别编译的 QPC；后续单程序切换见第六个探针。

[第四个探针](9.17.2026/runtime_constraints/COMBINED_ADAPTATION.md) 已在单常驻 QPC 中组合运行时 expert ID 与五种八组容量，验证逐次切换、无状态 overflow/replay 及 MXFP6 路径。profile 库的常量共享依赖容量组合与精度：16/32 的三个 profile 几乎无额外权重存储，加入 128 或 64 可明显增加空间。该探针尚未覆盖当前层路由计数的提前获取、多卡及 KV 状态；当前计数与多卡的后续结果见下文。

[第五个探针](9.17.2026/runtime_constraints/CURRENT_COUNT_DISPATCH.md) 已验证先读取当前层 router 输出，再选择 expert ID 和容量：单卡 1 核 router + 15 核 expert 可同时保持激活，FP16/MXFP6 输出与 fused 对照逐位相同。相对 15 核 fused 对照，净边界成本约 0.30–0.61 ms；两种 ProgramGroup 切换协议均更慢。仍是合成权重、部分 expert 的无状态探针，MXFP6 参考误差约 6% 尚未解决。

**探索顺序（2026-09-21 更新）**：已按[单卡探索清单](9.17.2026/runtime_constraints/SINGLE_CARD_ROADMAP.md)完成 card 0 上的联合控制（expert ID、组宽、容量、当前计数）及规模测量，并完成首轮多卡实验。真实权重、KV/chunk 连续性及应用路由 trace 后续继续探索；保留此前所有多卡设计候选。

[第六个探针](9.17.2026/runtime_constraints/TOPOLOGY_SWITCHING.md) 已在单常驻程序内切换每组 1、2、4、8、16、32 expert，并同时改变 ID。输入 shape 决定编译时可解析的 If 分支；设备只执行选中拓扑，FP16/MXFP6 均与对应固定组宽对照逐位相同。六种布局的 packed constants 约为单独 width 4 的 4.8 倍，并有持续执行开销；输入数值决定的 If 与双输出分支在本 SDK 路径被拒绝。

[联合控制及规模探针](9.17.2026/runtime_constraints/JOINT_SCALING.md) 已完成本轮单卡步骤 1、2：当前计数驱动 expert ID、均匀/混合组宽与容量，覆盖 32/128 expert、64/128/256 token 和 1/2/4 层依赖的合成 MoE，共 44,400 次计时事务通过机制检查。四层双 profile 占用设备内存 FP16 9.516 GiB、MXFP6 3.959 GiB；MXFP6 约 6% 参考误差仍未解决。

[四卡独立 profile 实验](9.17.2026/runtime_constraints/MULTICARD_JOINT.md) 已完成 8,640 次四卡计时事务及 4,560 次匹配的单卡对照。每卡固定驻留 32 expert，可在同一 chunk 独立选择 padding；T=128 warm 输入使用 card 0/3 的 C32 与 card 1/2 的 C16。T=128 balanced 的 FP16 并发主机延迟为 7.928 ms，四卡保守对照为 18.304 ms。跨卡 ownership 变化、复制与两对卡 QPC 仍待测；本轮结束时四卡资源均已释放。

**真实模型保守基线**：[Qwen3-30B-A3B 全 48 层 prefill](9.17.2026/real_model/README.md) 使用真实权重、四卡单 QPC、统一 C128，暂未接入自适应 padding。同一 GSM8K prompt 41 的前 128 token，FP16/MXFP6 各完成 60 次计时，主机中位延迟分别为 572.223/525.103 ms。两者均预测与 FP32 参考相同的下一 token，但 logits 相对 L2 误差为 5.029%/18.842%，均未通过 5% 诊断门槛；不是模型准确率或多 chunk/KV 连续性的验证。

**真实模型静态 oracle**：[逐层最小安全 padding 与热/冷 expert 重分组](9.17.2026/real_model/oracle_padding/README.md) 已在全 48 层、四卡单 QPC、同一 128-token 输入上完成。与重新编译的同配置 C128 对照相比，FP16 为 572.298 → 501.623 ms（降 12.35%），MXFP6 为 496.480 → 425.481 ms（降 14.30%）。每个候选均完成 60 次计时、零溢出，counts/logits 与重分组后的 C128 对照逐位相同；最小容量在两种精度下均优于向上取整到 16。该结果仅覆盖原生两组各 64 expert 的离线 oracle，未计入运行时选择、重排、权重搬运或加载成本。重分组后的 FP16/MXFP6 参考误差为 3.096%/17.201%，仅前者通过此输入上的 5% 诊断门槛。

**逐层 profiling 与 2 的幂 padding**：[48 层完整对照](9.17.2026/real_model/oracle_padding/layer_profile/results/README.md) 已完成 6 个相同 `stats-level=70` 配置、360 次计时和 18 份 trace。FP16 的平均 MoE 时间为 C128 11.5466 → 最小容量 10.0475 → 2 的幂 9.7871 ms；MXFP6 为 10.1564 → 8.5871 → 8.0593 ms。2 的幂容量对 C128 的 MoE 加速为 1.180×/1.260×，但仅在 24/48 与 40/48 层优于最小容量，应保留两类候选逐层调优。这些是完整模型中的插桩时间，不应与上面的无插桩绝对时间混用；混合容量策略尚未编译测量。

**编译器修复、MXFP6、全模型与 combine 重设计**：[`-mdts-mos=1` 系列实验（E1–E14）](9.17.2026/real_model/oracle_padding/mdts_flag/README.md) 发现默认编译把 MoE down projection 按输出列切到四卡（每层 121.6 MiB 跨卡交换），`-mdts-mos=1` 使其真正 expert-parallel（layer 2 重放 7.27 → 5.00 ms，输出逐位相同），此后 T=128 下 padding 与重分组不再影响时延，跨卡均衡成为主要杠杆；随后完成更大 prefill chunk 与分阶段容量（T=256/512）、整套重放的 MXFP6 版本、从 checkpoint 重建的全 48 层 prefill（MXFP6：497 → 352 ms 仅加 flag，→ 243 ms 移植层级重写，→ 232 ms 加 token-centric combine，logits 与历史逐位相同；flag 会破坏 KV retained state 配对，全模型 flag 版本为 prefill-only），以及去掉稠密累加器的 token-centric combine（MXFP6 重放 T=128/256/512 层时延 −10%/−27%/−31%，逐位相同）和 profiling 轮。QPC 与 trace 留在本地，脚本在 `mdts_flag/scripts/`。

## 复现入口

- 单层微基准:`tools/moe_tier_bench.sh <name> <groups> [num_devices]`,导出器 `tools/moe_tier_bench_export.py`
- 两 QPC 并行 + C++ 主机循环:`目前最好的sol.md` 第二节
- 48 层端到端:`9.17.2026/e2e/build_layer.sh <layer>`(导出 + 编译)、`9.17.2026/e2e/run_final.sh`、驱动 `tools/moe_e2e_host.cpp`
- 环境:`source /home/chihao/qeff-venv/bin/activate; export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH`;SDK 1.21.6,QEfficient 1.23;模型 `/home/chihao/models/qwen3_30b_a3b/hf`
