# tools/ — 脚本索引

所有命令从 repo 根目录运行:`python3 tools/xxx.py`。脚本之间用 `sys.path.insert(0, dirname(__file__))` 互相导入,必须留在同一目录。

## 单层 MoE 微基准(tier_bench 系列)
| 脚本 | 用途 |
|---|---|
| `moe_tier_bench_export.py` | **核心导出器**:单层 EP MoE 图,支持分组画布 `--groups`、两轮宽度 `--stage-widths`、真实权重 `--real-weights --expert-order`、只导一组 lane `--lane-range`、复制 router `--dup-router`、不汇合 `--separate-outputs` |
| `moe_tier_bench.sh` | 导出 + 编译 + profiling 一条龙 |
| `moe_tier_bench_verify.sh` / `moe_tier_bench_verify.py` | fp16 编译,卡上输出 vs CPU 参考 |
| `moe_tier_bench_analyze.py` / `_phases.py` / `_nodespans.py` / `_coremap.py` | 从 qaic-opstats trace 读每卡 span、每核每节点时间、算子分类 |
| `moe_tier_bench_regen_x.py` | 给旧变体重生成非退化输入 |
| `moe_tier_permute.py`、`moe_tier_crossdomain.py` | expert 顺序置换、跨领域路由对比 |
| `moe_mdp_manual_partition.py` | 两分区 config(组 0 → 卡 0,1;组 1 → 卡 2,3) |
| `moe_mdp_partition3.py` | 三分区 / 零边分区 config(路一) |
| `moe_mdp_partition_spans.py` | 多分区 trace 里每分区每卡的启动/结束时刻 |

## 真实权重、真实输入
| 脚本 | 用途 |
|---|---|
| `moe_real_layer0_input.py` | GSM8K 某题经真实 embedding + layer 0 attention 得到 MoE 输入 x |
| `moe_layer0_router_oncard.py` | layer 0 router 在卡上跑 100 题,采每 expert 单题最大 token 数 |
| `moe_collect_routing.py` / `moe_collect_decode_routing.py` | CPU HF 模型采 48 层 prefill / decode 路由 |
| `moe_prefill_balance.py`、`moe_analyze_routing.py`、`moe_layer_expert_load.py`、`moe_top_experts_per_layer.py`、`moe_decode_step_similarity.py` | 路由统计分析 |

## 两 QPC 并行(路二)与主机循环
| 脚本 | 用途 |
|---|---|
| `moe_par2_wallclock.py` | Python qaicrt:热/冷 QPC 单独、串行、并行的主机 wall clock |
| `moe_par2_host.cpp` | **C++ C API 主机循环**:单线程异步 enqueue、堆 buffer 复用、F16C 相加;`--ref4=` 同口径量四卡 QPC |

## 48 层端到端(e2e)
| 脚本 | 用途 |
|---|---|
| `moe_e2e_cpu_ref.py` | CPU fp32 逐层参考,保存每层残差流、路由、logits |
| `moe_e2e_layer_export.py` | 一层 attention + 一组 MoE 的 2 卡图(用编译器 CustomRMSNorm) |
| `moe_e2e_verify_layers.py` | 每层 QPC 喂参考输入,热+冷相加 vs 参考 |
| `moe_e2e_host.cpp` | 48 层链式驱动(常驻 / lazy / swap 三种模式,naive 模式) |
| `moe_e2e_logits.py` | 最终 norm + lm_head,和 CPU 参考、naive 比 argmax / 相关 |

## 放置与机制探针
| 脚本 | 用途 |
|---|---|
| `moe_probe_placement.sh`、`moe_probe_lane2core.py`、`moe_probe_lane2device.py`、`moe_probe_tap2core.py` | lane → 卡 / 核 的放置探针 |
| `moe_proof_percard_ops.py`、`moe_sync_straggler.py`、`moe_ep_stage_timeline.py`、`moe_onnx_structure.py`、`moe_find_expert_bytes.py`、`moe_verify_expert_tensor.py` | 真模型 trace 解读、算子归属、权重定位 |
| `moe_ep_predict.py`、`moe_ep_batch_sweep.py`、`moe_blocksize_sweep.py`、`moe_decode_bmm_export.py`、`moe_tiny_ep_decode_proof.py` | EP 负载预测、decode 形态实验 |

## Qwen3-8B per-op profiling(perop 系列)
`perop_profile_qwen3.sh`、`perop_profile_qwen3_4card.sh`、`perop_generate.py`、`perop_make_input.py`、`perop_make_mdp.py`、`perop_percore.py`、`perop_percard_table.py`、`perop_by_projection.py`、`perop_crosscard_ops.py`、`perop_crosscard_flow.py`、`perop_cut_cost.py` — 见 `../perop/README.md`、`../perop4/README.md`

## 手机端(与 AI 100 无关)
`explain_qnn_profile.py`、`explain_qnn_lines.py`、`parse_qnn_profile.py`、`parse_opencl_profile.py`、`per_layer_decode_total.py` — 原始文件在 `../phone_perf/`
