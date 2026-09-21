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
| `moe_qwen3_baseline.py`、`moe_qwen3_baseline_host.cpp` | 真实 48 层模型的四卡保守 prefill 基线；逐次 logits 检查、FP32 参考误差、60 次计时及设备内存/释放检查；见 `9.17.2026/real_model/README.md` |
| `moe_qwen3_oracle.py` | 原生 48 层图的静态逐层 padding 与 expert 重排、设备路由校准、容量溢出审计及 full-model oracle 对照；见 `9.17.2026/real_model/oracle_padding/README.md` |
| `moe_qwen3_profile.py`、`moe_qwen3_profile_sweep.py`、`moe_qwen3_profile_report.py` | 相同 profiling 配置下的 C128、最小容量、2 的幂容量对照；逐层 MoE 时间、SDK trace 校验和图表；见 `9.17.2026/real_model/oracle_padding/layer_profile/README.md` |
| `moe_qwen3_routing_profile.py` | 从已保存 trace 中隔离路由列置换 Gather、关联复制和周边向量运算；校验原始 trace、四卡覆盖及 MoE 边界，输出逐层耗时与图表 |
| `moe_multilayer_export.py` | 将不同 padding 的完整 decoder 层连接成一个 ONNX；同权重 uniform/tuned 对照及路由计数输出 |
| `moe_multilayer_bench.py`、`moe_single_qpc_host.cpp` | 单 QPC 编译、卡上溢出/数值检查、预热后的 C++ 主机计时；见 `9.17.2026/multilayer/README.md` |
| `moe_grant_access.sh` | 管理员授予指定用户设备及实验数据访问权限，备份原 ACL |

## 放置与机制探针
| 脚本 | 用途 |
|---|---|
| `moe_capacity_probe.py`、`moe_capacity_probe_host.cpp`、`moe_capacity_audit.py` | 固定 128 token，单 QPC 运行时切换两轮容量；两卡/四卡、逐次输出验证、溢出检测、形状/权重包审计及设备 trace；见 `9.17.2026/runtime_constraints/README.md` |
| `moe_resident_selection.py`、`moe_resident_selection_host.cpp`、`moe_resident_audit.py` | 单卡常驻 32 expert，用运行时 ID 在两个固定容量组之间重分组；FP16/MXFP6、Qwen 矩阵尺寸、静态分组对照及压缩/设备 trace 审计；见 `9.17.2026/runtime_constraints/RESIDENT_SELECTION.md` |
| `moe_group_width_probe.py` | 复用同一常驻 expert bank、输入与参考，等 padding 工作量扫描每组 1–32 expert；逐次验证、FP16/MXFP6 延迟、实际打包形状及 HMX 核参与度；见 `9.17.2026/runtime_constraints/GROUP_WIDTH.md` |
| `moe_adaptive_probe.py`、`moe_adaptive_host.cpp` | 单常驻 QPC 同时切换 expert ID 与八组容量；四种切换模式、无状态 overflow/replay、profile 子集常量存储和设备 trace 审计；见 `9.17.2026/runtime_constraints/COMBINED_ADAPTATION.md` |
| `moe_current_count_probe.py`、`moe_current_count_host.cpp` | 单卡先获取当前路由计数，再选择 expert ID 和容量；1+15 核双常驻、两种 ProgramGroup 协议、15/16 核 fused 对照、逐次正确性和内存审计；见 `9.17.2026/runtime_constraints/CURRENT_COUNT_DISPATCH.md` |
| `moe_topology_probe.py`、`moe_topology_host.cpp` | 单常驻 QPC 以 shape-resolved If 切换 1–32 expert 组宽，并改变运行时 ID；固定拓扑对照、编译限制、布局子集存储及逐分支设备 trace；见 `9.17.2026/runtime_constraints/TOPOLOGY_SWITCHING.md` |
| `moe_joint_scale.py`、`moe_joint_host.cpp`、`moe_joint_runtime.h` | 联合切换当前计数驱动的 expert ID、均匀/混合组宽及容量；64/128/256 token、32/128 expert、保守对照、编译常量和分阶段设备内存审计 |
| `moe_layer_scale.py` | 1/2/4 层依赖的合成 MoE 链；128 expert、两种容量 profile、独立层权重；保留路由特征以控制负载，未包含 attention/KV |
| `moe_multicard_joint.py`、`moe_multicard_joint_host.cpp` | 四卡各驻留 32 expert，使用当前全局 router 输出，比较各卡独立/共同 profile 与保守 profile；四卡并发及串行执行对照、主机 FP32 求和和单卡参考 |
| `moe_probe_placement.sh`、`moe_probe_lane2core.py`、`moe_probe_lane2device.py`、`moe_probe_tap2core.py` | lane → 卡 / 核 的放置探针 |
| `moe_proof_percard_ops.py`、`moe_sync_straggler.py`、`moe_ep_stage_timeline.py`、`moe_onnx_structure.py`、`moe_find_expert_bytes.py`、`moe_verify_expert_tensor.py` | 真模型 trace 解读、算子归属、权重定位 |
| `moe_ep_predict.py`、`moe_ep_batch_sweep.py`、`moe_blocksize_sweep.py`、`moe_decode_bmm_export.py`、`moe_tiny_ep_decode_proof.py` | EP 负载预测、decode 形态实验 |

## Qwen3-8B per-op profiling(perop 系列)
`perop_profile_qwen3.sh`、`perop_profile_qwen3_4card.sh`、`perop_generate.py`、`perop_make_input.py`、`perop_make_mdp.py`、`perop_percore.py`、`perop_percard_table.py`、`perop_by_projection.py`、`perop_crosscard_ops.py`、`perop_crosscard_flow.py`、`perop_cut_cost.py` — 见 `../perop/README.md`、`../perop4/README.md`

## 手机端(与 AI 100 无关)
`explain_qnn_profile.py`、`explain_qnn_lines.py`、`parse_qnn_profile.py`、`parse_opencl_profile.py`、`per_layer_decode_total.py` — 原始文件在 `../phone_perf/`
