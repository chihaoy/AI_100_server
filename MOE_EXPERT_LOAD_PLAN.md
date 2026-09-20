# MoE Expert 负载观测计划 — Qwen3-30B-A3B on 4× AI 100

目标:在 AI 100 上跑 MoE 模型,**先观察、再提方法**。搞清楚 (1) compiler 把 expert 分到了哪张卡/哪个核,
(2) 不同 workload(数学 vs coding)下 expert 激活分布是否不同,(3) 这个静态划分在真实 routing 下会不会
产生 load imbalance / 计算浪费。

> **方法论原则**(来自 2026-08-28 讨论):不能上来就提方法。先跑真实 benchmark、先找 bottleneck,
> 有可观测的问题才有优化空间;如果 compiler 已经做得很好,那结论就是"在这个模型+benchmark 上没得做"。

---

## ⚠️ Phase 1 已完成 — 本文档下方多处前提已被实测推翻

2026-08-28 跑完 Phase 1,结果见 **[perop_moe/README.md](perop_moe/README.md)**。四条修正:

| 本文档下方的假设 | 实测 |
|---|---|
| expert 按 `e mod 16` 静态分到核 | ❌ 只在**非默认**的 chunked prefill 路径;默认导出无任何 expert→核 映射 |
| 能从 trace 做 per-expert 归因 | ❌ **结构上不可能** — 编译器 2943 个节点里 0 个带 expert 索引 |
| 四卡可能不均衡 | ❌ **实测卡间 cv = 0.226%,max/mean = 1.0035** |
| Phase 3(折算到卡/撞核) | ❌ 作废 —— 没有映射可折算 |

**真正观察到的问题换了位置**,见 [perop_moe/README.md](perop_moe/README.md) 的结果 3 和结果 5:
router 的 `TopK` 严重串行(cv 387%,有核为 0),attention 比 MoE 更不均衡(24% vs 7.5%),
以及权重按 (token,expert) 对取导致 prefill 在 `seq_len > 16` 时比稠密还贵。

---

## 核心矛盾(一句话说清这个课题在做什么)

MoE 里有三样东西是**静态**的,只有一样是**动态**的:

| 东西 | 什么时候定下来 | 推理时变不变 |
|---|---|---|
| expert 权重(128 × 3 个矩阵) | 训练时 | 不变 |
| router 权重(一个 `[128, 2048]` 矩阵) | 训练时 | 不变 |
| **expert → 核/卡 的放置** | **编译时,写死进 QPC** | **不变** |
| **router 的输出(选哪 8 个)** | **运行时,每个 token 现算** | **每一步都在变** |

router 的权重一辈子不变,但因为输入每个 token 都不同,它的**决定**每步都在变
(`y = Wx`,W 固定而 x 变)。**这是 MoE 全部动态性的唯一来源。**

于是:

```
编译器:在"什么都不变"的世界里,编译期把 expert 摆好位置,摆完写死在 QPC 里
router:在"每一步都变"的世界里,运行时每个 token 现挑 8 个
       ↑ 两边互不知情
```

**一个静态的放置,去承接一个动态的负载。**

而且导出成 ONNX 之后 "router" 这个名字就消失了 —— 在图里它只是一个普通的 `MatMul` 节点,
**编译器根本不知道这张图里存在 "routing" 这个概念**,自然也无从针对它优化。

在 AI 100 上这个错配比 GPU 更严重:GPU 是运行时 launch kernel 的,理论上还能在运行时重新调度;
**AI 100 的 QPC 是 AOT 编译死的,形状全静态,跑起来之后没有任何运行时调度器能对 routing 做出反应。**

> **编译的时候看不到数据,跑的时候改不了决定。** 这就是本课题要研究的那道缝。

---

## 执行进度总览

| Phase | 内容 | 状态 |
|---|---|---|
| 0. 环境与工具链侦察 | 卡容量、QEfficient MoE 支持、lowering 路径 | ✅ 完成(见下) |
| 1. 静态放置图 | expert → (card, core) 映射 + MDP 4 卡编译 | ✅ **完成 2026-08-28 — 见 [perop_moe/README.md](perop_moe/README.md)** |
| **1.5 权重搬运账** | **prefill 的 MoE 稀疏在这台硬件上是否白给** | **未开始(建议优先)** |
| 2. Routing 分布 | GSM8K / HumanEval 的 expert 激活直方图 | 未开始(**目的已变**,见下) |
| 3. 折算到卡 | 撞核分析 + capacity 浪费率 | ❌ **作废** — 无 expert→核 映射可折算 |
| 4. 敏感性验证 | `num_nsp` / `packed_chunk_size` 扫描 | 未开始 |

---

## Phase 0 实测结果(2026-08-28)

### 硬件

`qaic-util -q`,四张卡完全一致:

```
Dram Total : 31391744 KB   → 29.94 GiB / 卡     (4 卡合计 ~119.8 GiB)
Dram Free  : 31138383 KB   → 29.70 GiB / 卡     (当前全空)
Nsp Total  : 16      Vc: 16     Pc: 7
MCID Total : 3072    Semaphore: 32
Sku Type   : Pcie Ultra     NSP 595.20 MHz / DDR 2133.00 MHz
```

宿主机:**无 GPU**,根分区剩 1.1 T,内存 188 G。

### 容量结论

| 精度 | Qwen3-30B-A3B (~30.5 B) | 结论 |
|---|---|---|
| bf16 | ~61 GB | 4 卡(120 GB)宽松;2 卡装不下 |
| **MXFP6**(现有 pipeline 的格式) | **~25 GB** | **单卡即可装下**;4 卡时每卡仅 ~6 GB |

第三行是关键。现有流水线编出来的权重就是 MXFP6(`blockdequantize_mxfp6` 占每个 projection
55–70% 的 cycle,见 `perop/README.md`),所以 30B-A3B **单卡可跑**。三个后果:

1. **有 1 卡 baseline** —— 可照搬 `1-card-vs-4-card` 分支的对比方法。
2. **用 4 张卡是"选择"不是"必须"** —— expert 跨卡放置(EP)成为可探索的设计空间。
3. **容量完全不紧张 → compiler 默认多半仍是 TP**(每卡持有全部 128 expert 的 1/4),
   而不是 EP。**那样卡间天然均衡**,不均衡只发生在一张卡的 16 个核之间。

### 模型配置(已核实,2026-08-28 拉自 HuggingFace `config.json`)

Qwen3-30B-A3B 的关键字段,**不再是估计值**:

| 字段 | 值 | 含义 |
|---|---:|---|
| `num_experts` | **128** | 每层 128 个 expert |
| `num_experts_per_tok` | **8** | 每 token 选 8 个(top-k 固定,不随内容变) |
| `hidden_size` | 2048 | |
| `moe_intermediate_size` | **768** | 单个 expert 的 MLP 中间层宽度 |
| `intermediate_size` | **6144** | = 8 × 768,"若这层是 dense 该有多宽" |
| `num_hidden_layers` | 48 | |
| `num_attention_heads` / `num_key_value_heads` | 32 / 4 | GQA 8:1 |
| `decoder_sparse_step` / `mlp_only_layers` | **1 / []** | **48 层全部是 MoE,没有 dense 层** |
| `router_aux_loss_coef` | **0.001** | **训练时用了负载均衡辅助损失** |
| `norm_topk_prob` | true | top-8 权重重新归一化到和为 1 |
| `tie_word_embeddings` | false | embed 与 lm_head 各占 311 M |

推算(全部核对通过):

| | 总参数 | 每 token 激活 |
|---|---:|---:|
| MoE-MLP(48 层 × 128 expert) | 29.0 B | **1.81 B**(只走 8 个) |
| Attention(48 层) | 0.91 B | 0.91 B(**全额**) |
| Router(48 层) | 0.013 B | 0.013 B(全额) |
| Embedding + lm_head | 0.62 B | 0.62 B(全额) |
| **合计** | **30.5 B** | **3.35 B** |

对上 `30B-A3B` 这个名字。**注意稀疏只发生在第一行** —— attention / router / embedding 全额付费,
这也是为什么总账是 1/10 而不是 MLP 那层的 1/16。

两条对方案有直接影响的:

- **48 层全是 MoE** → Phase 2 要统计 `48 × 128 = 6144` 个 expert 实例,不是几百个。
- **`router_aux_loss_coef: 0.001`** → 模型训练时**被推着往均衡走**。这直接下调 H3 的预期:
  全局平均大概率比较平,别指望看到某个 expert 被用 10 倍。真正的机会在**瞬时**和**特定 workload**,
  见 Phase 2 / Phase 3a。

---

### 软件栈

- `qeff-venv`:QEfficient **1.22.0.dev0** + transformers 5.5.4 + torch(Python 3.10)
- 已支持的 MoE:`qwen3_moe`、`qwen3_5_moe`、`glm4_moe`、`gpt_oss`、`mixtral_moe`、`granitemoe`、`qwen3_vl_moe`
- HF cache 现有:Qwen3-1.7B、Qwen3-8B(均为 dense),MoE 需现下

### MoE 在这条工具链上的 lowering(源码读出,待 ONNX 复核)

`QEfficient/transformers/models/qwen3_moe/modeling_qwen3_moe.py` 里 **prefill 和 decode 是两条完全
不同的路径**:

| | Prefill(expert blocking) | Decode(gather-bmm) |
|---|---|---|
| 类 | `QEffPrefillChunkedQwen3MoeSparseMoeBlock` (:207) | `QEffQwen3MoeSparseMoeBlock` (:275) |
| expert→核 | `view(local_experts, num_nsp, …).transpose(0,1)` → **expert e → NSP core (e mod num_nsp)** | 无固定映射,按 router indices gather |
| `num_nsp` | `= num_cores`,默认 **16**(`modeling_auto.py:3722-3724`) | — |
| 容量 | `packed_chunk_size = 256`(`constants.py:20`),chunk 数导出时定死,多余行 `torch.where` mask 掉 | — |
| 数据依赖 | **无** —— 静态形状,付 worst-case | **有** —— `experts.gate_proj[idx]` 按 top-k gather 权重再 bmm |

**Phase 0 已经回答了导师的第一个问题:compiler 的 expert 划分是静态 round-robin,完全不看 workload。**

### ⚠️ 必须绕开的方法论陷阱

`perop/README.md` 已写明:per-op cycle count 由 **shape 和 op graph** 决定,**不由数值决定**。
所以直接换 benchmark 跑 `qaic-opstats`,四张卡的 cycle 数不会变 —— 不均衡**不会**以
"某张卡的 op 变慢"的形式出现。

不均衡的代价实际藏在两处:

- **prefill**:固定 capacity 下,静态算掉的行数 ≫ 实际路由过来的行数 → **padding 浪费**
- **decode**:top-k 个 expert 的 `e mod 16` 可能撞到同一个核 → 该核串行做多个 expert,其余核 idle
  → **关键路径变长**

不绕开这个陷阱,很容易跑出"四张卡很均衡、没得做"的假结论。

---

## 模型选择

| 候选 | 规格 | 用途 | 风险 |
|---|---|---|---|
| **Qwen3-30B-A3B** ✅ 首选 | 128 E / top-8,48 层(**config 已核实**) | 正式实验;规格与讨论中举的例子**完全一致**,汇报时不用解释为什么换规格 | ~61 GB 下载;CPU 上 router probe 很慢 |
| gpt-oss-20b | 32 E / top-4,MXFP4 | 备选,QEfficient 原生支持 expert blocking | 量化格式给 MDP 编译带来额外变量 |
| Qwen1.5-MoE-A2.7B | 60 E / top-4 | **先跑通工具链**:脚本、hook、统计、出图 | 架构较老,结论不能直接外推 |

**策略**:小模型调通全部脚本 → 换 Qwen3-30B-A3B 出正式数据。CPU-only 环境下大模型每次前向是
分钟级,调脚本的迭代成本承受不起。

---

## Phase 1 — 静态放置图(compiler 侧,不需要 benchmark)

**产物**:`expert → (card, core)` 映射表 + 它 workload-agnostic 的证据

1. `QEFFAutoModelForCausalLM.from_pretrained(...).export()` 出 ONNX,分别导
   `enable_chunking=True / False` 两版
2. 从 ONNX 的 initializer 形状和节点名,验证权重是否按 `[num_nsp, local_experts, H, I]` 重排
   → **证实/证伪 `expert e → core (e mod 16)`**
3. 仿照 `qwen3_8b_4card/mdp_ts_4.json` 写 MoE 的 4 卡 MDP config,`-stats-level=70` 编译
4. 跑 `tools/perop_profile_qwen3_4card.sh` → 每卡 per-op 表 + merged trace
5. 跑 `tools/perop_crosscard_flow.py` → 看 MoE 的 AllReduce 落在哪些 op
   (预期在 expert 输出求和之后,而不是 `down_proj`)

> **H1(可证伪)**:expert→core 映射是静态的,与输入无关。

**⭐ 优先做步骤 2–3 的一个子问题:默认切法是 TP 还是 EP。** 见"命门"一节。

---

## Phase 1.5 — 权重搬运账:prefill 的 MoE 稀疏是否白给(建议优先做)

**这一条不依赖 EP/TP 的答案,也不依赖 routing 是否不均衡 —— 它是 MoE + 权重瓶颈硬件的结构性后果。**

### 论证

"激活 3.35 B" 是一个 **per-token** 的概念,但**权重搬运是 per-batch 的**:只要 batch 里**有任何一个**
token 选中了 expert 57,那 4.72 MB 权重就必须从 DDR 搬进来解开。

| | 一次算几个 token | 被选中 expert 的并集 | 权重真省了吗 |
|---|---|---|---|
| **decode** | 1 | 8 / 128 | ✅ **真省,只搬 6.25%** |
| **prefill** | 128 | 128 token × top-8 = 1024 个槽位 → **几乎必然覆盖全部 128** | ❌ **一点没省,全搬** |

而 `perop/README.md` 已经测出这台硬件 **55–70% 的 cycle 花在 `blockdequantize_mxfp6`**
(搬权重 + 解权重)上 —— **瓶颈是搬运,不是算力**。

于是推论:

> **prefill 时,MoE 省下的是 FLOPs,而 FLOPs 不是瓶颈;真正的瓶颈——权重搬运——按 30 B 全额付。
> `A3B` 这个标称值在 prefill 阶段基本是个幻觉。**

换句话说,这笔"付内存换质量、计算量不变"的交易,在 GPU 上划算(算力贵、HBM 带宽高),
**在 AI 100 上正好踩在痛点上**。

### 怎么测(现有流水线直接就能出数)

1. 跑 `tools/perop_profile_qwen3_4card.sh`,prefill 与 decode 各一次
2. 从 `op_level_*.summary.txt` 取 `blockdequantize_mxfp6` 的 **cycle 数**和 **out KB**
3. 对比三组数:
   - MoE prefill vs MoE decode 的权重搬运量
   - MoE vs 现有 dense Qwen3-8B 的**每 token 搬运量**(已有基线,见 `perop/`)
   - 实测搬运量 vs "理论上只需搬 8/128" 的差距
4. 用 Phase 2 的真实 routing 算 prefill 的 **expert 覆盖率**:
   `|batch 内被选中的 expert 并集| / 128`,按 seq_len 扫描(16/32/64/128/256)
   —— 找出**覆盖率饱和的临界 batch 大小**

> **H4**:prefill 的 expert 覆盖率在 seq_len ≥ 某个不大的值时就饱和到 ~100%,
> 此后 MoE 相对 dense 不再节省任何权重搬运。

### 为什么建议排在 Phase 3 前面

- 不需要先回答 TP/EP 那个"命门"
- 不需要 routing 不均衡这个前提成立
- 用小 MoE 就能验证方法,不必等 30 B 下完
- 如果 H4 成立,量级是**倍数级**的,比负载不均衡那几十个百分点更硬

---

## Phase 2 — Routing 分布(workload 侧,纯 CPU)

**产物**:`(benchmark, phase, layer, expert) → count` 的 CSV

- **benchmark**:GSM8K test 取 50 题;coding 用 HumanEval(164 题取 50)或 MBPP
  - **不建议起步用 SWE-bench** —— 要 clone repo、跑测试,重得离谱,而我们根本不看正确率
- **跑法**:transformers CPU bf16,hook `Qwen3MoeTopKRouter.forward` 拿 `router_indices`;
  每题只需 prefill + 生成 ~128 token
- **必须分开统计 prefill 和 decode** —— 两条 lowering 完全不同,混在一起的直方图没有意义
- 不看正确率。我们做的是 efficiency,不是 accuracy

> **H2**:GSM8K 与 coding 的 expert 分布显著不同(JS divergence / χ² 检验)。
> **H3**:分布本身不均匀(Gini、max/mean、top-10 expert 的 token 占比)。

**⚠️ 对 H2/H3 的预期要校准。** config 里 `router_aux_loss_coef: 0.001` 证实训练时加了负载均衡
辅助损失 —— 模型**被明确训练成均衡的**。但那个损失只约束**训练集上的全局平均**,不约束:

| 不被约束的维度 | 对应的观测 |
|---|---|
| 单个 workload(GSM8K 的 token 分布 ≠ 训练集分布) | **H2 —— JS divergence 是本方案最关键的一个数** |
| 单个 batch / 单步 decode(8 个球扔 16 个桶,撞核是概率必然) | **Phase 3a** |
| 单层(通常深层比浅层更专门化) | 按 layer 拆开看,别只看 48 层聚合 |

所以:**H3(全局不均匀)大概率会被证伪,而这不代表方案失败。** 要找的问题不在"平均"里,
在"瞬时"和"特定 workload"里。

**H2 的判决意义**:若两个 benchmark 的热门 expert 显著不同,则说明
**不存在一个静态分配方案能同时对两种 workload 都最优** —— 问题是结构性的,不是调参能解决的,
这才是有研究价值的发现。反之若分布几乎一致,那就只是个"换个静态分配一劳永逸"的工程问题。

---

## Phase 3 — 折算到卡(Phase 1 × Phase 2,核心结论在这里)

用 H1 的映射把实测 expert 计数折算成 core / card 负载。

### 3a. decode 撞核分析(最有希望的点)

每个 decode step 的 top-8 expert 各自落到核 `e mod 16`。均匀随机时,**8 个球扔进 16 个桶,
最大桶期望约 2** → 关键路径已经是平均负载的 ~2×。真实 routing 有相关性(同层热门 expert
倾向共现),只会更糟。

指标:每 step 的 `max_core_load / mean_core_load`,跨 step 求分布;再折算到卡。

### 3b. prefill capacity 浪费率

```
waste = 1 −  Σ_e (实际路由到 e 的 token 数)  /  (num_experts × seq_len)
```

top-8/128 的理论下界就是 **93.75% 浪费**。**先用导出的 ONNX 节点数复核 chunk 循环是不是真的
覆盖全部 `seq_len`** —— 源码读起来是这样,若属实,这本身就是一个比负载不均衡大得多的发现。

---

## Phase 4 — 敏感性验证(替代 on-device 直测)

既然 cycle 与数值无关,无法通过换 benchmark 在设备上直接测出差异。改成:

- 对 `expert_blocking_num_nsp` 和 `packed_chunk_size` 做扫描,测"capacity 每降一档省多少 cycle"
- 再用 Phase 3 的真实 routing 折算成**可达收益**

---

## ⭐ 命门:默认是 TP 还是 EP

这决定整个问题的表述方式,**必须在投入其余工作前先回答**:

| 默认切法 | 每卡持有 | 卡间是否会不均衡 | 问题该表述成 |
|---|---|---|---|
| **TP**(切 expert 内部维度) | 全部 128 expert 的 1/4 | **否,天然均衡** | 核间撞核 + capacity 浪费 |
| **EP**(切 expert) | 32 个完整 expert | **是** | 卡间 load imbalance(讨论中设想的场景) |

Phase 0 的容量结论(每卡 30 GB,MXFP6 下模型只要 25 GB)强烈暗示 **compiler 没有理由切 EP**,
所以大概率是 TP。若如此,"两张卡 idle"不会出现,但**核间撞核和 capacity 浪费依然成立且可测**,
只是汇报口径要改。

---

## 待确认的未知数

| # | 未知数 | 怎么查 | 状态 |
|---|---|---|---|
| 0 | Qwen3-30B-A3B 的 config | `curl` HF `config.json` | ✅ 已核实,见 Phase 0 |
| 1 | 每卡 DDR 容量 | `qaic-util -q` | ✅ 29.94 GiB/卡,见 Phase 0 |
| 2 | QEfficient 1.22 的 MoE 是否支持 MDP 多卡编译 | 试编译小 MoE + 查 compile 参数 | 未查 |
| 3 | 4 卡下 expert 权重切 TP 还是 EP | Phase 1 步骤 2–3 | 未查(倾向 TP) |
| 4 | decode gather 的 expert 权重常驻 DDR 还是能进 VTCM | merged trace 的 `opMemory` 字段 | 未查 |

---

## 风险

- **第一堵墙很可能是编译期,不是显存。** 8B dense 编出来是 1852 个 compiler 节点
  (`perop4/CUTTING.md`);128 expert × 48 层的图大得多,`Vc:16 / Pc:7 / MCID:3072 /
  Semaphore:32` 这些**每卡静态资源**都可能先撞上。`CUTTING.md` 记的"三个会咬人的约束"没覆盖这些。
- 已知 `--aic-num-activations 2` 会因为 16 个 NSP 被网络全占而失败 —— MoE 上大概率同样。
- CPU-only 跑 30B-A3B 的 router probe 很慢,必须靠小样本(50 题)+ 短生成(~128 token)控制时间。
- MXFP6 量化本身会不会改变 routing?理论上会有微小偏移。若要严谨,Phase 2 的 routing 应
  **在 bf16 CPU 上采**,并注明与设备上 MXFP6 的 routing 可能有差异。

---

## 节奏与汇报口径

- **第 1 周**:小 MoE 打通 Phase 1 + **Phase 1.5**,回答"命门"(TP/EP)和其余未知数;同时写 Phase 2 脚本
- **第 2 周**:Qwen3-30B-A3B,跑出 GSM8K / HumanEval 两套 routing 数据
- **第 3 周**:Phase 3 出图,给结论

**Phase 1.5 排在第 1 周是有意的** —— 它不依赖 TP/EP 的答案,也不依赖 routing 不均衡,
是唯一一条"无论其他结论如何都能出数"的路线。万一命门查出来是 TP、routing 又很均衡,
Phase 1.5 仍能独立支撑一个结论。

最终结论应该是这个形式的一句话:

> 在 Qwen3-30B-A3B / GSM8K vs HumanEval 上,expert→core 是静态 round-robin;decode 每步 top-8
> 撞核使关键路径达到平均负载的 **Z×**;prefill 固定 capacity 使 **W%** 的 MoE 计算是 padding;
> prefill 的 expert 覆盖率在 seq_len ≥ **S** 时饱和,此后 MoE 相对 dense **不再节省权重搬运**;
> 两个 benchmark 的 expert 分布 JS 散度为 **D**。

—— 或者诚实地是"没观察到不均衡,因为 X"。那也是有效结论。

---

## 相关文件

| 文件 | 作用 |
|---|---|
| `perop/README.md` | 1 卡 per-op profiling 流水线(dense Qwen3-8B) |
| `perop4/README.md` | 4 卡 TP profiling + 跨卡数据流 |
| `perop4/CUTTING.md` | MDP partition config 怎么写、三个会咬人的约束 |
| `perop4/COMMUNICATION.md` | 卡内 NOC / 卡间 P2P 用的是哪些算子 |
| `tools/perop_profile_qwen3_4card.sh` | 4 卡 profiling 驱动脚本 |
| `tools/perop_crosscard_flow.py` | 从 merged trace 提卡间传输矩阵 |
| `tools/perop_make_mdp.py` | 生成 MDP partition config |
