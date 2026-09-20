# Phase 1 结果 — Qwen3-30B-A3B (MoE) 在 4× AI 100 上的 expert 放置

**日期**:2026-08-28  **结论**:默认配置下 **不存在 expert→卡/核 的静态映射**,四卡负载差异 **0.35%**。

计划见 [../MOE_EXPERT_LOAD_PLAN.md](../MOE_EXPERT_LOAD_PLAN.md)。

## 怎么跑出来的

```
1. 下载    Qwen/Qwen3-30B-A3B -> /home/chihao/models/qwen3_30b_a3b/hf   (57 GB, 7.6 min)
2. 导出    export.py -> export-126f04c76d4a0568/Qwen3MoeForCausalLM.onnx (57 GB, 5.8 min)
             ⚠ 必须 torch_dtype=float16 + 释放 __qeff_init__ 的死张量,否则 OOM(见下)
3. Pass A  qaic-compile -mdp-dump-partition-config  -> moe_mdp_dump.json   (编译器自己的默认划分)
4. Pass B  qaic-compile -mdp-load-partition-config=mdp_ts_4.json -stats-level=70
             -> qpc_decode/ (25 GB, 3m51s)
5. Pass C  qaic-runner -D 0:1:2:3 --aic-profiling-type raw_device_stats
             qaic-opstats --summary --trace --merge-mq-traces true
6. 分析    tools/perop_percore.py <merged.trace.json>          ← 本次新写
```

脚本在 `/home/chihao/models/qwen3_30b_a3b/4card/pass_{a,b,c}_*.sh`。

## 结果 1 — 卡间负载:完全均衡

decode(seq_len=1, ctx=256),4 卡 × 16 核 TP,单次 inference 的 merged trace:

| | 总 pcycles | 占均值 |
|---|---:|---:|
| card0 | 75,933,606 | 99.99% |
| card1 | 75,732,728 | 99.72% |
| card2 | 76,209,443 | 100.35% |
| card3 | 75,893,782 | 99.94% |

**卡间 cv = 0.226%,max/mean = 1.0035。**

实测吞吐:**14.3 ms/token,70 tok/s**(`ExecTimeUs` 四卡分别 14276/14285/14276/14304 µs)。

## 结果 2 — expert 的计算与取数在 64 个核上是均匀的

| op | 作用 | 事件数 cv | **cycle cv** | max/mean |
|---|---|---:|---:|---:|
| `/mlp/MatMul` | expert bmm (gate) | 0.00% | **0.95%** | 1.022 |
| `/mlp/MatMul_1` | expert bmm (up) | 0.00% | **1.19%** | 1.029 |
| `/mlp/MatMul_2` | expert bmm (down) | 0.00% | **2.98%** | 1.068 |
| `/mlp/Gather_3` | 取 gate_proj 权重 | 0.69% | 4.75% | 1.093 |
| `/mlp/Gather_4` | 取 up_proj 权重 | 0.00% | 8.02% | 1.337 |
| `/mlp/Gather_5` | 取 down_proj_t 权重 | 0.00% | 8.57% | 1.151 |

全部 5889 个 op-site 都出现在全部 **64 个 (card, core)** 上。

> ⚠ **诚实说明**:这个均匀性是**结构性保证**,不是测出来的偶然。因为 lowering 是索引式、形状固定的
> (见结果 4),cycle 数与输入无关。测量是**确认**了它,而不是发现了它。

## 结果 3 — 真正不均匀的是 router 和 attention,不是 expert

| op | cycle cv | max/mean | 占总 cycles |
|---|---:|---:|---:|
| `/mlp/gate/TopK` | **387%** | **16.1×**(有核为 0) | ~0.3% |
| `/mlp/gate/MatMul`(router) | **119%** | 5.8× | ~0.9% |
| `/mlp/Einsum`(top-8 求和) | 107% | 3.4× | ~0.5% |
| `/self_attn/MatMul`(注意力分数) | 53% | — | 2.1% |
| **`/self_attn/` 全部** | **24.0%** | 1.50 | — |
| **`/mlp/` 全部(MoE)** | **7.5%** | 1.24 | — |

两个反直觉的点:

1. **attention 比 MoE 更不均衡**(24% vs 7.5%)。32 个 Q 头 / 4 个 KV 头切到 64 个核上除不尽。
2. **routing 决策本身是严重串行的** —— `TopK` 在部分核上 cycles 为 **0**,最忙的核是均值的 16 倍。
   它在关键路径上(Gather 依赖它),但目前只占 ~1.7% cycles。

## 结果 4 — 为什么"expert→卡/核 映射"不存在

默认导出走 `QEffQwen3MoeSparseMoeBlock`(gather-bmm),每层的 MoE 只有:

```
router : gate/MatMul -> Softmax -> TopK -> Einsum -> Div
gather : Gather_3/4/5  <- experts.{gate_proj,up_proj,down_proj_t}  [128, 2048, 768]
compute: MatMul, MatMul_1 -> Sigmoid*Mul -> Mul_2 -> MatMul_2
combine: Mul_3 -> Einsum
```

- **每层只有 3 个 MatMul,不管选中哪 8 个 expert**
- **128 个 expert 融成一个 3D 张量**,靠 TopK 索引 Gather
- 编译器 dump 的 2943 个节点里,**带 `experts.<数字>` 的有 0 个**

expert 身份只作为**张量数据**存在,从不作为**节点名**存在 → 从 trace 做 per-expert 归因**在结构上不可能**。

### 两种切法都不产生 EP

| 切法 | expert 在哪 | 会不会某些卡 idle |
|---|---|---|
| 编译器默认(`-mdp-dump`:2 卡 pipeline,layers 0-23 / 23-47) | 第 N 层的全部 128 个 expert 在拥有该层的卡上 | ❌ |
| QEfficient 默认(`mdp_ts_4.json`:单 partition 4 卡 TP) | 每个 expert 切 4 份,每卡持 1/4 | ❌ |

`generate_mdp_partition_config` 只发单 partition + 全部设备,**没有 `nodeList`**;`nodeList` 在整个
QEfficient 里不存在。**EP 不是默认,也无法用 QEfficient 表达。**

## 结果 5 — 权重按 (token, expert) 对取,不是按 expert 取

ONNX shape inference 实测:

```
gate/TopK_output_1  [T, 8]                ← 选中的 expert 索引
Reshape_1_output_0  [T*8]
Gather_3_output_0   [T*8, 2048, 768]      ← ⚠ 每个 (token,expert) 对各取一份完整权重
Reshape_2_output_0  [T*8, 1, 2048]
```

同一个 expert 被 100 个 token 选中,它的权重就被取 100 遍。单层 expert 权重搬运量
(fp16,一个 expert 的三个矩阵 = 9.0 MiB):

| seq_len | Gather 量 | 对比"128 个 expert 全载一遍"(1.125 GiB) |
|---:|---:|---|
| 1 (decode) | 72 MiB | **6.25%** ✅ 真省 16× |
| **16** | 1152 MiB | **1.00×** ← 收支平衡点 = `num_experts / top_k` |
| 32 | 2.25 GiB | 2× |
| 128 | 9.0 GiB | **8×,比稠密算完 128 个 expert 还贵** |

**这张图只适合 decode。** prefill 需要 `prefill_only=True, enable_chunking=True` 的 chunked lowering
(那条路径才有 `expert e -> NSP core (e mod 16)` 的静态映射)。

## 踩过的坑

| 症状 | 原因 | 解法 |
|---|---|---|
| 导出 `EXIT=137`(OOM),238 GB vs 188 GB | ① bf16 被强制转 fp32([modeling_auto.py:134](file:///home/chihao/qeff-venv/lib/python3.10/site-packages/QEfficient/transformers/models/modeling_auto.py)) ② `__qeff_init__` 克隆 expert 权重却不释放源张量 | 显式 `torch_dtype=torch.float16` + 手动 `del` `gate_up_proj`/`down_proj`(峰值降到 118 GB;导出时 173 GB) |
| `hf download` 报 `No module named 'click'` | CLI 缺依赖 | 用 `huggingface_hub.snapshot_download` |
| runner `past_key.0 does not have entry in buffer identifier vector` | 编译时加了 `-retained-state`,KV 变成设备驻留、不再可绑定 | 删掉 `bindings.json` 用随机输入(cycle 只由 shape 决定),或编译时去掉 `-retained-state` |
| `-mdp-dump` 与 `-mdp-load` 互斥 | 编译器限制 | 分两次编译 |

## 文件

| 文件 | 内容 |
|---|---|
| `moe_mdp_dump.json` | 编译器自己的默认划分(2943 节点,2 partition) |
| `mdp_ts_4.json` | QEfficient 的 4×16 TP 配置(实际用的) |
| `decode/percore_decode.csv` | 每个 (site, card, core) 的事件数 + pcycles |
| `decode/op_level_decode_slice00.summary.txt` | SDK 原样输出的 card0 per-op 表 |
| `decode/out/*.trace.json` | merged Perfetto trace(343 MB × 4,git 忽略) |
| `../tools/perop_percore.py` | 本次新写的 per-(card,core,engine) 归因工具 |
| `../tools/moe_onnx_structure.py` | 本次新写的 MoE ONNX 结构 dump |

---

# Phase 2 结果 — routing 分布(2026-08-28)

在 CPU 上用 transformers hook `Qwen3MoeTopKRouter.forward` 采集,**不在卡上**
(Phase 1 已证明卡上测不出 workload 差异:真实 token vs 随机输入,MoE 各 op 差 ≤0.64%)。

| benchmark | 题数 | token | 路由决策 |
|---|---:|---:|---:|
| GSM8K (openai/grade-school-math test.jsonl) | 100 | 7,123 | 2,735,232 |
| HumanEval (openai/human-eval) | 100 | 12,876 | 4,944,384 |

工具:`tools/moe_collect_routing.py`(采集) / `tools/moe_analyze_routing.py`(分析)。
数据:`routing/routing_{gsm8k,humaneval}.npz`、`routing/expert_counts.csv`。

## ⚠️ 聚合会骗人:必须逐层看

`expert 5` 在每一层是**完全独立的参数**。把 48 层加起来求"expert 5 的总使用量"没有物理意义,
而且会把信号抹平:

| 统计量 | 48 层聚合 | **逐层** | 差 |
|---|---:|---:|---:|
| Gini(使用不均衡度) | 0.177 | **0.620** | 3.5x |
| JS divergence(两 workload 差异) | 0.029 | **0.325** | 11x |

`moe_analyze_routing.py` 内置的判读规则用的是**聚合** JS,因此会误判为"差异不大"。**以逐层为准。**

## H3 — expert 使用极度不均衡 ✅ 成立

GSM8K,第 6 层(均匀的话每个 expert 应被选 445 次):

```
最热的 5 个: 5858, 4417, 4007, 2590, 2072    ← 最热是均值的 13.2x
最冷的 10 个: 全为 0                          ← 一次都没被选中
前 10% 的 expert(12 个) 吃掉 53.0% 的 token   (均匀时 10%)
前 25% 的 expert(32 个) 吃掉 81.5% 的 token   (均匀时 25%)
```

逐层 Gini:GSM8K mean 0.620(max 0.725 @L6),HumanEval mean 0.583(max 0.720 @L19)。
零使用 expert:每层 8–17 个。

训练时的 `router_aux_loss_coef: 0.001` 只压平了**训练集全局平均**,不约束单 workload 单层。

## H2 — 两个 workload 用不同的 expert ✅ 显著

```
逐层 JS: mean=0.3245  median=0.3207  p10=0.2434  p90=0.4576
超过 0.05(显著阈值)的层: 48/48
```

逐层 top-k 热门 expert 重叠率:

| top-k | 实测重叠 | 随机期望 | 最差的层 |
|---:|---:|---:|---|
| **8** | **11.2%** | 6.2% | **0.0% (L6)** |
| 16 | 19.0% | 12.5% | 6.2% (L3) |
| 32 | 36.1% | 25.0% | 18.8% (L1) |
| 64 | 61.6% | 50.0% | 53.1% (L0) |

**最热的 8 个 expert 的重叠率只比随机瞎猜高一点。**

### 可操作的代价

把热门 expert 常驻快速存储,用 GSM8K 调,换到 HumanEval:

| 保留 | GSM8K 命中 | HumanEval 命中 | 损失 |
|---:|---:|---:|---:|
| 16/128 | 48.1% | **16.7%** | **65.3%** |
| 32/128 | 69.6% | 32.2% | 53.7% |
| 48/128 | 83.7% | 46.1% | 44.9% |
| 64/128 | 92.2% | 58.3% | 36.8% |

**结论:不存在一个静态 expert 放置/缓存方案能同时对数学和代码都最优。这是结构性问题,不是调参问题。**

## 覆盖率曲线 & 权重搬运浪费(Phase 1.5 的答案)

一个 prompt 的 prefill 会碰到多少个不同的 expert(逐层,跨 prompt 平均):

| seq_len | GSM8K | HumanEval | 均匀模型 |
|---:|---:|---:|---:|
| 1 | 6.2% | 6.2% | 6.1% |
| 16 | 40.0% | 36.9% | 63.4% |
| 64 | 58.0% | 60.7% | 98.2% |
| 128 | **64.6%** | **70.3%** | 100.0% |
| 256 | — | 76.4% | 100.0% |

**真实覆盖率饱和在 65–76%,不是均匀模型说的 100%** —— routing 高度集中,一道题有 24–35% 的
expert 从头到尾没被碰过。

结合 Phase 1 测到的"权重按 (token, expert) 对 Gather"(`Gather_3_output_0 = [T*8, 2048, 768]`),
用 GSM8K 的实测覆盖率算浪费:

| seq_len | 实际搬运 | 真正需要 | **浪费倍数** |
|---:|---:|---:|---:|
| **1**(decode) | 72 MiB | 71 MiB | **1.01x** ← 几乎最优 |
| 16 | 1152 MiB | 461 MiB | 2.50x |
| 64 | 4608 MiB | 668 MiB | 6.90x |
| **128**(prefill) | 9216 MiB | 744 MiB | **12.38x** |

- 实际搬运 = `T x top_k x 9.0 MiB`(每个 (token,expert) 对各载一次)
- 真正需要 = `覆盖率 x 128 x 9.0 MiB`(每个 distinct expert 只载一次)
- 浪费倍数 = **平均每个被选中的 expert 被多少个 token 共用**

decode(T=1)的 8 次选择落在 8 个**互不相同**的 expert 上 → 零浪费。
prefill(T=128)的 1024 次选择只落在 83 个 expert 上 → 每个热门 expert 的 9 MiB 被重复搬 12.4 遍。

**MoE 的稀疏性在 decode 上兑现得完美,在 prefill 上被这条 lowering 完全吃掉。**

> 唯一还能推翻这张表的机制:编译器在运行时把已取的 expert 权重缓存在 VTCM 里。它编译期看不见
> 哪些索引重复,所以只能靠运行时缓存。验证办法:编一个 prefill QPC,把实测的
> `blockdequantize_mxfp6` / DMA 量跟上表预测对比。**这是下一步。**

---

# 受控实验:证明一个 expert 精确平分在四张卡上(2026-08-28)

前面的论证都是从计时/字节数**反推**放置,属于统计推断。下面是一个受控实验。

## 方法

把 ONNX 里 48 个 router 的权重张量(`onnx::MatMul_*`,各 `[2048,128]` fp16,均为独立外部文件)
**全部置零** -> 所有 router logits 相等 -> `TopK` 按索引返回前 8 个 -> **强制永远只选 expert 0–7**。

用硬链接复制导出目录再打断那 48 个文件的链接,额外磁盘仅 24 MB,不需重新导出:

```bash
cp -al export-126f04c76d4a0568/. export_forced/
# 然后把 48 个 router 外部权重文件覆盖成全零
```

产物:`/home/chihao/models/qwen3_30b_a3b/export_forced/`、`4card/qpc_forced/`(编译 3m47s)。

## 判据

expert 0–7 是一个**连续块**。在任何"每卡 32 个 expert"的 EP 方案下,它们**必然全部落在 card0**。

| | EP 的预测 | TP 的预测 |
|---|---|---|
| `Gather_3` 每卡 KB | **9600 / 0 / 0 / 0** | 2400 / 2400 / 2400 / 2400 |
| card0 的 MoE cycles | ~4x 基线 | 与基线一致 |

## 结果

```
op (layer 0)     card0      card1      card2      card3        合计     极差
Gather_3        2400.1     2400.0     2400.0     2400.0      9600.1    0.00%
Gather_4        2400.3     2400.6     2400.5     2400.4      9601.8    0.01%
Gather_5        2399.9     2400.0     2400.1     2400.4      9600.5    0.02%
MatMul          7329.0     7327.9     7328.0     7328.1     29313.0    0.01%
MatMul_2        7495.7     7495.8     7496.4     7495.8     29983.8    0.01%

InfPCycles: 15,635,222 / 15,607,427 / 15,611,109 / 15,611,165   卡间极差 0.178%
TopK 事件数 1728 -> 路由确实执行了,未被常量折叠
```

自然路由的基线:`Gather_3` = 2400.2 / 2400.0 / 2399.9 / 2399.9,`InfPCycles` 极差 0.166%。
**强制版与基线的每卡分布完全一致。**

## 结论

```
合计 9600 KB / 8 个 expert = 1200 KB per expert   (gate_proj 在 MXFP6 下)
1200 KB / 4 张卡           =  300 KB per expert per card   <- 精确 1/4
```

**一个 expert 精确平分在四张卡上,每卡 1/4;且与被选中的是哪些 expert 无关。**

这是受控实验而非统计推断:改变了唯一的自变量(被选中的 expert 集合),观测量(每卡字节分布)纹丝不动。

## 顺带被推翻的一条论证

先前我用"`Gather` 从不触发跨卡传输"论证过 EP 不成立。**那条论证有洞**:EP 下若 8 个 expert
恰好每卡分到 2 个,Gather 也是本地的(每卡只取自己那 2 个),跨卡合并发生在后面的 `Add_1`——
而 `Add_1` 确实有跨卡流量。所以那条观察不能单独排除 EP。以本节的受控实验为准。

---

# Phase 3 — Expert Parallelism 实测(2026-08-29)

> ⚠️ **本节用 QEfficient 1.23.0.dev0 (git f5fc080)**;上面 Phase 1/2 全部是 1.22.0.dev0
> (git 7b620bb) 测的。升级前的环境快照见 `env_before_upgrade.txt`,
> 旧 venv 备份在 `/home/chihao/qeff-venv.bak-7b620bb`。

1.23 新增了 `QEfficient/transformers/moe/` 模块,提供 `MoEFlavour.EXPERT_PARALLEL`。
执行计划见 `EP_PLAN.txt`。

## 怎么跑的

```python
m = QEFFAutoModelForCausalLM.from_pretrained(
    HF, torch_dtype=torch.float16,
    qaic_config={"moe_config": {"flavour": "expert_parallel", "cores_per_expert": 1,
                                "tree_reduce": False, "expert_parallel_chunk_size": 256}})
m.compile(prefill_only=True, enable_chunking=True, prefill_seq_len=128, ctx_len=256,
          batch_size=1, num_devices=4, num_cores=16, mxfp6_matmul=True,
          mos=1, aic_enable_depth_first=True)
```

`prefill_only=True` 是必需的 —— `select_moe_flavour` 只在 prefill 选 EP,decode 恒为 DECODE_BMM。
脚本 `/home/chihao/models/qwen3_30b_a3b/ep/run_ep.py`。

导出 25 min(57 GB ONNX),编译 10m28s(25 GB QPC,4 个 slice)。

## EP 布局(用 1.23 的 _resolve_expert_parallel_layout 算)

`num_experts=128, num_devices=4, num_cores=16, cores_per_expert=1`:

```
num_pipeline_stages      = 2      循环 2 次
num_parallelized_experts = 64     64 条并行 lane
experts_per_soc          = 16     每卡同时 16 个 expert
每卡 lane                = 16     -> 每卡拥有 32 个 expert

lane p = e mod 64 ;  card d = p // 16
  card0 {0..15,  64..79}    card1 {16..31, 80..95}
  card2 {32..47, 96..111}   card3 {48..63, 112..127}
```

## 结果 1 — EP 在图层面确证生效

| 判据 | gather-bmm (1.22) | **EP (1.23)** |
|---|---|---|
| ONNX 节点数 | 15,266 | **27,975** (1.83x) |
| layer0 `/mlp/` 节点 | 69 | **313** (4.5x) |
| expert 权重张量 | `[128, 2048, 768]` 一个 | **`[64, 2048, 768]` x 2 组** |
| 首维含义 | expert | **lane** |
| `CtxGather3D` | 无 | **5 个/层**(打包 token) |
| `Einsum` | 1 个/层 | **4 个/层**(lane/device 归约) |

**首维 128(expert) -> 64(lane)** 就是 `pack_moe_weights_for_expert_parallel` 的
`view(2, 64, ...).transpose(0,1)`;`for slot in range(2)` 被展开成两组张量。
张量切分器因此切 lane 轴(64/4 = 每卡 16 条),而不是 hidden 轴。

## 结果 2 — 权重搬运:实测命中 chunked 预测,比 gather 路径省 7.7x

单次推理的 `blockdequantize_mxfp6` 总量(`out KB` x 16 核 x 4 卡):

| | tokens | 总量 | 每 token |
|---|---:|---:|---:|
| TP decode (1.22) | 1 | 6.59 GiB | **6.586 GiB** |
| **EP prefill (1.23)** | **128** | **56.30 GiB** | **0.440 GiB** |

**prefill 每 token 的权重搬运便宜 15.0 倍。**

单层对账:

```
EP 实测          1.173 GiB/层
chunked 预测     1.125 GiB/层   (128 expert x 9 MiB, 与 T 无关)   <- 命中,差 4%(attention)
gather-bmm 预测  9.0   GiB/层   (T x 8 x 9 MiB, T=128)           <- 省 7.7x
```

**Phase 1.5 那个 12.4x 浪费的推论,在这里得到了反向验证**:换成 chunked/EP 之后,
权重搬运确实变成与 T 无关的常数。

## 结果 3 — 算子构成才是 TP/EP 的判据,卡间均匀度不是

card0 的 c0 核,`out KB`:

| 算子 | TP decode | EP prefill | 说明 |
|---|---:|---:|---|
| `blockdequantize_mxfp6` | 107,904 | 922,368 | 权重解码 |
| `aicgather HVX DDR (uindex8` | 21,600 | **消失** | **按 (token,expert) 对取权重 -> 没有了** |
| `cumsum HVX` | **无** | **3,072** | **token 打包(EP 特征)** |
| `aicbatchedreduceadd` | 96 | **24,576** | **lane/device 归约(EP 特征)** |
| `aicconvolutiond32 HMX` | 37,666 | 488,770 | 真正的矩阵乘 |

**`aicgather` 消失 + `cumsum` 出现**,是 chunked/EP 最直接的硬件层证据。

### ⚠️ 判据修正:要看"干了多少活",不是"跑了多久"

`EP_PLAN.txt` 阶段 5b 原本写"EP 生效则四卡 events 不再相同"。方向对,但指标要选对。

**wall-clock 完全均匀**(`ExecTimeUs` 四卡极差 0.006%) —— 因为四张卡互相等,同时结束。
**但实际工作量差 16%**(累加各 op 的 pcycles):

```
card0 116.2% of mean    card1 97.9%    card2 89.4%    card3 96.5%
卡间 cv = 9.90%   max/mean = 1.162
```

差额全变成了等待。**用 wall-clock 判断会得出"完全均衡"的假结论。**

对比 TP decode 的卡间 max/mean = 1.0035 —— **EP 确实引入了卡间不均衡。**

## 结果 4 — 吞吐:prefill 只比 decode 快 3.7x

```
EP prefill (128 token)  497 ms/次  ->  257 tok/s
TP decode  (1 token)    14.3 ms    ->   70 tok/s
```

card0 c0 核的时间去向:

```
sync HMX              94.56%   <- 仍然是同步瓶颈,跟 decode 一样
aicbatchedreduceadd   15.39%   <- 归约
aiccopytovtcm         14.28%
cumsum                 7.71%
blockdequantize_mxfp6  7.12%
aicconvolutiond32      4.17%   <- 真正的矩阵乘只占 4%
```

**chunked/EP 的 16x padding 冗余在这里显形**:算力大量花在被 mask 掉的行上。
`expert_blocking_num_packed_chunks` 是死代码,循环上界本该是
`ceil(T*top_k/num_experts/chunk)` 而不是 `ceil(T/chunk)` —— 这是最直接的改进点。

## 结果 4b — 核利用率腰斩,核间不均衡是 TP 的两倍

分母 = `InfPCycles x 16 核`,分子 = 该卡所有 op 的 pcycles 之和(方法同 Phase 1)。

| | 利用率 | 卡间 max/mean | 核间 cv | 核间 max/mean |
|---|---:|---:|---:|---:|
| TP decode | **30.2%** | 1.0035 | 10.8% | 1.232 |
| **EP prefill** | **15.9%** | **1.162** | **36-44%** | **2.39-2.70** |

每卡明细:

```
card0  18.5%   最忙核 44.0%  最闲核 15.8%
card1  15.5%   最忙核 39.6%  最闲核 13.4%
card2  14.2%   最忙核 38.3%  最闲核 12.0%
card3  15.3%   最忙核 39.3%  最闲核 13.3%
```

核间不均衡的原因很直接:`num_parallelized_experts=64`、每卡 16 条 lane ->
**每个核只负责 1 条 lane = 2 个 expert**。逐层 Gini 0.62 之下,只分到 2 个 expert 的核
冷热差异极大。

哪些算子不均匀,进一步坐实:

```
mlp/MatMul_1, MatMul_3    cv 0.00%      <- expert 矩阵乘,padding 抹平了 routing
mlp/MatMul,   MatMul_4    cv 0.05-0.13%
mlp/ScatterElements       cv 5.02%   ⚠
mlp/CtxGather3D_1/n14     cv 8.35%   ⚠  <- token 打包量由 routing 决定
```

**矩阵乘均匀、打包不均匀** —— 这是 EP 在硬件上最直接的签名。

## 结果 5 — EP 的卡间负载:静态预测(不碰卡)

用已采的 routing 套 `card d = (e mod 64)//16`(`tools/moe_ep_predict.py`,
产物 `ep_predicted_card_load.txt`):

| | GSM8K | HumanEval |
|---|---:|---:|
| 48 层聚合 max/mean | 1.058 | 1.086 |
| **逐层平均 max/mean** | **1.296** | **1.273** |
| 最差层 | 1.649 (L7) | 1.615 (L9) |
| 超过 1.2 的层 | **36/48** | 29/48 |
| 有用功占比 | 6.07-6.61% | 5.76-6.79% (理想 6.25%) |
| 容量截断需要的 capacity factor | **1.65** | 1.61 |

**教授设想的卡间不均衡在 EP 下第一次成立**(TP 下实测 0.166%,不可能发生)。
但因为 padding,它当前只表现为"有用功占比不同",不是 idle。

最后一行是最有价值的:现在每卡每层算 `32 x T` 行、有用的只有 6.25%
(capacity factor 相当于 **16**);若改成按 per-card 容量截断,**只需 1.65** ->
**计算量可削减约 9.7 倍**,且不需要改并行方式。

## 结果 6 — decode batch 扫描:EP 在任何 batch 下都不划算

`tools/moe_ep_batch_sweep.py`,产物 `ep_batch_sweep.txt`。从真实 routing 抽样:

| batch | 覆盖率 | 卡间 max/mean | 某卡全 idle 概率 |
|---:|---:|---:|---:|
| 1 | 6.2% | **1.741** | **30.8%** |
| 4 | 19.4% | 1.452 | 0.2% |
| 32 | 56.3% | 1.314 | 0.0% |
| 128 | 74.6% | 1.312 | 0.0% |
| 256 | 78.8% | 1.309 | 0.0% |

(GSM8K;HumanEval 同形状,平台值 1.28)

- **小 batch**: 不均衡 1.74,**31-37% 的概率某张卡完全没活干**(8 个 expert 扔 4 桶的组合必然)
- **大 batch**: idle 消失,但 `max/mean` **停在 1.28-1.31 而不是 1.0** ——
  剩下的是 routing 倾斜带来的**系统性**偏差(逐层 Gini 0.62),batch 再大也平均不掉。
  这个平台值与结果 5 的逐层预测(1.296/1.273)**独立吻合**。
- 而且大 batch 下覆盖率 75-87%,**权重搬运与 TP 相同** -> EP 不再带来任何好处

**两端都不划算,中间没有甜点区。**

## 新增的坑

**#8 QEfficient 1.23 生成的 specializations.json,SDK 1.21.6 不认**

```
1.23 生成: {"specializations":[{"name":"Prefill","symbols":{"batch_size":"1",...}}]}
1.21.6 要: {"specializations":[{"batch_size":"1","ctx_len":"256","seq_len":"128"}]}
报错     : [json.exception.type_error.302] type must be string, but is object
```

`m.compile()` 会在最后一步 `qaic-compile` 上失败(ONNX 已经导出成功)。绕法:把
`specializations.json` 的 `symbols` 拍平后手动调 `qaic-compile`,见
`/home/chihao/models/qwen3_30b_a3b/ep/compile_ep.sh`。

## 本节文件

| 文件 | 内容 |
|---|---|
| `ep_predicted_card_load.txt` | EP 卡间负载的静态预测 |
| `ep_batch_sweep.txt` | 不同 decode batch 下的负载与覆盖率 |
| `ep_prefill/out/*.summary.txt` | SDK 原样 per-card op 表 |
| `ep_prefill/percore_ep.csv` | 每个 (op, card, core) 的事件数与 pcycles |
| `../tools/moe_ep_predict.py` | 静态预测工具 |
| `../tools/moe_ep_batch_sweep.py` | batch 扫描工具 |

QPC: `/home/chihao/models/qwen3_30b_a3b/ep/qeff_cache/.../qpc-6ff7c7675b1fcae0/qpc/` (25 GB)
