# 测量方法学 —— 这些数据是怎么来的

模型:Qwen3-30B-A3B(128 experts, top-8, 48 层, hidden 2048, moe_intermediate 768)
硬件:4 x Qualcomm Cloud AI 100(每卡 16 核)
软件:QEfficient 1.22(Phase 1/2)与 1.23.0.dev0(Phase 3),AIC SDK 1.21.6

本文只讲**怎么测的**和**每种测量的边界在哪**。结论在 `README.md` 和
`ADVISOR_DIRECTION.md`,MegaBlocks 的可行性判断在 `MEGABLOCKS_VERDICT.md`。

---

## 〇、四类数据,先分清哪些是硬件测的

| 数据 | 怎么来的 | 硬件测量? | 文件 |
|---|---|---|---|
| **路由**(每 token 选了哪 8 个 expert) | CPU 上 HuggingFace 前向 + forward hook | **否** | `routing/routing_*.npz` |
| **per-op / per-core / per-card 周期数** | 卡上跑 QPC + SDK profiling | **是** | `ep_prefill/`, `decode_realtok/` |
| **op 执行次数 / 数据搬运量** | 同上,取 SDK summary 的 `count` / `out KB` | **是(且是静态量)** | `*/out/*.qaic-opstats.summary.txt` |
| **expert 权重放置** | ONNX 张量切片与 checkpoint 逐比特比对 | 否(图分析) | `proof_expert_split/` |

**最容易混淆的一点**:路由统计全部来自 CPU,不是从卡上采的。
router 是确定性计算(hidden x gate 权重 -> softmax -> topk),权重同源,
所以路由结果与卡上一致;但它不是硬件测量值。凡是引用路由数字的结论,
说的都是"模型会怎么路由",不是"卡上跑出了什么"。

---

## 一、路由数据(CPU,HuggingFace)

**目的**:知道每层每个 expert 实际收到多少 token。这是分档表、跨域实验、
周期-12 发现的共同输入。

**工具**:`tools/moe_collect_routing.py`

### 方法

在每一层的 `mlp.gate`(即 router)上挂 forward hook,取其输出的第 3 项:

```python
def mk(idx):
    def hook(mod, inp, out):
        captured[idx] = out[2].detach().to(torch.int16).cpu().numpy()
    return hook
for i, layer in enumerate(model.model.layers):
    gate = layer.mlp.gate
    handles.append(gate.register_forward_hook(mk(i)))
```

`mk(i)` 用工厂函数闭包,避免 late-binding 把所有 hook 绑到同一个 `i`。

### 为什么 out[2] 就是 expert 索引

`transformers/models/qwen3_moe/modeling_qwen3_moe.py:254` 的
`Qwen3MoeTopKRouter.forward`:

```python
router_logits = F.linear(hidden_states, self.weight)          # [seq, 128]
router_logits = softmax(router_logits, dtype=torch.float, dim=-1)
router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
return router_logits, router_scores, router_indices
#         out[0]         out[1]          out[2]   <- 取这个
```

而上层 `Qwen3MoeSparseMoeBlock.forward` 拿它直接当 `selected_experts`
喂给 `self.experts(...)`。**记录的正是模型真正用来选 expert 的张量,不是重算的。**

### 前向怎么跑

```python
ids = tok(s, return_tensors="pt", truncation=True, max_length=a.max_len)   # 不 padding
model(**ids, use_cache=False)        # 一次喂完整 prompt = prefill
```

`use_cache=False`,一次性喂全部 token —— **这是 prefill,没有 decode**。
断言 `len(captured) == 48` 且 `arr.shape == (L, T, K)`,任一不满足直接退出。

### 输出

`routing_<bench>.npz`,键 `p0..p99`,每个是 `[48, T, 8]` int16;
另有 `meta` 记录 bench / model / layers / experts / top_k / chat_template
与每条 prompt 的 token 数。**存的是完整长度,不截断。**

### 统计

```python
np.bincount(arr[layer].ravel(), minlength=128)   # 第 e 格 = expert e 收到几个 token
```

自检:总和必须等于 `T * top_k`。实测 prompt 0 layer 0:616 = 77 x 8。

### 边界

- **CPU,非卡上测量**(见第〇节)
- **只有 prefill,无 decode**
- **`chat_template=True`**,每条 prompt 前带 Qwen 对话模板 token,这部分计入统计
- 语料:GSM8K 100 条 / HumanEval 100 条

### 复现

```bash
python3 tools/moe_collect_routing.py --bench gsm8k     --n 100 --out perop_moe/routing/
python3 tools/moe_collect_routing.py --bench humaneval --n 100 --out perop_moe/routing/
```

---

## 二、硬件 per-core / per-card 归因(卡上)

**目的**:知道每张卡、每个核实际执行了哪些 op、多少周期。

**工具**:`tools/perop_percore.py` -> `tools/perop_percard_table.py`

### 方法

SDK profiling 产出的合并 trace 里,每个事件带一个 `tid`。
**仓库里已有的工具全部丢弃 `tid`,于是盲目地把 64 个 (卡, 核) 加在一起。**
真正的连接键是 trace 自带的 `thread_name` 元数据:

```
tid  ->  "QAicGraph_slice<N>_Core_<M>_<HVX|HMX|DMAIssue>[_Barrier|_DMA]"
```

`slice N` = 卡号,`Core M` = 核号。`perop_percore.py` 两遍扫描:先建
`tid -> (card, core, engine)` 映射,再按此归因每个事件。不做 tid 算术,
并断言推出的格点自洽。

### 两种指标,用途不同

| 指标 | 来源 | 性质 | 用途 |
|---|---|---|---|
| `pcycles` | trace 事件 | 含计时抖动(单核 ±5%) | 看时间去向 |
| `count` / `out KB` | SDK summary | **静态量,不抖动** | 证明放置与执行次数 |

**要证明"谁在算什么"必须用 count / out KB,不能用 pcycles。**

### op-site 过滤

`perop_percard_table.py` 只保留路径含 `/model/` 或 `/lm_head` 的 op-site。
原始 5889 个 "site" 里有 5799 个是运行时簿记(`inputseminc_*` 之类),
不过滤会把结论淹掉。

### 边界

- `decode_realtok/` 的卡间数字来自**单个 token 的一次推理**,不是 benchmark。
  这一点在 `decode_realtok/PROVENANCE.md` 里单独记了。
- 输入绑定用 `perop_make_input.py --layers 0`,只绑 `input_ids`/`position_ids`,
  绕开 `-retained-state` 冲突。实测真实 token vs 随机输入,MoE 各 op 差 <= 0.64%
  —— 周期数由 shape 决定,对计时无影响。

---

## 三、受控实验:强制路由

**目的**:证明"没被选中的卡也在做同样的 op"。这是推翻教授原始假设的关键实验。

**方法**:把 48 个 router 权重文件全部置零 -> softmax 输出均匀 -> TopK
恒定返回 expert 0-7(全部落在 card0)-> 重新导出编译成
`ep_forced/`,在卡上跑,比较四卡的 op `count`。

**为什么这样有效**:它把"路由"这个变量按住不动,如果四卡仍然做一样的事,
就说明执行量与路由无关。

**结果**(同一二进制、同一次运行):

| op | card0 | card1 | card2 | card3 |
|---|---:|---:|---:|---:|
| `aicconvolutiond32` | 3155 | 3155 | 3155 | 3155 |
| `blockdequantize_mxfp6` | 2654 | 2654 | 2654 | 2653 |
| `cumsum` | 48 | 48 | 48 | 48 |

card1/2/3 一个被选中的 expert 都没有,执行次数完全相同。
(card3 的 2653 少 1 是采样抖动,非结构差异。)

**注意**:`ep_prefill` 与 `ep_forced` 是两次独立编译,**跨编译比 op count 不成立**
(实测 `cumsum` 96 vs 48)。只能比同一二进制内的卡间差异。

---

## 四、expert 权重放置的位比对

**目的**:证明每个 expert 被 1/4 均分到四张卡(而不是整个放在某一张)。

**方法**:`tools/moe_verify_expert_tensor.py` 把 ONNX 里 expert 张量的第 e 个切片
与 checkpoint 的 `experts.e` 逐比特比对。测了 layer 0 与 layer 24,
gate_proj 与 up_proj,expert 0/5/91/127/3/17/64。同时做了错配对照,
确认该方法有区分力(错配必然不等)。

**失败的方法也记一下**:先试过字节指纹搜索(`moe_find_expert_bytes.py`),
失败 —— matmul 权重被 d32 分块重排,layernorm 权重能在 32B/16B/8B 找到,
matmul 权重在所有长度都找不到。这是**方法失败,不是数据不存在**。

---

## 五、分析工具链

| 工具 | 输入 | 输出 |
|---|---|---|
| `moe_collect_routing.py` | HF 模型 + 语料 | `routing_*.npz` |
| `moe_analyze_routing.py` | npz | Gini / JS / 覆盖率 |
| `moe_layer_expert_load.py` | npz | 单层 per-expert 负载 + split-half 稳定性 |
| `moe_top_experts_per_layer.py` | npz | 逐层最忙 N 个 expert(表 + CSV) |
| `moe_tier_crossdomain.py` | npz x2 | 分档表跨域迁移丢弃率 |
| `moe_tier_permute.py` | npz x2 | 档位表能否分解为「形状 x 排名」 |
| `moe_blocksize_sweep.py` | npz | MegaBlocks 式分块在本形状上的收益 |
| `moe_ep_predict.py` | npz | EP 静态分配下的每卡负载预测 |
| `moe_ep_batch_sweep.py` | npz | EP 负载 vs decode batch size |
| `perop_percore.py` | 合并 trace | per-(卡,核,引擎) 归因 CSV |
| `perop_percard_table.py` | percore CSV | per-(op, 卡) 汇总表 |
| `moe_proof_percard_ops.py` | 多份 trace | per-(op, 卡) 证据表 |
| `moe_onnx_structure.py` | ONNX | MoE 结构与逐 expert 可归因性判定 |
| `moe_verify_expert_tensor.py` | ONNX + checkpoint | 张量切片位比对 |

> `moe_analyze_routing.py` 内置的 verdict **用的是聚合 JS,是错的** ——
> 要看逐层数值,别信它的结论行。

---

## 六、方法学陷阱(全部是本项目实际踩过并纠正的)

### 1. 聚合陷阱 —— 出现过三次

逐层统计与 48 层聚合统计差 3.5-11 倍。三个实例:

| | 聚合值 | 逐层值 |
|---|---:|---:|
| Gini | 0.177 | **0.620** |
| 跨域 JS 散度 | 0.029 | **0.325** |
| 分档表下的卡间 max/mean | 1.026 | **1.222** |

层是串行执行的,每层都要等最慢的卡,**逐层才作数**。
凡是"每层一个数"的量,先看逐层分布再看聚合。

### 2. wall-clock 掩盖不均衡

最初报告"四张卡完全均匀(0.006%)"——那是 wall-clock。卡之间会同步,
所以墙钟必然接近。实际工作量差 16%(card0 116.2%,card2 89.4%)。
**要看不均衡必须用 pcycles 或 op count,不能用墙钟。**

### 3. padding 污染路由统计

有一次分析复刻了卡上的补齐输入(65 真 token + 63 个零),结果那 63 个 padding
贡献了 **49.2%** 的激活,且极度集中(某个 expert 独吞 63 个)。
真实 token 的正确数字是 mean 4.06 / max 64 / 46.9% 为零。

`moe_collect_routing.py` **不做 padding**(只 truncation),所以 Phase 2 数据干净。
但任何复刻卡上输入的分析都必须先剔除 padding。

### 4. 长度混淆

GSM8K prompt 中位 66 token,HumanEval 中位 119。直接比跨域路由差异,
测到的是长度不是路由。**所有跨域实验都用定长窗口**(截各 prompt 前 W 个 token)。

### 5. 自评的乐观偏差

用同一批 prompt 既拟合档位表又评估,丢弃率会偏低。
所有分档实验都把每个 benchmark 切 fit/eval 两半,域内也用 held-out。

### 6. 跨编译比较不成立

`ep_prefill` 与 `ep_forced` 是两次独立编译,`cumsum` count 96 vs 48。
**只能比同一二进制内的差异。**

### 7. 相关系数被跨度抬高

split-half r = 0.984 很高,部分是因为真实负载跨度大(0 到 10.01)。
更贴近决策的指标是"用一半拟合的档位表评另一半丢多少"(2.20%)。
两个数一致,互相印证 —— 但不要只报 r。

### 8. 分母错误

`moe_ep_predict.py` 曾把"有用比例"的分子按 48 层求和、分母按单层算,
得出 293% 这种不可能的数。修正(乘 `L`)后是 6.07-6.79%(理想 6.25%)。
**任何超过 100% 的比例先怀疑分母。**

### 9. 曾经报错又撤回的结论

留档以免重犯:

| 曾经的说法 | 实际 |
|---|---|
| 四卡完全均匀 | 那是 wall-clock,工作量差 16% |
| 核间 2.5x 不均衡来自 routing | 来自 core0 的串行工作,与 expert 激活相关性 r=+0.018 |
| 图里没有 `If` | 有 94 个,但全是形状簿记,expert MatMul 上游没有 |
| 越界 gather 返回 0 | `INT32_MAX` 被钳到索引 0,读的是 token 0 的**真实**隐状态,算完再置零 |
| 改 chunk 循环上界就能拿 9.7x | 每 expert 所需容量 14.4-15.8(上限 16),只有按卡池化才行(cf=2.04) |
| `blockdequantize_mxfp6` 占 55-70% | EP prefill 实测 7.1-7.8% |

---

## 七、复现清单

```bash
# 路由数据(CPU,约 1 小时/benchmark)
python3 tools/moe_collect_routing.py --bench gsm8k     --n 100 --out perop_moe/routing/
python3 tools/moe_collect_routing.py --bench humaneval --n 100 --out perop_moe/routing/

# 单层负载 + 稳定性
python3 tools/moe_layer_expert_load.py perop_moe/routing/routing_gsm8k.npz -l 0 -W 48

# 逐层最忙 expert
python3 tools/moe_top_experts_per_layer.py perop_moe/routing/routing_gsm8k.npz \
        -W 48 -n 5 --csv perop_moe/routing/top_experts_per_layer_gsm8k.csv

# 分档表跨域迁移
python3 tools/moe_tier_crossdomain.py -W 48
python3 tools/moe_tier_permute.py

# MegaBlocks 分块收益
python3 tools/moe_blocksize_sweep.py \
        perop_moe/routing/routing_gsm8k.npz perop_moe/routing/routing_humaneval.npz

# 卡上 profiling 见 EP_PLAN.txt(7 个阶段 + 7 个已知坑)
```

卡上流程与操作层面的坑在 `EP_PLAN.txt`,本文不重复。
