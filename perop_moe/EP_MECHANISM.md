# QEfficient EXPERT_PARALLEL 机制拆解

模型 Qwen3-30B-A3B(E=128 experts, top-8, 48 层, H=2048, I=768)
硬件 4 × Cloud AI 100(每卡 16 核)
软件 QEfficient 1.23.0.dev0 —— 与上游 main 逐字节核对过,**算法一致**

> 仅 `flavours.py` / `qflavours.py` 的调用风格不同:上游用薄包装
> `ctx_gather_3d_generalized(...)`(内部 `select_interface` 在 legacy/dynamo 两条
> 导出路径间二选一),本地直接 `CtxGatherFunc3DGeneralized.apply(...)`。
> `customop/ctx_scatter_gather.py` 两边字节完全相同(17158 B)。
> 我们走 legacy 导出,行为相同。

本文只讲**机制**。结论与研究方向见 `ADVISOR_DIRECTION.md`,
测量方法见 `METHODOLOGY.md`,逐步形状与代价见 `EP_DATAFLOW.md`。

---

## 一、代码路径

```
MoEFlavour.EXPERT_PARALLEL
  → moe_expert_parallel(...)                              flavours.py
       for slot in range(num_pipeline_stages):            # Python 循环,导出时展开
           cumsum_scatter_gather_update_expert_blocked(...)
                → build_matched_idx_from_cumsum(...)
                → CtxGather3D / expert_mlp / mask / CtxScatter3D
```

**坑**:`MoEFlavour.EXPERT_BLOCKED = "expert_parallel"`,与 `EXPERT_PARALLEL`
是同一个枚举值(向后兼容别名)。所以函数名里的 `expert_blocked` 就是 EP,
不是另一种 flavour。

flavour 由 `select_moe_flavour` 决定:**prefill 选 `EXPERT_PARALLEL`,
decode 选 `DECODE_BMM`**。

---

## 二、编译期:expert → lane → card

`_resolve_expert_parallel_layout(num_experts=128, num_devices=4, num_cores=16,
cores_per_expert=1)`(`transformers/models/pytorch_transforms.py:1444`)实跑:

```
总核数                    = 64      (4 卡 × 16 核)
num_pipeline_stages      = 2       128 expert × 1 核 / 64 核 = 要 2 轮
num_parallelized_experts = 64      lane 数
experts_per_soc          = 16      每卡 lane 数
```

**映射不是任何调度器算的,是两个 reshape 定死的**:

```python
rw.view(2, 64, T)          # expert e = stage*64 + lane  ⇒  stage = e//64, lane = e%64
out.view(4, 16, T, H)      # lane p  ⇒  card p//16
```

于是每卡持有 32 个 expert,分两轮:

| | stage 0 | stage 1 |
|---|---|---|
| card0 | e0–15 | e64–79 |
| card1 | e16–31 | e80–95 |
| card2 | e32–47 | e96–111 |
| card3 | e48–63 | e112–127 |

**这是静态的,与数据无关。改不了,除非改 Python 源码。**

---

## 三、运行时:token 怎么进去

### 3.1 稀疏性先被摊平

router 给的是紧凑的 `(topk_indices [T,8], topk_weights [T,8])`,但 EP 要稠密矩阵:

```python
def densify_topk(topk_indices, topk_weights, num_experts):
    routing_weights = topk_weights.new_zeros((T, num_experts))   # [T,128] 全零
    routing_weights.scatter_(1, topk_indices, topk_weights)      # 只填 8 个
    return routing_weights
```

`[T,8]` → `[T,128]`(120 个零)→ `T2Ei = rw > 0` 再把 mask 找回来。
**sparse → dense → mask,绕了一圈。**

### 3.2 每个 token 的 8 个 expert 散在多张卡上

真实数据(p0 第 12 层前几个 token):

```
token0: e10(c0) e18(c1) e24(c1) e36(c2) e78(c0) e84(c1) e86(c1) e119(c3)  → card[0,1,2,3]
token3: e2(c0)  e4(c0)  e8(c0)  e24(c1) e46(c2) e56(c3) e111(c2) e113(c3) → card[0,1,2,3]
token7: e40(c2) e44(c2) e46(c2) e47(c2) e53(c3) e61(c3) e66(c0) e113(c3)  → card[0,2,3]
```

**一个 token 的计算被拆到几乎所有卡** —— 这是"不会有卡闲着"的直接原因,
也是教授最初那个"某些卡会 idle"假设不成立的机制解释。

---

## 四、matched_idx 是怎么建出来的

这是 EP 的核心。**四步纯索引运算,一个隐状态都不碰。**

```python
def build_matched_idx_from_cumsum(T2Ei):                    # T2Ei: [64, T] bool
    token_idx    = arange(T).unsqueeze(0).expand(64, -1)    # [64,T] = 0,1,2,...,T-1
    valid_prefix = torch.cumsum(T2Ei.to(int32), dim=1)      # ① 前缀和 = 排第几
    valid_dest   = valid_prefix - 1                         # ② 转 0-based
    scatter_pos  = where(T2Ei, valid_dest, INT32_MAX)       # ③ 没选的作废
    matched_idx  = INT32_MAX.expand_as(token_idx)           # ④ 全 INT32_MAX 的画布
    matched_idx  = CtxScatter3DInt(matched_idx, scatter_pos, token_idx)   # ⑤ 反转
    return matched_idx
```

### 实例(QEfficient 真函数输出,4 lane × 8 token)

```
T2Ei              t0  t1  t2  t3  t4  t5  t6  t7
  lane0:           0   1   0   1   1   0   0   1     token 1,3,4,7 选了它
  lane1:           1   1   1   0   0   0   0   0     token 0,1,2
  lane2:           0   0   0   0   0   0   0   0     没有 token
  lane3:           1   1   1   1   1   1   1   1     全部 8 个(最坏情况)

① cumsum                                             前缀和 = 排第几(1-based)
  lane0:           0   1   1   2   3   3   3   4
  lane1:           1   2   3   3   3   3   3   3
  lane2:           0   0   0   0   0   0   0   0
  lane3:           1   2   3   4   5   6   7   8

② −1                                                 转 0-based 行号
  lane0:          -1   0   0   1   2   2   2   3

③ scatter_pos = where(T2Ei, dest, INT32_MAX)         正向表: token → 行
  lane0:           .   0   .   1   2   .   .   3
  lane1:           0   1   2   .   .   .   .   .
  lane2:           .   .   .   .   .   .   .   .
  lane3:           0   1   2   3   4   5   6   7

⑤ matched_idx                                        反向表: 行 → token
  lane0:           1   3   4   7   .   .   .   .     4 行真, 4 行 padding
  lane1:           0   1   2   .   .   .   .   .     3 行真, 5 行 padding
  lane2:           .   .   .   .   .   .   .   .     全空, 但照样跑完全部 GEMM
  lane3:           0   1   2   3   4   5   6   7     填满, 0 padding  ← 缓冲必须开 T 宽的理由
```

四种情形一次覆盖:部分占用 / 恰好在前面 / 全空 / 填满。

### 第 ⑤ 步在做"映射反转"

```
scatter_pos:  token t  →  该去第几行         (正向, cumsum 天然给出)
matched_idx:  第 r 行   →  装的是哪个 token   (反向, gather 需要的)
```

手法:**把 `token_idx`(0,1,2,…)当作"要写入的值",`scatter_pos` 当作"写到哪"。**

对照 lane0:`scatter_pos` 里 `t3→1`,`matched_idx` 里 `r1→3` —— 同一条信息翻了个面。

### INT32_MAX 在两侧语义不同(易错点)

**scatter 侧(建表)—— 跳过不写**

```python
valid = position_ids != torch.iinfo(torch.int32).max
data[batch_idx[valid], position_ids[valid].long()] = updates[valid]
```

没被写到的格子保持 `INT32_MAX`,那就是空行标记。

**gather 侧(用表)—— 钳到 0**

```python
ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
return data[batch_indices, ctx_indices]
```

空行**不是读到零,是读到 token 0 的真实隐状态**,算出真实结果,再靠 mask 抹掉。
这是"算满再 mask"的字面含义:不是先判断再跳过,是先全算再擦掉。

### 卡上跑的是 ONNX 版,没有 valid 过滤

```python
@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxScatter3DInt(data, position_ids, updates):
    ...
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)     # [64, T, 2] 的 (lane, 行) 坐标
    return ops.ScatterND(data, indices, updates)         # INT32_MAX 直接当坐标传进去
```

依赖硬件把越界写丢弃。PyTorch 版显式过滤只是为了让 CPU 参考跑出相同结果。

> **这是个隐含契约,不是 ONNX 规范保证的**(规范里越界索引是未定义行为)。
> 我们验证过卡上与 CPU 的 argmax / top-5 一致,所以在这个模型上确实一致。
> **改档位宽度时必须重新验这一条。**

### 为什么压缩要在索引上做

| | 每条 | T=8, H=2048, fp16 |
|---|---:|---:|
| 索引 | 4 B | 32 B |
| 隐状态 | H×2 = 4096 B | 32,768 B |

**差 1024 倍。** 所以压缩全在索引上做,隐状态只搬一次、**直接落到最终位置**。
"先放进 array 再前移"那种做法要搬两遍数据,贵 2048 倍。

---

## 五、expert_out 是什么,归约到最终输出

### 5.1 它装的是什么

```
expert_out[p, t, :]  =  w · Expert_e(x_t)      e 是 lane p 在该 stage 对应的 expert
```

**每条 lane 对每个 token 的贡献,且已经乘过路由权重。** 它存在的唯一目的是算出

```
MoE(x_t) = Σ_{e ∈ top8(t)}  w_{e,t} · Expert_e(x_t)
```

为什么要这么大一个暂存区?因为**多条 lane 会同时贡献同一个 token**
(每个 token 有 8 个 expert,散在不同 lane、不同卡上),
而静态图里**没有原子加**。只能每条 lane 各存一份,最后求和。
**用 32 MB 空间换掉写冲突。**

### 5.2 三级归约刚好合完 128 个 expert

```
2 stage  ×  16 lane/卡  ×  4 卡  =  128
   ↑            ↑            ↑
 累加器      einsum ⑮      einsum ⑯
```

**第一级**是 `expert_out` 自己的读-改-写。注意它的第一维是 **lane(64)而非 expert(128)**:

```python
expert_out = x.new_zeros((64, T, H))          # 循环外创建一次
for slot in range(2):
    expert_out = cumsum_scatter_gather_update_expert_blocked(..., expert_out=expert_out, ...)
```

```
lane p 在 stage0 是 expert p        ┐
lane p 在 stage1 是 expert p+64     ┘ → 两轮结果累加进同一个 expert_out[p]
```

累加是必须的,不能覆盖。函数内部是读-改-写三步:

```python
expert_out_chunk = gather(expert_out, matched_idx)                   # 读旧值
updated_chunk    = expert_out_chunk + down_chunk                     # 累加本次贡献
...mask...                                                           # 垃圾行清零
expert_out       = scatter(expert_out, matched_idx, updated_chunk)   # ⑭ 写回
```

权重布局印证:`weights.gate` 是 **4 维 `[64, 2, 2048, 768]`** = `[lane, stage, H, I]`
(`validate_expert_parallel_moe_weights` 强制检查),`W_g[:, slot]` 取出 `[64, 2048, 768]`。

**第二、三级**是两个 einsum:

```python
expert_out = einsum("dpth->dth", expert_out.view(4, 16, T, H))   # ⑮ 卡内 16 lane 求和
return       einsum("dth->th",  expert_out)                      # ⑯ 跨 4 卡求和
```

实测 op 计数印证这条链:

```
aicbatchedreduceadd  count=96  out 1024 KB/次 = [128,2048] fp32  → ⑮ 卡内 (48层 × 2 stage)
aicbatchedreduceadd  count=48  out  512 KB/次 = [128,2048] fp16  → ⑯ 跨卡 (48层 × 1)
```

### 5.3 出了 MoE 之后去哪

```
   [T, 2048]                          = MoE(x_t), 8 个 expert 的加权和
      │  block.forward: out.view(B, S, H)
      ↓
Qwen3MoeDecoderLayer.forward:
      residual      = hidden_states                  # attention 之后的值
      hidden_states = post_attention_layernorm(hidden_states)
      hidden_states = self.mlp(hidden_states)        # ← 就是上面那个 [T, 2048]
      hidden_states = residual + hidden_states       # ★ 残差相加
      return hidden_states
      │
      ↓  下一层(重复 48 次)
      ↓
   final norm  →  lm_head [2048, 151936]  →  logits [T, 151936]  →  argmax/sample
```

**所以 `expert_out` 的最终价值就是那个 `residual + hidden_states`** ——
它是这一层 MLP 对隐状态的修正量。MoE 相对普通 MLP 的唯一区别,
是这个修正量由 8 个专家加权合成,而不是一个固定 MLP 算出来。

### 5.4 归约本身也是 16× 浪费,且分档不省

`expert_out` 32 MB,但**任何时刻只有 6.25% 非零** —— 每个 token 只被 8 个 expert 选中,
其余 120 个对它的贡献恒为零,却都占着位置参与求和。

`aicbatchedreduceadd` 实测占 **19%**,就是在把这 93.75% 的零加起来。
而且**分档方案一分不省**(`expert_out` 的中间维是 token 轴,压不掉,见第七节)。

要动它得换归约方式 —— 代码里有 `reduce_nsp_tree`(tree reduction),
但只在 Qwen3-VL-MoE 上启用,我们这条路径 `tree_reduce=False`。
**这是分档之外的另一个独立方向,尚未评估它在这台硬件上是否更省。**

---

## 六、为什么必须 padding

因果链,每一环都成立:

```
硬件没有运行时调度器
  (VTCM 手工管理 / 16 核静态分工 / DMA 描述符预建 / 无分支预测)
   → 形状必须编译期定死
权重按 [64, 2048, 768] 连续存放, 且要合成一次批量 GEMM
   → 64 条 lane 必须打包进「一个」矩形张量
矩形 → 共用宽度 → 取所有 lane 的最大
编译期定死 → 再取所有未来输入的最大
一个 expert 可以吃下全部 T 个 token
   → 宽度 = T = 128
gather 输出宽度 = 索引宽度 (GatherND batch_dims=1)
   → x_chunk = [64, 128, 2048] = 32 MB
```

### 两重最大值叠加

单看任何一重都不至于这么浪费,合起来才是 16×:

```
宽度 ≥ max over 所有 lane      (矩形约束: 上例 lane0 只要 4 行, lane3 要 8 行, 同一张量)
     ≥ max over 所有可能输入    (编译期定死, 要伺候所有未来 prompt)
     =  T
```

### 最坏情况确实可达,但极其罕见

实测 6144 个 (层, expert) 格,历史最大 token 数分布(W=48 窗口,GSM8K):

| 区间 | 格数 | 占比 |
|---|---:|---:|
| 0 | 675 | 11.0% |
| 1–2 | 772 | 12.6% |
| 3–4 | 569 | 9.3% |
| 5–8 | 1127 | 18.3% |
| 9–16 | 1668 | 27.1% |
| 17–32 | 870 | 14.2% |
| 33–48 | 463 | 7.5% |

**达到 T 的只有 2 格(0.03%),中位数是 8。为了 2 个格子,6144 个格子全按 48 开。**
(第 25 层 expert 104 在某条 prompt 拿到 100% 的 token —— 上界确实可达。)

### 什么能剪,什么不能

| 想剪到 | 是什么量 | 可行? |
|---|---|---|
| 本 lane 的实际需求 | 运行时、逐 lane | ✗ 矩形约束 |
| 本次输入所有 lane 的最大 | 运行时 | ✗ 动态形状 |
| **历史统计的分位数** | **编译期常数** | **✓ 分档方案** |
| T(最坏情况) | 编译期常数 | ✓ 现状 |

第二行不可行有直接证据:强制路由实验中 card1/2/3 零个 expert 被选中,
仍精确执行 **3155 次 `aicconvolutiond32`**,与 card0 相同。
**执行次数在编译完成那一刻冻结,运行时没有计数器能让它少走几圈。**

---

## 七、`[64, T, H]` 里的 T 有两种含义(重要)

| 张量 | 中间轴含义 | 分档后 |
|---|---|---|
| `x_chunk` / gate,up / down / mask | **打包后第几行** | 变窄 ✓ |
| `expert_out`(累加器) | **第几个 token** | **不变** ✗ |
| 归约 `einsum` | **第几个 token** | **不变** ✗ |

代码依据:`matched_idx` 装的是 **token 编号**,而

```python
expert_out_chunk = gather(expert_out, matched_idx)           # 用 token 号读
expert_out       = scatter(expert_out, matched_idx, updated)  # 用 token 号写
```

所以 `expert_out` 的中间轴必须能索引任意 token,宽度只能是 T。

另注:`cumsum` **无条件对全部 T 列做**,与该 lane 是否有 token 无关
(上例 lane2 全空也要扫 8 列),所以它**不随实际 token 数缩放**,分档后不受益。

---

## 八、对分档方案的含义

### 改一处,下游全跟着变

`matched_idx` 的宽度由 ⑥ 决定,之后所有张量的形状都是从它推导出来的
(gather 输出宽度 = 索引宽度)。**不用去动 gather,不用去动 GEMM。**

```python
# 现在
matched_idx = INT32_MAX.expand_as(token_idx)             # [64, T]
scatter_pos = where(T2Ei, valid_dest, INT32_MAX)

# 分档 (tier 是编译期常数)
matched_idx = INT32_MAX.expand(n_lanes_t, tier)          # [该档 lane 数, tier]
scatter_pos = where(T2Ei & (valid_dest < tier), valid_dest, INT32_MAX)
#                          └──── 超出档位的 token 在此被丢弃
```

**丢弃机制现成就有**:`scatter_pos = INT32_MAX` 的位置本来就不会被写入,
超容量 token 只要在 `where` 里多一个条件就自动"没被 scatter 进去",
后面 gather 也读不到。**不需要新算子。**

这正是 GShard/Switch 的标准 capacity 做法 —— cumsum 给排位、和容量比、超了就丢。
**QEfficient 等于把容量硬设成了 T(容量因子 = E/K = 16),分档只是把它改成逐格可变。**

### 唯一麻烦:档位不同的 lane 不能放进同一张量

```
64 条 lane → 按 tier 分成 7 组
   组 t: matched_idx_t 形状 [该组 lane 数, t]
         GEMM 跑 [该组 lane 数, t, I] 而不是 [64, T, I]
```

每层从 1 次批量 GEMM 变成 7 次小 GEMM。形状仍全是编译期常数,静态形状没被破坏。
矩形约束还在,但从"1 个 64 宽的矩形"变成"7 个更窄的矩形",总面积 6144 → 918。

**但这个权衡必须实测**:这台硬件上开销大头是搬运和同步(`sync HMX` 94.4%),
7 次小 GEMM 的额外同步开销可能吃掉一部分收益。7 次是可接受的量级(不是 64 次),
但不能靠推理下结论。

### 收益要打折 —— 别把行数比例当加速比

| | 现状 | 分档后 |
|---|---|---|
| GEMM | 3,711 GMAC | ~554 GMAC(省 85%)✓ |
| `expert_out` | 32 MB/stage | 不变 ✗ |
| 归约(实测 19%) | – | 不变 ✗ |
| `cumsum`(实测 7.7%) | – | 不变 ✗ |

对照实测时间分布:

```
aicbatchedreduceadd   19.0%   ← 不受益
aiccopytovtcm         14.2%   ← 部分受益 (x_chunk 32MB → ~5MB)
cumsum                 7.7%   ← 不受益
aicconvolutiond32      4.3%   ← 全额受益
```

**被砍掉的 85% 落在只占 4.3% 的那一项上。**
先前报的 5.17× / 6.7× 是**行数比例,不是时间比例**。
要给可信的时间估计,需按张量维度重新归因 profiling 数据,把各 op 分成
(a) 随打包行数缩放 (b) 随 token 数缩放 (c) 固定开销。**尚未做。**
