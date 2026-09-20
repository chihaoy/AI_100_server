# EP 单层数据流 —— 逐步形状与代价

Qwen3-30B-A3B / 4×Cloud AI 100 / QEfficient 1.23。
T=128(静态 seq_len), H=2048, I=768, E=128, top_k=8, lane P=64, stage S=2, card D=4。

下表是**一层、一个 stage**。`for slot in range(2)` 是 Python 循环,导出时完全展开,
所以每层实际跑两遍 ⑤–⑭。

最后一列的硬件 op 来自实测 `ep_prefill/out/*.qaic-opstats.summary.txt`(card0 c0 核)。

| | 步骤 | 输入 | 输出 | MAC | 张量 | padding? | 对应硬件 op |
|---|---|---|---|---:|---:|---|---|
| ① | router: `x @ gate.W^T` | `[128,2048] @ [2048,128]` | `[128,128]` | 33.6 M | 32 KB | 否 | `aicconvolutiond32` |
| ② | `softmax` + `topk(8)` | `[128,128]` | 2×`[128,8]` | – | 8 KB | 否 | **`aictopk`** count=48 = 48层×1 |
| ③ | `densify_topk`(scatter) | 2×`[128,8]` | `[128,128]` | – | 32 KB | 否 | – |
| ④ | `view/transpose` → lane 布局 | `[128,128]` | `[64,2,128]` | – | 32 KB | 否 | (编译期消解) |
| ⑤ | `T2Ei = rw[:,slot,:] > 0` | `[64,2,128]` | `[64,128]` bool | – | 8 KB | 否(掩码) | – |
| ⑥ | **`cumsum` → `matched_idx`** | `[64,128]` | `[64,128]` int32 | – | 32 KB | **定形 ←** | **`cumsum`** count=96 = 48×2<br>out 32 KB/次 = 本张量 |
| ⑦ | **`CtxGather3D` → `x_chunk`** | `[64,128,2048]` gather | `[64,128,2048]` | – | 32.0 MB | **是 ←** | 搬运计入 `aiccopytovtcm` |
| ⑧ | GEMM1 `gate = x @ W_g` | `[64,128,2048] @ [64,2048,768]` | `[64,128,768]` | 12.88 G | 12.0 MB | 是 | `aicconvolutiond32` (HMX) |
| ⑨ | GEMM2 `up = x @ W_u` | 同上 | `[64,128,768]` | 12.88 G | 12.0 MB | 是 | 同上 |
| ⑩ | `h = up * silu(gate)` | `[64,128,768]` | `[64,128,768]` | – | 12.0 MB | 是 | `elementmul` count=480 = 48×10 |
| ⑪ | GEMM3 `down = h @ W_d` | `[64,128,768] @ [64,768,2048]` | `[64,128,2048]` | 12.88 G | 32.0 MB | 是 | `aicconvolutiond32` (HMX) |
| ⑫ | 乘路由权重 + 累加 | `[64,128,2048]` | `[64,128,2048]` | – | 32.0 MB | 是 | `elementmul` |
| ⑬ | **mask `row<valid_rows`→0** | `[64,128,2048]` | `[64,128,2048]` | – | 32.0 MB | **垃圾行在此清零** | – |
| ⑭ | `CtxScatter3D` 写回 | `[64,128,2048]` | `[64,128,2048]` | – | 32.0 MB | 是 | – |
| | **每 stage 合计** | | | **38.69 G** | | | |
| ⑮ | 卡内归约 `einsum("dpth->dth")` | `[4,16,128,2048]` | `[4,128,2048]` | – | | | `aicbatchedreduceadd` count=96=48×2<br>out 1024 KB/次 = `[128,2048]` fp32 |
| ⑯ | 跨卡归约 `einsum("dth->th")` | `[4,128,2048]` | `[128,2048]` | – | | | `aicbatchedreduceadd` count=48=48×1<br>out 512 KB/次 = `[128,2048]` fp16 |

## 账

```
每 stage       38.69 GMAC
两 stage       77.38 GMAC
48 层       3,714.07 GMAC
其中有用          6.25%   = T*K/S / (P*T) = 512 / 8192
浪费        3,481.94 GMAC   <- 一次 128-token prefill 白算的量
```

## ⑯ 之后:出了 MoE 去哪

```
   [128, 2048]                        = MoE(x_t) = Σ_{e∈top8(t)} w · Expert_e(x_t)
      │  block.forward: out.view(B, S, H)
      ↓
Qwen3MoeDecoderLayer.forward:
      residual      = hidden_states                  # attention 之后的值
      hidden_states = post_attention_layernorm(hidden_states)
      hidden_states = self.mlp(hidden_states)        # ← 就是上面那个 [128, 2048]
      hidden_states = residual + hidden_states       # ★ 残差相加
      return hidden_states
      │
      ↓  下一层(重复 48 次)
      ↓
   final norm → lm_head [2048, 151936] → logits [128, 151936] → argmax/sample
```

**整条 EP 流水的最终产物,就是那个 `residual + hidden_states` 的修正量。**
MoE 相对普通 MLP 的唯一区别,是这个修正量由 8 个专家加权合成。

⑭ 写回的 `expert_out [64,128,2048]` 是**跨 stage 复用的累加器**,第一维是
**lane(64)而非 expert(128)**:lane p 在 stage0 是 expert p、stage1 是 expert p+64,
两轮累加进同一格。加上 ⑮⑯ 两级 einsum,`2 stage × 16 lane/卡 × 4 卡 = 128`,
刚好合完全部 expert。

> `expert_out` 32 MB 但任何时刻只有 6.25% 非零 —— `aicbatchedreduceadd` 那 19%
> 就是在把 93.75% 的零加起来,**且分档方案一分不省**(中间维是 token 轴)。
> 详见 `EP_MECHANISM.md` 第五节。

---

## 读这张表的四个要点

### 1. 需要 padding 矩阵的是 ⑧⑨⑪ 三个 GEMM

就是 `profiles.py` 里 `silu_glu_mlp` 的三行:

```python
def silu_glu_mlp(x, W_g, W_u, W_d, ...):
    gate = x @ W_g                      # ⑧
    up   = x @ W_u                      # ⑨
    return (up * F.silu(gate)) @ W_d    # ⑪
```

这里没有任何"跳过空行"的机制。`[64,128,2048] @ [64,2048,768]` 是一次完整的
批量矩阵乘,64 条 lane × 128 行全算,包括 41% 完全没有 token 的 lane。

### 2. 分界在 ⑥/⑦ 之间

**①–⑥ 全是索引和掩码**,合计 144 KB。⑥ 只是**定形**(32 KB)。
**⑦ 是分水岭** —— gather 把隐状态搬进 `[64,128,2048]`,padding 在这一刻
从"一个宽度常数"变成 **32 MB 实体张量**。

依赖方向是 ⑥ → ⑦,不是反过来:`GatherND(batch_dims=1)` 的输出宽度
等于索引宽度。**所以改 ⑥ 一处,⑦ 及之后全部自动跟着变窄。**

### 3. ⑦⑫⑬⑭ 在 32 MB 张量上来回搬,一个乘法都不做

四步合计 128 MB 的搬运,零 MAC。这与实测 op 分布吻合:

```
sync HMX              94.4%   <- 矩阵引擎在等
aicbatchedreduceadd   19.0%   <- ⑮⑯ 归约
aiccopytovtcm         14.2%   <- ⑦⑭ 搬运
cumsum                 7.7%   <- ⑥
blockdequantize_mxfp6  7.8%   <- 权重解量化
aicconvolutiond32      4.3%   <- ⑧⑨⑪ 真正的矩阵乘
```

**这台硬件上 padding 更贵的不是 ⑧⑨⑪ 的 3.48 TMAC,而是 ⑦⑫⑬⑭ 反复搬那 128 MB。**

### 4. `[64, T, H]` 里的 T 有两种含义

| 张量 | 中间轴 | 分档后 |
|---|---|---|
| ⑦ `x_chunk` / ⑧⑨ gate,up / ⑪ down / ⑬ mask | **打包后第几行** | 变窄 ✓ |
| `expert_out`(累加器) | **第几个 token** | 不变 ✗ |
| ⑮⑯ 归约 | **第几个 token** | 不变 ✗ |

代码依据:`matched_idx` 装的是 token 编号,`expert_out` 用 token 号寻址
(`gather(expert_out, matched_idx)` / `scatter(expert_out, matched_idx, ...)`),
所以它的中间轴必须能索引任意 token,宽度只能是 T。

另:`cumsum` ⑥ 无条件扫全部 T 列(空 lane 也扫),**不随实际 token 数缩放**,分档后不受益。

**推论:先前报的 5.17x / 6.7x 是行数比例,不是时间比例。**
被分档砍掉的 85% 落在只占 4.3% 的 `aicconvolutiond32` 上。
可信的时间估计需按张量维度重新归因 profiling 数据,把各 op 分成
(a) 随打包行数缩放 (b) 随 token 数缩放 (c) 固定开销。**尚未做。**

---

机制细节(matched_idx 怎么建、为什么必须 padding)见 `EP_MECHANISM.md`。
