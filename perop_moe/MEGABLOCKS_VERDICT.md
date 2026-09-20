# MegaBlocks 能否在 QEfficient / Cloud AI 100 上实现 —— 前置检查报告

日期:2026-09-08
模型:Qwen3-30B-A3B (128 experts, top-8, 48 层, hidden 2048, moe_intermediate 768)
软件:QEfficient 1.23.0.dev0 + Qualcomm AIC SDK 1.21.6
硬件:4 x Qualcomm Cloud AI 100 (每卡 16 核)
论文:MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (arXiv 2211.15841)

背景:教授在会上要求"看一看那个 MegaBlocks 能不能写",并预判"大概率不太行"。
本文是这项前置检查的结论与证据。

---

## 结论

**不行。** 而且理由比"接口写不出来"更强,分三层:

1. **写不出来** —— 三个硬阻塞,见第一节。
2. **就算写出来也不省** —— MegaBlocks 的块粒度(128)比本模型每个 expert 的实际
   工作量(平均 7.1 行)大一个数量级,实测只有 1.5-1.6x,见第二节。
3. **就算省了也是解错问题** —— MegaBlocks 的前提是 padding 浪费算力;
   本硬件上算力只占 4.3%,浪费实际发生在数据搬运和同步上,见第三节。

第三点反而是这个课题真正的价值所在,见第四节。

---

## 一、三个硬阻塞

### ① QEfficient 的"自定义算子"不是硬件 kernel,只是标准 ONNX 算子的组合

`QEfficient/customop/onnxscript_utils.py:50` 的 `qeff_custom_op` 装饰器,
只是把一个 onnxscript 函数编译成 legacy / dynamo 两个变体:

```python
def qeff_custom_op(domain: str, version: int):
    """Compile one custom op body into legacy and dynamo ONNXScript variants."""
    custom_opset = onnxscript.values.Opset(domain, version)
    def decorator(fn):
        legacy_func = _compile_with_default_opset(fn, custom_opset, ONNX_LEGACY_EXPORT_OPSET)
        dynamo_func = _compile_with_default_opset(fn, custom_opset, ONNX_DYNAMO_EXPORT_OPSET)
        setattr(legacy_func, _DYNAMO_FUNC_ATTR, dynamo_func)
        return legacy_func
    return decorator
```

被它包装的算子体全部是标准 ONNX 算子。例如 MoE 路径上最"自定义"的
`CtxGather3D`(`QEfficient/customop/ctx_scatter_gather.py`):

```python
def CtxGather3D(data, ctx_indices):
    batch_size  = ops.Slice(ops.Shape(data), starts=[0], ends=[1], axes=[0])
    idx_seq_len = ops.Slice(ops.Shape(ctx_indices), starts=[1], ends=[2], axes=[0])
    expand_shape = ops.Concat(batch_size, idx_seq_len, axis=0)
    ctx_indices = ops.Expand(ctx_indices, expand_shape)
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=1)
```

`Slice + Shape + Concat + Expand + Unsqueeze + GatherND`,全部标准。

**含义**:在 QEfficient 这一层,你没有"写一个新 kernel"的能力,只能用 ONNX
已有算子拼装。而 MegaBlocks 的全部价值就在它手写的 SDD / DSD 块稀疏 CUDA kernel。
这一层递不下去。

复现:
```bash
SP=/home/chihao/qeff-venv/lib/python3.10/site-packages
sed -n '/def qeff_custom_op/,/return decorator/p' $SP/QEfficient/customop/onnxscript_utils.py
sed -n '/def CtxGather3D/,/^$/p'                  $SP/QEfficient/customop/ctx_scatter_gather.py
```

### ② SDK 里没有块稀疏 GEMM 算子

编译器 `libQAicCompiler.so` 中全部 **126** 个 `aic*` 算子,矩阵乘只有两个,都是稠密的:

```
aicmatmul
aicbatchmatmul
```

稀疏相关的算子只有以下几个,没有一个能做块稀疏矩阵乘:

| 算子 | 实际用途 |
|---|---|
| `SparseLengthsSum` / `SparseLengthsWeightedSum` | DLRM 推荐系统 embedding-bag |
| `SparseToDense` / `SparseToDenseMask` | 稀疏索引转稠密张量 |
| `aicfusedrowwisequantizedsparselengthsweightedsum` | 同上,量化版 |
| `aicsparseconvolution` | 空间稀疏卷积(点云类),Glow `lib/Backends/AIC/internal/kernels/sparseconv.cpp` |

前几个是推荐系统的 embedding 查表算子(按 index 取行再求和),
最后一个是空间稀疏卷积。**都不是 `[T, E*I]` 形状的块稀疏矩阵乘。**

也**没有 2:4 结构化稀疏**。搜索 "prune" 命中的全部是 LLVM 编译器 pass
(`prune-eh`、`pipeliner-prune-deps`、`pipeliner-prune-loop-carried`),与权重剪枝无关。

复现:
```bash
SO=/opt/qti-aic/dev/lib/x86_64/libQAicCompiler.so
strings $SO | grep -oE "^aic[a-z0-9]+" | sort -u | wc -l          # -> 126
strings $SO | grep -oE "^aic[a-z0-9]+" | sort -u | grep -iE "matmul|sparse|gemm"
strings $SO | grep -xiE "sparse[a-z]*|blocksparse[a-z]*" | sort -u
strings $SO | grep -iE "2:4|structured.?sparsit" | sort -u        # -> 空
```

### ③ 静态形状与运行时构造的稀疏元数据冲突

MegaBlocks 的 `topology = make_topology(indices)` 在**运行时**由 routing 结果
构造 BCSR 块结构,非零块的数量随输入数据变化。论文还明确要求
"allowing for the blocks in the block diagonal matrix to have a **variable number of rows**"。

QEfficient 走的是 AOT(提前编译)路径,所有形状在编译期固定。SDK 的
`-dynamic-shape-input` 只作用于**图的输入**,没有动态形状的 MatMul kernel,
而且它与 QEfficient 始终传入的 `-network-specialization-config` 互斥。

**含义**:块数可变这一条,AOT 编译根本无法表达。

#### 关键分界:数据决定"值"可以,数据决定"形状"不行

QEfficient **已经在用运行时决定的索引** —— `CtxGather3D` 的 `ctx_indices`
由 `cumsum` 在运行时算出,编译器完全不知道里面装什么值。这没问题。

它做不了的是**运行时决定的形状**:`ctx_indices` 的 shape 恒为 `[B, T]`,只有内容变。
AOT 编译器必须在编译期定下 DDR/VTCM 缓冲大小、DMA 次数、16 核分工、循环上界;
BCSR 的 `indices`/`indptr` 长度随 routing 变化,这些全都定不下来。

这正解释了 QEfficient 为何补齐到 T:它主动把变化从"形状"挤进"值"
(形状锁死 `[E, T, I]`,越界填 `INT32_MAX`,算完再 mask)。分档 padding 方案可行
也是同一个道理 —— 几种形状在编译期定死,运行时只选图。

#### "按最坏情况静态分配"救不回来

自然的补救是给 BCSR 按上界静态开缓冲。上界为 `T*K + E*(B-1)`
(最坏情况:E 个 expert 每个都有一个只装 1 行的残块):

| 块大小 B | 最坏界 | vs 当前 16384 | 实测 p99 (GSM8K) | p99 静态分配 |
|---:|---:|---:|---:|---:|
| **128(论文所用)** | **17280** | **0.95x** | 14080 | 1.16x |
| 64 | 9088 | 1.80x | - | - |
| 32 | 4992 | 3.28x | - | - |
| **16** | **2944** | **5.57x** | 1888 | 8.68x |
| 8 | 1920 | 8.53x | - | - |

**B=128 静态化后是负收益**(17280 > 16384):`E*(B-1) = 16256` 这一项
直接压垮了 `T*K = 1024`。放宽到实测 p99 也只有 1.16x。

而 **B=16 的最坏界 2944,与分档 padding 方案实测的 2900 行几乎相同** ——
这不是巧合:静态化之后的 MegaBlocks 就退化成了分档补齐,本来就是同一个想法。

**因此**:MegaBlocks 中唯一能过静态形状这一关的,就是"补齐到小块的倍数",
而这部分我们已经有了。它真正独有的东西 —— 运行时 BCSR、可变块高、不丢 token ——
全部依赖动态形状,一个都过不来。

---

## 二、就算能写,MegaBlocks 的块大小在本模型形状上也几乎不省

用实测路由数据做块大小 sweep(GSM8K / HumanEval 各 100 prompts x 48 层,
数据由 `tools/moe_collect_routing.py` 在 CPU 上采集,未做 padding):

对照基线是 QEfficient `EXPERT_PARALLEL` 的实际行为:每层执行
`E x T_pad = 128 x 128 = 16384` 行,**与路由无关**。

| 块大小 | GSM8K 行/层 | vs 当前 | 利用率 | HumanEval 行/层 | vs 当前 | 利用率 |
|---|---:|---:|---:|---:|---:|---:|
| 当前 EP | 16384 | 1.00x | 3.4% | 16384 | 1.00x | 4.6% |
| **MB-128(论文所用)** | 10094 | **1.62x** | 5.5% | 10954 | **1.50x** | 6.9% |
| MB-64 | 5064 | 3.24x | 11.0% | 5515 | 2.97x | 13.7% |
| MB-32 | 2604 | 6.29x | 21.5% | 2871 | 5.71x | 26.4% |
| MB-16 | 1427 | 11.48x | 39.2% | 1656 | 9.89x | 45.8% |
| MB-8 | 917 | 17.86x | 60.9% | 1137 | 14.42x | 66.7% |
| MB-4 | 701 | 23.37x | 79.7% | 908 | 18.05x | 83.5% |
| 理想(无补齐) | 559 | 29.32x | 100% | 758 | 21.61x | 100% |

**MegaBlocks 自己的 128x128 只给 1.5-1.6x**,比我们已经量化过的
4 档静态 padding 方案(0/16/64/128 行,5.6x,丢 token 1.02%)还差。

原因很直接:

- 128 是 NVIDIA tensor core 的 tile 尺寸,MegaBlocks 的场景是**训练 / 大 batch**,
  每个 expert 动辄上千行。
- 本配置 T=128、top-8、128 experts,实测**每层平均只有 78.9 个 expert 活跃**,
  有效行数 559,即**平均每个活跃 expert 只有 7.1 行**(HumanEval 8.9 行)。
- 用 128 行的块去装 7 行,补齐问题只是从"补到 T"变成"补到 128",没有解决。

要让 MB-128 真正有效,需要每 expert >= 128 行,即
`T x top_k / E >= 128` => **T >= 2048**。这正好对应教授提到的"大 batch"场景 ——
也说明 MegaBlocks 的适用区间与我们当前测量的区间(batch=1,T=128)不重叠。

**唯一可迁移的**是"补齐到小块的倍数,而不是补齐到 T"这个思想,
而那基本就是分档 padding 方案的推广(MB-16 = 11.5x 与分档方案同量级)。

复现:
```bash
python3 tools/moe_blocksize_sweep.py \
    perop_moe/routing/routing_gsm8k.npz perop_moe/routing/routing_humaneval.npz
```

---

## 三、更根本的问题:本硬件上 padding 的代价不在"算"

> 更正:先前记录的"`blockdequantize_mxfp6` 占 55-70%"是错的,实测为 7.1-7.8%。

EP prefill 实测,card0 c0 核的时间去向
(`ep_prefill/out/aic-profiling-program-19-inf-2-QAicGraph_slice00-*.summary.txt`):

```
sync HMX              94.4%   <- 矩阵引擎在等
aicbatchedreduceadd   19.0%   <- 归约
aiccopytovtcm         14.2%   <- 搬进 VTCM
cumsum                 7.7%   <- 打包索引,且只在 core0 上跑
blockdequantize_mxfp6  7.8%   <- 权重解量化(不随行数缩放)
aicconvolutiond32      4.3%   <- 真正的矩阵乘
```

(各引擎 HMX / HVX / DMA 并发,百分比按引擎分别计算,故不求和为 100%。)

**真正的矩阵乘只占 4.3%,而矩阵引擎 94.4% 的时间在 sync 等待。**

这意味着:把执行行数砍 11x,直接受益的只有那 4.3% 加归约的一部分。
MegaBlocks 的整个立论前提 —— "padding 浪费的是 FLOPs" —— 在这台硬件上不成立。

这里浪费的是:
- **数据搬运**:`aiccopytovtcm` 14.2%,DMA 侧 `aicmulticastvtcm_in` 占 36.6%
- **同步**:`sync HMX` 94.4%
- **串行**:`cumsum` 只在 core0 上执行(其余 15 核精确为 0)

---

## 四、这个结论对课题的意义

教授给的判据是"要找硬件特有的 challenge,而不是重复 GPU 上做过的事"。
这次前置检查恰好给出了一个:

| | GPU (MegaBlocks 的前提) | Cloud AI 100 (实测) |
|---|---|---|
| padding 浪费什么 | 算力(FLOPs) | 数据搬运 + 同步 |
| 矩阵乘占比 | 主导 | 4.3% |
| 能否运行时构造稀疏结构 | 能(动态 kernel) | 不能(AOT 静态形状) |
| 有效块粒度 | 128(tensor core tile) | 需要 8-32,且受 d32 布局约束 |
| 适用 token 规模 | T >= 2048(训练) | 当前 T=128,batch=1 |

**建议的方向调整**:路线 ① (profile-guided 静态优化) 的目标函数,
应当从"减少浮点运算"改写为"**减少 VTCM 搬运量与同步次数**"。
这是 GPU MoE 文献里没有人做的题,而且是被这台硬件的成本结构逼出来的。

---

## 五、附:证据文件清单

| 内容 | 位置 |
|---|---|
| 块大小 sweep 脚本 | `tools/moe_blocksize_sweep.py` |
| 路由原始数据 (per-token, 无 padding) | `perop_moe/routing/routing_{gsm8k,humaneval}.npz` |
| 路由采集脚本 | `tools/moe_collect_routing.py` |
| EP prefill 逐核 op 统计 | `perop_moe/ep_prefill/out/*.qaic-opstats.summary.txt` |
| EP prefill 逐核归并表 | `perop_moe/ep_prefill/percore_ep.csv` |
| expert 四卡均分的证据包 | `perop_moe/proof_expert_split/` |
| 完整实验记录 | `perop_moe/README.md` |
| 会议方向记录(本文为其第九节的独立版) | `perop_moe/ADVISOR_DIRECTION.md` |
