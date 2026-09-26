# 最终跨卡求和改为 16 核并行:方法、结果与原设计的对比

日期 2026-09-26。四张 AI 100(每卡 16 核),SDK 1.21.6,Qwen3-30B-A3B layer 2 replay,MXFP6 专家权重,`-mdts-mos=1`,
prompt-41 路由(T=256/512 为合成路由),hot/cold 放置与 oracle 容量(T=128: 92/2,T=256: 158/4,T=512: 296/32)。
起点是原 README 3.13 的最终设计 token-owned combine。所有脚本、graph、profile 在 `/home/chihao/testing`。

## 1. 结论

- token-owned 设计的尾部(卡 1 到 3 结束之后卡 0 还要做的部分,T=512 时 1.23 ms)不是跨卡搬运,而是卡 0 上
  **4 个核串行执行 16 个归约 tile 的计算**(1.03 ms),跨卡搬运(6 MiB,0.53 ms)藏在它下面。
- 把最终求和改写成按 token 行切分的元素级 Add,编译器把它铺到卡 0 的 16 个核上,相加从 1.03 ms 降到 8 µs。
  单层主机时延 T=512 从 5.57 ms 降到 4.91 ms(**−12%**),尾部从 1.23 降到 0.71 ms。
- 相加去掉之后,尾部的下限就是链路:6 / 3 / 1.5 MiB 各需 0.53 / 0.25 / 0.125 ms(T=512/256/128),约 **12 GB/s**,
  严格按字节计费。这是 combine 里唯一剩下的开销,再压缩只能少传字节或换拓扑。
- 副作用:卡 0 的热 stage 慢了约 0.4 ms(T=512),吃掉了约一半的收益。原因指向 6 MiB 的 P2P 落地缓冲静态驻留在卡 0
  的 TCM。四种减小驻留的写法都没有消除它;把热 stage 那份提前传(stagesplit)能消除,但字节翻倍、卡 0 的冷 stage 被拖慢,整体更差。

## 2. 原设计的尾部到底在做什么

token-owned(原 README 3.13)在每张卡本地把每个 token 的 8 行加好之后,剩下的是把四张卡的 [T, 2048] 部分和加成一个。
原图把它写成 16 个 `Einsum('dth->th')` tile(每个 32 行)加 Concat。编译器把每个 Einsum 降成同一模板:

```
1、2、3 号核各加一份(一卡对一核)→ multicast 给 0 号核 → 0 号核再加(两次 reduce-add)
```

16 个 tile 共用这 4 个核和同一个汇合点,所以一个 tile 做完才能做下一个。T=512 的 trace(`profile/K_T512_hc_296_32_tokenowned_mxfp6_s70_flag`):

| 项 | 数值 |
|---|---:|
| 卡 1 到 3 结束 | 3.14 ms |
| 三张卡发出部分和(48 次 P2P,6 MiB) | 3.136 到 3.151 ms 发起 |
| 16 个 tile,每个 5 个算子在 0 到 3 号核 | 3.135 到 4.162 ms,每 tile 73 µs |
| 0 号核输出写回 2 MiB | 4.163 到 4.230 ms |
| 卡 0 结束 | 4.36 ms |

链路给一个 tile 送 384 KiB 只要 33 µs,比 tile 的 73 µs 快,而且 6 MiB 在 3.67 ms 就全部到齐,后面的 tile 都在已到达的数据上跑。
所以这 1.03 ms 是串行计算,不是传输。原 README 3.12 把它描述为"串行 tile 流水线、根在卡 0",3.14 说 fp16 部分和不省时间,与此一致。

## 3. 改法:按 token 行切分的元素级相加

`scripts/make_final_combine.py <token-owned graph> <out> addtree`:

```
p_c = Slice(to_partials, card c)          # 四份 [T, 2048]
out = (p0 + p1) + (p2 + p3)               # 元素级 Add
```

编译器对元素级 Add 按行分核:卡 0 的 16 个核各加自己的 32 行,核之间没有依赖、没有 multicast(清单:elementadd 48 个算子,
卡 0 全部 16 核)。并行轴从"来源卡"(4 → 4 个核)换成了"token 行"(512 → 16 个核),和每卡本地归约(绿色)用的是同一种写法。

试过的其他写法:`Sum` 算子和 `Transpose + ReduceSum` 都被编译器退回 4 核 `aicbatchedreduceadd` 加一次 DDR 归约,无效。

## 4. 结果

### 4.1 时延(T=512,同会话,3 轮 × 100,主机中位)

| 版本 | 主机 ms | 相对 token-owned | device ms(3 样本) | 卡 0 尾部 |
|---|---:|---:|---|---:|
| anchor,dense combine(3.8) | 9.735 | | | |
| token-owned(3.13) | 5.573 | – | 4.38 / 4.36 / 4.34 | 1.23 ms |
| **addtree(本文)** | **4.912** | **−12%** | 4.38 / 4.29 / 4.11 | 0.71 ms |
| tileadd,16 段各自元素级相加 | 5.140 | −8% | 4.31 / 4.15 / 4.21 | 0.67 ms |
| addtree + fp16 部分和 | 5.192 | −7% | 4.35 / 4.32 / 4.01 | 0.71 ms |

不带探针、SDK runner 连跑 40 次的每次总时长:token-owned 5.90 ms,addtree 5.39,tileadd 5.22,和主机计时一致。
输出与 anchor 的相对 L2 为 3.5e-4(fp16 元素级相加的结合顺序),token-owned 为 1.3e-5。

### 4.2 尾部分解(device,中位样本)

| T | 部分和字节 | 到齐前空档(链路) | 相加 | 输出写回 | 尾部:token-owned → addtree |
|---:|---:|---:|---:|---:|---|
| 128 | 1.5 MiB | 0.125 ms | 8 µs | | 0.30 → 0.16 ms |
| 256 | 3.0 MiB | 0.25 ms | 8 µs | | 0.58 → 0.32 ms |
| 512 | 6.0 MiB | 0.53 ms | 8 µs | 0.17 ms | 1.21 → 0.71 ms |

空档三个样本各自精确到 0.01 ms,与字节数严格成正比,约 12 GB/s。trace 里的 P2P 事件(12 到 15 µs)记录的是发起时刻,
到达时间只能从接收端第一个消费算子反推。

### 4.3 线段图:改前(上)与改后(下),T=512

![token-owned vs addtree](figures/T512_tokenowned_vs_addtree.png)

读图(横轴 ms,四个子图是四张卡,每行一个核;蓝/橙热冷 GEMM,紫权重 DMA,棕解量化,粉 gather 与 index build,绿本地归约,红最终求和,灰等待):

- 上图卡 0 右端:0 到 3 号核上四条红色长条,3.15 到 4.16 ms,其余 12 核灰色等待。
- 下图卡 0 右端:16 行各一个红点(4.11 ms,8 µs)。红点左边 3.59 到 4.11 ms 四张卡全灰,是 6 MiB 在链路上的时间。
- 下图卡 0 的热 stage(蓝)比上图长约 0.4 ms(1.99 → 2.42),其他三张卡不变;因此粉色的 index build 和橙色冷 stage 整体右移 0.45 ms,
  卡 1 到 3 的结束从 3.14 变成 3.58。

单张图:`figures/T512_tokenowned_cores.png`、`figures/T512_fc_addtree_cores.png`、`figures/T512_fc_tileadd_cores.png`。

## 5. 副作用:卡 0 热 stage 变慢

| 观察 | 数据 |
|---|---|
| 算子结构不变 | 卡 0 每核 36 个 GEMM、36 个解量化、36 次权重 DMA,字节相同 |
| 每个算子均匀变慢 | GEMM 每次 21.7 → 49.8 µs,解量化 12.2 → 21.6 µs;HMX 等权重的时间翻倍;卡 1 到 3 不变 |
| 随 T 增长 | 热 stage 结束时间差:T=128 约 0,T=256 约 +0.1 ms,T=512 +0.3 到 +0.5 ms;对应部分和 1.5 / 3 / 6 MiB |

为消除它试的变体(T=512,device,中位样本;`scripts/fc_compare.py`):

| 变体 | device | 卡 0 热 stage 结束 | 卡 0 尾部 | 结果 |
|---|---:|---:|---:|---|
| addtree | 4.29 | 2.42 | 0.71 | 基准 |
| rev,(p3+p2)+(p1+p0) | 4.25 | 2.37 | 0.70 | 编译器仍把相加放在卡 0,无效 |
| tileadd 32 段 | 4.39 | 2.56 | 0.66 | 无效 |
| half,分两半 | 4.46 | 2.57 | 0.71 | 无效 |
| stagesplit,热 stage 部分和提前传 | 4.50 | **2.06** | 0.71 | 热 stage 恢复,但跨卡 12 MiB,卡 0 冷 stage 被推迟到 3.27 ms,整体更差 |

解释:P2P 落地缓冲是静态分配的,发送方直接写卡 0 的 TCM 地址,这块区域整层期间都保留。原设计 16 个 tile 串行,只需一个 384 KiB 的
落地区复用 16 次;元素级写法让所有段同时可执行,6 MiB 全部驻留,每核少了约 384 KiB 的 TCM,热 stage 的权重预取深度下降。
切 32 段或分两半不改变"全部同时落地",所以无效;stagesplit 要落地 12 MiB,超出编译器放进 TCM 的量,改走 DDR,热 stage 恢复,
代价换成冷 stage 期间的 DDR 争用。这是从 trace 反推的,编译器的分配表看不到。
可验证的修法:16 段元素级相加但段间加数据依赖,让编译器回到"一个落地缓冲复用 16 次",同时保留 16 核并行;预期热 stage 回到
1.99 ms、尾部不变,整层收益从 −12% 到约 −20%。

## 6. 现在还剩什么

改后 T=512 一层(device 约 4.2 ms)的构成:热 stage 约 2.1 ms(含副作用 0.4),stage 间索引交换约 0.5,冷 stage 约 0.5,
每卡本地归约 0.05,**跨卡搬运 0.53**,相加 0.01,输出写回 0.17。跨卡搬运是 combine 里唯一剩下的项,占整层 12%,只能靠:

1. 少传字节:只传该卡上有 expert 参与的 token 行。现在的放置下每个 token 平均碰 3.8 张卡,只能省 7%(sorted 布局 39%,但它在 flag
   下慢 0.7 ms),必须和放置一起设计,并与 balance 对冲。
2. 重叠:热 stage 那份提前传。stagesplit 表明接收方要付带宽代价,除非冷 stage 那份是稀疏的、字节很少。
3. 换拓扑:四张卡各收四分之一的 token(reduce-scatter),每卡 1.5 MiB、约 0.13 ms。原 README 3.14 和本轮都被编译器折叠回卡 0,需编译器支持。

## 7. 复现

```
scripts/make_final_combine.py combine/T512_hc_296_32_tokenowned combine/T512_to_fc_addtree addtree     # 或 tileadd [段数] | rev | half | stagesplit | sum | treduce
scripts/compile_any.sh combine/T512_to_fc_addtree F_T512_fc_addtree_mxfp6_s0_flag mxfp6 0 flag          # s70 为带探针
scripts/timing.py timing/S11b_fc_spec.json timing/S11b_fc.json 3 100                                    # 同会话计时
scripts/profile_case.sh qpc/F_T512_fc_addtree_mxfp6_s70_flag profile/F_T512_fc_addtree_mxfp6_s70_flag F_T512_fc_addtree 512 combine/T512_to_fc_addtree/input_f16.bin
plotvenv/bin/python scripts/detail_plot2.py profile/F_T512_fc_addtree_mxfp6_s70_flag/analysis/sample1 "title" figures/x.png 0,1,2,3
scripts/fc_compare.py                                                                                   # 变体对比表
```

数据:`timing/S11b_fc*.json`(计时),`profile/F_T*_fc_*`(trace、每核 CSV、分析),`inventory/F_T512_fc_*`(编译期算子清单),
`combine/T*_to_fc_*`(graph)。
