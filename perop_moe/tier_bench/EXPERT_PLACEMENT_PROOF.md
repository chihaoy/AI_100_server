# Expert 到卡、到核的放置规则:推导过程与证据
> **2026-09-22:** `/home/chihao/models/qwen3_30b_a3b/{ep,4card}` 与三个 `qeff*_cache` 已删除以腾磁盘(共 518 GB)。原始权重 `hf/` 保留;重建脚本、spec、custom_io 和日志见 [`build_recipes/`](../build_recipes/README.md)。

日期 2026-09-09。硬件 Qualcomm Cloud AI 100 × 4 卡(`/dev/accel0..3`),每卡 16 核。
**所有结论均来自卡上实测**(编译产物、qaic-runner 输出张量、qaic profiler trace),
不是读源码推出来的。读源码的部分只用于解释"为什么",单独标出。

本文可独立发出。每一条结论后面跟着:实验设计 → 原始数据(文件路径 + 数字)→ 从数据到结论的一步推导 → 证据等级。

---

## 0. 结论

对 QEfficient 的 EP(expert-parallel)图,64 条 lane(每条 lane = 一个 expert 的 gate/up/down 三个矩阵)在 4 卡 × 16 核上的放置:

| 层级 | 规则 | 证据等级 |
|---|---|---|
| lane → 卡 | lane p 在卡 **p // 16**(连续块 0–15 / 16–31 / 32–47 / 48–63) | **数值直读**(§2) |
| 每核持有几个 expert | **恰好 1 个整 expert**,不切列、不与别的 expert 共享 | **判决实验直读**(§4) |
| 卡内 lane → 核 | lane p 在核 **p % 16** | 单卡 lane 0–7 直读(§3)+ 整块连续规则(§4)推出 |
| 编译器是否调度节点 | **否**。只切张量;独立节点各自摊满 16 核并串行 | **直读**(§5) |

一句话:**expert 放在哪个核,由它在 batched 权重张量 `[64, H, I]` 首维上的位置决定;编译器把首维整块等分给核,不做任何"哪个 expert 去哪个核"的决策。**

### "一核一 expert"的两个限定

直接回答"最终是不是一个核上一个 expert":**在默认配置(每卡 16 核、每卡 16 条 lane)下是**,但必须带两个限定,否则会误读:

1. **这是巧合,不是编译器的设计意图。** 编译器没有"expert"这个概念,它只把 batched 张量的首维整块等分给核。
   lane 数恰好等于核数,才落成一核一 expert。核数一变就不成立:12 核时 gate/up 变成 8 个核各 2 个 expert、4 个核空着(§4)。
   反过来,如果 expert 身份不在同一张量的首维上(拆成独立节点),一核一 expert 立刻消失,每个 expert 摊满 16 核(§5)。

2. **只对 gate / up 严格成立;down 的分配粒度是 1/4 个 expert。** 12 核时 down 是 11 核各 1.25 个 expert 的列块、core0 2.25 个(§4)。
   16 核时 16 lane × 4 块 = 64 块,64 / 16 = 每核 4 块 = 恰好 1 个整 expert,所以看起来也是一核一 expert,但机制是"列块整除",与 gate/up 的"整 expert 不可分"不同。

对课题真正有用的是后半句:放置由权重张量的排列决定,改放置只能在 Python 打包时置换 expert 顺序;
"两个 expert 共用一个核"或"按负载给不同核分不同数量的 expert",这个编译器表达不出来(§7)。

---

## 1. 被测对象与公共设置

单层 MoE 微基准(无 attention、无 KV、随机权重),形状与 Qwen3-30B-A3B 一致:

```
T=128  H=2048  I=768  E=128  K=8  S=2(stage)  P=64(lane)  num_devices=4
```
(`u128_c12/info.json`、`probe_partial/info.json`)

图结构与 QEfficient EP 一致:128 个 expert 打包成 2 组 `[64, 2048, 768]` 张量,首维是 lane 而不是 expert。
每个 MatMul 节点是一个 batched GEMM,`/MatMul_1,2,3` = stage 0 的 gate/up/down,`/MatMul_4,5,6` = stage 1。

编译:`qaic-compile -aic-hw-version=ai100 -convert-to-fp16 [-mxfp6-matmul] -aic-num-cores=$CORES -mos=1 -aic-enable-depth-first -mdp-load-partition-config=mdp.json`
(`tools/moe_tier_bench.sh:47`、`tools/moe_probe_placement.sh:19`)

4 卡 MDP:`mdp.json` 一个 partition,4 个 device,p2p 连接(`probe_partial/mdp.json`)。

运行:`qaic-runner -t <qpc> -D 0:1:2:3 -i x.bin --write-output-dir ...`,profiling 开启后 `out/` 下得到每卡的 `*.qaic-opstats.trace.json`,
trace 事件按线程名 `QAicGraph_slice<card>_Core_<core>` 归到卡和核。

基线延迟(`run.log` 的 `ExecTimeUs`,均值):

| 变体 | 卡数 | 核数 | 说明 | 延迟 |
|---|---:|---:|---|---:|
| `c1_u128` | 1 | 16 | batched,单卡跑一张卡的份额(16 lane × 2 stage) | 3.26 ms |
| `u128` | 4 | 16 | batched,完整 EP | 9.91 ms(card0) |

一个 expert 的一个矩阵(gate / up / down 任一)fp16 大小:`2048 × 768 × 2 B = 3072 KB`。这个数是后面所有核级推断的尺子。

---

## 2. lane → 卡:每卡 partial sum 的数值回归(`probe_partial`)

### 设计
图与 `u128` 完全相同,只把**跨卡归约前**的 `[4, T, H]` 每卡 partial 直接作为输出(`emit_partial: true`)。fp16 编译(不量化,消除数值噪声)。
CPU 上算出 64 条 lane 各自的贡献 `y_lane[p]`(`y_lane.npy`,64 MB)。因为权重随机,64 个贡献近似正交,可以对每个输出 slice d 做最小二乘:

```
partial[d] ≈ Σ_p c_{d,p} · y_lane[p]
```

**不预设任何假设**——不假定连续、不假定交错——让 64 个系数自己说话。

### 原始数据
- 卡上输出:`probe_partial/outdir_fp16/partial-activation-0-inf-0.bin`,2,097,152 B = 4 × 128 × 2048 × 2 B ✓,由 `run_fp16.log` 中 qaic-runner 写出
- 分析:`tools/moe_probe_lane2device.py`(第 15 行 `np.fromfile` 读上面的 .bin)
- 输出:`probe_partial/probe_lane2device.txt`,全文如下

```
D=4 T=128 H=2048 P=64   |partial| max=1.41   sum-of-slices vs y_ref: max|err|=0.000919
slice 0: lanes with coef>0.5 -> [0..15]   residual max|err|=0.000761   coef range on those 1.000..1.000, others |max| 0.000
slice 1: lanes with coef>0.5 -> [16..31]  residual max|err|=0.000796   coef range on those 1.000..1.000, others |max| 0.000
slice 2: lanes with coef>0.5 -> [32..47]  residual max|err|=0.000861   coef range on those 1.000..1.000, others |max| 0.000
slice 3: lanes with coef>0.5 -> [48..63]  residual max|err|=0.000834   coef range on those 1.000..1.000, others |max| 0.000

coefficient matrix (rows = device slice, cols = lane), rounded:
  d0: ################................................................
  d1: ................################................................
  d2: ................................################................
  d3: ................................................################

matches contiguous (lane p -> slice p//16): True
matches strided    (lane p -> slice p%4): False
```

### 推导
64 × 4 个系数全部精确为 1.000 或 0.000,每个 slice 的残差 < 1e-3,四个 slice 之和与参考输出差 9e-4(fp16 精度)。
slice d 恰好等于 lane 16d..16d+15 的和。**卡 d 持有 lane 16d 到 16d+15,连续块;交错假设(p mod 4)被显式排除。**

### 边界
读的是输出张量的逻辑 slice d;slice ↔ 物理卡靠 MDP tensor-slice 的约定。fp16 编译,与主线 MXFP6 编译不同(只影响数值精度,不影响切分)。

---

## 3. 卡内 lane → 核(单卡直读):带名字的标量 tap(`c1_tap`)

### 设计
batched GEMM 完全不动,只在 down-proj 输出 `[64, T, H]` 后面按 lane 切片,每片乘一个标量,节点命名 `/taps.<g>.<p>/Mul`。
trace 里这个乘法带着 lane 编号。**持有 lane p 输出的那个核,做这个乘法时不需要 `aicmulticastvtcm` 把数据搬过来;其余核都要先收一次多播。**
所以数一下每个核上 tap 相关事件数:本地核只有乘法事件,远程核 = 乘法 + 多播,事件数多一倍。

### 原始数据
- 编译产物:`c1_tap/qpc/programqpc.bin`,136 MB,11:17;trace:`c1_tap/out/*.qaic-opstats.trace.json`,4.7 MB,11:17
- 分析:`tools/moe_probe_tap2core.py`;输出 `c1_tap/probe_tap2core.txt`:

```
c1_tap card0: tap op kinds {'aicmulsplat': 768, 'aicmulticastvtcm': 720}
lane -> {core: n_events}
  lane 0: {0: 6,  1: 12, 2: 12, ..., 15: 12}
  lane 1: {0: 12, 1: 6,  2: 12, ..., 15: 12}
  lane 2: {0: 12, 1: 12, 2: 6,  ..., 15: 12}
  lane 3: {..., 3: 6, ...}
  lane 4: {..., 4: 6, ...}
  lane 5: {..., 5: 6, ...}
  lane 6: {..., 6: 6, ...}
  lane 7: {..., 7: 6, ...}
```

### 推导
768 个乘法 = 8 lane × 16 核 × 6;720 个多播 = 8 lane × 15 核 × 6。每条 lane 恰好有 **一个**核只有 6 个事件(无多播),其余 15 核 12 个。
那个核对 lane 0..7 依次是 core 0..7。**lane p 的 down 输出在核 p 上。**

### 边界与顺带解决的疑问
- trace 里只出现 lane 0..7 与 stage 0 的节点:单卡编译把结构相同的 (stage, lane 半区) **卷成循环**,复用第一份的名字。
  输出数值证明 32 个 expert 全算了(MXFP6 QPC 输出对 y_ref 相关 0.996;只算 1/4 时应约 0.35)。
  这同时解释了更早的未解项"c1 trace 只有 3 个 MatMul / 8 核":循环卷叠,不是少算。
- 四卡版 `tap4`(`tap4/probe_tap2core.txt`)读不出 GEMM 布局:64 个 tap 在四张卡上都跑了,每卡"无需多播"的核 = lane//4
  (lane 0–3 → core0,lane 4–7 → core1,…),延迟 9.9 → 12.6 ms。说明编译器先把 down 输出**跨卡交换**再按每核 4 lane 重新布局后才做乘法,
  tap 读到的是 tap 自己的布局。所以四卡上的核编号**没有**逐核直读,见 §4 的推导。

---

## 4. 每核持有整个 expert 还是列切片:12 核判决实验(`u128_c12`)

### 为什么需要这一步
16 核时 coremap(`tools/moe_tier_bench_coremap.py perop_moe/tier_bench/u128 --card 0`)给出:

```
node        tiles cores  deqKB/core
/MatMul_1     192    16   3072..3072    gate
/MatMul_2     192    16   3072..3072    up
/MatMul_3      32    16   3071..3072    down
/MatMul_4..6  同上
```

每核解量化 3072 KB = 1 个矩阵。**但两种假设都给出 3072:**
- (A) 每核 1 个整 expert
- (B) 16 个 expert 每个切成 16 份列,每核拿 16 × 1/16 = 1 个矩阵的量

16 核 / 16 lane 分不出来。

### 设计
同一 ONNX,同一 4 卡 MDP,只把 `-aic-num-cores` 改成 **12**(`mdp.json` 四卡 numCores 均为 12)。16 条 lane 不被 12 整除:
- (A) 整 expert:必然不均,有的核 2 个、有的核 1 个、或有核空着;每核 KB 必是 3072 的整数倍
- (B) 列切:12 核各 16 × 3072 / 12 = **4096 KB**,完全均匀

两种假设的预测互斥。

### 原始数据
- 编译:`u128_c12/compile.log`,QPC `u128_c12/qpc/programqpc.bin` 525 MB,12:48
- trace:`u128_c12/out/*slice00*-merged.qaic-opstats.trace.json`,10.2 MB,12:48
- `python3 tools/moe_tier_bench_coremap.py perop_moe/tier_bench/u128_c12 --card 0 --per-core`(card0,其余三卡相同):

```
/MatMul_1 (gate)   24 tiles   8 cores   3 tiles/core   deqKB/core 6144..6145
    core 0: 6145 KB   core 1: 6144   core 2: 6144   core 3: 6144
    core 4: 6145 KB   core 5: 6144   core 6: 6144   core 7: 6144
    core 8..11: 无 tile
/MatMul_2 (up)     同 gate:8 核 × 6144 KB,core 8..11 空
/MatMul_3 (down)   26 tiles  12 cores
    core 0: 4 tiles  6912 KB
    core 1..11: 各 2 tiles  3839..3841 KB
/MatMul_4,5,6 (stage 1)  与 stage 0 逐核相同
```

延迟:`u128_c12` card0 11.09 ms vs `u128` 9.91 ms(+12%)。

### 推导
- gate/up:8 核 × 6144 KB = 8 × 2 × 3072 = 16 个整矩阵 ✓。每核 **恰好 2 个整 expert**,**4 个核完全空着**。
  假设 (B) 预测的 "12 核各 4096" 不成立。编译器宁可空 4 个核,也不把一个 expert 切开。
- down:11 核 × 3840 + 6912 = 42240 + 6912 = 49152 = 16 × 3072 ✓。3840 = 1.25 × 3072,6912 = 2.25 × 3072。
  down 的分配单位是 **1/4 个 expert(512 列)**:16 lane × 4 块 = 64 块,64 / 12 = 5 余 4,11 核各 5 块(1.25),余数堆给 core0(9 块 = 2.25)。
  余数落在 core0 且其余核完全相同,说明是 **lane 主序连续分**。
- 延迟 +12% 与 "gate/up 每核算 2 个 expert、但 down 只多 25%" 一致。

**结论:编译器把 batched 节点按首维(lane)整块分给核。gate/up 以整 expert 为单位,down 以 1/4 expert 列块为单位。
16 核时两者都恰好是 lane p → 核 p。** 这是 §0 表里"每核 1 个 expert"和"四卡 lane p → 核 p%16"的依据:
整块 + 连续 + 16 lane / 16 核,只有一种排法,且与 §3 单卡直读的 lane 0..7 → core 0..7 一致。

### 边界
核级数据来自 profiler 记录的每核解量化字节数,是"从旁证推断",不像 §2 那样直接读输出数值。
但实验设计使两种假设的预测互斥,实测与 (A) 吻合、与 (B) 不吻合。

---

## 5. 编译器不做节点级调度:每 lane 独立节点(`c1_split`、`split4`)

### 设计
`--split-lanes`:把 batched GEMM 拆开,每条 lane 的 gate/up/down 是自己的 MatMul 节点(`/lanes.<g>.<p>[_1]/MatMul*`)。
如果编译器会做"节点到核"的分配,32 个互相独立的节点应当被分到不同的核上并行跑。

### 原始数据
`c1_split/probe_lane2core.txt`(单卡,32 lane):

```
conv tiles per (core, lane):   16 核 × 32 lane,每格都是 9
dequant KB per (core, lane):   每格 496..752 KB(≈ 3072 × 3 / 16 = 576 的上下)
lane 之间:严格串行,每条 ~80 µs
```

`split4/probe_lane2core.txt`(4 卡,card0):

```
128 lanes seen(全部 128 个 expert 都出现在这张卡上)
conv tiles per (core, lane):   16 核 × 128 lane,每格都是 3
每 lane 每卡解量化 ~2300 KB / 9216 KB(= 三个矩阵之和)≈ 1/4
```

延迟:

| 变体 | 卡 | 每核用几条 lane | 延迟 | vs batched |
|---|---:|---|---:|---:|
| `c1_u128`(batched) | 1 | 每核 1 lane | 3.26 ms | — |
| `c1_split` | 1 | 每 lane 摊满 16 核,32 节点串行 | 5.35 ms | 1.6× |
| `u128`(batched) | 4 | 每卡 16 lane,每核 1 lane | 9.91 ms | — |
| `split4` | 4 | 每卡 128 lane,每 lane 每卡 1/4 | 26.3 ms | 2.7× |

### 推导
1. 32 个独立节点,每一个都被切成 16 份摊到 16 核(每格 9 tile,没有一格是 0),然后节点一个接一个串行。
   **编译器不做节点级的核分配**,只会把每个节点的张量切开。
2. 4 卡时每张卡看到全部 128 条 lane、每卡持有每个 expert 的 1/4:MDP tensor-slice 切的是**节点的张量**而不是节点。
   expert 身份一旦不在同一个张量的首维上,EP 就退化成 TP(每卡 1/4 权重,全部 expert),慢 2.7×。
3. 反过来说明 §3、§4 看到的"一核一 expert"**完全是 batched 张量首维被 16 等分的副产品**,不是编译器识别出 expert 并做的放置。

---

## 6. 机制解释(这部分来自读源码,不是实测)

lane 顺序由 QEfficient 的 `pack_moe_weights_for_expert_parallel`
(`QEfficient/transformers/moe/weights.py:204`)决定:`[128, H, I]` 经 `view(2, 64, ...).transpose(0, 1)` 变成 2 组 `[64, H, I]`,
首维是 lane;归约写成 `view(4, 16, T, H)`,卡的归属 = 首维的连续 16 份(`perop_moe/README.md` 结果 1)。

把 §2–§5 的实测和这个机制合起来:

```
expert e 在哪个槽位 p     ← Python 打包时的排列(源码)
槽位 p 在哪张卡          ← p // 16     (实测 §2)
槽位 p 在卡内哪个核      ← p % 16      (实测 §3 + §4)
编译器参与了什么         ← 只把首维整块等分;没有任何决策(实测 §4 + §5)
```

---

## 7. 对课题的含义

- **改 expert 放置 = 在 Python 里置换 expert 顺序再导出。** 编译器不参与,没有 flag 可调。
- **两个 expert 共用一个核、或让一个核空着,在这个编译器下无法表达**(12 核实验里它宁可空核也不共享)。
  分档 padding 后若想按预算做卡间 / 核间装箱,唯一自由度是 lane 顺序。
- **不能把 lane 拆成独立节点**来获得放置自由度:那会让 EP 退化成 TP,慢 2.7×。
- `ADVISOR_DIRECTION.md` 路线 ①(b)"聪明放 expert"在此基础上的精确表述:可做,但只能通过权重张量排列做,且只能按部署/批次切换(换排列 = 换权重槽位 = 重新装载)。

---

## 8. 复现

```bash
# §2 lane -> 卡(fp16,4 卡,输出 partial)
tools/moe_probe_placement.sh partial          # 生成 probe_partial/,内部调用 moe_probe_lane2device.py

# §3 单卡 tap(batched 图 + --tap-lanes;c1_tap = 1 卡、16 lane;tap4 = 4 卡、64 lane)
EXTRA="--tap-lanes" tools/moe_tier_bench.sh c1_tap 128:16 1
EXTRA="--tap-lanes" tools/moe_tier_bench.sh tap4   128:64 4
python3 tools/moe_probe_tap2core.py perop_moe/tier_bench/c1_tap
python3 tools/moe_probe_tap2core.py perop_moe/tier_bench/tap4

# §4 12 核判决(同一 ONNX,-aic-num-cores=12)
CORES=12 tools/moe_tier_bench.sh u128_c12 128:64 4
python3 tools/moe_tier_bench_coremap.py perop_moe/tier_bench/u128_c12 --card 0 --per-core
python3 tools/moe_tier_bench_coremap.py perop_moe/tier_bench/u128     --card 0 --per-core   # 16 核对照

# §5 独立节点
tools/moe_probe_placement.sh split            # 生成 c1_split/、split4/,内部调用 moe_probe_lane2core.py
```

---

## 9. decode 能不能走 EP:代码判定 + 小模型卡上验证(2026-09-15)

### 9.1 结论

| 问题 | 答案 | 证据等级 |
|---|---|---|
| 原版 QEfficient 1.23 的 decode 是 EP 吗 | **不是**。decode 恒为 `decode_bmm`(gather + bmm,TP);EP 只在 `prefill_only=True` 时被默认选中 | 源码直读(§9.2) |
| `prefill_only=False` 会让 prefill 和 decode 都用什么 | 同一份 ONNX 服务两个 specialization,flavour 只选一次;默认两者都是 `decode_bmm` | 源码直读 + 小模型复现(§9.2、§9.4) |
| 能不能让 decode 用 EP | **能**,不改 QEfficient 源码:把 `qaic_config={"moe_config": {"flavour": "expert_parallel", ...}}` **同时传给 `compile()`**(只传 `from_pretrained` 无效) | 小模型端到端(§9.3–§9.5) |
| decode 用 EP 划算吗(单层、4 卡) | batch ≤ 64 都不划算;batch 128 才持平(§9.6) | 卡上实测 |

### 9.2 源码:flavour 由 `prefill_only` 决定,不由"当前是不是 decode"决定

```
QEfficient/transformers/moe/flavours.py:310
    preferred = MoEFlavour.EXPERT_PARALLEL if is_prefill else MoEFlavour.DECODE_BMM
QEfficient/transformers/models/pytorch_transforms.py:1641-1646
    flavour = select_moe_flavour(..., is_prefill=prefill_only, requested_flavour=requested_flavour)
    module._moe_flavour = flavour
QEfficient/base/modeling_qeff.py:985-992      (compile() 每次都重跑这个 transform)
    OptimizedMoETransform.apply(self.model, prefill_only=bool(compiler_options.get("prefill_only", False)),
                                ..., qaic_config=qaic_config, ...)        # <- 用的是 compile() 收到的 qaic_config
QEfficient/transformers/moe/block.py:51        _moe_flavour: MoEFlavour = MoEFlavour.DECODE_BMM   # 类默认
QEfficient/transformers/moe/block.py:181-187   forward: [B,S,H] -> view(B*S,H) -> execute_moe_flavour(x, routing)
QEfficient/transformers/models/modeling_auto.py:4217   build_decode_specialization: 同一 ONNX, seq_len=1
```

推导:`is_prefill` 就是编译参数 `prefill_only`,不是 token 级判断。`prefill_only=False` 的 QPC 用一份 ONNX 同时编 prefill(seq_len=128)和 decode(seq_len=1)两个 specialization,flavour 在导出前只选一次,所以两者一致。
唯一的出口是 `flavours.py:315-328`:显式给了 `requested_flavour` 就照给的返回,不看 `is_prefill`。
但 `compile()` 会用**自己收到的** `qaic_config` 重新选一次(`modeling_qeff.py:985`);只在 `from_pretrained` 里给 flavour,到 compile 时 `requested_flavour=None`,又落回默认。
之前的 EP prefill 实验(`run_ep.py`)之所以是 EP,是因为 `prefill_only=True` 的默认就是 EP,不是显式 flavour 起了作用。

四种组合:

| `prefill_only` | `compile()` 收到的 flavour | prefill spec | decode spec |
|---|---|---|---|
| True | 无 / expert_parallel | EP | 不编 |
| False | 无 | decode_bmm | decode_bmm |
| False | expert_parallel | EP | **EP** |

### 9.3 小模型端到端(正例)

对象:随机权重 Qwen3-MoE,2 层、8 expert、top-2、H=256、I=128。走**真实** QEfficient 1.23 路径:
`from_pretrained(qaic_config=EP)` → `compile(prefill_only=False, batch_size=8, prefill_seq_len=32, ctx_len=64, num_devices=4, num_cores=16, mxfp6_matmul=True, qaic_config=EP)`。
QEfficient 自己的 qaic-compile 一步在本 SDK 上因 spec JSON 格式失败(与 `run_ep.py` 相同),ONNX 与 `custom_io.yaml`/`mdp_ts_4.json` 已生成;之后用拍平的 spec `{"batch_size":"8","ctx_len":"64","seq_len":"1"}` 手动 `qaic-compile` 得到 **decode-only** QPC。
脚本 `tools/moe_tiny_ep_decode_proof.py`;产物 `perop_moe/ep_decode_proof/`。

原始数据:
- transform 之后每层 `_moe_flavour=expert_parallel`,stages=1,lanes=8,experts_per_soc=16;`hash_params["moe_prefill_flavour"]="expert_parallel"`
- 导出 ONNX layer0 `/mlp/` 节点:`CumSum` ×1、`CtxScatter3DInt` ×1、`CtxGather3D` ×3、`CtxScatter3D` ×1、`Einsum` ×5(lane/device 归约)——EP 打包图
- QEfficient 写出的 `specializations.json`(`ep_decode_proof/qefficient_specializations.json`):**只有一个 spec,名为 "Decode",batch_size=8,seq_len=1**
- 手动编译 12.5 s,`programqpc.bin` 8 MB,4 卡 MDP;`qaic-runner -D 0:1:2:3` 每步 0.59 ms(`ep_decode_proof/run_decode_B8.log`)
- 该 seq_len=1 QPC 的 card0 core0 opstats(`ep_decode_proof/opstats_slice00_decode_B8.summary.txt`):

```
cumsum HVX TCM (index32, bool)                  2 次
aicbatchedreduceadd HVX TCM                     4 个 site
aicscatternd HVX DDR/TCM                        3 个 site
blockdequantize_mxfp6 HVX TCM     160 KB        (逐 lane 全量解量化)
aicgather HVX DDR (float16, ui8)  0.38 KB       (activation 打包,不是权重)
无 aicgather (uindex8, ...)                     (decode_bmm 的按 (token,expert) 取权重在这里不存在)
```

对照:同形状 decode_bmm 微基准(`tier_bench/dbmm_B1`)card0 core0 是 `aicgather HVX DDR (uindex8, ui8, index32)` 450 KB,batch 16 时 11,100 KB;没有 `cumsum`。

### 9.4 负例(不传 compile-time qaic_config)

同一脚本,只去掉 `compile(..., qaic_config=...)`:每层 `_moe_flavour=decode_bmm`,`hash_params["moe_prefill_flavour"]="decode_bmm"`,
expert 权重 initializer 为 `[8, 256, 128]`(即 `[E, H, I]`,未打包),layer0 `/mlp/` 没有 `CumSum`/`CtxGather3D`。
这一步证明 §9.2 的"compile 重新选 flavour"不是读源码的猜测。

### 9.5 数值验证:decode-only EP QPC 算得对

`ep_decode_proof/verify_decode_B8.py`:8 个真实 token id `[17,250,511,777,3,999,64,128]`,batch 8、seq_len 1、position 0,
只绑定 `input_ids`/`position_ids`(KV 是 device retained state),`--write-output-dir` 取 `logits` `[8,1024]` fp32,与 CPU 上 fp32 PyTorch 前向比:

| token | corr | top-1 一致 | top-5 交集 |
|---:|---:|:--:|:--:|
| 17 | 0.998 | 否 | 5/5 |
| 250 | 0.998 | 是 | 4/5 |
| 511 | 0.998 | 是 | 5/5 |
| 777 | 0.998 | 是 | 5/5 |
| 3 | 0.998 | 是 | 5/5 |
| 999 | 0.997 | 是 | 5/5 |
| 64 | 0.998 | 是 | 4/5 |
| 128 | 0.996 | 否 | 5/5 |

错位对照(QPC 第 b 行 vs 参考第 b+1 行)corr 在 −0.30..0.18,说明每行确实是各自 token 的输出。两个 top-1 不一致发生在随机权重、logits 几乎平坦的模型上,是 MXFP6 噪声量级。
PyTorch 给出的 layer0 top-2 expert 为 `[[2,5],[7,2],[0,5],[3,7],[7,2],[1,7],[4,3],[6,3]]`,8 个 expert 全部被命中,即 8 条 lane、4 张卡在这一步 decode 里都有真实工作,跨卡归约结果正确。

### 9.6 代价:EP vs decode_bmm,单层、4 卡、Qwen3-30B-A3B 形状(`tier_bench/ep_vs_dbmm_batch.txt`)

decode batch B 进 MoE 就是 T=B 行(`block.py:183`),所以 EP 列直接复用 §1 的微基准改 `-T`;decode_bmm 列是新写的 `tools/moe_decode_bmm_export.py`(与 `flavours.py::moe_decode_bmm` 同图),同一编译/运行脚本(`tools/moe_tier_bench.sh`,`EXPORTER=` 切换)。

| batch | EP(ms) | decode_bmm(ms) | 目录 |
|---:|---:|---:|---|
| 1 | 6.16 | 0.14 | `u128_T1` / `dbmm_B1` |
| 8 | 6.27 | 0.63 | `u128_T8` / `dbmm_B8` |
| 16 | 6.61 | 2.76 | `u128_T16` / `dbmm_B16` |
| 32 | 6.63 | 5.10 | `u128_T32` / `dbmm_B32` |
| 64 | 8.18 | 5.22 | `u128_T64` / `dbmm_B64` |
| 128 | 9.91 | 10.32 | `u128` / `dbmm_B128` |

推导:EP 每层几乎是常数,因为每个核不管有没有 token 路由到它都把自己 2 个 expert 解量化一遍(T=1 时 core0 `blockdequantize` 18,560 KB,T=128 时 18,432 KB,`aicconvolutiond32` 都是 53 tile)。
decode_bmm 随 batch 线性长,因为它按 (token, expert) 对 gather B×8 份权重。B×8=128 时两者搬同样多的权重,交点落在 batch 128。
**batch 1 时 EP decode 慢 44×,batch 64 仍慢 1.6×。**

### 9.7 30B 真模型:同样三项检查(2026-09-15 01:30–01:50)

导出:`run_ep_decode.py`(`prefill_only=False, batch_size=8, num_devices=4, num_cores=16`,两处 `qaic_config`),峰值 RSS 167 GB,ONNX 57 GB,
落在 `/home/chihao/models/qwen3_30b_a3b/ep/Qwen3MoeForCausalLM/Qwen3MoeForCausalLM-40c66dadb1e495df/`。
`hashed_export_params.json`:`moe_prefill_flavour=expert_parallel`,64 lane × 2 stage,chunk 256。
QEfficient 自己写的 `specializations.json`:**只有 "Decode",batch_size=8,seq_len=1**(`ep_decode_proof/qwen3_30b/qefficient_specializations.json`)。
手动 decode-only 编译(`compile_ep_decode.sh`,spec batch 8 / ctx 256 / seq_len 1):11 min,QPC 24.6 GB。

**(1) 卡上算子签名**(`ep_decode_proof/qwen3_30b/opstats_slice00_decode_B8.summary.txt`,card0 core0):

```
blockdequantize_mxfp6   851 次   937,344 KB     (= 48 层 × 2 expert × 3 矩阵 × 3072 KB ≈ 884 MB + attention;逐 lane 全量解量化)
cumsum                   96 次                  (48 层 × 2 stage,token 打包)
aicbatchedreduceadd      96 × 4 个 site         (lane / device 归约)
aicgather (float16)      仅 2.5–3 KB 级         (activation 打包)
无 aicgather (uindex8)                          (TP decode 的按 (token,expert) 取权重在这里不存在)
```

延迟(`ExecTimeUs` card0):**batch 8:118.3 ms / step**(14.8 ms/token);**batch 1:86.4 ms / step**(`qpc_decode_B1`,同一 ONNX 换 spec 重编,10 min);对照 TP decode QPC batch 1 为 14.3 ms/token,即 **batch 1 时 EP decode 慢 6.0×**。
与 §9.6 微基准一致:EP 每步把 128 个 expert 全搬一遍,batch 8 只是把这个固定成本摊到 8 个 token 上。

**(2) 行独立与确定性**:8 行全填 token 50744 时 8 行 logits 两两 corr=1.000、top-1 全为 594;同一混合输入跑两次 max|diff|=0;
混合 batch 里 token 50744 那一行与全 50744 batch 的第 0 行 corr=1.0000。batch 内各行互不影响。

**(3) 数值验证**(`ep_decode_proof/qwen3_30b/{ep_50744,tp_50744,ep_mixA,cpu_ref_*}.npy`):

| 对比 | corr | top-1 | top-5 交集 | max\|diff\| |
|---|---:|---|:--:|---:|
| token 50744:**EP decode QPC(batch 8)vs TP decode QPC(1.22,batch 1,Phase 1 基线)** | **1.0000** | 594 / 594 | 5/5 | 0.06(logit 量级 18) |
| token 50744:EP decode QPC **batch 1** vs TP decode QPC | **1.0000** | 594 / 594 | 5/5 | 0.06 |
| token 50744:EP decode QPC batch 1 vs EP batch 8 第 0 行 | 1.0000 | 594 / 594 | 5/5 | 0.000 |
| token 50744:TP decode QPC vs HF bf16 CPU | 0.848 | 594 / 323 | 4/5 | 9.85 |
| token 50744:EP decode QPC vs HF bf16 CPU | 0.848 | 594 / 323 | 4/5 | 9.87 |
| 8 token 混合 batch:EP decode QPC vs HF bf16 CPU | 0.74–0.92 | 4/8 一致 | 0–3/5 | — |

推导:EP decode QPC 和 Phase 1 一直当基线用的 TP decode QPC 算出**同一个函数**(corr 1.0000,差 0.06)。
两者对 HF bf16 CPU 的偏差**完全相同**(0.848 vs 0.848),所以这个偏差属于 QEfficient 的 MXFP6/fp16 图本身(或 HF bf16 CPU 参考),与 EP 无关。
第一次比对时误把 TP 的输入当成 9707(实际绑定的是 50744,见 `4card/qpc_decode/bind/in_0.bin`),得到 corr 0.65,已纠正。

### 9.7b 边界

- batch 1 与 batch 8 的 EP decode 输出逐 bit 相同(max|diff|=0.000),batch 只改变 padding 行数,不改变每行的计算。
- 只验证了 position 0 的一步 decode;多步生成的 KV 逻辑没在 EP 图上跑过。EP 只改 MoE 部分,这一步足以判定。
- QPC 与 HF bf16 的 0.85 相关是两条 QEfficient 路线共有的,没有进一步归因(MXFP6 路由翻转 / fp16 / bf16 CPU 参考)。
- decode_bmm 微基准与真模型 TP decode 的关系:真模型 48 层 14.3 ms/token,微基准 0.14 ms/层 × 48 ≈ 6.7 ms,差额是 attention、router 之外的算子与同步。

### 9.8 复现

```bash
# 正例 + 负例(把脚本里 compile(..., qaic_config=...) 去掉即负例)
python3 tools/moe_tiny_ep_decode_proof.py            # 导出 + 打印 flavour / ONNX op / specializations.json
# decode-only 手动编译、运行、验证见 perop_moe/ep_decode_proof/{compile_decode_B8.log,run_decode_B8.log,verify_decode_B8.py}
# 代价表
for B in 1 8 16 32 64; do EXTRA="-T $B" tools/moe_tier_bench.sh u128_T$B 128:64 4; done
for B in 1 8 16 32 64 128; do EXPORTER=tools/moe_decode_bmm_export.py EXTRA="-T $B" tools/moe_tier_bench.sh dbmm_B$B 128:64 4; done
```

### 9.9 改了什么、改在哪个文件

**QEfficient 源码一行没改。** `qaic_config={"moe_config": {...}}` 不是文件,是 Python 字典,作为关键字参数传进两个调用。改动只在我们自己的脚本里:

| 文件 | 改动 | 作用 |
|---|---|---|
| `/home/chihao/models/qwen3_30b_a3b/ep/run_ep_decode.py`(新,基于 `run_ep.py`) | 第 22 行 `from_pretrained(..., qaic_config={"moe_config": MOE})`(原来就有);**第 37 行 `compile(..., qaic_config={"moe_config": MOE})`(新增,决定 flavour 的那一处)**;`prefill_only=False`、`batch_size=B`(环境变量 `B`,默认 8) | 30B 真模型:导出含 decode specialization 的 EP 图 |
| `/home/chihao/models/qwen3_30b_a3b/ep/build_ep_decode.sh`(新) | 跑上面的脚本(QEfficient 自带的 qaic-compile 一步在本 SDK 因 spec JSON 格式失败,与 `run_ep.py` 一样),然后用拍平的 `spec_decode_B{8,1}.json`(`batch_size=B, ctx_len=256, seq_len=1`)手动 `qaic-compile` 出 decode-only QPC `qpc_decode_B8/`、`qpc_decode_B1/` | 30B 真模型:decode-only EP QPC |
| `tools/moe_tiny_ep_decode_proof.py`(新) | 第 20 行、第 23 行同样两处 `qaic_config={"moe_config": MOE}`;第 23 行去掉即 §9.4 负例 | 小模型正例 / 负例 |
| `tools/moe_decode_bmm_export.py`(新) | 与 `QEfficient/transformers/moe/flavours.py::moe_decode_bmm` 同图的单层微基准导出器 | §9.6 decode_bmm 列 |
| `tools/moe_tier_bench.sh` | 新增 `EXPORTER=` 环境变量(默认仍是 `moe_tier_bench_export.py`) | 复用同一编译/运行/profiling 流程跑 decode_bmm |

`MOE` 字典本身:

```python
MOE = {"flavour": "expert_parallel", "cores_per_expert": 1,
       "tree_reduce": False, "expert_parallel_chunk_size": 256}
```

QEfficient 在哪里读它:`base/modeling_qeff.py:985` 把 `compile()` 收到的 `qaic_config` 交给 `OptimizedMoETransform.apply`;
`transformers/models/pytorch_transforms.py:1601-1602` 取 `moe_config["flavour"]` 作为 `requested_flavour`;
合法取值是 `transformers/moe/flavours.py:35-37` 的 `MoEFlavour`:`"simple_loop"`、`"decode_bmm"`、`"expert_parallel"`。
只传给 `from_pretrained` 的那份会被 compile 时的重新选择覆盖(§9.2、§9.4),所以**必须两处都传**。

## 10. 文件索引

| 路径 | 内容 |
|---|---|
| `probe_partial/outdir_fp16/partial-activation-0-inf-0.bin` | §2 卡上输出张量(2 MB) |
| `probe_partial/probe_lane2device.txt` | §2 回归结果 |
| `c1_tap/out/*.trace.json`, `c1_tap/probe_tap2core.txt` | §3 trace 与统计 |
| `tap4/probe_tap2core.txt` | §3 四卡 tap(读不出 GEMM 布局的反例) |
| `u128/out/*.trace.json`, `u128_c12/out/*.trace.json` | §4 16 核 / 12 核 trace |
| `u128_c12/compile.log`, `u128_c12/mdp.json`, `u128_c12/run.log` | §4 编译、MDP、延迟 |
| `c1_split/probe_lane2core.txt`, `split4/probe_lane2core.txt` | §5 |
| `*/run.log` | 各变体 `ExecTimeUs` |
| `tools/moe_probe_placement.sh`, `moe_probe_lane2device.py`, `moe_probe_tap2core.py`, `moe_probe_lane2core.py`, `moe_tier_bench_coremap.py` | 全部脚本 |
| `README.md` 结果 7、结果 8 | 同一组实验的原始记录 |
| `ep_vs_dbmm_batch.txt`, `u128_T{1,8,16,32,64}/`, `dbmm_B*/` | §9.6 EP vs decode_bmm 逐 batch 延迟 |
| `../ep_decode_proof/` | §9.3–§9.5 小模型 decode-only EP QPC:spec、opstats、logits、验证脚本 |
| `tools/moe_tiny_ep_decode_proof.py`, `tools/moe_decode_bmm_export.py` | §9 脚本 |
