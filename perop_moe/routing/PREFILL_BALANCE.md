# Prefill 时每个 expert 分到的 token 数量平均吗?——GSM8K 实测(2026-09-15)

## 0. 问题与结论

**配置**:Qwen3-30B-A3B(48 层 MoE,128 expert,top-8),routing 来自 CPU bf16 HF 模型的 **prefill** 前向;GSM8K test 300 题(21,790 token);
**batch = 1 个 prompt 一次 prefill**,T = 真实 prompt 长度 36–146(比例按 T 归一,对卡上 pad 到 128 的图同样适用);另外模拟把 B 个独立 prompt 合成一个 batch。
**适用范围**:所有关于 padding 的结论只对 **expert-parallel 图** 有意义(QEfficient 1.23 `prefill_only=True, flavour=expert_parallel`,4 卡 × 16 核,64 lane × 2 stage,每个 expert 一个 [T, H] 的独立 padded 槽)。默认 decode 图是 decode_bmm(TP,按 (token, expert) gather 权重),没有 per-expert 槽,不适用;decode 若按 `EXPERT_PLACEMENT_PROOF.md` §9 编成 EP,分档要用 `DECODE_STEP_SIMILARITY.md` 的 batch 32 / batch 8 decode 数据另算。

教授的假设(9/7 微信):"prefill 多 batch 的情况下,每个 expert 分到的 token 数量应该挺平均的,所以一个稍微保守一点的统一 padding 可能就够了。"

**实测结论:**
1. **单个 prompt 的一次 prefill,每层 expert 负载极不平均**:最忙 expert 拿到约 11 倍平均份额(0.70 T 的 token,p90 0.87 T),同时 49/128 个 expert 一个 token 都没有,Gini 0.71。
2. **把多个独立 prompt 合成一个 batch 不会变平均**:同一 workload(GSM8K)合 100 个 prompt,最忙仍是 9.8 倍平均,Gini 0.62,从 B=8 起就不再下降。热 expert 是系统性的(每层 61% 的 prompt 最忙的 expert 就是全体最忙的那个),同类 prompt 叠加而不抵消。
3. **混合两个 workload**(GSM8K + HumanEval 各半)能降到 6.7 倍,仍远不平均。
4. **统一的"保守 padding"没有便宜的选择**:溢出 ≤1% 需要 padding 达到 T 的 63–69%(只省约 1/3 的行);padding 设为 2 倍平均份额(省 88% 行)会让 24% 的 token 溢出。多 batch 对此几乎没有帮助。
5. **但确实不需要把每个 expert 都 pad 到 T**:按 expert、按层分别定 padding(前 200 题取每个 expert 的 p99,后 100 题测),只用 **18% 的行**就把溢出压到 **0.58%**;用每 expert 历史最大值 × 1.25,28% 的行、溢出 0.04%。也就是说同样 ≤1% 溢出,分档比统一 padding 省 82% 的行。形态是"每层约 10 个 expert 要 pad 到 >T/2、约 30 个 >T/4、约 39 个 ≤T/16、约 13 个可以直接 pad 到 0",不是"大家都缩一点"。最忙的 expert 仍要接近 T(p99 = 0.98 T)。
6. **唯一接近平均的是卡这一级**(EP 每卡 32 个 expert 热冷相抵,卡间 max/mean ≈ 1.3)。教授的"平均"在卡粒度成立,在 expert 粒度不成立,而 padding 关心的是后者。

## 1. 怎么测

| 项 | 设置 |
|---|---|
| 模型 | Qwen3-30B-A3B(48 层 MoE,每层 128 expert,top-8,`norm_topk_prob=True`),HF 权重 `/home/chihao/models/qwen3_30b_a3b/hf` |
| 数据 | GSM8K test(`/home/chihao/models/bench/gsm8k_test.jsonl`)前 300 题,套 chat template;对照 HumanEval 前 100 题(8 月 28 日采集) |
| 怎么跑 | CPU bf16,只做 prefill 前向(不生成)。hook `Qwen3MoeTopKRouter.forward`,记录每 token、每层的 top-8 expert 索引。工具 `tools/moe_collect_routing.py --bench gsm8k --n 300`,20.5 min,输出 `routing_gsm8k_300.npz`(每题一个 `[48, T, 8]` int16 数组) |
| 规模 | 300 题,T 每题 36–146(中位 68),共 21,790 token,8,367,360 个路由决策 |
| 为什么在 CPU 上而不在卡上 | 卡上的 EP QPC 不输出 router 结果;Phase 1 已证明卡上 cycle 数与输入无关,routing 只能从 torch 模型读。见 §4 限定 |
| 统计 | `tools/moe_prefill_balance.py routing_gsm8k_300.npz`。对每个 (prompt p, 层 l):`counts[e]` = 该层被路由到 expert e 的 token 数(Σ_e = T·8)。均匀时每个 expert 应得 T·8/128 = T/16。所有统计**逐层**做,不跨层聚合(不同层的 expert 是不同参数) |
| batch 模拟 | 随机抽 B 个 prompt,把它们同一层的 counts 相加(等价于 B 个 prompt 合成一个 prefill batch),200–300 次抽样取中位数 |

## 2. 数据

### 2.1 单个 prompt(batch = 1),每层;14,400 个 (prompt, layer) 样本的分布

| 指标 | 中位数 | p10 | p90 | max |
|---|---:|---:|---:|---:|
| 最忙 expert 的 token 数 / 均匀份额 (T/16) | **11.17** | 7.53 | 13.87 | 15.72 |
| 最忙 expert 的 token 数 / T | **0.70** | 0.47 | 0.87 | 0.98 |
| 零 token 的 expert 数(共 128) | **49** | 36 | 63 | 82 |
| Gini(128 个 expert) | 0.71 | 0.63 | 0.79 | 0.85 |
| 最忙 8 个 expert 的 token 份额(均匀 6.2%) | 39% | 30% | 49% | 67% |
| token 数 > T/2 的 expert 个数 | 2 | 0 | 4 | 7 |
| token 数 > T/4 的 expert 个数 | 6 | 4 | 9 | 13 |

逐层(L0…L47)"最忙/均匀"的均值:
```
4.1 6.0 9.4 8.8 12.4 11.6 13.2 12.7 10.6 10.5 9.5 10.2 13.0 10.3 14.2 13.0 11.9 8.8 12.7 11.8 10.3 8.9 9.3 8.6 13.6 11.9 13.4 14.1 11.1 9.1 12.5 11.7 11.3 9.5 9.5 9.5 12.9 13.4 13.9 13.6 11.7 8.9 10.7 10.6 10.6 11.0 10.2 5.4
```
只有第 0 层(4.1×)和第 47 层(5.4×)相对平,中间各层 9–14×。

### 2.2 合并 B 个独立 prompt(同一 workload,GSM8K)

| B | 最忙/均匀 | Gini | 零 token expert | EP 卡间 max/mean | 合并后 token 数 |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.14 | 0.707 | 48.0 | 1.325 | 68 |
| 2 | 10.41 | 0.674 | 38.5 | 1.304 | 146 |
| 4 | 10.00 | 0.654 | 32.0 | 1.290 | 287 |
| 8 | 9.86 | 0.639 | 26.0 | 1.280 | 575 |
| 16 | 9.74 | 0.632 | 21.0 | 1.273 | 1,159 |
| 32 | 9.83 | 0.627 | 17.0 | 1.269 | 2,321 |
| 64 | 9.83 | 0.625 | 14.0 | 1.266 | 4,653 |
| 100 | 9.83 | 0.624 | 12.0 | 1.267 | 7,267 |

为什么不收敛:每层里,61%(中位数;最差层 2%,最好层 100%)的 prompt 其最忙 expert 就是全体合并后最忙的那个。

### 2.3 混合 workload(一半 GSM8K + 一半 HumanEval)

| batch 组成 | B=8 | B=32 | B=64 |
|---|---:|---:|---:|
| 只 GSM8K:最忙/均匀(Gini) | 9.86 (0.639) | 9.83 (0.627) | 9.83 (0.625) |
| 只 HumanEval | 8.58 (0.609) | 8.22 (0.592) | 8.27 (0.588) |
| 各一半 | **6.87 (0.541)** | **6.66 (0.526)** | **6.66 (0.523)** |

跨 workload 部分抵消(两者热 expert 不同,README Phase 2 H2:top-8 热 expert 重叠率只有 11%),但抵消不完。

### 2.4 统一"保守 padding"的代价

给每个 expert 统一 padding = k × 均匀份额(k=16 即今天 pad 到 T)。溢出 = 超出 padding 的 (token, expert) 分配占全部分配的比例,300 次随机 batch 的均值:

| batch | k=2(12% 行) | k=4(25%) | k=6(38%) | k=8(50%) | k=12(75%) | k=16(100%) |
|---:|---:|---:|---:|---:|---:|---:|
| B=1 | 31.9% | 15.0% | 8.0% | 4.0% | 0.4% | 0 |
| B=8 | 24.8% | 10.3% | 4.9% | 2.3% | 0.3% | 0 |
| B=32 | 23.8% | 9.8% | 4.6% | 2.2% | 0.3% | 0 |

| 目标溢出 | 需要的 k(B=1 / 8 / 32) | padding 占 T | 省下的行 |
|---|---|---:|---:|
| ≤ 1% | 11 / 10 / 10 | 63–69% | 31–37% |
| ≤ 0.1% | 13.5 / 13 / 13 | 81–84% | 16–19% |
| = 0 | 16 / 15 / 14.5 | 91–100% | 0–9% |

多 batch 对此几乎没有帮助(B=8 与 B=32 一样)。

### 2.5.0 §2.5 的配置(明确写清)

| 项 | 值 |
|---|---|
| 模型 | Qwen3-30B-A3B,HF 权重 `/home/chihao/models/qwen3_30b_a3b/hf`,48 层 MoE,每层 128 expert,top-8,`norm_topk_prob=True` |
| routing 从哪来 | CPU bf16 的 HF 模型 prefill 前向(`tools/moe_collect_routing.py`),不是卡上 MXFP6 图 |
| 阶段 | **只有 prefill**(prompt 一次性前向);decode 不在此表内 |
| batch | **1 个 prompt 一次 prefill**(batch_size=1)。T = 该 prompt 的真实 token 数(36–146,套 chat template);比例按 T 归一,所以对卡上 pad 到 prefill_seq_len=128 的图同样适用 |
| 数据划分 | GSM8K test 前 200 题定每层每个 expert 的 padding,第 201–300 题测溢出与行数 |
| "行数"的含义 | EP prefill 图里每个 expert(lane)要算的 padded 行数之和,相对"每个 expert 都 pad 到 T"(= 128 × T 行/层) |
| 什么时候是 expert parallelism | 这张表对应 **QEfficient 1.23 的 EP prefill 图**(`prefill_only=True`,`flavour=expert_parallel`,4 卡 × 16 核,64 lane × 2 stage,每 lane 1 个 expert/stage,每 lane 一个 [T, H] 的 padded 槽)。只有 EP 图里 expert 才各自有独立的 padding 槽,"按 expert 分档 padding"才有意义。**默认的 decode 图是 decode_bmm(TP,按 (token, expert) 对 gather 权重),没有 per-expert padding 槽,这张表不适用于它**;若用 §9(EXPERT_PLACEMENT_PROOF)的方法把 decode 也编成 EP,则 decode 的分档要用 `DECODE_STEP_SIMILARITY.md` 里 batch 32 / batch 8 的 decode 数据另算 |
| 溢出的含义 | 超过该 expert padding 槽的 (token, expert) 分配占全部分配(T×8)的比例;溢出的 token 在当前静态图里会被丢掉(§PREFILL 的 §3、教授:丢 token 不可接受),所以要么留余量、要么做绕行 |

### 2.5 按 expert 分别定 padding 能省多少(profile-guided,GSM8K 前 200 题定 padding,后 100 题测)

每层每个 expert 的 padding = 它在 200 题里 token 数/T 的分位数(上限 1.0);行数相对"全部 pad 到 T":

| 规则 | padded 行数 | 留出集溢出 | pad 到 0 的 expert |
|---|---:|---:|---:|
| 逐 prompt 精确(oracle 下限) | 6.2% | 0 | 49/128 |
| 每 expert p50 | 5.4% | 28.3% | 49 |
| 每 expert p90 | 11.1% | 5.1% | 26 |
| 每 expert p95 | 13.4% | 2.5% | 21 |
| **每 expert p99** | **18.2%** | **0.58%** | 13 |
| 每 expert max | 22.7% | 0.17% | 11 |
| 每 expert max × 1.25 | 28.2% | 0.04% | 11 |

对照 §2.4 的统一 padding:溢出 ≤1% 要 63–69% 的行,≤0.1% 要 81–84%。
**按 expert 分档:同样 ≤1% 溢出只要 18% 的行(省 82%),≤0.1% 只要 28%(省 72%)。**
每层需要 pad 到 >T/2 的 expert 约 10 个,>T/4 约 30 个,≤T/16 约 39 个;最忙的 expert 仍要 pad 到接近 T(p99 达 0.98 T),所以是"少数几个大槽 + 一大半小槽或零槽",不是"大家都缩一点"。

## 3. 对课题的含义
- "按 batch 摊平后统一保守 padding"这条捷径走不通;溢出和节省是硬性 trade-off。
- 同一层里最忙 expert 需要接近 T(p90 为 0.87 T),一半 expert 需要 0。这是按 expert 分档 padding 的空间所在,也是必须处理"溢出/绕行"的原因。
- 卡间负载天然接近平均(1.3×),所以 EP 的 lane 置换在卡粒度收益有限;收益要在核/expert 粒度找。

## 4. 限定
- routing 来自 CPU bf16 的 HF 模型。卡上 MXFP6 图与 bf16 CPU 的 logits 相关只有 0.85(`tier_bench/EXPERT_PLACEMENT_PROOF.md` §9.7),个别 top-8 可能翻转,但抹不平 10 倍量级的偏斜。
- 只统计 prefill(prompt 前向),不含 decode;decode 的相邻 step 相似性要另做生成实验。
- Qwen3-30B-A3B 的 `router_aux_loss_coef=0.001`,负载均衡约束弱;load-balance loss 更强的模型可能偏斜更小,需另测。
- T 是真实 prompt 长度(36–146);EP QPC 编译的 prefill_seq_len=128 会 pad 到 128,比例结论不变。
- 100 题集(8 月 28 日)是 300 题集的前 100 题,routing 逐 bit 相同;两轮统计一致(见 `prefill_balance_gsm8k_100.txt`)。

## 5. 复现
```bash
source /home/chihao/qeff-venv/bin/activate
python3 tools/moe_collect_routing.py --bench gsm8k --n 300 --out perop_moe/routing/routing_gsm8k_300.npz   # ~20 min CPU
python3 tools/moe_prefill_balance.py perop_moe/routing/routing_gsm8k_300.npz --out perop_moe/routing/prefill_balance_gsm8k_300.txt
```
§2.3 与 §2.4 的两段临时脚本逻辑同 `moe_prefill_balance.py`(混合 workload:两份 npz 各抽 B/2;统一 padding:`max(counts − k·T·K/E, 0)` 求和 / (L·T·K))。

## 6. 文件
| 路径 | 内容 |
|---|---|
| `routing_gsm8k_300.npz`、`collect_gsm8k_300.log` | 本次 300 题 routing 与采集日志 |
| `routing_gsm8k.npz`、`routing_humaneval.npz` | 8 月 28 日的 100 题 GSM8K、100 题 HumanEval |
| `prefill_balance_gsm8k_{100,300}.txt` | `moe_prefill_balance.py` 完整输出 |
| `prefill_expert_distribution_gsm8k.txt` | **48 层 × 128 expert 每个 expert 的 token 份额分布**(300 题上的 mean/p50/p90/p99/max/零占比),带配置说明 |
| `prefill_counts_gsm8k_300.csv` | 逐 prompt 逐层的 128 个 expert token 数(14,400 行:prompt, layer, T, e0..e127) |
| `../../tools/moe_collect_routing.py`、`../../tools/moe_prefill_balance.py` | 采集与统计脚本 |
