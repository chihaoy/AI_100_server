# 证据包:一个 expert 精确平分在四张卡上

**待证命题**:Qwen3-30B-A3B 在 4 张 AI 100 上按 QEfficient 默认配置(`mdp_ts_4.json`,单 partition
4 卡 x 16 核)编译后,**每一个 expert 都被切成四等份、四张卡各持 1/4**;不存在任何 expert 到卡的分配。

**输入固定**:两次运行都喂同一个 token `9707`(`perop_make_input.py --token-ids 9707`),
decode 规格 `seq_len=1, ctx_len=256`。

---

## 实验设计

只改一个自变量:**被 router 选中的 expert 集合**。

| | NATURAL | FORCED |
|---|---|---|
| ONNX | `export-126f04c76d4a0568/` | `export_forced/`(硬链接副本,48 个 router 权重文件置零) |
| router | 原始权重 | **全零** -> logits 全相等 -> `TopK` 返回索引前 8 个 |
| 实际选中的 expert | 由 token 9707 决定的任意 8 个 | **恒为 {0,1,2,3,4,5,6,7}** |
| QPC | `4card/qpc_decode/` | `4card/qpc_forced/` |

`expert 0-7` 是**连续块**。在任何"每卡 32 个 expert"的专家并行(EP)方案下,它们**必然全部落在
card0**。所以两种放置方式给出截然不同的预测:

| 观测量 | **EP 的预测** | **TP 的预测** |
|---|---|---|
| `mlp/Gather_3` 每卡 KB | **9600 / 0 / 0 / 0** | 2400 / 2400 / 2400 / 2400 |
| 卡间极差 | 无穷大 | ~0% |
| card0 的 MoE cycles | ~4x | 与 NATURAL 一致 |

---

## 结果

```
### NATURAL router (token 9707, router 未改)
op (layer 0)                     card0       card1       card2       card3          合计     极差%
mlp/gate/MatMul                  446.3       446.3       446.3       446.3      1785.3   0.01%
mlp/Gather_3                    2400.1      2400.1      2400.2      2400.0      9600.3   0.01%
mlp/Gather_4                    2400.5      2400.1      2399.9      2399.8      9600.3   0.03%
mlp/Gather_5                    2400.1      2400.0      2399.9      2400.2      9600.2   0.01%
mlp/MatMul                      7328.1      7328.7      7328.6      7328.6     29314.1   0.01%
mlp/MatMul_1                    9376.4      9375.9      9376.3      9377.0     37505.7   0.01%
mlp/MatMul_2                    7496.0      7496.3      7496.0      7495.9     29984.3   0.00%
mlp/Einsum                        32.0        32.0        32.0        32.0       128.0   0.00%
self_attn/q_proj/MatMul         5508.4      5507.9      5508.2      5507.8     22032.3   0.01%

### FORCED router (token 9707, router 权重全零 -> 强制只选 expert 0-7)
op (layer 0)                     card0       card1       card2       card3          合计     极差%
mlp/gate/MatMul                  446.4       446.3       446.3       446.3      1785.3   0.01%
mlp/Gather_3                    2400.1      2400.0      2400.0      2400.0      9600.1   0.00%
mlp/Gather_4                    2400.3      2400.6      2400.5      2400.4      9601.8   0.01%
mlp/Gather_5                    2399.9      2400.0      2400.1      2400.4      9600.5   0.02%
mlp/MatMul                      7329.0      7327.9      7328.0      7328.1     29313.0   0.01%
mlp/MatMul_1                    9375.9      9375.5      9376.4      9375.1     37502.9   0.01%
mlp/MatMul_2                    7495.7      7495.8      7496.4      7495.8     29983.8   0.01%
mlp/Einsum                        32.0        32.0        32.0        32.0       128.0   0.00%
self_attn/q_proj/MatMul         5508.2      5508.4      5508.7      5508.2     22033.4   0.01%
```

`InfPCycles`(qaic-runner 报告,30 次迭代平均):

| | card0 | card1 | card2 | card3 | 卡间极差 |
|---|---:|---:|---:|---:|---:|
| NATURAL | 15,719,938 | 15,690,576 | 15,694,282 | 15,694,622 | 0.187% |
| FORCED | 15,635,222 | 15,607,427 | 15,611,109 | 15,611,165 | 0.178% |

**折叠检验**:FORCED 运行的 trace 里 `gate/TopK` 事件数 = 1728,说明 router 确实执行了 TopK,
没有被编译器常量折叠成静态索引。

---

## 结论

强制把被选中的 expert 从"任意 8 个"改成"恒为 0-7"后,**每卡字节分布纹丝不动**,
9 个 op 的卡间极差全部 <= 0.03%。EP 预测的 `9600/0/0/0` 没有出现。

单个 expert 的账:

```
9600 KB (8 个 expert 的 gate_proj, MXFP6) / 8 = 1200 KB per expert
1200 KB / 4 张卡                              =  300 KB per expert per card  <- 精确 1/4
```

**每个 expert 被切成四等份,四张卡各持 1/4,且与被选中的是哪些 expert 无关。**

---

## 数值怎么来的

`KB = opPCycle / PCyclesPerKB`,两个字段都是 `qaic-opstats` 写进 trace 的原始输出,没有任何模型假设。
`op_level_*_slice0N.summary.txt` 是 SDK `qaic-opstats --summary` 的**逐字节原样输出**,可独立核对。

## 自己复核

```bash
bash perop_moe/proof_expert_split/verify.sh          # 从 trace 重新生成本表
grep -c aictopk perop_moe/decode_forced/out/*slice00*.summary.txt   # 确认 TopK 存在
```

## 本目录文件

| 文件 | 内容 |
|---|---|
| `per_card_moe_ops.txt` / `.csv` | 上面那张证据表(csv 含 events / kb / pcycles 三列) |
| `op_level_natural_slice0{0..3}.summary.txt` | SDK 原样输出,自然路由,每卡一份 |
| `op_level_forced_slice0{0..3}.summary.txt` | SDK 原样输出,强制路由,每卡一份 |
| `verify.sh` | 从 trace 重新生成证据表 |
| `../../tools/moe_proof_percard_ops.py` | 提取脚本 |

原始 trace(343 MB x 8,git 忽略)在 `perop_moe/decode_realtok/out/` 和 `perop_moe/decode_forced/out/`。
重新生成:`/home/chihao/models/qwen3_30b_a3b/4card/pass_{b,c,d}_*.sh`。
