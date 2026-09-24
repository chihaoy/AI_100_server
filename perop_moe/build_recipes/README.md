# build_recipes/ — 整模型 QPC 的重建配方

`/home/chihao/models/qwen3_30b_a3b/{ep,4card}` 于 2026-09-22 删除(356 GB,为腾磁盘)。
这里保留了里面所有 2 MB 以下的编译脚本、specialization、custom_io、mdp 配置和日志,
足以重建当时的 QPC。原始权重 `/home/chihao/models/qwen3_30b_a3b/hf` 仍在。

| 目录 | 内容 |
|---|---|
| `ep/` | EP prefill/decode 的导出与编译脚本(`compile_ep.sh` 等)、日志、QEfficient 缓存的 export params |
| `4card/` | 更早的四卡路线:`pass_a`–`pass_e` 各阶段脚本与日志、`custom_io.yaml`、`mdp_ts_4.json`、`moe_mdp_dump.json`、三个 spec json |

## 重建 naive 整模型 prefill QPC(我们所有整模型对照的来源)

1. 按 `../README.md` "怎么跑的" 一节用 QEfficient 1.23 导出(约 25 min,57 GB ONNX):
   `prefill_only=True, enable_chunking=True, prefill_seq_len=128, ctx_len=256, batch_size=1,
    num_devices=4, num_cores=16, mxfp6_matmul=True, mos=1, aic_enable_depth_first=True`
2. 把 `specializations.json` 的 `symbols` 拍平以兼容 SDK 1.21.6(见 `../README.md` 坑 #8),
   然后用 `ep/compile_ep.sh` 里的 `qaic-compile` 命令行编译(约 10m28s,25 GB QPC)。

被删掉的实测结果已记录在:`../README.md`、`../EP_MECHANISM.md`、`../EP_DATAFLOW.md`、
`../../9.17.2026/结论总结/端到端prefill_naive_vs_两QPC.md`(naive 一次 prefill 520 ms 主机 / 498.6 ms 卡上)。
