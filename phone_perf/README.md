# phone_perf/ — 手机端(Snapdragon NPU / OpenCL / CPU)profiling 遗留资料

与 AI 100 无关,是更早在手机上做 Qwen3 推理 profiling 时的原始文件,从 repo 根目录收进来。

| 文件 | 内容 |
|---|---|
| `README-profile-opencl.md` | OpenCL profiling 的测法 |
| `README-bench-npu-attn.md` | NPU attention 微基准计划 |
| `NPU_SPEC_DECODE_PLAN.md` | NPU 上 n-gram 投机解码的阶段计划 |
| `qnn_profile_breakdown.txt`、`qnn_profile_old.csv`、`qnn_profile_ctx512.csv` | QNN profiler 输出及逐层拆解 |
| `qwen3*.perf`、`qwen3_aot.perfetto-trace`、`qwen3_cpu/`、`qwen3-cpu-perf/`、`qwen3-npu-perf/`、`viewer_out.csv`、`qwen3_opencl_norma;.csv` | 各后端的 perf 原始数据 |
| `adb_cpu_push.sh`、`adb_npu_push.sh`、`cmake_android_qnn.log` | 推到手机的脚本和 Android 构建日志 |
| `NIAH1_1k.txt` | 测试用的长文本 |

解析脚本在 `../tools/`:`parse_qnn_profile.py`、`explain_qnn_profile.py`、`explain_qnn_lines.py`、`parse_opencl_profile.py`、`per_layer_decode_total.py`。
