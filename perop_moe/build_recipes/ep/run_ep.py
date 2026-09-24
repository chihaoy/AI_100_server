"""EP_PLAN 阶段 2+3: 用 QEfficient 1.23 的 expert_parallel 导出并编译 4 卡 prefill QPC."""
import time, json, resource, torch, os, sys

t0 = time.time()
def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[{time.time()-t0:7.0f}s | peakRSS {rss:6.1f} GB]", *a, flush=True)

from QEfficient import QEFFAutoModelForCausalLM

HF = "/home/chihao/models/qwen3_30b_a3b/hf"
MOE = {"flavour": "expert_parallel", "cores_per_expert": 1,
       "tree_reduce": False, "expert_parallel_chunk_size": 256}

log("QEfficient", __import__("QEfficient").__version__)
log("moe_config =", json.dumps(MOE))
log("loading model (float16) ...")
m = QEFFAutoModelForCausalLM.from_pretrained(
    HF, torch_dtype=torch.float16, continuous_batching=False,
    qaic_config={"moe_config": MOE},
)
log("loaded, dtype =", next(m.model.parameters()).dtype)

log("compile(prefill_only=True, enable_chunking=True, num_devices=4, num_cores=16) ...")
qpc = m.compile(
    prefill_only=True,
    enable_chunking=True,
    prefill_seq_len=128,
    ctx_len=256,
    batch_size=1,
    num_devices=4,
    num_cores=16,
    mxfp6_matmul=True,
    mos=1,
    aic_enable_depth_first=True,
)
log("COMPILE_OK ->", qpc)

# ---- 验证 EP 真的生效了 ----
log("=== 生效的 MoE 配置 ===")
for name, mod in m.model.named_modules():
    if hasattr(mod, "_moe_flavour"):
        log(f"  module              : {name}")
        log(f"  _moe_flavour        : {getattr(mod,'_moe_flavour',None)}")
        for k in ("num_devices","cores_per_expert","total_avl_cores","num_pipeline_stages",
                  "num_parallelized_experts","experts_per_soc","tree_reduce",
                  "expert_parallel_num_packed_chunks"):
            log(f"  {k:<26}: {getattr(mod,k,'<未设置>')}")
        break
else:
    log("  ⚠ 没有任何模块带 _moe_flavour —— EP 可能没生效")
