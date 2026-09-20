"""EP for DECODE: same expert_parallel flavour as run_ep.py, but prefill_only=False so the QPC gets a
decode specialization (seq_len=1) that also runs the EP graph, and batch_size=B so decode sees T=B rows.
No QEfficient code change: select_moe_flavour honours an explicit flavour regardless of is_prefill."""
import time, json, resource, torch, os, sys

B = int(os.environ.get("B", "8"))
t0 = time.time()
def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[{time.time()-t0:7.0f}s | peakRSS {rss:6.1f} GB]", *a, flush=True)

from QEfficient import QEFFAutoModelForCausalLM

HF = "/home/chihao/models/qwen3_30b_a3b/hf"
MOE = {"flavour": "expert_parallel", "cores_per_expert": 1,
       "tree_reduce": False, "expert_parallel_chunk_size": 256}

log("QEfficient", __import__("QEfficient").__version__, "| decode batch B =", B)
log("moe_config =", json.dumps(MOE))
m = QEFFAutoModelForCausalLM.from_pretrained(
    HF, torch_dtype=torch.float16, continuous_batching=False,
    qaic_config={"moe_config": MOE},
)
log("loaded")

log(f"compile(prefill_only=False, batch_size={B}, num_devices=4, num_cores=16) ...")
qpc = m.compile(
    prefill_only=False,
    prefill_seq_len=128,
    ctx_len=256,
    batch_size=B,
    num_devices=4,
    num_cores=16,
    mxfp6_matmul=True,
    mos=1,
    aic_enable_depth_first=True,
    qaic_config={"moe_config": MOE},   # REQUIRED: compile-time transform re-selects the flavour from THIS qaic_config
)
log("COMPILE_OK ->", qpc)

for name, mod in m.model.named_modules():
    if hasattr(mod, "_moe_flavour"):
        log(f"  module: {name}  _moe_flavour: {getattr(mod,'_moe_flavour',None)}")
        for k in ("num_pipeline_stages","num_parallelized_experts","experts_per_soc","expert_parallel_num_packed_chunks"):
            log(f"  {k:<36}: {getattr(mod,k,'<unset>')}")
        break
log("hash_params:", {k: v for k, v in m.hash_params.items() if "moe" in k or k in ("prefill_only",)})
