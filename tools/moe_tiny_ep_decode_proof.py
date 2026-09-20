"""Proof at small scale: EP flavour is selected for a NON-prefill-only build and lands in the graph that
serves decode. Tiny random Qwen3-MoE, real QEfficient transforms/export, then a decode-only compile."""
import os, sys, json, torch
S = os.path.dirname(os.path.abspath(__file__)); os.environ["QEFF_HOME"] = S
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
HF = f"{S}/hf"
if not os.path.exists(f"{HF}/config.json"):
    cfg = Qwen3MoeConfig(vocab_size=1024, hidden_size=256, intermediate_size=512, moe_intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                         num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
                         norm_topk_prob=True, max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(0); Qwen3MoeForCausalLM(cfg).half().save_pretrained(HF)
from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.transformers.moe import MoEFlavour, select_moe_flavour
from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import QEffQwen3MoeSparseMoeBlock
print("[A] select_moe_flavour, is_prefill=False:")
print("    no request  ->", select_moe_flavour(supported_flavours=QEffQwen3MoeSparseMoeBlock.supported_moe_flavours, is_prefill=False))
print("    'expert_parallel' ->", select_moe_flavour(supported_flavours=QEffQwen3MoeSparseMoeBlock.supported_moe_flavours, is_prefill=False, requested_flavour="expert_parallel"))
MOE = {"flavour": "expert_parallel", "cores_per_expert": 1, "tree_reduce": False, "expert_parallel_chunk_size": 256}
m = QEFFAutoModelForCausalLM.from_pretrained(HF, torch_dtype=torch.float16, continuous_batching=False, qaic_config={"moe_config": MOE})
try:
    m.compile(prefill_only=False, prefill_seq_len=32, ctx_len=64, batch_size=8, num_devices=4, num_cores=16,
              mxfp6_matmul=True, mos=1, aic_enable_depth_first=True, qaic_config={"moe_config": MOE})
    print("[B] QEfficient compile succeeded")
except Exception as e:
    print("[B] QEfficient compile step failed (expected on this SDK):", str(e).splitlines()[-1][:120])
print("[C] per-module flavour after transform (prefill_only=False):")
for name, mod in m.model.named_modules():
    if hasattr(mod, "_moe_flavour"):
        print(f"    {name}: _moe_flavour={mod._moe_flavour}  stages={getattr(mod,'num_pipeline_stages',None)} lanes={getattr(mod,'num_parallelized_experts',None)} experts_per_soc={getattr(mod,'experts_per_soc',None)}")
print("[D] hash_params:", {k: v for k, v in m.hash_params.items() if "moe" in k or k == "prefill_only"})
print("[E] onnx:", m.onnx_path)
import onnx
g = onnx.load(m.onnx_path, load_external_data=False).graph
l0 = [n for n in g.node if "/layers.0/mlp/" in n.name]
ops = {}
for n in l0: ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("    layer0 /mlp/ op counts:", ops)
gathers = [n for n in l0 if n.op_type == "Gather"]
print("    Gather nodes in layer0 mlp:", [(n.name, list(n.input)) for n in gathers][:4])
for init in g.initializer:
    if "layers.0" in init.name and "expert" in init.name.lower():
        print("    expert weight", init.name, list(init.dims))
print("[F] specializations written by QEfficient:")
import glob; q = sorted(glob.glob(f"{S}/**/qpc-*", recursive=True), key=os.path.getmtime)[-1]; print("   ", q); print(open(f"{q}/specializations.json").read())
open(f"{S}/Q", "w").write(q)
