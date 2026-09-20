#!/usr/bin/env python3
"""逐比特验证:ONNX 的 [128, H, I] expert 张量,第 e 片 == checkpoint 的 experts.<e>。

这是"trace 里的某个 op 处理的就是那 128 个 expert"这条链上唯一需要数值验证的一环;
其余各环(节点名 -> 张量名 -> 外部文件 -> thread_name -> card/core)都是名字匹配。

用法:
  python3 moe_verify_expert_tensor.py [--layer 0] [--experts 0,5,91,127] [--proj gate_proj]
"""
import argparse, json, os, numpy as np, torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--hf",   default="/home/chihao/models/qwen3_30b_a3b/hf")
ap.add_argument("--onnx-dir", default="/home/chihao/models/qwen3_30b_a3b/export-126f04c76d4a0568")
ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--experts", default="0,5,91,127")
ap.add_argument("--proj", default="gate_proj", choices=["gate_proj", "up_proj"])
a = ap.parse_args()

L = a.layer
ext = os.path.join(a.onnx_dir, f"model.layers.{L}.mlp.experts.{a.proj}")
sz = os.path.getsize(ext)
E, H, I = 128, 2048, 768
exp = E * H * I * 2
print(f"ONNX 外部权重 {os.path.basename(ext)}")
print(f"  大小 {sz:,} B   期望 128x2048x768x2 = {exp:,}   {'OK' if sz == exp else 'MISMATCH'}")
if sz != exp:
    raise SystemExit("形状不符,后续比对无意义")
w = torch.from_numpy(np.array(np.memmap(ext, dtype=np.float16, mode="r").reshape(E, H, I)))

idx = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
def ck(e):
    k = f"model.layers.{L}.mlp.experts.{e}.{a.proj}.weight"
    with safe_open(os.path.join(a.hf, idx[k]), framework="pt") as f:
        return f.get_tensor(k)

print(f"\ncheckpoint  model.layers.{L}.mlp.experts.<e>.{a.proj}.weight   bf16 [{I}, {H}]")
print(f"{'expert':>8s}  {'逐比特相同':>10s}  {'max|diff|':>12s}")
ok = True
es = [int(x) for x in a.experts.split(",")]
for e in es:
    b = ck(e).to(torch.float16).T.contiguous()
    same = torch.equal(w[e], b)
    d = (w[e].float() - b.float()).abs().max().item()
    ok &= same
    print(f"{e:8d}  {'YES' if same else 'NO':>10s}  {d:12g}")

# 对照:错位比对必须不同,否则说明比对本身没有区分力
e0, e1 = es[0], es[1] if len(es) > 1 else es[0] + 1
b = ck(e1).to(torch.float16).T.contiguous()
ctrl = not torch.equal(w[e0], b)
print(f"\n对照 ONNX[{e0}] vs checkpoint expert {e1}: "
      f"{'不同 (比对有区分力)' if ctrl else '相同 (比对无效!)'}   "
      f"max|diff|={(w[e0].float()-b.float()).abs().max().item():g}")

print("\n结论:", "ONNX 的 [128,H,I] 张量第 e 片确实就是 checkpoint 的 expert e"
      if (ok and ctrl) else "验证未通过")
