#!/usr/bin/env python3
"""在每张卡的编译权重里,直接找某一个 expert 的字节,判定它是否被四等分。

前提:QPC 必须用 -convert-to-fp16 且【不加】-mxfp6-matmul 编译,否则权重被重新编码,
     字节对不上。expert 权重的 fp16 源来自 ONNX 外部文件(已逐比特验证 == checkpoint)。

方法:从 expert e 的 [2048, 768] 矩阵按 4x4 网格取 16 个签名(每个 16 个连续 fp16 = 32 字节,
     实际唯一),在四个 slice 的 constants.bin 里搜。

  * 若 expert 被四等分,每张卡应命中约 1/4 的签名
  * 签名的落点还能判定切分轴:
      同一【列区】的 4 个签名落同一张卡  -> 沿 dim1(768) 切,column-parallel
      同一【行区】的 4 个签名落同一张卡  -> 沿 dim0(2048) 切,row-parallel

用法:
  python3 moe_find_expert_bytes.py --expert 91 --slices <dir with QAicGraph_slice0N_dir/constants.bin>
"""
import argparse, os, mmap, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--onnx-dir", default="/home/chihao/models/qwen3_30b_a3b/export-126f04c76d4a0568")
ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--proj", default="gate_proj")
ap.add_argument("--expert", type=int, default=91)
ap.add_argument("--slices", required=True, help="包含 QAicGraph_slice0N_dir/constants.bin 的目录")
ap.add_argument("--sig-len", type=int, default=16, help="每个签名多少个 fp16")
a = ap.parse_args()

E, H, I = 128, 2048, 768
ext = os.path.join(a.onnx_dir, f"model.layers.{a.layer}.mlp.experts.{a.proj}")
w = np.memmap(ext, dtype=np.float16, mode="r").reshape(E, H, I)
M = np.array(w[a.expert])                        # [2048, 768]
print(f"expert {a.expert}  layer {a.layer}  {a.proj}   矩阵 {M.shape}  fp16")

ROWS = [256, 768, 1280, 1792]                    # dim0(2048) 的四个 1/4 的中心
COLS = [96, 288, 480, 672]                       # dim1(768)  的四个 1/4 的中心
sigs = []
for ri, r in enumerate(ROWS):
    for ci, c in enumerate(COLS):
        s = M[r, c:c + a.sig_len].tobytes()
        sigs.append((ri, ci, r, c, s))
print(f"取了 {len(sigs)} 个签名,每个 {a.sig_len} 个 fp16 = {a.sig_len*2} 字节\n")

slices = []
for n in range(4):
    for cand in (os.path.join(a.slices, f"QAicGraph_slice0{n}_dir", "constants.bin"),
                 os.path.join(a.slices, f"QAicGraph_slice0{n}", "constants.bin")):
        if os.path.exists(cand):
            slices.append((n, cand)); break
if len(slices) != 4:
    raise SystemExit(f"只找到 {len(slices)} 个 constants.bin —— 先 qaic-qpc extract")
for n, p in slices:
    print(f"  slice0{n}: {os.path.getsize(p)/2**30:6.2f} GiB  {p}")

hits = {}
for n, p in slices:
    with open(p, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        for (ri, ci, r, c, s) in sigs:
            if mm.find(s) != -1:
                hits.setdefault((ri, ci), []).append(n)
        mm.close()
    print(f"  slice0{n} 扫描完成")

print(f"\n=== 每个签名落在哪张卡 (行=dim0 的 1/4, 列=dim1 的 1/4) ===")
print(f"{'':14s}" + "".join(f"col区{ci} (c={c:3d})".rjust(16) for ci, c in enumerate(COLS)))
for ri, r in enumerate(ROWS):
    row = f"row区{ri} (r={r:4d})".ljust(14)
    for ci in range(len(COLS)):
        h = hits.get((ri, ci), [])
        row += ("未找到" if not h else "card" + ",".join(map(str, h))).rjust(16)
    print(row)

per = {n: 0 for n in range(4)}
for h in hits.values():
    for n in h: per[n] += 1
found = len(hits)
print(f"\n=== 命中统计 ===")
print(f"  找到的签名: {found}/{len(sigs)}")
for n in range(4):
    print(f"  card{n}: {per[n]:2d} 个签名  ({per[n]/max(found,1)*100:5.1f}%)")

print(f"\n=== 判定 ===")
if found == 0:
    print("  一个签名都没找到 -> 权重被重新编码或重排(检查是否误用了 -mxfp6-matmul)")
else:
    bycol = all(len({tuple(hits.get((ri, ci), [])) for ri in range(len(ROWS))}) == 1
                for ci in range(len(COLS)) if any((ri, ci) in hits for ri in range(len(ROWS))))
    byrow = all(len({tuple(hits.get((ri, ci), [])) for ci in range(len(COLS))}) == 1
                for ri in range(len(ROWS)) if any((ri, ci) in hits for ci in range(len(COLS))))
    if bycol and not byrow:
        print("  同一【列区】的签名落同一张卡 -> 沿 dim1 (768) 切,column-parallel")
    elif byrow and not bycol:
        print("  同一【行区】的签名落同一张卡 -> 沿 dim0 (2048) 切,row-parallel")
    else:
        print("  落点模式不是简单的行/列四等分 —— 见上表")
    ok = all(2 <= per[n] <= 6 for n in range(4))
    print(f"  四张卡各命中 {[per[n] for n in range(4)]} -> "
          f"{'均匀,expert 被四等分 ✅' if ok else '不均匀 ⚠'}")
