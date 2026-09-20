#!/usr/bin/env python3
"""Layer-0 prefill routing of Qwen3-30B-A3B measured ON THE AI 100 CARD, GSM8K prompts.

Graph = embed_tokens -> layer0 input_layernorm -> layer0 self-attention (causal, RoPE, q/k norm) -> residual
        -> post_attention_layernorm -> router (linear + softmax + top-8), REAL checkpoint weights.
Output = top-8 expert indices per token (int32 [T, 8]) and the top-8 probabilities.  Layer-0 routing does not
depend on layer-0's experts, so the experts are not in the graph.  Prompts are right-padded to T; the causal
mask makes padding invisible to the real tokens, and padded rows are dropped when reading the result.

Stages (each cached in --out):
  export   : load shard, build module, export ONNX (fp32, external weights), CPU sanity check vs routing_gsm8k.npz
  compile  : qaic-compile, fp16 (+ optional -mxfp6-matmul, like the real deployment)
  run      : qaic-runner one prompt at a time on --device, raw bins in/out
  compare  : per-prompt overlap card vs CPU (bf16 HF run), per-expert token counts from the card -> CSV
  python3 moe_layer0_router_oncard.py --out 9.17.2026/layer0_router_oncard [--T 160] [--n 100] [--mxfp6]
"""
import argparse, json, os, subprocess, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HF = "/home/chihao/models/qwen3_30b_a3b/hf"
GSM = "/home/chihao/models/bench/gsm8k_test.jsonl"
CPU_NPZ = "/home/chihao/mllm/perop_moe/routing/routing_gsm8k.npz"
QAIC = "/opt/qti-aic/exec"


class Layer0Router(nn.Module):
    def __init__(self, cfg, sd, T):
        super().__init__()
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm, Qwen3MoeRotaryEmbedding
        H, nh, nkv, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.nh, self.nkv, self.hd, self.T, self.K = nh, nkv, hd, T, cfg.num_experts_per_tok
        f = lambda k: sd[k].float()
        self.embed = nn.Embedding.from_pretrained(f("model.embed_tokens.weight"), freeze=True)
        self.in_norm = Qwen3MoeRMSNorm(H, eps=cfg.rms_norm_eps); self.in_norm.weight.data = f("model.layers.0.input_layernorm.weight")
        self.post_norm = Qwen3MoeRMSNorm(H, eps=cfg.rms_norm_eps); self.post_norm.weight.data = f("model.layers.0.post_attention_layernorm.weight")
        self.q_norm = Qwen3MoeRMSNorm(hd, eps=cfg.rms_norm_eps); self.q_norm.weight.data = f("model.layers.0.self_attn.q_norm.weight")
        self.k_norm = Qwen3MoeRMSNorm(hd, eps=cfg.rms_norm_eps); self.k_norm.weight.data = f("model.layers.0.self_attn.k_norm.weight")
        lin = lambda k, o, i: nn.Linear(i, o, bias=False)
        self.q = lin("q", nh * hd, H); self.q.weight.data = f("model.layers.0.self_attn.q_proj.weight")
        self.k = lin("k", nkv * hd, H); self.k.weight.data = f("model.layers.0.self_attn.k_proj.weight")
        self.v = lin("v", nkv * hd, H); self.v.weight.data = f("model.layers.0.self_attn.v_proj.weight")
        self.o = lin("o", H, nh * hd); self.o.weight.data = f("model.layers.0.self_attn.o_proj.weight")
        self.router_w = nn.Parameter(f("model.layers.0.mlp.gate.weight"), requires_grad=False)      # [E, H]
        rot = Qwen3MoeRotaryEmbedding(cfg)
        cos, sin = rot(torch.zeros(1, T, H), torch.arange(T)[None])                                  # [1, T, hd]
        self.register_buffer("cos", cos.float()); self.register_buffer("sin", sin.float())
        self.register_buffer("mask", torch.full((T, T), float("-inf")).triu(1))

    @staticmethod
    def rope(x, cos, sin):                                                     # x [1, h, T, hd]
        h = x.shape[-1] // 2
        rx = torch.cat((-x[..., h:], x[..., :h]), dim=-1)
        return x * cos[:, None] + rx * sin[:, None]

    def forward(self, input_ids):                                              # [1, T] int32
        x = self.embed(input_ids)                                              # [1, T, H]
        h = self.in_norm(x)
        T = self.T
        q = self.q_norm(self.q(h).view(1, T, self.nh, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k(h).view(1, T, self.nkv, self.hd)).transpose(1, 2)
        v = self.v(h).view(1, T, self.nkv, self.hd).transpose(1, 2)
        q, k = self.rope(q, self.cos, self.sin), self.rope(k, self.cos, self.sin)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-1, -2)) * (self.hd ** -0.5) + self.mask
        att = F.softmax(att, dim=-1, dtype=torch.float32)
        a = (att @ v).transpose(1, 2).reshape(1, T, self.nh * self.hd)
        x = x + self.o(a)
        h = self.post_norm(x)[0]                                               # [T, H]
        if getattr(self, "return_h", False):
            return h                                                            # MoE-block input (what the experts see)
        logits = F.linear(h, self.router_w)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_p, top_i = torch.topk(probs, self.K, dim=-1)
        return top_i.to(torch.int32), top_p


def prompts(tok, n, T):
    items = [json.loads(l) for l in open(GSM)][:n]
    out = []
    for it in items:
        s = tok.apply_chat_template([{"role": "user", "content": it["question"]}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        ids = tok(s, return_tensors="np")["input_ids"][0]
        assert len(ids) <= T, (len(ids), T)
        out.append(ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--T", type=int, default=160)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--mxfp6", action="store_true", help="also -mxfp6-matmul (deployment flags)")
    ap.add_argument("--stage", default="all", choices=["all", "export", "compile", "run", "compare"])
    a = ap.parse_args(); D = os.path.abspath(a.out); os.makedirs(D, exist_ok=True)
    t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:6.1f}s]", *x, flush=True)
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(HF); tok.pad_token = tok.pad_token or tok.eos_token
    pad_id = tok.pad_token_id
    ids = prompts(tok, a.n, a.T); log(f"{len(ids)} GSM8K prompts, max {max(map(len, ids))} tokens, T={a.T}")
    onnx_path = os.path.join(D, "layer0_router.onnx"); qpc = os.path.join(D, "qpc")

    if a.stage in ("all", "export") and not os.path.exists(onnx_path):
        from safetensors.torch import load_file
        cfg = AutoConfig.from_pretrained(HF)
        sd = load_file(os.path.join(HF, "model-00001-of-00016.safetensors"))
        sd = {k: v for k, v in sd.items() if "experts" not in k and (k.startswith("model.layers.0.") or k.startswith("model.embed"))}
        m = Layer0Router(cfg, sd, a.T).eval(); del sd
        # CPU sanity: this module (fp32) vs the bf16 full-model hook capture, prompt 0..2, real tokens only
        z = np.load(CPU_NPZ, allow_pickle=True)
        for p in range(3):
            x = np.full((1, a.T), pad_id, np.int32); x[0, :len(ids[p])] = ids[p]
            with torch.no_grad(): ti, _ = m(torch.from_numpy(x))
            ref = z[f"p{p}"][0]; got = ti[:len(ids[p])].numpy()
            assert ref.shape == got.shape, (ref.shape, got.shape)
            ov = np.mean([len(set(ref[t]) & set(got[t])) / 8 for t in range(len(ref))])
            log(f"  cpu check prompt {p}: top-8 overlap fp32-module vs bf16-HF = {ov:.3f}  (top-1 same {np.mean(ref[:,0]==got[:,0]):.3f})")
        x = torch.from_numpy(np.full((1, a.T), pad_id, np.int32))
        torch.onnx.export(m, (x,), onnx_path, input_names=["input_ids"], output_names=["top_idx", "top_prob"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
        import onnx
        mo = onnx.load(onnx_path); onnx.save_model(mo, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True,
                                                   location="layer0_router.data", size_threshold=1024)
        log(f"exported {onnx_path}")

    if a.stage in ("all", "compile") and not os.path.exists(os.path.join(qpc, "programqpc.bin")):
        cmd = [f"{QAIC}/qaic-compile", "-aic-hw", "-aic-hw-version=ai100", f"-m={onnx_path}", "-convert-to-fp16",
               "-aic-num-cores=16", "-mos=1", "-aic-enable-depth-first", f"-aic-binary-dir={qpc}"] + (["-mxfp6-matmul"] if a.mxfp6 else [])
        subprocess.run(["rm", "-rf", qpc]); r = subprocess.run(cmd, capture_output=True, text=True)
        open(os.path.join(D, "compile.log"), "w").write(r.stdout + r.stderr)
        if not os.path.exists(os.path.join(qpc, "programqpc.bin")): sys.exit("COMPILE FAILED:\n" + (r.stdout + r.stderr)[-2000:])
        log("compiled " + ("fp16+mxfp6" if a.mxfp6 else "fp16"))

    RUN = os.path.join(D, "run"); os.makedirs(RUN, exist_ok=True)
    if a.stage in ("all", "run"):
        for p, seq in enumerate(ids):
            od = os.path.join(RUN, f"p{p}")
            if os.path.exists(os.path.join(od, "done")): continue
            os.makedirs(od, exist_ok=True)
            x = np.full((1, a.T), pad_id, np.int32); x[0, :len(seq)] = seq; x.tofile(os.path.join(od, "input_ids.bin"))
            r = subprocess.run([f"{QAIC}/qaic-runner", "-t", qpc, "-d", str(a.device), "-i", os.path.join(od, "input_ids.bin"),
                                "--write-output-dir", od, "--num-iter", "1"], capture_output=True, text=True)
            outs = [f for f in os.listdir(od) if f.startswith("top_idx")]
            if not outs: sys.exit(f"prompt {p}: no output. {r.stdout[-800:]} {r.stderr[-800:]}")
            open(os.path.join(od, "done"), "w").write(outs[0])
            if p % 10 == 0: log(f"  ran prompt {p}")
        log("all prompts run on card")

    if a.stage in ("all", "compare"):
        z = np.load(CPU_NPZ, allow_pickle=True); E = 128
        import csv
        rows, ov_all, top1_all, tot_tok = [], [], [], 0
        card_cnt = np.zeros(E, np.int64); cpu_cnt = np.zeros(E, np.int64)
        for p, seq in enumerate(ids):
            od = os.path.join(RUN, f"p{p}"); f = open(os.path.join(od, "done")).read()
            got = np.fromfile(os.path.join(od, f), np.int32).reshape(a.T, 8)[:len(seq)]
            ref = z[f"p{p}"][0].astype(np.int32)
            ov = np.array([len(set(ref[t]) & set(got[t])) / 8 for t in range(len(seq))])
            ov_all.append(ov.mean()); top1_all.append(np.mean(ref[:, 0] == got[:, 0])); tot_tok += len(seq)
            c = np.bincount(got.ravel(), minlength=E); card_cnt += c; cpu_cnt += np.bincount(ref.ravel(), minlength=E)
            rows.append([p, len(seq), round(ov.mean(), 4), round(float(np.mean(ref[:, 0] == got[:, 0])), 4)] + c.tolist())
        with open(os.path.join(D, "layer0_prefill_gsm8k_oncard.csv"), "w") as fo:
            w = csv.writer(fo); w.writerow(["prompt", "tokens", "top8_overlap_vs_cpu", "top1_same_vs_cpu"] + [f"e{i}" for i in range(E)]); w.writerows(rows)
        tot_assign = tot_tok * 8
        diff = np.abs(card_cnt - cpu_cnt)
        summ = [f"prompts {len(ids)}  tokens {tot_tok}  assignments {tot_assign}",
                f"per-token top-8 overlap card vs CPU(bf16): mean {np.mean(ov_all):.4f}  min prompt {np.min(ov_all):.4f}",
                f"top-1 expert identical: {np.mean(top1_all):.4f}",
                f"per-expert count |card - cpu|: total {diff.sum()} of {tot_assign} ({100*diff.sum()/tot_assign:.2f}%), max single expert {diff.max()}",
                "top-10 experts on card: " + " ".join(f"e{i}:{card_cnt[i]}" for i in np.argsort(-card_cnt)[:10]),
                "top-10 experts on cpu : " + " ".join(f"e{i}:{cpu_cnt[i]}" for i in np.argsort(-cpu_cnt)[:10])]
        open(os.path.join(D, "summary.txt"), "w").write("\n".join(summ) + "\n"); print("\n".join(summ))


if __name__ == "__main__":
    main()
