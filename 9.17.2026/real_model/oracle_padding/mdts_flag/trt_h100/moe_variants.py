#!/usr/bin/env python3
"""MoE-block variants for the TensorRT/H100 experiments (RANKTIER_TENSORRT_H100_EXPERIMENT_PLAN.md, section 5), written
with ONNX-exportable primitives so that TensorRT compiles every arm from the same operator set.

One Qwen3-30B-A3B MoE layer on one device that owns all E = 128 experts, chunk length T fixed per graph. Inputs: x [T, d]
(the layer's MoE input), ids [T, k] (selected experts, replayed), w [T, k] (normalized routing weights, replayed). Outputs:
y [T, d] (the routed-expert sum) and ovf [1] (1 if some expert received more tokens than its lane's capacity).

Shared routing metadata in every arm: one-hot counts, exclusive prefix sums (each assignment's slot in its expert's rows),
a flat row per assignment, a token table and a weight table scattered from those rows. Arms:
  capacity  'uniform'  one tier of E lanes at capacity T (drops nothing)
            'tiered'   tiers (lanes, capacity) by load rank, calibrated (calibrate_tiers.py)
  combine   'dense'    repaired dense combine: each tier scatters its weighted rows into a zero-filled [lanes, T+1, d]
                       accumulator (row T absorbs padding) and sums over lanes; no read of the zeros, no transpose
            'token'    token-owned: all tiers' weighted rows in one flat buffer, each token gathers its k rows and sums them
  ranked    True       experts are ordered by run-time count (ties: lower index first); lanes gather their weights
            False      fixed native order with constant weights (only meaningful for 'uniform')
Weights: gate and up are concatenated to one [E, d, 2f] tensor in every arm. Padding rows read a zero token row and carry
weight 0, so they contribute exact zeros. The overflow flag compares every assignment's slot with its lane's capacity;
overflowing assignments are routed to a dummy row (dropped) and must be recomputed by the caller (uniform engine)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def tier_tables(tiers, E):
    L = [l for l, _ in tiers]; C = [c for _, c in tiers]
    assert sum(L) == E, (L, E)
    a = np.cumsum([0] + L)[:-1]; off = np.cumsum([0] + [l * c for l, c in tiers])
    tier_of = np.concatenate([[i] * l for i, l in enumerate(L)]); lane_of = np.concatenate([np.arange(l) for l in L])
    cap = np.array([C[i] for i in tier_of]); offr = np.array([off[i] + lane_of[r] * C[i] for r, i in enumerate(tier_of)])
    return a.tolist(), off.tolist(), cap, offr


class MoEBlock(nn.Module):
    def __init__(self, wgu, wd, T, tiers, combine, ranked=True, k=8):
        super().__init__()
        self.register_buffer('wgu', wgu)                   # [E, d, 2f]
        self.register_buffer('wd', wd)                     # [E, f, d]
        self.E, self.d, self.f = wgu.shape[0], wgu.shape[1], wd.shape[1]
        self.T, self.k, self.tiers, self.combine, self.ranked = T, k, list(tiers), combine, ranked
        assert combine in ('dense', 'token') and (ranked or len(self.tiers) == 1)
        self.a, self.off, cap, offr = tier_tables(self.tiers, self.E)
        self.R = int(self.off[-1])
        self.register_buffer('cap_of_rank', torch.tensor(cap, dtype=torch.float32))
        self.register_buffer('off_of_rank', torch.tensor(offr, dtype=torch.int64))
        self.register_buffer('experts', torch.arange(self.E, dtype=torch.int64))
        self.register_buffer('tie', torch.arange(self.E - 1, -1, -1, dtype=torch.float32))
        self.register_buffer('tok', torch.arange(self.T, dtype=torch.int64).unsqueeze(1).expand(self.T, k).reshape(-1).clone())

    def forward(self, x, ids, w):
        T, k, E, d, f, R = self.T, self.k, self.E, self.d, self.f, self.R
        ids = ids.to(torch.int64)
        oh = (ids.unsqueeze(-1) == self.experts).to(torch.float32)          # [T, k, E]
        M = oh.sum(1)                                                       # [T, E]
        n = M.sum(0)                                                        # [E] tokens per expert
        slot = torch.gather(torch.cumsum(M, 0) - M, 1, ids)                 # [T, k] slot within the expert's rows
        if self.ranked:
            perm = torch.topk(n * E + self.tie, E).indices                  # rank -> expert, ties: lower expert first
            rank = torch.zeros(E, dtype=torch.int64, device=x.device).scatter(0, perm, self.experts)   # expert -> rank
            r = torch.gather(rank, 0, ids.reshape(-1)).reshape(T, k)
        else:
            perm, r = None, ids
        cap = torch.gather(self.cap_of_rank, 0, r.reshape(-1)).reshape(T, k)
        valid = slot < cap
        row = torch.where(valid, torch.gather(self.off_of_rank, 0, r.reshape(-1)).reshape(T, k) + slot.to(torch.int64),
                          torch.full_like(r, R))                            # [T, k] flat row, R = dummy
        ovf = (1.0 - valid.to(torch.float32)).amax().reshape(1)
        rf = row.reshape(-1)
        tbl = torch.full((R + 1,), T, dtype=torch.int64, device=x.device).scatter(0, rf, self.tok)     # token of each row (T = zero row)
        wt = torch.zeros(R + 1, dtype=x.dtype, device=x.device).scatter(0, rf, w.reshape(-1).to(x.dtype))
        xp = torch.cat([x, torch.zeros(1, d, dtype=x.dtype, device=x.device)], 0)            # [T+1, d]
        outs, y = [], None
        for i, (L, C) in enumerate(self.tiers):
            o = self.off[i]
            ti = tbl[o:o + L * C].reshape(L, C)
            X = xp[ti]                                                      # [L, C, d]
            if self.ranked:
                e = perm[self.a[i]:self.a[i] + L]
                Wgu, Wd = self.wgu[e], self.wd[e]                           # gathered weights [L, d, 2f], [L, f, d]
            else:
                Wgu, Wd = self.wgu[self.a[i]:self.a[i] + L], self.wd[self.a[i]:self.a[i] + L]
            h = torch.matmul(X, Wgu)
            Y = torch.matmul(F.silu(h[..., :f]) * h[..., f:], Wd)          # [L, C, d]
            Y = Y * wt[o:o + L * C].reshape(L, C, 1)
            if self.combine == 'token':
                outs.append(Y.reshape(L * C, d))
            else:
                # zero-filled at run time: x[0,0] * 0 cannot be folded (NaN/Inf semantics), so TensorRT writes zeros instead of
                # storing a [L, T+1, d] zero constant in the engine and copying it in on every call
                acc = (x[:1, :1] * 0).reshape(1, 1, 1).expand(L, T + 1, d)
                lane = torch.arange(L, device=x.device).unsqueeze(1).expand(L, C)
                acc = acc.index_put((lane, ti), Y)                          # ScatterND; padding lands in row T
                yi = acc[:, :T].sum(0)
                y = yi if y is None else y + yi
        if self.combine == 'token':
            Yf = torch.cat(outs + [torch.zeros(1, d, dtype=x.dtype, device=x.device)], 0)    # [R+1, d], row R = 0
            y = Yf[row].sum(1)                                              # [T, k, d] -> [T, d]
        return y, ovf


def reference(x, ids, w, wg, wu, wd):
    """Untiered FP32 reference: every routed assignment, expert by expert (torch tensors, any device)."""
    x, w = x.float(), w.float(); y = torch.zeros_like(x)
    for e in torch.unique(ids).tolist():
        t, j = torch.nonzero(ids == e, as_tuple=True)
        h = F.silu(x[t] @ wg[e].float()) * (x[t] @ wu[e].float())
        y.index_add_(0, t, (h @ wd[e].float()) * w[t, j].unsqueeze(1))
    return y
