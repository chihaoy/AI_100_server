#!/usr/bin/env python3
"""Single-layer MoE (expert-parallel flavour) micro-benchmark exporter with PER-LANE PADDING TIERS.

Purpose: answer "if every expert gets a different padding width, is that inefficient on AI 100?"
by exporting the *same* EP graph QEfficient 1.23 emits (cumsum -> CtxScatter3DInt -> CtxGather3D
-> 3 GEMM -> mask -> CtxScatter3D -> einsum reduce) but with lanes grouped by a compile-time
padding width ("tier"). The uniform case (one group, tier = T) is byte-for-byte the QEfficient
structure and serves as the baseline.

Graph = router (MatMul + Softmax + TopK + normalise + densify) + EP MoE, one layer, no attention.
Only op *shapes* matter for cycle counts on this hardware (verified earlier: real vs random input
differ <= 0.64%), so weights and inputs are random.

Lane layout (mirrors QEfficient): E = P * S experts, lane p at stage s = expert s*P + p.
Tier groups are contiguous lane ranges; each group has its own weights [n_g, S, H, I] and its own
accumulator [n_g, T, H]. Both stages of a lane share the lane's tier (experts are paired by tier) unless
--stage-widths gives each stage its own width (see 9.17.2026/结论总结/STAGE_BATCHING.md).

  python3 moe_tier_bench_export.py --out DIR --groups 128:16                 # uniform (baseline)
  python3 moe_tier_bench_export.py --out DIR --groups 128:1,64:2,32:5,16:4,8:4
  python3 moe_tier_bench_export.py --out DIR --groups 128:64 --stage-widths 128,32   # stage 0 128 rows, stage 1 32 rows
  python3 moe_tier_bench_export.py --out DIR --groups 128:128 -S 1                   # one stage, 2 experts per core in one node
"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/home/chihao/qeff-venv/lib/python3.10/site-packages")
from QEfficient.customop.ctx_scatter_gather import (
    CtxGatherFunc3DGeneralized, CtxScatterFunc3DGeneralized, CtxScatterFunc3DInt)

INT32_MAX = torch.iinfo(torch.int32).max


class CtxScatterFunc3DIntDrop(CtxScatterFunc3DInt):
    """Same ONNX op as CtxScatterFunc3DInt (com.qualcomm.cloud::CtxScatter3DInt); the CPU forward
    additionally drops positions >= data.shape[1], mirroring the hardware's out-of-bounds ScatterND
    behaviour that the QEfficient EP path already relies on for INT32_MAX rows."""

    @staticmethod
    def forward(data, position_ids, updates):
        data = data.clone()
        valid = (position_ids != INT32_MAX) & (position_ids < data.shape[1])
        batch_idx = torch.arange(data.shape[0]).view(-1, 1).expand_as(position_ids)
        data[batch_idx[valid], position_ids[valid].long()] = updates[valid]
        return data


def parse_groups(s):
    """'128:1,64:2,32:5' -> [(128,1),(64,2),(32,5)]  (tier, n_lanes), kept in given order."""
    out = []
    for item in s.split(","):
        t, n = item.split(":")
        out.append((int(t), int(n)))
    return out


class LaneMLP(nn.Module):
    """One lane's expert MLP as its own submodule so the exported ONNX node names carry the lane
    index (/lanes.<p>/MatMul ...). Used by --split-lanes to read lane -> core off the trace."""

    def __init__(self, S, H, I, g):
        super().__init__()
        self.Wg = nn.Parameter((torch.randn(S, H, I, generator=g) * 0.02).half())
        self.Wu = nn.Parameter((torch.randn(S, H, I, generator=g) * 0.02).half())
        self.Wd = nn.Parameter((torch.randn(S, I, H, generator=g) * 0.02).half())

    def forward(self, xc, s):                                                # xc [1, cap, H]
        return (F.silu(xc @ self.Wg[s]) * (xc @ self.Wu[s])) @ self.Wd[s]


class LaneTap(nn.Module):
    """Per-lane scalar multiply appended after the batched down-proj (--tap-lanes). Its node name
    /taps.<g>.<p>/Mul tags lane p's slice of the batched GEMM output, so the trace shows which core holds it."""

    def __init__(self, p):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(1.0 + 1e-3 * (p + 1)).half())

    def forward(self, x):
        return x * self.k


class MoELayerEP(nn.Module):
    def __init__(self, T, H, I, K, S, groups, num_devices=1, seed=0, capacity_check="oob",
                 split_lanes=False, emit_partial=False, tap_lanes=False, stage_widths=None, shared_weights=False, real=None,
                 lane_range=None, dup_router=False, separate_outputs=False):
        super().__init__()
        # --dup-router: groups >= 1 route with their own copy of the router weight (same values, separate ONNX
        # initializer so the compiler cannot merge the two chains) -> the second lane group has no data dependency
        # on the first (its own router + densify + index chain).  --separate-outputs: no final combine; the QPC
        # outputs one reduced [T,H] per group (y0, y1, ...).  Together they give a graph whose groups can be put in
        # partitions with no edge between them (path 1 of naive_vs_64_16, 2026-09-20).
        self.dup_router, self.separate_outputs = dup_router, separate_outputs
        self.capacity_check = capacity_check
        # --lane-range l0,l1 (needs --real-weights): the router still scores ALL experts of the full model (E_router =
        # len(expert_order)), but only lanes [l0, l1) of the lane-major layout are materialised in this graph. Used to
        # export one lane group (e.g. the hot 64 experts of a 64/16 split) as its own QPC whose output is that group's
        # partial sum; two such QPCs run on disjoint cards give naive's output when their outputs are added on the host.
        self.lane0 = lane_range[0] if lane_range else 0
        # --stage-widths: per-stage padding width overriding the group tier (stage s of every group uses
        # stage_widths[s]).  Tests whether the two experts a core holds (stage 0 / stage 1) must share padding.
        self.stage_widths = list(stage_widths) if stage_widths else None
        self.split_lanes, self.emit_partial, self.tap_lanes = split_lanes, emit_partial, tap_lanes
        self.taps = nn.ModuleList([nn.ModuleList([LaneTap(p) for p in range(n)]) for _, n in groups]) if tap_lanes else None
        g = torch.Generator().manual_seed(seed)
        self.T, self.H, self.I, self.K, self.S = T, H, I, K, S
        self.groups = groups
        self.P = sum(n for _, n in groups)
        self.E = self.P * self.S
        self.E_router, self.P_full = self.E, self.P          # router width / lanes of the full model (== this graph unless --lane-range)
        self.D = num_devices
        # router weight [E, H] (F.linear convention, same as Qwen3MoeTopKRouter)
        self.router = nn.Parameter((torch.randn(self.E, H, generator=g) * 0.02).half())
        self.Wg, self.Wu, self.Wd = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        self.lanes = nn.ModuleList()
        if real is not None and not split_lanes:
            # real checkpoint weights, lane-major: real["router"] [E,H]; real["Wg"/"Wu"] [P,S,H,I]; real["Wd"] [P,S,I,H]
            self.router = nn.Parameter(real["router"].half().clone(), requires_grad=False)
            if dup_router:
                # rows permuted (fixed seed) so the compiler cannot CSE this chain into the original one (it did, when
                # the weights were byte-identical: the second Softmax/TopK/Transpose chain was folded into the first
                # and the "independent" partition silently depended on the first again).  Columns are un-permuted
                # after densify, so lane order is unchanged.
                E = real["router"].shape[0]
                self.dup_perm = torch.randperm(E, generator=torch.Generator().manual_seed(12345))
                self.dup_inv = torch.argsort(self.dup_perm)
                self.router_dup = nn.Parameter(real["router"][self.dup_perm].half().clone(), requires_grad=False)
            self.E_router = self.router.shape[0]; self.P_full = self.E_router // self.S
            assert self.lane0 + self.P <= self.P_full, (self.lane0, self.P, self.P_full)
            a = 0
            for tier, n in groups:
                self.Wg.append(nn.Parameter(real["Wg"][a:a + n].half().clone())); self.Wu.append(nn.Parameter(real["Wu"][a:a + n].half().clone()))
                self.Wd.append(nn.Parameter(real["Wd"][a:a + n].half().clone())); a += n
            return
        if shared_weights and not split_lanes:
            # --shared-weights: draw the full [P, S, H, I] tensors once and slice per group, so a grouped
            # variant has exactly the same expert weights as the 1-group (uniform) variant -> outputs are
            # comparable token by token (they differ only where the smaller capacity drops tokens).
            # For a single group this is identical to the per-group draw below.
            Wg = (torch.randn(self.P, S, H, I, generator=g) * 0.02).half()
            Wu = (torch.randn(self.P, S, H, I, generator=g) * 0.02).half()
            Wd = (torch.randn(self.P, S, I, H, generator=g) * 0.02).half()
            a = 0
            for tier, n in groups:
                self.Wg.append(nn.Parameter(Wg[a:a + n].clone())); self.Wu.append(nn.Parameter(Wu[a:a + n].clone()))
                self.Wd.append(nn.Parameter(Wd[a:a + n].clone())); a += n
            return
        for tier, n in groups:
            if split_lanes:
                self.lanes.append(nn.ModuleList([LaneMLP(S, H, I, g) for _ in range(n)]))
            else:
                self.Wg.append(nn.Parameter((torch.randn(n, S, H, I, generator=g) * 0.02).half()))
                self.Wu.append(nn.Parameter((torch.randn(n, S, H, I, generator=g) * 0.02).half()))
                self.Wd.append(nn.Parameter((torch.randn(n, S, I, H, generator=g) * 0.02).half()))

    def cap(self, tier, s):
        """padding width of stage s in a group whose tier is `tier`."""
        return self.stage_widths[s] if self.stage_widths else tier

    def weights(self, gi, p, s):
        """(Wg, Wu, Wd) of lane p, stage s in group gi -- same for both layouts."""
        if self.split_lanes:
            m = self.lanes[gi][p]
            return m.Wg[s], m.Wu[s], m.Wd[s]
        return self.Wg[gi][p, s], self.Wu[gi][p, s], self.Wd[gi][p, s]

    # ---- router: identical to QEffQwen3MoeTopKRouter + densify_topk ----------------
    def route(self, x, W=None):
        logits = F.linear(x, self.router if W is None else W)                # [T, E]
        probs = F.softmax(logits, dtype=torch.float, dim=-1).to(logits.dtype)
        top_w, top_i = torch.topk(probs, self.K, dim=-1)
        top_w = top_w / torch.einsum("bk->b", top_w).unsqueeze(-1)          # norm_topk_prob
        top_w = top_w.to(x.dtype)
        rw = top_w.new_zeros((x.shape[0], self.E_router))
        rw.scatter_(1, top_i, top_w)                                          # dense [T, E]
        return rw

    # ---- QEfficient build_matched_idx_from_cumsum, with a capacity (tier) ------------
    def matched_idx(self, T2Ei, cap):
        n, T = T2Ei.shape
        i32max = torch.tensor(INT32_MAX, dtype=torch.int32)
        token_idx = torch.arange(T, dtype=torch.int32).unsqueeze(0).expand(n, -1)
        valid_prefix = torch.cumsum(T2Ei.to(torch.int32), dim=1)
        valid_dest = valid_prefix - 1
        canvas = i32max.expand(n, cap)                                       # [n, cap]
        if self.capacity_check == "compare" and cap < T:
            # explicit drop: rows whose packed position >= cap are marked INT32_MAX
            keep = T2Ei & (valid_dest < cap)
            scatter_pos = torch.where(keep, valid_dest, i32max)              # [n, T]
            m = CtxScatterFunc3DInt.apply(canvas.unsqueeze(-1), scatter_pos, token_idx.unsqueeze(-1))
        else:
            # QEfficient's exact graph; positions >= cap fall outside the [n, cap] canvas and are
            # dropped by ScatterND's out-of-bounds behaviour (same mechanism as INT32_MAX rows)
            scatter_pos = torch.where(T2Ei, valid_dest, i32max)             # [n, T]
            m = CtxScatterFunc3DIntDrop.apply(canvas.unsqueeze(-1), scatter_pos, token_idx.unsqueeze(-1))
        return m.squeeze(-1)                                                 # [n, cap]

    # ---- QEfficient cumsum_scatter_gather_update_expert_blocked (num_packed_chunks=1) --
    def stage_update(self, x, T2Ei, gi, s, rw_unsq, expert_out, cap, x_unsq=None):
        n, T = T2Ei.shape
        midx = self.matched_idx(T2Ei, cap)
        valid_rows = torch.einsum("ij->i", T2Ei.to(torch.int32)).unsqueeze(1)   # [n,1]
        x_exp = (x.unsqueeze(0) if x_unsq is None else x_unsq).expand(n, -1, -1)
        row_range = torch.arange(cap, dtype=torch.int32).unsqueeze(0)
        x_chunk = CtxGatherFunc3DGeneralized.apply(x_exp, midx)              # [n, cap, H]
        if self.split_lanes:
            down = torch.cat([self.lanes[gi][p](x_chunk[p:p + 1], s) for p in range(n)], 0)
        else:
            W_g, W_u, W_d = self.Wg[gi][:, s], self.Wu[gi][:, s], self.Wd[gi][:, s]
            gate = x_chunk @ W_g
            up = x_chunk @ W_u
            down = (up * F.silu(gate)) @ W_d                                 # [n, cap, H]
            if self.tap_lanes:
                down = torch.cat([self.taps[gi][p](down[p:p + 1]) for p in range(n)], 0)
        rw_chunk = CtxGatherFunc3DGeneralized.apply(rw_unsq, midx)           # [n, cap, 1]
        down = down * rw_chunk
        old = CtxGatherFunc3DGeneralized.apply(expert_out, midx)
        upd = old + down
        cvr = torch.clamp(valid_rows, min=torch.zeros_like(valid_rows),
                          max=torch.full_like(valid_rows, cap))
        upd = torch.where((row_range < cvr).unsqueeze(-1), upd, torch.zeros_like(upd))
        return CtxScatterFunc3DGeneralized.apply(expert_out, midx, upd)

    def forward(self, x):
        T, H = x.shape
        def lane_weights(W):
            rw = self.route(x, W)                                                                        # [T, E]
            if W is not None:
                rw = rw[:, self.dup_inv]                                                                  # back to lane-major expert order
            rw = rw.transpose(0, 1).contiguous().view(self.S, self.P_full, T).transpose(0, 1).contiguous()  # [P_full,S,T]
            if self.P != self.P_full:
                rw = rw[self.lane0:self.lane0 + self.P].contiguous()                                    # [P,S,T] this graph's lanes
            return rw, rw.unsqueeze(-1)                                                                 # [P,S,T], [P,S,T,1]
        rw, rw_unsq = lane_weights(None)
        if self.dup_router:
            rw_d, rw_unsq_d = lane_weights(self.router_dup)                  # second, independent routing chain for groups >= 1
        # Reduce as QEfficient does: intra-device over lanes ("dpth->dth"), then across devices
        # ("dth->th"). With several tier groups the per-group intra-device partials [D,T,H] are
        # summed first (device-local adds) so the cross-device reduce still happens once.
        partial = None; outs = []
        a = 0
        for gi, (tier, n) in enumerate(self.groups):
            b = a + n
            if self.dup_router and gi >= 1:
                rw_g, rwu_g = rw_d[a:b], rw_unsq_d[a:b]
                # the exporter merges identical x.unsqueeze(0) nodes into one shared node, which would be an edge from
                # group 0's partition into group 1's; give group 1 its own (Reshape) node instead
                x_unsq = x.reshape(1, T, H)
            else:
                rw_g, rwu_g = rw[a:b], rw_unsq[a:b]; x_unsq = None
            expert_out = x.new_zeros((n, T, H))
            if tier > 0:
                for s in range(self.S):
                    T2Ei = rw_g[:, s, :] > 0
                    expert_out = self.stage_update(x, T2Ei, gi, s, rwu_g[:, s], expert_out, self.cap(tier, s), x_unsq)
                if self.D > 1 and n % self.D == 0:
                    red = torch.einsum("dpth->dth", expert_out.view(self.D, n // self.D, T, H))
                else:
                    red = torch.einsum("nth->th", expert_out).unsqueeze(0)      # [1,T,H]
                partial = red if partial is None else partial + red
                outs.append(torch.einsum("dth->th", red) if red.shape[0] > 1 else red[0])
            a = b
        if self.separate_outputs:
            return tuple(outs)                                               # one reduced [T,H] per group, no combine
        if self.emit_partial:
            return partial                                                   # [D,T,H] per-device partials
        if partial.shape[0] > 1:
            return torch.einsum("dth->th", partial)
        return partial[0]


def reference(model, x, per_lane=False):
    """Plain per-expert loop with the same capacity-drop rule; fp32.
    per_lane=True returns [P, T, H]: each lane's contribution (both stages summed)."""
    T = x.shape[0]
    rw = model.route(x)                                                      # [T, E] fp16
    rw32 = rw.float()
    out = torch.zeros(model.P, T, model.H) if per_lane else torch.zeros(T, model.H)
    a = 0
    for gi, (tier, n) in enumerate(model.groups):
        for p in range(n):
            for s in range(model.S):
                e = s * model.P_full + model.lane0 + a + p
                sel = torch.nonzero(rw32[:, e] > 0).squeeze(1)
                sel = sel[:model.cap(tier, s)]                               # capacity drop (first-come)
                if len(sel) == 0:
                    continue
                xs = x[sel].float()
                Wg, Wu, Wd = (w.float() for w in model.weights(gi, p, s))
                y = (F.silu(xs @ Wg) * (xs @ Wu)) @ Wd
                if per_lane:
                    out[a + p, sel] += y * rw32[sel, e].unsqueeze(1)
                else:
                    out[sel] += y * rw32[sel, e].unsqueeze(1)
        a += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", required=True, help="tier:n_lanes,... e.g. 128:16 (uniform)")
    ap.add_argument("-T", type=int, default=128)
    ap.add_argument("-H", type=int, default=2048)
    ap.add_argument("-I", type=int, default=768)
    ap.add_argument("-K", type=int, default=8)
    ap.add_argument("-S", type=int, default=2, help="pipeline stages (experts per lane)")
    ap.add_argument("--num-devices", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="verify torch forward vs reference loop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capacity-check", choices=["oob", "compare"], default="oob",
                    help="how rows beyond a lane's tier are dropped (oob = rely on ScatterND OOB drop, zero extra ops)")
    ap.add_argument("--split-lanes", action="store_true",
                    help="placement probe: one MatMul node per lane (names /lanes.<p>/...) instead of one batched node per group")
    ap.add_argument("--tap-lanes", action="store_true",
                    help="placement probe: keep the batched GEMMs, append a named per-lane Mul on the down-proj output")
    ap.add_argument("--stage-widths", default=None,
                    help="comma list of S padding widths, one per stage, overriding the group tier (e.g. 128,32): "
                         "stage 0 of every group is padded to 128 rows, stage 1 to 32")
    ap.add_argument("--real-weights", default=None, help="safetensors shard with model.layers.<L>.mlp.* (Qwen3-MoE) -> real router + expert weights")
    ap.add_argument("--real-layer", type=int, default=0)
    ap.add_argument("--expert-order", default=None, help="json list, len E: original expert id for lane-major index e = s*P + p (default identity = QEfficient layout)")
    ap.add_argument("--x-npy", default=None, help="real input x [T,H] (fp16 npy) instead of random")
    ap.add_argument("--dup-router", action="store_true", help="groups >= 1 get their own router copy + index chain (no edge from group 0)")
    ap.add_argument("--separate-outputs", action="store_true", help="no final combine: outputs y0, y1, ... (one per group)")
    ap.add_argument("--lane-range", default=None,
                    help="l0,l1 (with --real-weights): export only lanes [l0,l1) of the full lane-major layout given by --expert-order "
                         "(router over all experts); output is this lane group's partial sum")
    ap.add_argument("--shared-weights", action="store_true",
                    help="draw expert weights as one [P,S,H,I] tensor and slice per group (grouped variant == uniform variant weights)")
    ap.add_argument("--emit-partial", action="store_true",
                    help="placement probe: output the per-device partial sums [D,T,H] instead of the reduced [T,H]; also dumps y_lane.npy")
    a = ap.parse_args()

    groups = parse_groups(a.groups)
    sw = [int(v) for v in a.stage_widths.split(",")] if a.stage_widths else None
    assert sw is None or len(sw) == a.S, "--stage-widths needs exactly S values"
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    real = None
    if a.real_weights:
        from safetensors.torch import load_file
        P = sum(n for _, n in groups); E = P * a.S
        lr = [int(v) for v in a.lane_range.split(",")] if a.lane_range else [0, P]
        assert lr[1] - lr[0] == P, f"--lane-range spans {lr[1]-lr[0]} lanes but --groups has {P}"
        order = json.load(open(a.expert_order)) if a.expert_order else list(range(E))
        E_full = len(order); P_full = E_full // a.S
        assert sorted(order) == list(range(E_full)), "expert-order must be a permutation of range(E_full)"
        assert a.lane_range or E_full == E, "expert-order length must equal E unless --lane-range is given"
        sd = load_file(a.real_weights); L = a.real_layer
        router = sd[f"model.layers.{L}.mlp.gate.weight"].float()[order]                       # [E_full,H], row e = lane-major expert
        g = lambda e, k: sd[f"model.layers.{L}.mlp.experts.{order[e]}.{k}.weight"].float().T  # gate/up: [768,2048].T -> [H,I]; down: [2048,768].T -> [I,H]
        lanes = range(lr[0], lr[1])
        Wg = torch.stack([torch.stack([g(s * P_full + p, "gate_proj") for s in range(a.S)]) for p in lanes])   # [P,S,H,I]
        Wu = torch.stack([torch.stack([g(s * P_full + p, "up_proj") for s in range(a.S)]) for p in lanes])
        Wd = torch.stack([torch.stack([g(s * P_full + p, "down_proj") for s in range(a.S)]) for p in lanes])  # [P,S,I,H]
        real = dict(router=router, Wg=Wg, Wu=Wu, Wd=Wd); del sd
        assert Wg.shape == (P, a.S, a.H, a.I) and Wd.shape == (P, a.S, a.I, a.H), (Wg.shape, Wd.shape)
    m = MoELayerEP(a.T, a.H, a.I, a.K, a.S, groups, a.num_devices, seed=a.seed, capacity_check=a.capacity_check,
                   split_lanes=a.split_lanes, emit_partial=a.emit_partial, tap_lanes=a.tap_lanes, stage_widths=sw,
                   shared_weights=a.shared_weights, real=real,
                   lane_range=[int(v) for v in a.lane_range.split(",")] if a.lane_range else None,
                   dup_router=a.dup_router, separate_outputs=a.separate_outputs).eval()
    assert not a.dup_router or a.real_weights, "--dup-router is implemented for --real-weights graphs"
    assert not a.separate_outputs or not a.emit_partial
    # x must NOT come from the same RNG stream as the router weight: with torch.manual_seed(seed) and the model's
    # Generator(seed) both starting at seed, x[t] == router[t]/0.02 when T == E, so token t routes to expert t
    # with fp16 weight 1.0 and every expert gets exactly one token (found 2026-09-19; all variants before that
    # date have this degenerate routing -- timings are unaffected (shape-only) but capacity drops never occurred).
    gx = torch.Generator().manual_seed(a.seed + 1000)
    x = (torch.randn(a.T, a.H, generator=gx) * 1.0).half()
    if a.x_npy:
        x = torch.from_numpy(np.load(a.x_npy)).half(); assert tuple(x.shape) == (a.T, a.H), x.shape
    rows = sum(m.cap(t, s) * n for t, n in groups for s in range(a.S))
    info = dict(T=a.T, H=a.H, I=a.I, K=a.K, S=a.S, P=m.P, E=m.E, num_devices=a.num_devices, capacity_check=a.capacity_check,
                split_lanes=a.split_lanes, emit_partial=a.emit_partial, tap_lanes=a.tap_lanes, real_weights=a.real_weights, expert_order=a.expert_order, x_npy=a.x_npy,
                lane_range=a.lane_range, E_router=m.E_router, dup_router=a.dup_router, separate_outputs=a.separate_outputs,
                groups=groups, stage_widths=sw, padded_rows_total=rows, padded_rows_uniform=a.T * m.P * a.S,
                row_ratio=rows / (a.T * m.P * a.S))
    print(json.dumps(info, indent=1))

    with torch.no_grad():
        y = m(x)
        if a.separate_outputs:
            for i, yi in enumerate(y):
                np.save(os.path.join(a.out, f"y{i}_torch.npy"), yi.float().numpy())
            y = sum(yi.float() for yi in y)                                  # total for y_torch / --check
        # dump the input and CPU outputs so the QPC can be checked numerically on the card
        x.numpy().tofile(os.path.join(a.out, "x.bin"))
        np.save(os.path.join(a.out, "y_torch.npy"), y.float().numpy())
        np.save(os.path.join(a.out, "y_ref.npy"), reference(m, x).numpy())
        if a.emit_partial:
            np.save(os.path.join(a.out, "y_lane.npy"), reference(m, x, per_lane=True).numpy())   # [P,T,H]
        if a.check and not a.emit_partial:
            ref = reference(m, x)
            err = (y.float() - ref).abs().max().item()
            scale = ref.abs().max().item()
            print(f"CHECK max|err|={err:.4g}  max|ref|={scale:.4g}  rel={err/max(scale,1e-9):.3g}")
            assert err / max(scale, 1e-9) < 5e-2, "torch forward mismatch vs reference"
            info["check_rel_err"] = err / max(scale, 1e-9)

    onnx_path = os.path.join(a.out, "moe_layer.onnx")
    t0 = time.time()
    with torch.no_grad():
        out_names = [f"y{i}" for i in range(len(groups))] if a.separate_outputs else ["partial" if a.emit_partial else "y"]
        torch.onnx.export(m, (x,), onnx_path, input_names=["x"], output_names=out_names,
                          dynamo=False, opset_version=17, do_constant_folding=True)
    import onnx
    from QEfficient.base.onnx_transforms import CustomOpTransform, FP16ClipTransform
    model = onnx.load(onnx_path)
    CustomOpTransform.apply(model)
    fmax = float(np.finfo(np.float16).max)
    for init in model.graph.initializer:              # clip fp32 constants to fp16 range
        FP16ClipTransform.apply(init, a.out, fmax, -fmax)
    big = sum(len(t.raw_data) for t in model.graph.initializer) > 1.5 * 2**30
    if big:   # a single protobuf file must stay < 2 GB; qaic-compile reads external weights fine
        for f in os.listdir(a.out):
            if f.endswith(".onnx.data") or (f.startswith("onnx__") and not f.endswith(".onnx")):
                os.remove(os.path.join(a.out, f))
        onnx.save(model, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True,
                  location="moe_layer.onnx.data", size_threshold=1024)
    else:
        onnx.save(model, onnx_path)
    print(f"EXPORT_OK {onnx_path}  ({os.path.getsize(onnx_path)/2**20:.0f} MB, {time.time()-t0:.0f}s)")
    ops = {}
    for nd in model.graph.node:
        ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
    info["onnx_nodes"] = len(model.graph.node)
    info["onnx_ops"] = ops
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1)
    print("nodes:", len(model.graph.node), {k: v for k, v in sorted(ops.items(), key=lambda kv: -kv[1])[:14]})


if __name__ == "__main__":
    main()
