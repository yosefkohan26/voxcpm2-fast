"""P2.5.1 preflight — per-stage breakdown of one causal layer's forward.

Decomposes ``FusedLayer(causal=True)`` at base_lm shape (hidden=2048,
intermediate=6144, N=100) into its 11 individual ops and measures each
with CUDA events. Compares against the theoretical floor (compute-bound
or HBM-bound, whichever is larger) per stage.

This tells us where the ~714 µs/layer gap from physics floor lives — which
in turn decides which kernel to attack first in P2.5.1.
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"
MODEL_DIR = REPO_ROOT / "models" / "VoxCPM2"

sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))
sys.path.insert(0, str(NANOVLLM_ROOT))


# RTX 5090 practical peaks (from notes/physics_floor_c1.md).
TFLOPS_BF16 = 178e12        # 85% of 209 TFLOPS peak
HBM_BW      = 1.52e12       # 85% of 1.792 TB/s peak
FLASHATTN_FLOORS_US = 10    # conservative; per-call minimum at N=100


def _init_dist():
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("gloo", rank=0, world_size=1)


def _build_layer_and_weights():
    _init_dist()
    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.model import Cpm4DecoderLayer
    from safetensors.torch import safe_open

    cfg = VoxCPM2Config.model_validate_json((MODEL_DIR / "config.json").read_text())
    lm_cfg = cfg.lm_config.model_copy(deep=True)
    lm_cfg.use_mup = False

    torch.set_default_dtype(torch.bfloat16)
    layer = Cpm4DecoderLayer(lm_cfg, is_causal=True).cuda().eval()
    torch.set_default_dtype(torch.float32)

    # Load base_lm.layers.0 weights
    prefix = "base_lm.layers.0."
    loaded: dict[str, torch.Tensor] = {}
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            if not k.startswith(prefix):
                continue
            loaded[k[len(prefix):]] = f.get_tensor(k)
    q = loaded["self_attn.q_proj.weight"]
    k_ = loaded["self_attn.k_proj.weight"]
    v = loaded["self_attn.v_proj.weight"]
    loaded["self_attn.qkv_proj.weight"] = torch.cat([q, k_, v], dim=0)
    g = loaded["mlp.gate_proj.weight"]
    u = loaded["mlp.up_proj.weight"]
    loaded["mlp.gate_up_proj.weight"] = torch.cat([g, u], dim=0)

    sd = layer.state_dict()
    with torch.no_grad():
        for name, tensor in loaded.items():
            if name in sd:
                sd[name].copy_(tensor.to(torch.bfloat16).cuda())

    return layer, lm_cfg, loaded


def _build_fused_layer(layer, lm_cfg, loaded):
    from fused_layer_chained import FusedLayer
    rope = layer.self_attn.rotary_emb
    cos_cache = rope.cos_cached.to(torch.float32).contiguous()
    sin_cache = rope.sin_cached.to(torch.float32).contiguous()
    keep = {"input_layernorm.weight", "self_attn.qkv_proj.weight",
            "self_attn.o_proj.weight", "post_attention_layernorm.weight",
            "mlp.gate_up_proj.weight", "mlp.down_proj.weight"}
    weights = {k: v.to(torch.bfloat16).cuda().contiguous()
               for k, v in loaded.items() if k in keep}
    return FusedLayer(
        weights=weights,
        rope_cos_cache=cos_cache,
        rope_sin_cache=sin_cache,
        hidden=lm_cfg.hidden_size,
        intermediate=lm_cfg.intermediate_size,
        causal=True,
        rms_eps=lm_cfg.rms_norm_eps,
    )


def _time_stage(stage_fn, iters=200, warmup=20):
    """Return p50 in microseconds for a zero-arg callable."""
    for _ in range(warmup):
        stage_fn()
    torch.cuda.synchronize()
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record()
        stage_fn()
        e1[i].record()
    torch.cuda.synchronize()
    xs_us = sorted(e0[i].elapsed_time(e1[i]) * 1000.0 for i in range(iters))
    return xs_us[len(xs_us) // 2]


def main(N=100, iters=200, warmup=20):
    torch.manual_seed(1)
    layer, lm_cfg, loaded = _build_layer_and_weights()
    ours = _build_fused_layer(layer, lm_cfg, loaded)

    H = lm_cfg.hidden_size
    I = lm_cfg.intermediate_size
    H_n = lm_cfg.num_attention_heads
    H_kv = lm_cfg.num_key_value_heads
    D = lm_cfg.kv_channels
    Q_DIM = H_n * D
    KV_DIM = H_kv * D
    QKV_DIM = Q_DIM + 2 * KV_DIM

    positions = torch.arange(N, device="cuda", dtype=torch.int32)
    hs = (torch.randn(N, H, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()

    from fused_layer_chained import _gemm, _pad_M_to_16
    import fused_layer_chained_ext as _ext
    from flash_attn import flash_attn_func

    # Pre-compute intermediate activations so each stage can run standalone.
    # We'll re-run through the pipeline to give each stage its proper input.
    def run_through(stop_at: str):
        """Execute the pipeline, returning the output of stage `stop_at`
        and the intermediate tensors needed as inputs for later stages."""
        with torch.inference_mode():
            x0 = hs.contiguous()
            ln_out = _ext.rmsnorm(x0, ours.w_in_ln, ours.rms_eps)
            if stop_at == "rmsnorm_in": return {"in": x0, "out": ln_out}
            qkv = _gemm(ln_out, ours.w_qkv)
            if stop_at == "qkv_gemm":    return {"in": ln_out, "out": qkv, "x0": x0}
            qkv_r = qkv.clone()
            _ext.rope_inplace(qkv_r, ours.cos, ours.sin, positions, H_n, H_kv, D)
            if stop_at == "rope":        return {"in": qkv, "out": qkv_r, "x0": x0}
            q = qkv_r[:, :Q_DIM].view(1, N, H_n, D).contiguous()
            k = qkv_r[:, Q_DIM:Q_DIM + KV_DIM].view(1, N, H_kv, D).contiguous()
            v = qkv_r[:, Q_DIM + KV_DIM:].view(1, N, H_kv, D).contiguous()
            attn = flash_attn_func(q, k, v, causal=True, softmax_scale=D ** -0.5)
            attn_2d = attn.view(N, Q_DIM).contiguous()
            if stop_at == "attention":   return {"q": q, "k": k, "v": v, "out": attn_2d, "x0": x0}
            o_out = _gemm(attn_2d, ours.w_o)
            if stop_at == "o_gemm":      return {"in": attn_2d, "out": o_out, "x0": x0}
            res = x0.clone()
            _ext.residual_add(res, o_out)
            if stop_at == "residual_1":  return {"in": o_out, "out": res}
            ln2 = _ext.rmsnorm(res, ours.w_post_ln, ours.rms_eps)
            if stop_at == "rmsnorm_post": return {"in": res, "out": ln2, "residual": res}
            gu = _gemm(ln2, ours.w_gu)
            if stop_at == "gate_up":     return {"in": ln2, "out": gu, "residual": res}
            mid = _ext.silu_mul(gu)
            if stop_at == "silu_mul":    return {"in": gu, "out": mid, "residual": res}
            mlp_out = _gemm(mid, ours.w_dn)
            if stop_at == "down_gemm":   return {"in": mid, "out": mlp_out, "residual": res}
            _ext.residual_add(res, mlp_out)
            return {"out": res}

    # Isolate the inputs for each stage.
    ctx_rmsnorm_in  = run_through("rmsnorm_in")     # x0
    ctx_qkv         = run_through("qkv_gemm")       # ln_out
    ctx_rope        = run_through("rope")           # qkv
    ctx_attn        = run_through("attention")      # q, k, v  (but we time qkv_r directly so re-run)
    ctx_o           = run_through("o_gemm")         # attn_2d
    ctx_res1        = run_through("residual_1")     # o_out, x0
    ctx_rms2        = run_through("rmsnorm_post")   # res
    ctx_gu          = run_through("gate_up")        # ln2
    ctx_silu        = run_through("silu_mul")       # gu
    ctx_dn          = run_through("down_gemm")      # mid
    x0 = hs.contiguous().clone()
    ln_out = ctx_qkv["in"].clone()
    qkv_for_rope = ctx_rope["in"].clone()  # qkv (pre-rope copy)
    qkv_post_rope = ctx_rope["out"].clone()
    q_a = qkv_post_rope[:, :Q_DIM].view(1, N, H_n, D).contiguous()
    k_a = qkv_post_rope[:, Q_DIM:Q_DIM + KV_DIM].view(1, N, H_kv, D).contiguous()
    v_a = qkv_post_rope[:, Q_DIM + KV_DIM:].view(1, N, H_kv, D).contiguous()
    attn_2d = ctx_o["in"].clone()
    o_out = ctx_res1["in"].clone()
    res_for_rms2 = ctx_rms2["in"].clone()
    ln2 = ctx_gu["in"].clone()
    gu = ctx_silu["in"].clone()
    mid = ctx_dn["in"].clone()
    mlp_out = ctx_dn["out"].clone()
    res_for_add = x0.clone()

    # ---------- stage lambdas ----------
    stages: list[tuple[str, callable, float, float]] = []

    # (name, fn, bytes_HBM, FLOPs)
    def rmsnorm_in_fn():
        _ = _ext.rmsnorm(x0, ours.w_in_ln, ours.rms_eps)
    bytes_rms = N * H * 2 * 2 + H * 2  # read x + write y (and w is tiny)
    flops_rms = N * H * 3              # ~3 ops/elem (sq, acc, scale)
    stages.append(("rmsnorm (in)",       rmsnorm_in_fn, bytes_rms, flops_rms))

    def qkv_gemm_fn():
        _ = _gemm(ln_out, ours.w_qkv)
    bytes_qkv = QKV_DIM * H * 2 + N * H * 2 + N * QKV_DIM * 2   # weight + act read + write
    flops_qkv = 2.0 * N * H * QKV_DIM
    stages.append((f"qkv_gemm  ({QKV_DIM}×{H})",  qkv_gemm_fn, bytes_qkv, flops_qkv))

    def rope_fn():
        r = qkv_for_rope.clone()
        _ext.rope_inplace(r, ours.cos, ours.sin, positions, H_n, H_kv, D)
    bytes_rope = N * (Q_DIM + KV_DIM) * 2 * 2  # read + write Q and K slices
    flops_rope = N * (Q_DIM + KV_DIM) * 6      # ~6 fp ops/elem
    stages.append(("rope (Q+K inplace)", rope_fn, bytes_rope, flops_rope))

    def attn_fn():
        _ = flash_attn_func(q_a, k_a, v_a, causal=True, softmax_scale=D ** -0.5)
    # FLOPs: 2 * N * N * (H_n * D)  (Q*K^T) + 2 * N * N * (H_n * D)  (SV)
    flops_attn = 2.0 * (2.0 * N * N * H_n * D)
    # HBM: Q + K + V read, O write
    bytes_attn = N * (Q_DIM + 2 * KV_DIM) * 2 + N * Q_DIM * 2
    stages.append(("flash_attn (causal)", attn_fn, bytes_attn, flops_attn))

    def o_gemm_fn():
        _ = _gemm(attn_2d, ours.w_o)
    bytes_o = H * Q_DIM * 2 + N * Q_DIM * 2 + N * H * 2
    flops_o = 2.0 * N * Q_DIM * H
    stages.append((f"o_gemm    ({H}×{Q_DIM})",   o_gemm_fn, bytes_o, flops_o))

    def res1_fn():
        r = res_for_add.clone()
        _ext.residual_add(r, o_out)
    bytes_res = N * H * 2 * 3  # read a, b; write a
    stages.append(("residual_add (post-attn)", res1_fn, bytes_res, 0.0))

    def rmsnorm_post_fn():
        _ = _ext.rmsnorm(res_for_rms2, ours.w_post_ln, ours.rms_eps)
    stages.append(("rmsnorm (post)",     rmsnorm_post_fn, bytes_rms, flops_rms))

    def gu_gemm_fn():
        _ = _gemm(ln2, ours.w_gu)
    bytes_gu = (2 * I) * H * 2 + N * H * 2 + N * (2 * I) * 2
    flops_gu = 2.0 * N * H * (2 * I)
    stages.append((f"gate_up   ({2*I}×{H})",    gu_gemm_fn, bytes_gu, flops_gu))

    def silu_fn():
        _ = _ext.silu_mul(gu)
    bytes_silu = N * (2 * I) * 2 + N * I * 2
    flops_silu = N * I * 4
    stages.append(("silu_mul",           silu_fn, bytes_silu, flops_silu))

    def dn_gemm_fn():
        _ = _gemm(mid, ours.w_dn)
    bytes_dn = H * I * 2 + N * I * 2 + N * H * 2
    flops_dn = 2.0 * N * I * H
    stages.append((f"down_gemm ({H}×{I})",      dn_gemm_fn, bytes_dn, flops_dn))

    def res2_fn():
        r = res_for_add.clone()
        _ext.residual_add(r, mlp_out)
    stages.append(("residual_add (post-mlp)",   res2_fn, bytes_res, 0.0))

    # Total p50 of the full layer forward (for cross-check).
    def full_fn():
        _ = ours.forward(hs, positions)

    print("=" * 100)
    print(f"P2.5.1 preflight — base_lm.layers.0 per-stage breakdown")
    print(f"N={N}  H={H}  I={I}  heads={H_n}/{H_kv}  head_dim={D}  QKV_DIM={QKV_DIM}")
    print("=" * 100)

    total_meas = 0.0
    total_floor = 0.0
    print(f"{'stage':32s}  {'p50 µs':>10s}  {'floor µs':>10s}  {'gap':>6s}  {'limit':>8s}")
    print(f"{'-'*32}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*8}")
    for name, fn, B, F in stages:
        us = _time_stage(fn, iters=iters, warmup=warmup)
        total_meas += us
        compute_us = F / TFLOPS_BF16 * 1e6 if F > 0 else 0.0
        bw_us      = B / HBM_BW * 1e6
        floor = max(compute_us, bw_us)
        if name.startswith("flash_attn"):
            floor = max(floor, FLASHATTN_FLOORS_US)
        total_floor += floor
        limit = "compute" if F > 0 and compute_us >= bw_us else "bw"
        if floor < 0.5:
            limit = "tiny"
        gap = us / floor if floor > 0 else 0.0
        print(f"{name:32s}  {us:10.2f}  {floor:10.2f}  {gap:5.1f}x  {limit:>8s}")
    print(f"{'-'*32}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*8}")
    print(f"{'sum of stages':32s}  {total_meas:10.2f}  {total_floor:10.2f}  {total_meas/total_floor:5.1f}x")

    full_us = _time_stage(full_fn, iters=iters, warmup=warmup)
    launch_overhead = full_us - total_meas
    print(f"{'full FusedLayer.forward':32s}  {full_us:10.2f}  "
          f"{'':10s}  {'':6s}  {'':8s}")
    print(f"{'  (overhead vs stage sum)':32s}  {launch_overhead:10.2f}  "
          f"{'':10s}  {'':6s}  launch/glue")
    print("=" * 100)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=100)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    main(N=args.N, iters=args.iters, warmup=args.warmup)
