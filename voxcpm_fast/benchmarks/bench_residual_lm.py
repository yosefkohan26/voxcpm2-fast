"""P2.5.3 — residual_lm (8-layer causal, no RoPE) bench.

Shape: hidden=2048, intermediate=6144, 16/2 heads, head_dim=128, NO RoPE.
Same per-layer GEMM shape as base_lm. Runs once per decode step on the
fusion of enc_outputs and feat_embeds, so it sits on every steady-state
decode step.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"
MODEL_DIR = REPO_ROOT / "models" / "VoxCPM2"

sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))
sys.path.insert(0, str(NANOVLLM_ROOT))


def _init_dist():
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("gloo", rank=0, world_size=1)


def _build_upstream_residual_lm():
    _init_dist()
    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.model import Cpm4Model
    from safetensors.torch import safe_open

    cfg = VoxCPM2Config.model_validate_json((MODEL_DIR / "config.json").read_text())
    r_cfg = cfg.lm_config.model_copy(deep=True)
    r_cfg.num_hidden_layers = cfg.residual_lm_num_layers
    r_cfg.use_mup = False
    r_cfg.vocab_size = 0  # no embed for residual_lm
    use_rope = not cfg.residual_lm_no_rope

    torch.set_default_dtype(torch.bfloat16)
    model = Cpm4Model(r_cfg, is_causal=True, use_rope=use_rope).cuda()
    torch.set_default_dtype(torch.float32)

    prefix = "residual_lm."
    loaded: dict[str, torch.Tensor] = {}
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                loaded[k[len(prefix):]] = f.get_tensor(k)

    assembled = dict(loaded)
    for i in range(r_cfg.num_hidden_layers):
        q = loaded.get(f"layers.{i}.self_attn.q_proj.weight")
        k_ = loaded.get(f"layers.{i}.self_attn.k_proj.weight")
        v = loaded.get(f"layers.{i}.self_attn.v_proj.weight")
        if q is not None and k_ is not None and v is not None:
            assembled[f"layers.{i}.self_attn.qkv_proj.weight"] = torch.cat([q, k_, v], dim=0)
        g = loaded.get(f"layers.{i}.mlp.gate_proj.weight")
        u = loaded.get(f"layers.{i}.mlp.up_proj.weight")
        if g is not None and u is not None:
            assembled[f"layers.{i}.mlp.gate_up_proj.weight"] = torch.cat([g, u], dim=0)

    sd = model.state_dict()
    with torch.no_grad():
        for name, t in assembled.items():
            if name in sd:
                sd[name].copy_(t.to(torch.bfloat16).cuda())
    model.eval()
    return model, r_cfg, assembled, use_rope


def _time_iters(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record()
        fn()
        e1[i].record()
    torch.cuda.synchronize()
    return sorted(e0[i].elapsed_time(e1[i]) for i in range(iters))


def main(N=100, warmup=30, iters=200):
    torch.manual_seed(17)
    model, r_cfg, weights, use_rope = _build_upstream_residual_lm()

    from fused_layer_chained import FusedCpm4Model

    if use_rope:
        rope = model.layers[0].self_attn.rotary_emb
        cos_cache = rope.cos_cached.to(torch.float32).contiguous()
        sin_cache = rope.sin_cached.to(torch.float32).contiguous()
    else:
        cos_cache = sin_cache = None

    need = set()
    for i in range(r_cfg.num_hidden_layers):
        for k in ("input_layernorm.weight", "self_attn.qkv_proj.weight",
                  "self_attn.o_proj.weight", "post_attention_layernorm.weight",
                  "mlp.gate_up_proj.weight", "mlp.down_proj.weight"):
            need.add(f"layers.{i}.{k}")
    need.add("norm.weight")
    fw = {k: v.to(torch.bfloat16).cuda().contiguous()
          for k, v in weights.items() if k in need}

    ours = FusedCpm4Model(
        weights=fw,
        rope_cos_cache=cos_cache,
        rope_sin_cache=sin_cache,
        hidden=r_cfg.hidden_size,
        intermediate=r_cfg.intermediate_size,
        num_layers=r_cfg.num_hidden_layers,
        causal=True,
        use_rope=use_rope,
        rms_eps=r_cfg.rms_norm_eps,
    )

    positions = torch.arange(N, device="cuda", dtype=torch.int32)
    hs = (torch.randn(N, r_cfg.hidden_size, device="cuda",
                      dtype=torch.bfloat16) * 0.02).contiguous()

    import flash_attn
    print("=" * 72)
    print(f"P2.5.3 — residual_lm {r_cfg.num_hidden_layers}-layer (causal, use_rope={use_rope}) bench")
    print(f"date        : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"flash-attn  : {flash_attn.__version__}")
    print(f"N={N}  hidden={r_cfg.hidden_size}  intermediate={r_cfg.intermediate_size}")
    print(f"warmup={warmup}  iters={iters}")
    print("=" * 72)

    from nanovllm_voxcpm.utils.context import set_context, reset_context
    cu = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    slot = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    def upstream_step():
        set_context(True, cu, cu, N, N, slot, None, None)
        try:
            with torch.inference_mode():
                return model(hs, positions.to(torch.int64))
        finally:
            reset_context()

    def ours_step():
        with torch.inference_mode():
            return ours.forward(hs, positions)

    with torch.inference_mode():
        up_out = upstream_step()
        our_out = ours_step()
    max_abs = (up_out.float() - our_out.float()).abs().max().item()
    max_val = up_out.float().abs().max().item()
    print(f"numerics: max_abs={max_abs:.4e}  max_val={max_val:.3e}"
          f"  max_rel={max_abs/max(max_val,1e-9):.3e}")
    print()

    up_ms = _time_iters(upstream_step, warmup, iters)
    our_e_ms = _time_iters(ours_step, warmup, iters)

    cs = torch.cuda.Stream()
    cs.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cs), torch.inference_mode():
        for _ in range(5):
            _ = ours.forward(hs, positions)
    torch.cuda.current_stream().wait_stream(cs)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        graph_out = ours.forward(hs, positions)
    with torch.inference_mode():
        ref = ours.forward(hs, positions).clone()
    g.replay()
    torch.cuda.synchronize()
    gve = (graph_out.float() - ref.float()).abs().max().item()
    print(f"graph-vs-eager numerics: max={gve:.4e}")

    our_g_ms = _time_iters(lambda: g.replay(), warmup, iters)

    def pct(xs, p): return xs[int(round(p / 100.0 * (len(xs) - 1)))]

    print()
    print(f"{'phase':28s}  {'p50':>7s}  {'p95':>7s}  {'p99':>7s}  {'mean':>7s}  (ms)")
    print(f"{'-'*28}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for name, xs in [("upstream eager", up_ms),
                     ("ours eager", our_e_ms),
                     ("ours graphed", our_g_ms)]:
        print(f"{name:28s}  {pct(xs,50):7.3f}  {pct(xs,95):7.3f}  {pct(xs,99):7.3f}  {statistics.mean(xs):7.3f}")

    print()
    up50, ou50, og50 = pct(up_ms,50), pct(our_e_ms,50), pct(our_g_ms,50)
    print(f"ours eager    vs upstream : {up50/ou50:.2f}x")
    print(f"ours graphed  vs upstream : {up50/og50:.2f}x")
    print(f"launch-overhead saved by graph: {ou50 - og50:.3f} ms  ({100.0*(1 - og50/ou50):.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    main(N=args.N, warmup=args.warmup, iters=args.iters)
