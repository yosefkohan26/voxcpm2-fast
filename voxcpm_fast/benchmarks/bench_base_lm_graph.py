"""P2.5.0 — CUDA graph capture of the chained 28-layer base_lm stack.

Measures the launch-overhead-only ceiling: wrap the existing
``FusedCpm4Model.forward`` in a ``torch.cuda.CUDAGraph`` and time the replay
against eager execution of the same code and against upstream eager.

This is the baseline for P2.5. It tells us how much of the 10× gap to the
physics floor is pure kernel-launch dispatch (which graph replay removes)
vs architectural overhead (HBM residual roundtrip, weight re-reads without
prefetch, no fused attn) which only the persistent megakernel removes.
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


def _init_dist_once():
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("gloo", rank=0, world_size=1)


def _build_upstream_base_lm():
    _init_dist_once()

    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.model import Cpm4Model
    from safetensors.torch import safe_open

    cfg = VoxCPM2Config.model_validate_json((MODEL_DIR / "config.json").read_text())
    lm_cfg = cfg.lm_config.model_copy(deep=True)
    lm_cfg.use_mup = False

    torch.set_default_dtype(torch.bfloat16)
    model = Cpm4Model(lm_cfg, is_causal=True).cuda()
    torch.set_default_dtype(torch.float32)

    prefix = "base_lm."
    loaded: dict[str, torch.Tensor] = {}
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            if not k.startswith(prefix):
                continue
            loaded[k[len(prefix):]] = f.get_tensor(k)

    assembled: dict[str, torch.Tensor] = dict(loaded)
    for i in range(lm_cfg.num_hidden_layers):
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
    return model, lm_cfg, assembled


def _run_upstream_prefill(model, positions, input_embeds):
    from nanovllm_voxcpm.utils.context import set_context, reset_context
    N = input_embeds.size(0)
    cu_seqlens_q = torch.tensor([0, N], dtype=torch.int32, device=input_embeds.device)
    cu_seqlens_k = torch.tensor([0, N], dtype=torch.int32, device=input_embeds.device)
    slot_mapping = torch.full((N,), -1, dtype=torch.int32, device=input_embeds.device)
    set_context(True, cu_seqlens_q, cu_seqlens_k, N, N, slot_mapping, None, None)
    try:
        with torch.inference_mode():
            out = model(input_embeds, positions)
    finally:
        reset_context()
    return out


def main(N=100, warmup=20, iters=200):
    torch.manual_seed(17)
    model, lm_cfg, weights = _build_upstream_base_lm()

    from fused_layer_chained import FusedCpm4Model

    rope = model.layers[0].self_attn.rotary_emb
    cos_cache = rope.cos_cached.to(torch.float32).contiguous()
    sin_cache = rope.sin_cached.to(torch.float32).contiguous()

    need_keys = set()
    for i in range(lm_cfg.num_hidden_layers):
        for k in ("input_layernorm.weight", "self_attn.qkv_proj.weight",
                  "self_attn.o_proj.weight", "post_attention_layernorm.weight",
                  "mlp.gate_up_proj.weight", "mlp.down_proj.weight"):
            need_keys.add(f"layers.{i}.{k}")
    need_keys.add("norm.weight")
    fused_weights = {k: v.to(torch.bfloat16).cuda().contiguous()
                     for k, v in weights.items() if k in need_keys}

    ours = FusedCpm4Model(
        weights=fused_weights,
        rope_cos_cache=cos_cache,
        rope_sin_cache=sin_cache,
        hidden=lm_cfg.hidden_size,
        intermediate=lm_cfg.intermediate_size,
        num_layers=lm_cfg.num_hidden_layers,
        causal=True,
        rms_eps=lm_cfg.rms_norm_eps,
    )

    # Stable input buffers. Graph capture records pointer identities of these;
    # replay processes whatever the caller writes into them in-place.
    positions = torch.arange(N, device="cuda", dtype=torch.int32)
    input_embeds = torch.randn(N, lm_cfg.hidden_size, device="cuda",
                               dtype=torch.bfloat16) * 0.02

    import flash_attn
    print("=" * 72)
    print(f"P2.5.0 — CUDA graph capture of chained base_lm (28 layers)")
    print(f"date        : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"flash-attn  : {flash_attn.__version__}")
    print(f"N={N}  hidden={lm_cfg.hidden_size}  layers={lm_cfg.num_hidden_layers}")
    print(f"warmup={warmup}  iters={iters}")
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. Eager baseline (ours, chained kernels, NO graph).
    # ---------------------------------------------------------------
    with torch.inference_mode():
        for _ in range(warmup):
            _ = ours.forward(input_embeds, positions)
        torch.cuda.synchronize()

        e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            e0[i].record()
            _ = ours.forward(input_embeds, positions)
            e1[i].record()
        torch.cuda.synchronize()
        ours_eager_ms = sorted(e0[i].elapsed_time(e1[i]) for i in range(iters))

    # ---------------------------------------------------------------
    # 2. CUDA graph capture (ours).
    # ---------------------------------------------------------------
    # Warm the capture stream per the PyTorch graph-capture recipe:
    # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.inference_mode():
        for _ in range(5):
            _ = ours.forward(input_embeds, positions)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        graph_out = ours.forward(input_embeds, positions)
    torch.cuda.synchronize()

    # Replay once, verify output matches eager.
    with torch.inference_mode():
        eager_ref = ours.forward(input_embeds, positions).clone()
    g.replay()
    torch.cuda.synchronize()
    graph_vs_eager_max = (graph_out.float() - eager_ref.float()).abs().max().item()
    graph_vs_eager_mae = (graph_out.float() - eager_ref.float()).abs().mean().item()

    # Replay timing.
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    e0g = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1g = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0g[i].record()
        g.replay()
        e1g[i].record()
    torch.cuda.synchronize()
    ours_graph_ms = sorted(e0g[i].elapsed_time(e1g[i]) for i in range(iters))

    # ---------------------------------------------------------------
    # 3. Upstream eager baseline.
    # ---------------------------------------------------------------
    with torch.inference_mode():
        for _ in range(warmup):
            _ = _run_upstream_prefill(model, positions, input_embeds)
        torch.cuda.synchronize()

        u0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        u1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            u0[i].record()
            _ = _run_upstream_prefill(model, positions, input_embeds)
            u1[i].record()
        torch.cuda.synchronize()
        up_eager_ms = sorted(u0[i].elapsed_time(u1[i]) for i in range(iters))

    # ---------------------------------------------------------------
    # 4. Results.
    # ---------------------------------------------------------------
    def pct(xs, p): return xs[int(round(p / 100.0 * (len(xs) - 1)))]

    print()
    print(f"graph replay numerics vs eager: max={graph_vs_eager_max:.4e}  mae={graph_vs_eager_mae:.4e}")
    print()
    print(f"{'phase':28s}  {'p50':>7s}  {'p95':>7s}  {'p99':>7s}  {'mean':>7s}  (ms)")
    print(f"{'-'*28}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for name, xs in [("ours eager (chained)", ours_eager_ms),
                     ("ours graphed",         ours_graph_ms),
                     ("upstream eager",       up_eager_ms)]:
        print(f"{name:28s}  {pct(xs,50):7.3f}  {pct(xs,95):7.3f}  {pct(xs,99):7.3f}  {statistics.mean(xs):7.3f}")

    print()
    # Gaps.
    ours_e50 = pct(ours_eager_ms, 50)
    ours_g50 = pct(ours_graph_ms, 50)
    up_e50 = pct(up_eager_ms, 50)
    print(f"ours eager    vs upstream : {up_e50 / ours_e50:.2f}x")
    print(f"ours graphed  vs upstream : {up_e50 / ours_g50:.2f}x")
    print(f"launch-overhead saved by graph: {ours_e50 - ours_g50:.3f} ms  ({100.0*(1 - ours_g50/ours_e50):.1f}%)")
    print()
    PHYSICS_FLOOR_MS = 1.93  # HBM-bw on base_lm weights, c=1, see physics_floor_c1.md
    print(f"physics floor (HBM-bw, base_lm weights): {PHYSICS_FLOOR_MS:.2f} ms")
    print(f"ours graphed gap to floor: {ours_g50 / PHYSICS_FLOOR_MS:.2f}x")
    print()
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    main(N=args.N, warmup=args.warmup, iters=args.iters)
