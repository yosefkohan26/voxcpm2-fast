"""P2.5.3 — DiT layer, upstream vs ours (chained tuned GEMM), graphed.

Measures a single `VoxCPM2LocDiT.decoder.layers[k]` forward (Cpm4DecoderLayer
with is_causal=False, hidden=1024, intermediate=4096, heads 16/KV 2, head_dim
128) at the real DiT inference shape: batch=2 (CFG), seq=11 (1 + prefix 6 + P 4).

Compares:
    1. Upstream `Cpm4DecoderLayer(is_causal=False)` eager
    2. Ours via `FusedLayer(hidden=1024, intermediate=4096, causal=False)`
       eager and graphed

Inputs/outputs use the same weights and same RoPE cache as upstream.
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


def _build_upstream_dit_layer():
    """Instantiate one `Cpm4DecoderLayer(is_causal=False)` with DiT config, load
    layer-0 weights from the checkpoint, return (layer, lm_cfg, raw_weights)."""
    _init_dist_once()

    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.model import Cpm4Model
    from safetensors.torch import safe_open

    cfg = VoxCPM2Config.model_validate_json((MODEL_DIR / "config.json").read_text())
    # Build the DiT-sized Cpm4Config analogously to VoxCPM2LocDiT's ctor.
    dit_cfg = cfg.lm_config.model_copy(deep=True)
    dit_cfg.hidden_size = cfg.dit_config.hidden_dim
    dit_cfg.intermediate_size = cfg.dit_config.ffn_dim
    dit_cfg.num_attention_heads = cfg.dit_config.num_heads
    # DiT uses num_key_value_heads from same config as encoder/decoder.
    dit_cfg.num_hidden_layers = cfg.dit_config.num_layers
    dit_cfg.use_mup = False
    if cfg.dit_config.kv_channels is not None:
        dit_cfg.kv_channels = cfg.dit_config.kv_channels

    torch.set_default_dtype(torch.bfloat16)
    # Build one non-causal Cpm4 stack; we'll grab layer 0.
    model = Cpm4Model(dit_cfg, is_causal=False).cuda()
    torch.set_default_dtype(torch.float32)

    # Load weights from DiT estimator block.
    prefix = "feat_decoder.estimator.decoder."
    loaded: dict[str, torch.Tensor] = {}
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            if not k.startswith(prefix):
                continue
            loaded[k[len(prefix):]] = f.get_tensor(k)

    assembled = dict(loaded)
    for i in range(dit_cfg.num_hidden_layers):
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
    return model.layers[0], dit_cfg, assembled


def _time_iters(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))


def main(M=22, warmup=30, iters=300):
    torch.manual_seed(17)
    layer, dit_cfg, weights = _build_upstream_dit_layer()

    from fused_layer_chained import FusedLayer

    rope = layer.self_attn.rotary_emb
    cos_cache = rope.cos_cached.to(torch.float32).contiguous()
    sin_cache = rope.sin_cached.to(torch.float32).contiguous()

    # Extract layer-0 weights for FusedLayer.
    lw = {
        "input_layernorm.weight":          weights["layers.0.input_layernorm.weight"],
        "self_attn.qkv_proj.weight":       weights["layers.0.self_attn.qkv_proj.weight"],
        "self_attn.o_proj.weight":         weights["layers.0.self_attn.o_proj.weight"],
        "post_attention_layernorm.weight": weights["layers.0.post_attention_layernorm.weight"],
        "mlp.gate_up_proj.weight":         weights["layers.0.mlp.gate_up_proj.weight"],
        "mlp.down_proj.weight":            weights["layers.0.mlp.down_proj.weight"],
    }
    lw = {k: v.to(torch.bfloat16).cuda().contiguous() for k, v in lw.items()}

    ours = FusedLayer(
        weights=lw,
        rope_cos_cache=cos_cache,
        rope_sin_cache=sin_cache,
        hidden=dit_cfg.hidden_size,
        intermediate=dit_cfg.intermediate_size,
        causal=False,
        rms_eps=dit_cfg.rms_norm_eps,
    )

    # Inputs: (M, hidden) bf16.  M = batch*seq (the DiT flattens 2*11 → 22).
    positions = torch.arange(M, device="cuda", dtype=torch.int32)
    hs = (torch.randn(M, dit_cfg.hidden_size, device="cuda",
                      dtype=torch.bfloat16) * 0.02).contiguous()

    import flash_attn
    print("=" * 72)
    print(f"P2.5.3 — DiT layer (Cpm4DecoderLayer is_causal=False) bench")
    print(f"date        : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"flash-attn  : {flash_attn.__version__}")
    print(f"M={M}  hidden={dit_cfg.hidden_size}  intermediate={dit_cfg.intermediate_size}"
          f"  heads={dit_cfg.num_attention_heads}/{dit_cfg.num_key_value_heads}")
    print(f"warmup={warmup}  iters={iters}")
    print("=" * 72)

    # Upstream forward: expects (batch, seq, hidden). Use batch=1, seq=M.
    hs_3d = hs.unsqueeze(0).contiguous()
    positions_3d = positions.to(torch.int64)

    from nanovllm_voxcpm.utils.context import set_context, reset_context
    cu_q = torch.tensor([0, M], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, M], dtype=torch.int32, device="cuda")
    slot = torch.full((M,), -1, dtype=torch.int32, device="cuda")

    def upstream_step():
        set_context(True, cu_q, cu_k, M, M, slot, None, None)
        try:
            with torch.inference_mode():
                out, _ = layer(positions_3d, hs_3d, None)
        finally:
            reset_context()
        return out

    def ours_step():
        with torch.inference_mode():
            return ours.forward(hs, positions)

    # Numerics: upstream vs ours.
    with torch.inference_mode():
        up_out = upstream_step().view(M, -1).float()
        our_out = ours_step().float()
    max_abs = (up_out - our_out).abs().max().item()
    mae = (up_out - our_out).abs().mean().item()
    print(f"numerics: max_abs={max_abs:.4e}  mae={mae:.4e}  max_val={up_out.abs().max().item():.3e}")
    print()

    # Eager timings.
    up_ms = _time_iters(upstream_step, warmup, iters)
    our_e_ms = _time_iters(ours_step, warmup, iters)

    # Graph capture of ours.
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.inference_mode():
        for _ in range(5):
            _ = ours.forward(hs, positions)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        graph_out = ours.forward(hs, positions)

    # Verify graphed numerics.
    with torch.inference_mode():
        ref = ours.forward(hs, positions).clone()
    g.replay()
    torch.cuda.synchronize()
    gvsE = (graph_out.float() - ref.float()).abs().max().item()
    print(f"graph-vs-eager numerics: max={gvsE:.4e}")

    our_g_ms = _time_iters(lambda: g.replay(), warmup, iters)

    def pct(xs, p): return xs[int(round(p / 100.0 * (len(xs) - 1)))]

    print()
    print(f"{'phase':28s}  {'p50':>7s}  {'p95':>7s}  {'p99':>7s}  {'mean':>7s}  (ms)")
    print(f"{'-'*28}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for name, xs in [("upstream eager",  up_ms),
                     ("ours eager",      our_e_ms),
                     ("ours graphed",    our_g_ms)]:
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
    ap.add_argument("-M", type=int, default=22)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    main(M=args.M, warmup=args.warmup, iters=args.iters)
