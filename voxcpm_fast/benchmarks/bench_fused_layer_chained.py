"""Wall-time benchmark for the P2.2 chained fused non-causal layer.

Measures end-to-end time per call at N=100 tokens, c=1, with hot weights and
hot CUDA kernels. Uses CUDA events so we measure pure GPU time (no
host-side Python dispatch unless that's part of the call).
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


def main(N=100, warmup=20, iters=500):
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("gloo", rank=0, world_size=1)

    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.model import Cpm4DecoderLayer
    from safetensors.torch import safe_open

    cfg = VoxCPM2Config.model_validate_json((MODEL_DIR / "config.json").read_text())
    enc_cfg = cfg.lm_config.model_copy(deep=True)
    enc_cfg.hidden_size = cfg.encoder_config.hidden_dim
    enc_cfg.intermediate_size = cfg.encoder_config.ffn_dim
    enc_cfg.num_attention_heads = cfg.encoder_config.num_heads
    enc_cfg.num_hidden_layers = cfg.encoder_config.num_layers
    enc_cfg.kv_channels = cfg.encoder_config.kv_channels
    enc_cfg.vocab_size = 0

    torch.set_default_dtype(torch.bfloat16)
    layer = Cpm4DecoderLayer(enc_cfg, is_causal=False).cuda().eval()
    torch.set_default_dtype(torch.float32)

    # Load feat_encoder.encoder.layers.0 weights.
    prefix = "feat_encoder.encoder.layers.0."
    loaded: dict[str, torch.Tensor] = {}
    attn_comp: dict[str, torch.Tensor] = {}
    mlp_comp: dict[str, torch.Tensor] = {}
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            if not k.startswith(prefix):
                continue
            sub = k[len(prefix):]
            if sub in {
                "input_layernorm.weight", "self_attn.qkv_proj.weight",
                "self_attn.o_proj.weight", "post_attention_layernorm.weight",
                "mlp.gate_up_proj.weight", "mlp.down_proj.weight",
            }:
                loaded[sub] = f.get_tensor(k)
            for part in ("q_proj", "k_proj", "v_proj"):
                if sub == f"self_attn.{part}.weight":
                    attn_comp[part] = f.get_tensor(k)
            for part in ("gate_proj", "up_proj"):
                if sub == f"mlp.{part}.weight":
                    mlp_comp[part] = f.get_tensor(k)
    if "self_attn.qkv_proj.weight" not in loaded and len(attn_comp) == 3:
        loaded["self_attn.qkv_proj.weight"] = torch.cat(
            [attn_comp["q_proj"], attn_comp["k_proj"], attn_comp["v_proj"]], dim=0)
    if "mlp.gate_up_proj.weight" not in loaded and len(mlp_comp) == 2:
        loaded["mlp.gate_up_proj.weight"] = torch.cat(
            [mlp_comp["gate_proj"], mlp_comp["up_proj"]], dim=0)

    sd = layer.state_dict()
    with torch.no_grad():
        for name, tensor in loaded.items():
            sd[name].copy_(tensor.to(torch.bfloat16).cuda())

    from fused_layer_chained import FusedNonCausalLayer
    rope = layer.self_attn.rotary_emb
    cos_cache = rope.cos_cached.to(torch.float32).contiguous()
    sin_cache = rope.sin_cached.to(torch.float32).contiguous()
    weights = {k: v.to(torch.bfloat16).cuda().contiguous() for k, v in loaded.items()}
    ours = FusedNonCausalLayer(
        weights=weights, rope_cos_cache=cos_cache, rope_sin_cache=sin_cache,
        rms_eps=enc_cfg.rms_norm_eps,
    )

    positions = torch.arange(N, device="cuda", dtype=torch.int32)
    hs = torch.randn(N, 1024, device="cuda", dtype=torch.bfloat16) * 0.1

    # Env fingerprint.
    import flash_attn
    print("=" * 72)
    print(f"date        : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"torch       : {torch.__version__}")
    print(f"flash-attn  : {flash_attn.__version__}")
    print(f"script      : bench_fused_layer_chained.py  N={N} warmup={warmup} iters={iters}")
    print("=" * 72)

    with torch.inference_mode():
        # Warmup (JITs flash_attn, any torch graph overhead).
        for _ in range(warmup):
            _ = ours.forward(hs, positions)
        torch.cuda.synchronize()

        # Ours: timed with CUDA events.
        e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        torch.cuda.synchronize()
        for i in range(iters):
            e0[i].record()
            _ = ours.forward(hs, positions)
            e1[i].record()
        torch.cuda.synchronize()
        ours_us = sorted(e0[i].elapsed_time(e1[i]) * 1000.0 for i in range(iters))

        # Upstream baseline (eager, no CUDA graph — same apples-to-apples,
        # both run kernels individually on the stream).
        hs3d = hs.unsqueeze(0).contiguous()
        for _ in range(warmup):
            _ = layer(positions, hs3d, None)
        torch.cuda.synchronize()
        u0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        u1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            u0[i].record()
            _ = layer(positions, hs3d, None)
            u1[i].record()
        torch.cuda.synchronize()
        up_us = sorted(u0[i].elapsed_time(u1[i]) * 1000.0 for i in range(iters))

    def pct(xs, p): return xs[int(round(p / 100.0 * (len(xs) - 1)))]

    print()
    print(f"iters       : {iters}")
    print(f"N           : {N}")
    print()
    print("ours (chained kernels):")
    print(f"  p50 = {pct(ours_us, 50):8.2f} µs")
    print(f"  p95 = {pct(ours_us, 95):8.2f} µs")
    print(f"  p99 = {pct(ours_us, 99):8.2f} µs")
    print(f"  mean= {statistics.mean(ours_us):8.2f} µs")
    print()
    print("upstream (eager Cpm4DecoderLayer):")
    print(f"  p50 = {pct(up_us, 50):8.2f} µs")
    print(f"  p95 = {pct(up_us, 95):8.2f} µs")
    print(f"  p99 = {pct(up_us, 99):8.2f} µs")
    print(f"  mean= {statistics.mean(up_us):8.2f} µs")
    print()
    speedup = statistics.mean(up_us) / statistics.mean(ours_us)
    print(f"mean speedup vs upstream eager: {speedup:.2f}x")
    print()
    # Physics floor reminder: single-layer compute floor is ~68 µs (see
    # notes/physics_floor_c1.md). We're a chained implementation with host
    # launch overhead between stages — will never hit floor in this form;
    # P2.5 persistent megakernel does.
    print(f"physics floor (compute) : ~68 µs per layer (physics_floor_c1.md)")
    print(f"our gap vs floor        : {pct(ours_us, 50) / 68.0:.2f}x over floor")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()
    main(N=args.N, warmup=args.warmup, iters=args.iters)
