"""P2.5.3 — feat_encoder (12-layer non-causal Cpm4Model) bench.

Shape: batch=1, seq=5 (CLS + patch_size=4 feats), hidden=1024, intermediate=4096,
heads 16/KV 2. Identical per-layer shape to DiT decoder.

feat_encoder runs once per decode step on every stream, so its graph-capture
ceiling is important for steady-state throughput.
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


def _build_upstream_feat_encoder():
    _init_dist()
    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.model import Cpm4Model
    from safetensors.torch import safe_open

    cfg = VoxCPM2Config.model_validate_json((MODEL_DIR / "config.json").read_text())
    enc_cfg = cfg.lm_config.model_copy(deep=True)
    enc_cfg.hidden_size = cfg.encoder_config.hidden_dim
    enc_cfg.intermediate_size = cfg.encoder_config.ffn_dim
    enc_cfg.num_attention_heads = cfg.encoder_config.num_heads
    enc_cfg.num_hidden_layers = cfg.encoder_config.num_layers
    enc_cfg.use_mup = False
    if cfg.encoder_config.kv_channels is not None:
        enc_cfg.kv_channels = cfg.encoder_config.kv_channels

    torch.set_default_dtype(torch.bfloat16)
    model = Cpm4Model(enc_cfg, is_causal=False).cuda()
    torch.set_default_dtype(torch.float32)

    prefix = "feat_encoder.encoder."
    loaded: dict[str, torch.Tensor] = {}
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                loaded[k[len(prefix):]] = f.get_tensor(k)

    assembled = dict(loaded)
    for i in range(enc_cfg.num_hidden_layers):
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
    return model, enc_cfg, assembled


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


def main(batch=1, seq=5, warmup=30, iters=300):
    torch.manual_seed(17)
    model, enc_cfg, weights = _build_upstream_feat_encoder()

    from fused_layer_chained import FusedCpm4Model

    rope = model.layers[0].self_attn.rotary_emb
    cos_cache = rope.cos_cached.to(torch.float32).contiguous()
    sin_cache = rope.sin_cached.to(torch.float32).contiguous()

    need = set()
    for i in range(enc_cfg.num_hidden_layers):
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
        hidden=enc_cfg.hidden_size,
        intermediate=enc_cfg.intermediate_size,
        num_layers=enc_cfg.num_hidden_layers,
        causal=False,
        rms_eps=enc_cfg.rms_norm_eps,
    )

    N = batch * seq
    positions = torch.arange(seq, device="cuda", dtype=torch.int32).repeat(batch)
    # Use σ≈1 so RMSNorm eps doesn't dominate (real activations are normalized).
    hs = torch.randn(N, enc_cfg.hidden_size, device="cuda",
                     dtype=torch.bfloat16).contiguous()

    import flash_attn
    print("=" * 72)
    print(f"P2.5.3 — feat_encoder 12-layer bench")
    print(f"date        : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"flash-attn  : {flash_attn.__version__}")
    print(f"batch={batch}  seq={seq}  hidden={enc_cfg.hidden_size}"
          f"  intermediate={enc_cfg.intermediate_size}  layers={enc_cfg.num_hidden_layers}")
    print(f"warmup={warmup}  iters={iters}")
    print("=" * 72)

    hs_3d = hs.view(batch, seq, -1).contiguous()
    positions_seq = positions[:seq].to(torch.int64)

    from nanovllm_voxcpm.utils.context import set_context, reset_context
    cu_q = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    slot = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    def upstream_step():
        set_context(True, cu_q, cu_k, N, N, slot, None, None)
        try:
            with torch.inference_mode():
                return model(hs_3d, positions_seq)
        finally:
            reset_context()

    def ours_step():
        with torch.inference_mode():
            return ours.forward(hs, positions, batch_size=batch)

    with torch.inference_mode():
        up_out = upstream_step()
        our_out = ours_step()
    up_flat = up_out.reshape(N, -1).float()
    our_flat = our_out.float()
    max_abs = (up_flat - our_flat).abs().max().item()
    mae = (up_flat - our_flat).abs().mean().item()
    max_val = up_flat.abs().max().item()
    print(f"numerics: max_abs={max_abs:.4e}  mae={mae:.4e}  max_val={max_val:.3e}"
          f"  max_rel={max_abs/max(max_val,1e-9):.3e}")
    print()

    up_ms = _time_iters(upstream_step, warmup, iters)
    our_e_ms = _time_iters(ours_step, warmup, iters)

    cs = torch.cuda.Stream()
    cs.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cs), torch.inference_mode():
        for _ in range(5):
            _ = ours.forward(hs, positions, batch_size=batch)
    torch.cuda.current_stream().wait_stream(cs)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        graph_out = ours.forward(hs, positions, batch_size=batch)
    with torch.inference_mode():
        ref = ours.forward(hs, positions, batch_size=batch).clone()
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
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    main(batch=args.batch, seq=args.seq, warmup=args.warmup, iters=args.iters)
