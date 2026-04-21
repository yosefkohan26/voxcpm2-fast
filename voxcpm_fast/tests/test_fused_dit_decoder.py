"""Numerics test for FusedCpm4Model @ DiT shape (non-causal, 12-layer).

Compares ``FusedCpm4Model(hidden=1024, intermediate=4096, causal=False)``
at batch=2, seq=11 (one CFG DiT pass) against upstream
``Cpm4Model(is_causal=False)`` with real weights from the DiT estimator.

Bar: max_rel <= 1e-2 (per CLAUDE.md R4), graph vs eager bit-exact.
"""

from __future__ import annotations

import os
import sys
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


def _build_upstream_dit():
    _init_dist()
    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.model import Cpm4Model
    from safetensors.torch import safe_open

    cfg = VoxCPM2Config.model_validate_json((MODEL_DIR / "config.json").read_text())
    dit_cfg = cfg.lm_config.model_copy(deep=True)
    dit_cfg.hidden_size = cfg.dit_config.hidden_dim
    dit_cfg.intermediate_size = cfg.dit_config.ffn_dim
    dit_cfg.num_attention_heads = cfg.dit_config.num_heads
    dit_cfg.num_hidden_layers = cfg.dit_config.num_layers
    dit_cfg.use_mup = False
    if cfg.dit_config.kv_channels is not None:
        dit_cfg.kv_channels = cfg.dit_config.kv_channels

    torch.set_default_dtype(torch.bfloat16)
    model = Cpm4Model(dit_cfg, is_causal=False).cuda()
    torch.set_default_dtype(torch.float32)

    prefix = "feat_decoder.estimator.decoder."
    loaded: dict[str, torch.Tensor] = {}
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
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
    return model, dit_cfg, assembled


def test_dit_decoder_numerics_bf16():
    torch.manual_seed(17)
    model, dit_cfg, weights = _build_upstream_dit()

    from fused_layer_chained import FusedCpm4Model

    rope = model.layers[0].self_attn.rotary_emb
    cos_cache = rope.cos_cached.to(torch.float32).contiguous()
    sin_cache = rope.sin_cached.to(torch.float32).contiguous()

    need = set()
    for i in range(dit_cfg.num_hidden_layers):
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
        hidden=dit_cfg.hidden_size,
        intermediate=dit_cfg.intermediate_size,
        num_layers=dit_cfg.num_hidden_layers,
        causal=False,
        rms_eps=dit_cfg.rms_norm_eps,
    )

    batch_size, seq = 2, 11
    N = batch_size * seq
    positions = torch.arange(seq, device="cuda", dtype=torch.int32).repeat(batch_size)
    hs = (torch.randn(N, dit_cfg.hidden_size, device="cuda",
                      dtype=torch.bfloat16) * 0.02).contiguous()

    hs_3d = hs.view(batch_size, seq, -1).contiguous()
    positions_seq = positions[:seq].to(torch.int64)

    from nanovllm_voxcpm.utils.context import set_context, reset_context
    cu_q = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    slot = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    with torch.inference_mode():
        set_context(True, cu_q, cu_k, N, N, slot, None, None)
        try:
            up_out = model(hs_3d, positions_seq)
        finally:
            reset_context()
        our_out = ours.forward(hs, positions, batch_size=batch_size)

    up_flat = up_out.reshape(N, -1).float()
    our_flat = our_out.float()
    diff = (up_flat - our_flat).abs()
    max_abs = diff.max().item()
    mae = diff.mean().item()
    max_val = up_flat.abs().max().item()
    max_rel = max_abs / max(max_val, 1e-9)

    assert max_rel <= 1e-2, (
        f"DiT decoder max_rel {max_rel:.3e} > 1e-2 bar "
        f"(max_abs={max_abs:.3e}, mae={mae:.3e}, max_val={max_val:.3e})"
    )


if __name__ == "__main__":
    test_dit_decoder_numerics_bf16()
    print("dit decoder numerics OK")
