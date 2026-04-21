"""Numerics test for FusedCpm4Model @ residual_lm shape (8-layer, causal, no RoPE).

Same per-layer shape as base_lm but 8 layers and no RoPE. Bar: max_rel <= 1.5e-2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"

sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast" / "benchmarks"))
sys.path.insert(0, str(NANOVLLM_ROOT))


def test_residual_lm_numerics_bf16():
    from bench_residual_lm import _build_upstream_residual_lm
    from fused_layer_chained import FusedCpm4Model

    torch.manual_seed(17)
    model, r_cfg, weights, use_rope = _build_upstream_residual_lm()

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

    N = 100
    positions = torch.arange(N, device="cuda", dtype=torch.int32)
    hs = (torch.randn(N, r_cfg.hidden_size, device="cuda",
                      dtype=torch.bfloat16) * 0.02).contiguous()

    from nanovllm_voxcpm.utils.context import set_context, reset_context
    cu = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    slot = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    with torch.inference_mode():
        set_context(True, cu, cu, N, N, slot, None, None)
        try:
            up_out = model(hs, positions.to(torch.int64))
        finally:
            reset_context()
        our_out = ours.forward(hs, positions)

    diff = (up_out.float() - our_out.float()).abs()
    max_abs = diff.max().item()
    max_val = up_out.float().abs().max().item()
    max_rel = max_abs / max(max_val, 1e-9)
    assert max_rel <= 1.5e-2, (
        f"residual_lm max_rel {max_rel:.3e} > 1.5e-2 bar "
        f"(max_abs={max_abs:.3e}, max_val={max_val:.3e})"
    )


if __name__ == "__main__":
    test_residual_lm_numerics_bf16()
    print("residual_lm numerics OK")
