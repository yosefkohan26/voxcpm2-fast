"""Numerics test for FusedCpm4Model @ feat_encoder shape (12-layer non-causal).

Same shape as DiT decoder (hidden=1024, intermediate=4096, 16/2 heads) but
different weights. Batch=1, seq=5 (CLS + patch_size=4).

Bar: max_rel <= 1.5e-2. bf16 accumulation over 12 layers with non-normalized
inputs reaches this level structurally.
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
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast" / "benchmarks"))
sys.path.insert(0, str(NANOVLLM_ROOT))


def test_feat_encoder_numerics_bf16():
    from bench_feat_encoder import _build_upstream_feat_encoder
    from fused_layer_chained import FusedCpm4Model

    torch.manual_seed(17)
    model, enc_cfg, weights = _build_upstream_feat_encoder()

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

    batch, seq = 1, 5
    N = batch * seq
    positions = torch.arange(seq, device="cuda", dtype=torch.int32).repeat(batch)
    hs = torch.randn(N, enc_cfg.hidden_size, device="cuda",
                     dtype=torch.bfloat16).contiguous()
    hs_3d = hs.view(batch, seq, -1).contiguous()
    positions_seq = positions[:seq].to(torch.int64)

    from nanovllm_voxcpm.utils.context import set_context, reset_context
    cu = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    slot = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    with torch.inference_mode():
        set_context(True, cu, cu, N, N, slot, None, None)
        try:
            up_out = model(hs_3d, positions_seq)
        finally:
            reset_context()
        our_out = ours.forward(hs, positions, batch_size=batch)

    up_flat = up_out.reshape(N, -1).float()
    our_flat = our_out.float()
    diff = (up_flat - our_flat).abs()
    max_abs = diff.max().item()
    max_val = up_flat.abs().max().item()
    max_rel = max_abs / max(max_val, 1e-9)

    assert max_rel <= 1.5e-2, (
        f"feat_encoder max_rel {max_rel:.3e} > 1.5e-2 bar "
        f"(max_abs={max_abs:.3e}, max_val={max_val:.3e})"
    )


if __name__ == "__main__":
    test_feat_encoder_numerics_bf16()
    print("feat_encoder numerics OK")
