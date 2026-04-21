"""Integration numerics test: full VoxCPM2Model.forward with fused shims.

With both runs seeded identically, the stochastic feat_decoder sampling
converges. Bar: latents max_rel ≤ 5e-2 (compound bf16 through
10 Euler × 12-layer DiT = 120 layer-forwards), stop_flag bit-exact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast" / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "nanovllm-voxcpm"))


def test_voxcpm2_forward_numerics_bf16():
    from bench_voxcpm2_forward import _load_upstream_voxcpm2, _install_fast_shims
    from nanovllm_voxcpm.utils.context import set_context, reset_context

    torch.manual_seed(17)
    model_up, cfg = _load_upstream_voxcpm2()
    model_ours, _ = _load_upstream_voxcpm2()
    _install_fast_shims(model_ours, cfg)

    N = 100
    P = cfg.patch_size
    feat_dim = cfg.feat_dim
    positions = torch.arange(N, device="cuda", dtype=torch.int64)
    text_tokens = torch.randint(0, cfg.lm_config.vocab_size - 100,
                                (N,), device="cuda", dtype=torch.int64)
    feat = (torch.randn(N, P, feat_dim, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    feat_mask = torch.zeros(N, device="cuda", dtype=torch.bool)
    feat_mask[N // 2:] = True
    temperature = torch.tensor([0.7], device="cuda", dtype=torch.bfloat16)
    cfg_value = torch.tensor([1.5], device="cuda", dtype=torch.bfloat16)

    cu = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    slot = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    def _run(model, seed=17):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        set_context(True, cu, cu, N, N, slot, None, None)
        try:
            with torch.inference_mode():
                out = model(positions, text_tokens, feat, feat_mask,
                            temperature, cfg_value)
        finally:
            reset_context()
        return out

    up = _run(model_up)
    ours = _run(model_ours)

    # Latents (continuous) — compound bf16 through 120 DiT layer-forwards.
    u = up["latents"].float()
    o = ours["latents"].float()
    d = (u - o).abs()
    max_rel = d.max().item() / max(u.abs().max().item(), 1e-9)
    assert max_rel <= 5e-2, f"latents max_rel {max_rel:.3e} > 5e-2"

    # stop_flag (argmax → discrete).
    s_up = up["stop_flag"]
    s_our = ours["stop_flag"]
    assert torch.equal(s_up, s_our), "stop_flag differs"


if __name__ == "__main__":
    test_voxcpm2_forward_numerics_bf16()
    print("voxcpm2 forward numerics OK")
