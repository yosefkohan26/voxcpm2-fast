"""Numerics + timing test for fused silu_mul + down GEMM.

Reference: torch eager `silu_mul(gu) @ W_dn.T + residual` vs our
`_ext.gemm_bf16_tuned_silu_residual(gu, W_dn, residual)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # direct-run fallback
    pytest = None
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import fused_layer_chained_ext as _ext


def reference_silu_mul_then_gemm(
    gu: torch.Tensor, W: torch.Tensor, residual: torch.Tensor | None = None
) -> torch.Tensor:
    """Reference: silu(gate) * up, then @ W.T, optionally + residual. All bf16."""
    K = W.size(1)
    gate = gu[:, :K]
    up = gu[:, K:]
    mid = F.silu(gate.to(torch.float32)) * up.to(torch.float32)
    out = mid @ W.to(torch.float32).T
    if residual is not None:
        out = out + residual.to(torch.float32)
    return out.to(torch.bfloat16)


if pytest is not None:
    _parametrize_M = pytest.mark.parametrize("M", [64, 128, 192, 256])
else:
    def _parametrize_M(fn):
        return fn


@_parametrize_M
def test_silu_fused_matches_reference(M: int) -> None:
    torch.manual_seed(13)
    H, I = 2048, 6144  # base_lm down_proj shape
    gu = (torch.randn(M, 2 * I, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    W = (torch.randn(H, I, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    residual = (torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

    ref = reference_silu_mul_then_gemm(gu, W, residual)
    ours = _ext.gemm_bf16_tuned_silu_residual(gu, W, residual)

    max_abs = (ref.float() - ours.float()).abs().max().item()
    mean_abs = (ref.float() - ours.float()).abs().mean().item()
    max_val = ref.float().abs().max().item()
    assert max_abs <= 1e-2, (
        f"M={M}: max_abs={max_abs:.4e} exceeds bf16 bar 1e-2 "
        f"(mean_abs={mean_abs:.4e}, max_val={max_val:.4e})"
    )


if pytest is not None:
    _parametrize_M2 = pytest.mark.parametrize("M", [64, 128, 256])
else:
    def _parametrize_M2(fn):
        return fn


@_parametrize_M2
def test_silu_fused_no_residual(M: int) -> None:
    torch.manual_seed(17)
    H, I = 2048, 6144
    gu = (torch.randn(M, 2 * I, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    W = (torch.randn(H, I, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

    ref = reference_silu_mul_then_gemm(gu, W, residual=None)
    ours = _ext.gemm_bf16_tuned_silu(gu, W)

    max_abs = (ref.float() - ours.float()).abs().max().item()
    assert max_abs <= 1e-2, f"M={M}: max_abs={max_abs:.4e}"


if __name__ == "__main__":
    # Quick local run
    for M in (64, 128, 192, 256):
        test_silu_fused_matches_reference(M)
        test_silu_fused_no_residual(M) if M != 192 else None
        print(f"M={M} OK")
    print("all pass")
