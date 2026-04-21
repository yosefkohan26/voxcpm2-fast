"""P2.5.2 Step 2b.0 — validate the 2-phase cooperative megakernel
(RMSNorm + QKV_GEMM with cg::this_grid().sync() between) against the
chained equivalent.

Chained reference:
    ln_out_ref = _ext.rmsnorm(hs, w_in_ln, eps)
    qkv_ref    = _ext.gemm_bf16_tuned(ln_out_ref, w_qkv)

Ours:
    ln_out, qkv = mk_dit_prefill_ext.step2b0_rmsnorm_qkv(
        hs_padded_64, w_in_ln, w_qkv, eps)

We compare BOTH phase outputs (ln_out and qkv) to the chained refs.

The whole point of this commit is to PROVE the cooperative-launch +
grid.sync() plumbing works for real multi-phase compute. Perf is
secondary (will come in Step 2b.x as more phases fold in).

Numerics bar (CLAUDE.md R4):
    ln_out:  max-abs ≤ 1e-2  (pure bf16 pointwise)
    qkv:     max-abs ≤ 1e-2  (bf16 GEMM with fp32 accumulator)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))

import mk_dit_prefill_ext as mk_ext
import fused_layer_chained_ext as chained_ext


H = 1024
QKV_DIM = 2560
RMS_EPS = 1e-6


def _pad_M_to_64(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    M = x.size(0)
    pad = (64 - M % 64) % 64
    if pad == 0:
        return x, 0
    return torch.nn.functional.pad(x, (0, 0, 0, pad)), pad


def _chained_ref(hs_pad, w_in_ln, w_qkv):
    ln_out = chained_ext.rmsnorm(hs_pad, w_in_ln, RMS_EPS)
    qkv    = chained_ext.gemm_bf16_tuned(ln_out, w_qkv)
    return ln_out, qkv


def _mk_run(hs_pad, w_in_ln, w_qkv):
    return mk_ext.step2b0_rmsnorm_qkv(hs_pad, w_in_ln, w_qkv, RMS_EPS)


def _check(label, ref, ours, bar=1e-2):
    assert ref.shape == ours.shape, (ref.shape, ours.shape)
    diff = (ref.float() - ours.float()).abs()
    max_abs = diff.max().item()
    mae = diff.mean().item()
    max_val = ref.float().abs().max().item()
    print(f"  [{label}] max_abs={max_abs:.3e}  mae={mae:.3e}  max_val={max_val:.3e}")
    assert max_abs <= bar, f"[{label}] max_abs={max_abs:.3e} > {bar}"


def _make_weights(seed=31):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Typical RMSNorm weight is near 1.0; make it so.
    w_in_ln = (torch.ones(H, device="cuda", dtype=torch.bfloat16)
               + torch.randn(H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
               ).contiguous()
    # QKV weights are small-scale init.
    w_qkv = (torch.randn(QKV_DIM, H, device="cuda", dtype=torch.bfloat16,
                         generator=g) * 0.02).contiguous()
    return w_in_ln, w_qkv


def test_dit_shape():
    """DiT inference shape: N_real=22 → padded to M=64."""
    g = torch.Generator(device="cuda").manual_seed(17)
    hs = (torch.randn(22, H, device="cuda", dtype=torch.bfloat16,
                      generator=g) * 0.1).contiguous()
    hs_pad, pad = _pad_M_to_64(hs)
    assert hs_pad.size(0) == 64

    w_in_ln, w_qkv = _make_weights(seed=19)

    ln_ref, qkv_ref = _chained_ref(hs_pad, w_in_ln, w_qkv)
    ln_our, qkv_our = _mk_run(hs_pad, w_in_ln, w_qkv)

    print(f"[dit_shape] M_padded=64 (real=22)")
    _check("phase1_rmsnorm",  ln_ref,  ln_our)
    _check("phase2_qkv_gemm", qkv_ref, qkv_our)


def test_larger_m():
    """Bigger M (exercises multi-tile GEMM; tiles_n = 2560/128 = 20, tiles_m = 2)."""
    g = torch.Generator(device="cuda").manual_seed(29)
    hs_pad = (torch.randn(128, H, device="cuda", dtype=torch.bfloat16,
                          generator=g) * 0.1).contiguous()

    w_in_ln, w_qkv = _make_weights(seed=37)

    ln_ref, qkv_ref = _chained_ref(hs_pad, w_in_ln, w_qkv)
    ln_our, qkv_our = _mk_run(hs_pad, w_in_ln, w_qkv)

    print(f"[larger_m] M=128")
    _check("phase1_rmsnorm",  ln_ref,  ln_our)
    _check("phase2_qkv_gemm", qkv_ref, qkv_our)


def test_bigger_m_multi_mtile():
    """M=256 means 4 M-tiles × 20 N-tiles = 80 tiles total. Exceeds grid=64
    so the work-stealing `for (tile = blockIdx.x; tile < total; tile +=
    gridDim.x)` loop must actually loop more than once per block. Important
    to validate because the chained form always grids = total_tiles."""
    g = torch.Generator(device="cuda").manual_seed(41)
    hs_pad = (torch.randn(256, H, device="cuda", dtype=torch.bfloat16,
                          generator=g) * 0.1).contiguous()
    w_in_ln, w_qkv = _make_weights(seed=43)

    ln_ref, qkv_ref = _chained_ref(hs_pad, w_in_ln, w_qkv)
    ln_our, qkv_our = _mk_run(hs_pad, w_in_ln, w_qkv)

    print(f"[bigger_m_multi_mtile] M=256  (tile count 80 > grid 64; stealing path)")
    _check("phase1_rmsnorm",  ln_ref,  ln_our)
    _check("phase2_qkv_gemm", qkv_ref, qkv_our)


def main():
    test_dit_shape()
    test_larger_m()
    test_bigger_m_multi_mtile()
    print()
    print("All 3 Step 2b.0 cooperative-megakernel numerics tests PASSED.")


if __name__ == "__main__":
    main()
