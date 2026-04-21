"""P2.5.2 Step 2b.1 — validate the 5-phase cooperative megakernel
(RMSNorm + QKV GEMM + RoPE + non-causal attention + O GEMM+residual)
against a chained reference built from the same primitives.

This is the first megakernel commit that executes a non-trivial piece
of the DiT forward end-to-end. If it passes, we have high confidence
in the cooperative-kernel plumbing under real compute.

Numerics bar (CLAUDE.md R4): bf16 max-abs ≤ 1e-2.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))

import mk_dit_prefill_ext as mk_ext
import fused_layer_chained_ext as chained_ext

from flash_attn import flash_attn_func


H = 1024
QKV_DIM = 2560
Q_DIM = 2048
KV_DIM = 256
NUM_Q = 16
NUM_KV = 2
HEAD_DIM = 128
INTERMEDIATE = 4096
RMS_EPS = 1e-6
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _pad_M_to_64(x: torch.Tensor) -> torch.Tensor:
    M = x.size(0)
    pad = (64 - M % 64) % 64
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, pad))


def _make_inputs(B: int, S: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    N_real = B * S
    hs_real = (torch.randn(N_real, H, device="cuda", dtype=torch.bfloat16,
                           generator=g) * 0.1).contiguous()
    hs = _pad_M_to_64(hs_real).contiguous()
    positions_real = torch.arange(N_real, device="cuda", dtype=torch.int32) % 1024
    positions = torch.nn.functional.pad(positions_real, (0, hs.size(0) - N_real), value=0)
    return hs, positions, hs_real


def _make_weights(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w_in_ln = (torch.ones(H, device="cuda", dtype=torch.bfloat16)
               + torch.randn(H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
               ).contiguous()
    w_qkv = (torch.randn(QKV_DIM, H, device="cuda", dtype=torch.bfloat16,
                         generator=g) * 0.02).contiguous()
    w_o = (torch.randn(H, Q_DIM, device="cuda", dtype=torch.bfloat16,
                       generator=g) * 0.02).contiguous()
    return w_in_ln, w_qkv, w_o


def _make_rope_cache(max_pos=2048, seed=7):
    # A small RoPE cache; exact values don't matter as long as both refs and
    # ours use the same. Real DiT uses LongRoPE — our bit-exact numerics
    # downstream don't care which formula as long as it's identical.
    g = torch.Generator(device="cuda").manual_seed(seed)
    cos = torch.randn(max_pos, HEAD_DIM, device="cuda", dtype=torch.float32,
                      generator=g) * 0.5 + 0.5  # roughly in [-0.5, 1.5]
    sin = torch.randn(max_pos, HEAD_DIM, device="cuda", dtype=torch.float32,
                      generator=g) * 0.5
    return cos.contiguous(), sin.contiguous()


def _chained_ref(hs_padded, w_in_ln, w_qkv, w_o, cos, sin, positions, B, S):
    """Run phases 1-5 via the chained primitives. Mirrors FusedLayer.forward
    up through the o_proj+residual step."""
    # Phase 1: RMSNorm.
    ln_out = chained_ext.rmsnorm(hs_padded, w_in_ln, RMS_EPS)
    # Phase 2: QKV GEMM.
    qkv = chained_ext.gemm_bf16_tuned(ln_out, w_qkv)
    # Phase 3: RoPE in-place.
    qkv = qkv.clone()  # keep a clean copy for inspection; rope_inplace mutates.
    chained_ext.rope_inplace(qkv, cos, sin, positions,
                             NUM_Q, NUM_KV, HEAD_DIM)
    # Phase 4: non-causal batched attention.
    # Build (B, S, H, D) views on the REAL rows only.
    N_real = B * S
    qkv_real = qkv[:N_real].contiguous()
    q3 = qkv_real[:, :Q_DIM].view(B, S, NUM_Q, HEAD_DIM)
    k3 = qkv_real[:, Q_DIM:Q_DIM + KV_DIM].view(B, S, NUM_KV, HEAD_DIM)
    v3 = qkv_real[:, Q_DIM + KV_DIM:].view(B, S, NUM_KV, HEAD_DIM)
    attn = flash_attn_func(q3, k3, v3, causal=False, softmax_scale=ATTN_SCALE)
    attn_2d_real = attn.reshape(N_real, Q_DIM)
    # Pad attn up to M (padded rows are zero — the megakernel's scratch_a
    # residual-add reads from rows [0, M); padded rows don't affect real
    # rows of hs_out but their hs_out values are garbage-but-ignorable).
    M = hs_padded.size(0)
    attn_padded = torch.zeros(M, Q_DIM, device="cuda", dtype=torch.bfloat16)
    attn_padded[:N_real] = attn_2d_real
    # Phase 5: hs_out = attn_out @ w_o^T + hs (residual-fused).
    hs_out = chained_ext.gemm_bf16_tuned_residual(attn_padded, w_o, hs_padded)
    return hs_out


def _check(label, ref, ours, N_real, bar=1e-2):
    """Compare only real rows [0, N_real); padded rows don't matter."""
    diff = (ref[:N_real].float() - ours[:N_real].float()).abs()
    max_abs = diff.max().item()
    mae = diff.mean().item()
    max_val = ref[:N_real].float().abs().max().item()
    print(f"  [{label}] real_rows={N_real}  "
          f"max_abs={max_abs:.3e}  mae={mae:.3e}  max_val={max_val:.3e}")
    assert max_abs <= bar, f"[{label}] max_abs={max_abs:.3e} > {bar}"


def test_dit_shape():
    """DiT inference: B=2, S=11 → N_real=22, padded M=64."""
    hs, positions, hs_real = _make_inputs(B=2, S=11, seed=17)
    w_in_ln, w_qkv, w_o = _make_weights(seed=19)
    cos, sin = _make_rope_cache()
    B, S = 2, 11
    N_real = B * S

    ref  = _chained_ref(hs, w_in_ln, w_qkv, w_o, cos, sin, positions, B, S)
    ours = mk_ext.step2b1_partial_layer(
        hs, w_in_ln, w_qkv, w_o, cos, sin, positions, B, S,
        RMS_EPS, ATTN_SCALE)

    print("[dit_shape] B=2 S=11")
    _check("step2b1_hs_out", ref, ours, N_real)


def test_larger_batch():
    """Bigger batch: B=4, S=32 → N_real=128 (already 64-aligned)."""
    hs, positions, hs_real = _make_inputs(B=4, S=32, seed=29)
    w_in_ln, w_qkv, w_o = _make_weights(seed=31)
    cos, sin = _make_rope_cache()
    B, S = 4, 32
    N_real = B * S
    assert hs.size(0) == 128

    ref  = _chained_ref(hs, w_in_ln, w_qkv, w_o, cos, sin, positions, B, S)
    ours = mk_ext.step2b1_partial_layer(
        hs, w_in_ln, w_qkv, w_o, cos, sin, positions, B, S,
        RMS_EPS, ATTN_SCALE)

    print("[larger_batch] B=4 S=32")
    _check("step2b1_hs_out", ref, ours, N_real)


def main():
    test_dit_shape()
    test_larger_batch()
    print()
    print("All Step 2b.1 5-phase megakernel numerics tests PASSED.")


if __name__ == "__main__":
    main()
