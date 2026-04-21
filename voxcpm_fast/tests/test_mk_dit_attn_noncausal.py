"""P2.5.2 Step 2a — validate `vcpm_attention_noncausal_batched` numerics
vs flash_attn_func(..., causal=False) at the DiT shape.

Shapes tested:
  1. DiT inference shape: B=2, S=11, Hq=16, Hk=2, D=128 (what the megakernel
     will see inside one LocDiT forward at patch_size=4, prefix=6).
  2. Larger S to exercise the K-tile loop: B=1, S=100.
  3. Very small S to cover the one-iter K-loop edge: B=4, S=3.
  4. Strided Q,K,V built via `as_strided` on a packed QKV tensor — the
     shape our future layer wrapper will pass in.

Numerics bar (CLAUDE.md R4):
  bf16  max-abs-diff  ≤ 1e-2
  fp32  max-abs-diff  ≤ 1e-5   (not applicable here — we never compute an
                                 fp32 reference; flash_attn's fp32 path is
                                 not exposed to us).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))

import mk_dit_prefill_ext as ext

from flash_attn import flash_attn_func


NUM_Q = 16
NUM_KV = 2
HEAD_DIM = 128


def _make_qkv(B: int, S: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make Q, K, V at DiT-style shapes (contiguous)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(B, S, NUM_Q, HEAD_DIM, device="cuda",
                     dtype=torch.bfloat16, generator=g) * 0.1).contiguous()
    k = (torch.randn(B, S, NUM_KV, HEAD_DIM, device="cuda",
                     dtype=torch.bfloat16, generator=g) * 0.1).contiguous()
    v = (torch.randn(B, S, NUM_KV, HEAD_DIM, device="cuda",
                     dtype=torch.bfloat16, generator=g) * 0.1).contiguous()
    return q, k, v


def _make_qkv_strided(B: int, S: int, seed: int):
    """Mimic the layer wrapper's `as_strided` views into a packed qkv tensor
    (N, Q_DIM + 2*KV_DIM). This is the real stride pattern the chained
    FusedLayer.forward sends to flash_attn, so we want to validate it here."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    QKV_DIM = NUM_Q * HEAD_DIM + 2 * NUM_KV * HEAD_DIM
    N = B * S
    packed = (torch.randn(N, QKV_DIM, device="cuda",
                          dtype=torch.bfloat16, generator=g) * 0.1).contiguous()
    q_dim = NUM_Q * HEAD_DIM
    kv_dim = NUM_KV * HEAD_DIM
    stride0 = packed.stride(0)  # == QKV_DIM

    q3 = packed.as_strided(
        (B, S, NUM_Q, HEAD_DIM),
        (S * stride0, stride0, HEAD_DIM, 1), storage_offset=0)
    k3 = packed.as_strided(
        (B, S, NUM_KV, HEAD_DIM),
        (S * stride0, stride0, HEAD_DIM, 1), storage_offset=q_dim)
    v3 = packed.as_strided(
        (B, S, NUM_KV, HEAD_DIM),
        (S * stride0, stride0, HEAD_DIM, 1), storage_offset=q_dim + kv_dim)
    return q3, k3, v3


def _check(q, k, v, label: str):
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # Reference: flash_attn with the same strides the kernel sees.
    # flash_attn accepts non-contiguous as long as last-dim stride is 1.
    with torch.inference_mode():
        ref = flash_attn_func(q, k, v, causal=False, softmax_scale=scale)
        ours = ext.attention_noncausal_batched(q, k, v, scale)

    # Shapes should match: (B, S, Hq, D).
    assert ref.shape == ours.shape, (ref.shape, ours.shape)

    diff = (ref.float() - ours.float()).abs()
    max_abs = diff.max().item()
    mae = diff.mean().item()
    max_val = ref.float().abs().max().item()

    print(f"[{label}] B={q.size(0)} S={q.size(1)} Hq={q.size(2)} D={q.size(3)}  "
          f"max_abs={max_abs:.3e}  mae={mae:.3e}  max_val={max_val:.3e}")

    # bf16 numerics bar: ≤ 1e-2 per CLAUDE.md R4.
    assert max_abs <= 1e-2, (
        f"[{label}] bf16 bar violated: max_abs={max_abs:.3e} > 1e-2")


def test_dit_shape_contig():
    """DiT inference shape: B=2, S=11 (contiguous tensors)."""
    q, k, v = _make_qkv(B=2, S=11, seed=17)
    _check(q, k, v, "dit_shape_contig")


def test_dit_shape_strided():
    """DiT inference shape: B=2, S=11 via as_strided packed qkv view."""
    q, k, v = _make_qkv_strided(B=2, S=11, seed=23)
    _check(q, k, v, "dit_shape_strided")


def test_long_seq():
    """Larger S that triggers the K-tile loop (multi-iter softmax)."""
    q, k, v = _make_qkv(B=1, S=100, seed=31)
    _check(q, k, v, "long_seq")


def test_tiny_seq():
    """S smaller than K_BLOCK — many padded K rows masked to -inf."""
    q, k, v = _make_qkv(B=4, S=3, seed=41)
    _check(q, k, v, "tiny_seq")


def test_multi_batch():
    """Sanity: cross-batch isolation. If we mixed batches the max_abs blows."""
    q, k, v = _make_qkv(B=8, S=16, seed=53)
    _check(q, k, v, "multi_batch")


def main():
    test_dit_shape_contig()
    test_dit_shape_strided()
    test_long_seq()
    test_tiny_seq()
    test_multi_batch()
    print()
    print("All 5 non-causal attention numerics tests PASSED (bf16 bar ≤ 1e-2).")


if __name__ == "__main__":
    main()
