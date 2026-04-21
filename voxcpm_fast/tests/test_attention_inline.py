"""P2.5.1.b — numerics + perf for the inline causal attention kernel.

Compares `vcpm_attention_causal` against `flash_attn_func` (causal=True) at
the base_lm GQA shape (NUM_Q=16, NUM_KV=2, HEAD_DIM=128) across several N
to exercise partial tiles and causal-mask edge cases.

Gates:
  - max rel diff ≤ 1e-2  (one bf16 ULP at typical layer output magnitudes)
  - mean rel diff ≤ 1e-3
"""

from __future__ import annotations

import os
import sys
import statistics
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))
sys.path.insert(0, str(REPO_ROOT / "nanovllm-voxcpm"))


NUM_Q = 16
NUM_KV = 2
D = 128
Q_DIM = NUM_Q * D
KV_DIM = NUM_KV * D


@pytest.mark.parametrize("N", [32, 64, 96, 100, 128, 200])
def test_attention_causal_numerics(N):
    torch.manual_seed(7)
    import fused_layer_chained_ext as _ext
    from flash_attn import flash_attn_func

    q = (torch.randn(N, Q_DIM, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    k = (torch.randn(N, KV_DIM, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    v = (torch.randn(N, KV_DIM, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    scale = D ** -0.5

    # Reference: flash_attn_func expects [B, N, H, D]
    q3 = q.view(1, N, NUM_Q, D).contiguous()
    k3 = k.view(1, N, NUM_KV, D).contiguous()
    v3 = v.view(1, N, NUM_KV, D).contiguous()
    ref = flash_attn_func(q3, k3, v3, softmax_scale=scale, causal=True)
    ref_2d = ref.view(N, Q_DIM).contiguous()

    # Ours
    got = _ext.attention_causal(q, k, v, scale)

    diff = (got.float() - ref_2d.float()).abs()
    up_max = ref_2d.abs().max().item()
    max_abs = diff.max().item()
    mae = diff.mean().item()
    rel_max = max_abs / max(up_max, 1e-9)
    rel_mae = mae / max(up_max, 1e-9)

    print()
    print(f"=== attention N={N} ===")
    print(f"  up_max={up_max:.4f}  ours_max={got.abs().max().item():.4f}")
    print(f"  max={max_abs:.4e}  mae={mae:.4e}  rel_max={rel_max:.4e}  rel_mae={rel_mae:.4e}")

    assert rel_max <= 1e-2, f"N={N}: rel_max too high: {rel_max:.4e}"
    assert rel_mae <= 1e-3, f"N={N}: rel_mae too high: {rel_mae:.4e}"


def test_attention_causal_perf():
    import fused_layer_chained_ext as _ext
    from flash_attn import flash_attn_func

    N = 100
    q = (torch.randn(N, Q_DIM, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    k = (torch.randn(N, KV_DIM, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    v = (torch.randn(N, KV_DIM, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    scale = D ** -0.5

    q3 = q.view(1, N, NUM_Q, D).contiguous()
    k3 = k.view(1, N, NUM_KV, D).contiguous()
    v3 = v.view(1, N, NUM_KV, D).contiguous()

    # Warmup
    for _ in range(20):
        _ = _ext.attention_causal(q, k, v, scale)
        _ = flash_attn_func(q3, k3, v3, softmax_scale=scale, causal=True)
    torch.cuda.synchronize()

    iters = 300

    def bench(fn):
        e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            e0[i].record()
            fn()
            e1[i].record()
        torch.cuda.synchronize()
        xs = sorted(e0[i].elapsed_time(e1[i]) * 1000.0 for i in range(iters))
        return xs[len(xs) // 2]

    p50_ours = bench(lambda: _ext.attention_causal(q, k, v, scale))
    p50_fa = bench(lambda: flash_attn_func(q3, k3, v3, softmax_scale=scale, causal=True))

    print()
    print(f"=== attention perf N={N} ===")
    print(f"  flash_attn      : {p50_fa:7.2f} µs")
    print(f"  ours inline     : {p50_ours:7.2f} µs")
    print(f"  speedup         : {p50_fa/p50_ours:.2f}×")


if __name__ == "__main__":
    for N in [32, 64, 96, 100, 128, 200]:
        test_attention_causal_numerics(N)
    test_attention_causal_perf()
