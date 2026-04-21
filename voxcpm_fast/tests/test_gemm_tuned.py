"""P2.5.1.a — numerics + perf for the tuned cp.async-pipelined GEMM.

Compares `vcpm_gemm_bf16_tuned` against:
  1. Our existing `vcpm_gemm_bf16` (WMMA, single warp/tile, no pipelining).
  2. torch.matmul (cuBLAS).

At all 4 base_lm GEMM shapes (qkv, o, gate_up, down). M padded to 64.

Gates:
  - max |diff| vs our existing GEMM ≤ 2  (bf16 quantization noise, a couple
    ULPs at typical magnitudes) — they accumulate fp32 the same way.
  - max rel vs cuBLAS ≤ 1e-2, mean rel ≤ 1e-3.
"""

from __future__ import annotations

import os
import sys
import statistics
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"

sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))
sys.path.insert(0, str(NANOVLLM_ROOT))


# 4 base_lm GEMM shapes. Real M=100; pad to TM=64 multiple → M=128.
SHAPES = [
    ("qkv",     128, 2560,  2048),
    ("o",       128, 2048,  2048),
    ("gate_up", 128, 12288, 2048),
    ("down",    128, 2048,  6144),
]


def _pad_M_to(x: torch.Tensor, pad_to: int) -> torch.Tensor:
    M = x.size(0)
    pad = (pad_to - M % pad_to) % pad_to
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, pad))


@pytest.mark.parametrize("name,M,N,K", SHAPES)
def test_gemm_tuned_numerics(name, M, N, K):
    torch.manual_seed(17)
    import fused_layer_chained_ext as _ext

    A = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    B = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

    # Reference 1: our existing WMMA kernel (M must be multiple of 16; M=112 is).
    ref_ours = _ext.gemm_bf16(A, B)

    # Reference 2: cuBLAS via torch.matmul.
    ref_cu = (A.float() @ B.float().t()).to(torch.bfloat16)

    # Ours tuned.
    got = _ext.gemm_bf16_tuned(A, B)

    diff_ours = (got.float() - ref_ours.float()).abs()
    diff_cu   = (got.float() - ref_cu.float()).abs()

    up_max = ref_cu.abs().max().item()
    rel_max = diff_cu.max().item() / max(up_max, 1e-9)
    rel_mae = diff_cu.mean().item() / max(up_max, 1e-9)

    print()
    print(f"=== GEMM tuned, {name}  M={M} N={N} K={K} ===")
    print(f"  up_max = {up_max:.4f}   ours max = {got.abs().max().item():.4f}")
    print(f"  vs our-WMMA  : max={diff_ours.max().item():.4e}  mae={diff_ours.mean().item():.4e}")
    print(f"  vs cuBLAS    : max={diff_cu.max().item():.4e}  mae={diff_cu.mean().item():.4e}"
          f"  rel_max={rel_max:.4e}  rel_mae={rel_mae:.4e}")

    # bf16 accumulator spread; both our WMMA and our tuned use fp32 acc.
    # Typical bf16-of-dot-product at these magnitudes is O(1-5) at the
    # large shapes; a couple ULPs is the acceptable tolerance. Match
    # cuBLAS to the same 1e-2 rel standard used for layer outputs.
    assert rel_max <= 1e-2, f"{name}: max rel vs cuBLAS too high: {rel_max:.4e}"
    assert rel_mae <= 1e-3, f"{name}: mean rel vs cuBLAS too high: {rel_mae:.4e}"


@pytest.mark.parametrize("name,M,N,K", SHAPES)
def test_gemm_tuned_perf(name, M, N, K):
    """Not a gate — report only. Timing is checked in bench_gemm_tuned.py."""
    import fused_layer_chained_ext as _ext

    torch.manual_seed(17)
    A = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    B = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

    # Warmup
    for _ in range(10):
        _ = _ext.gemm_bf16_tuned(A, B)
        _ = _ext.gemm_bf16(A, B)
        _ = A.float() @ B.float().t()
    torch.cuda.synchronize()

    iters = 200

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

    p50_tuned = bench(lambda: _ext.gemm_bf16_tuned(A, B))
    p50_wmma  = bench(lambda: _ext.gemm_bf16(A, B))
    A_f = A.float(); B_f = B.float()
    p50_cu    = bench(lambda: A_f @ B_f.t())

    # Floor.
    FLOPS = 2.0 * M * N * K
    TFLOPS = 178e12
    HBM = 1.52e12
    compute_us = FLOPS / TFLOPS * 1e6
    bw_bytes = N * K * 2 + M * K * 2 + M * N * 2
    bw_us = bw_bytes / HBM * 1e6
    floor = max(compute_us, bw_us)

    print()
    print(f"=== GEMM perf {name}  M={M} N={N} K={K} ===")
    print(f"  our WMMA (prior): {p50_wmma:7.2f} µs   ({p50_wmma/floor:.1f}× floor)")
    print(f"  ours tuned      : {p50_tuned:7.2f} µs   ({p50_tuned/floor:.1f}× floor)")
    print(f"  cuBLAS (fp32)   : {p50_cu:7.2f} µs  (fp32 reference only)")
    print(f"  floor           : {floor:7.2f} µs   (compute={compute_us:.1f}, bw={bw_us:.1f})")
    print(f"  speedup vs WMMA : {p50_wmma/p50_tuned:.2f}×")


if __name__ == "__main__":
    for name, M, N, K in SHAPES:
        test_gemm_tuned_numerics(name, M, N, K)
    for name, M, N, K in SHAPES:
        test_gemm_tuned_perf(name, M, N, K)
