"""Sweep TN ∈ {64, 128} per-shape via gemm_bf16_tuned_tn to find the optimal
dispatch threshold. The current hard-coded dispatch picks TN=128 for N>=8192
and TN=64 below.
"""

from __future__ import annotations

import sys
import statistics
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SHAPES = [
    ("qkv",     128, 2560,  2048),
    ("o",       128, 2048,  2048),
    ("gate_up", 128, 12288, 2048),
    ("down",    128, 2048,  6144),
]


def main():
    import fused_layer_chained_ext as _ext

    def bench(fn, iters=300):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            e0[i].record()
            fn()
            e1[i].record()
        torch.cuda.synchronize()
        xs = sorted(e0[i].elapsed_time(e1[i]) * 1000.0 for i in range(iters))
        return xs[len(xs) // 2]

    print(f"{'shape':10s}  {'M':>4s} {'N':>6s} {'K':>5s}  {'TN=32 µs':>10s}  {'TN=64 µs':>10s}  {'TN=128 µs':>10s}  {'winner':>7s}")
    print("-" * 90)
    for name, M, N, K in SHAPES:
        torch.manual_seed(9)
        A = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
        B = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

        tn32 = tn64 = tn128 = None
        if N % 32 == 0:
            tn32 = bench(lambda: _ext.gemm_bf16_tuned_tn(A, B, 32))
        if N % 64 == 0:
            tn64 = bench(lambda: _ext.gemm_bf16_tuned_tn(A, B, 64))
        if N % 128 == 0:
            tn128 = bench(lambda: _ext.gemm_bf16_tuned_tn(A, B, 128))

        results = {}
        if tn32 is not None: results["TN=32"] = tn32
        if tn64 is not None: results["TN=64"] = tn64
        if tn128 is not None: results["TN=128"] = tn128
        winner = min(results, key=results.get)

        s = lambda v: f"{v:10.2f}" if v is not None else "n/a".rjust(10)
        print(f"{name:10s}  {M:>4d} {N:>6d} {K:>5d}  {s(tn32)}  {s(tn64)}  {s(tn128)}  {winner:>7s}")


if __name__ == "__main__":
    main()
