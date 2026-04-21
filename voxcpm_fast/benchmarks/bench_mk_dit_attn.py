"""P2.5.2 Step 2a — standalone perf bench for the non-causal attention kernel.

Measures:
  1. flash_attn_func(..., causal=False)         [reference]
  2. ext.attention_noncausal_batched(...)       [ours]
at the DiT inference shape (B=2, S=11).
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))

import mk_dit_prefill_ext as ext

from flash_attn import flash_attn_func


NUM_Q = 16
NUM_KV = 2
HEAD_DIM = 128


def _time(fn, warmup=30, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))


def main(B=2, S=11):
    torch.manual_seed(17)
    q = (torch.randn(B, S, NUM_Q, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    k = (torch.randn(B, S, NUM_KV, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    v = (torch.randn(B, S, NUM_KV, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    scale = 1.0 / math.sqrt(HEAD_DIM)

    def fa(): return flash_attn_func(q, k, v, causal=False, softmax_scale=scale)
    def ours(): return ext.attention_noncausal_batched(q, k, v, scale)

    print(f"B={B} S={S} Hq={NUM_Q} Hkv={NUM_KV} D={HEAD_DIM}")
    print(f"gpu={torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    fa_ms = _time(fa)
    ou_ms = _time(ours)

    def pct(xs, p): return xs[int(round(p / 100.0 * (len(xs) - 1)))]
    print(f"{'phase':30s}  {'p50':>7s}  {'p95':>7s}  {'p99':>7s}  (µs)")
    print(f"{'-' * 30}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
    for name, xs in [("flash_attn causal=False", fa_ms), ("ours noncausal_batched", ou_ms)]:
        print(f"{name:30s}  {pct(xs,50)*1000:7.1f}  {pct(xs,95)*1000:7.1f}  {pct(xs,99)*1000:7.1f}")

    print(f"ours vs flash_attn p50: {pct(fa_ms,50)/pct(ou_ms,50):.2f}x")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-B", type=int, default=2)
    ap.add_argument("-S", type=int, default=11)
    args = ap.parse_args()
    main(B=args.B, S=args.S)
