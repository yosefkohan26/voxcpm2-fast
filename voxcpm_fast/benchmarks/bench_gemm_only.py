"""Micro-benchmark: how much of the 5.30 ms graphed base_lm is JUST the 4 GEMMs?

If we replace the 28-layer forward with "just 4 GEMMs per layer, no RMSNorm, no
rope, no flash_attn, no silu, no residual", and capture a CUDA graph of that,
we get a hard lower bound on what the chained form can achieve.
"""

from __future__ import annotations

import os
import sys
import statistics
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))

import fused_layer_chained_ext as _ext


def main(N=100, num_layers=28, iters=200, warmup=20):
    torch.manual_seed(1)
    H = 2048
    I = 6144
    QKV_DIM = 2560

    M = (N + 63) // 64 * 64

    hs = (torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    W_qkv = (torch.randn(QKV_DIM, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    W_o   = (torch.randn(H, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    W_gu  = (torch.randn(2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    W_dn  = (torch.randn(H, I, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

    # Share weights across layers to match HBM-bandwidth profile of distinct layers.
    # (Actually we WANT distinct weights to exercise L2 misses as per real forward.)
    W_qkvs = [W_qkv.clone() for _ in range(num_layers)]
    W_os   = [W_o.clone() for _ in range(num_layers)]
    W_gus  = [W_gu.clone() for _ in range(num_layers)]
    W_dns  = [W_dn.clone() for _ in range(num_layers)]

    attn_out = (torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()

    # attn_slice has N=H, K=H but we use a full-M buffer.
    mid = (torch.randn(M, I, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    residual = hs.clone()

    def forward_gemms_only(x):
        for i in range(num_layers):
            qkv = _ext.gemm_bf16_tuned(x, W_qkvs[i])             # [M, QKV_DIM]
            # use a slice of qkv as attention_out simulator (H dim match)
            # For physics, we want 4 GEMM reads of HW size H, 2I, I, H
            o = _ext.gemm_bf16_tuned_residual(attn_out, W_os[i], residual)
            gu = _ext.gemm_bf16_tuned(o, W_gus[i])                # [M, 2I]
            # Manual silu on gu to produce mid-shaped intermediate
            x = _ext.gemm_bf16_tuned_residual(mid, W_dns[i], o)
        return x

    # warmup
    for _ in range(warmup):
        _ = forward_gemms_only(hs)
    torch.cuda.synchronize()

    # eager
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record()
        _ = forward_gemms_only(hs)
        e1[i].record()
    torch.cuda.synchronize()
    eager = sorted(e0[i].elapsed_time(e1[i]) for i in range(iters))
    eager_p50 = eager[len(eager) // 2]

    # Graph
    out = torch.empty_like(hs)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = forward_gemms_only(hs)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    with torch.cuda.graph(g):
        out = forward_gemms_only(hs)

    # warmup graph
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record()
        g.replay()
        e1[i].record()
    torch.cuda.synchronize()
    graphed = sorted(e0[i].elapsed_time(e1[i]) for i in range(iters))
    graphed_p50 = graphed[len(graphed) // 2]

    # Phys floor: 4 weights per layer × num_layers bytes / 1.52 TB/s
    bytes_w = (QKV_DIM * H + H * H + 2*I * H + H * I) * 2 * num_layers
    bytes_act_per_layer = (M * H * 2 * 4)  # input, attn_out, residual, mid (rough)
    floor_ms = (bytes_w + bytes_act_per_layer * num_layers) / 1.52e12 * 1e3

    print(f"M={M} H={H} I={I} QKV={QKV_DIM} num_layers={num_layers}")
    print(f"eager p50:   {eager_p50:.3f} ms")
    print(f"graphed p50: {graphed_p50:.3f} ms")
    print(f"HBM floor (weights + activations): {floor_ms:.3f} ms")
    print(f"gap to HBM floor (graphed): {graphed_p50/floor_ms:.2f}x")


if __name__ == "__main__":
    main()
