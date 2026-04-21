"""P2.5.2 Step 2b.2 — perf bench for the 9-phase cooperative megakernel vs
chained FusedLayer at DiT shape.

Measures:
  1. Chained FusedLayer.forward (eager)            — 9 kernel launches
  2. Chained FusedLayer.forward (CUDA graph)        — 9 replayed
  3. Megakernel step2b2_full_layer (eager)          — 1 cooperative launch
  4. Megakernel step2b2_full_layer (CUDA graph)     — 1 replayed
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

import mk_dit_prefill_ext as mk_ext

from fused_layer_chained import FusedLayer


H = 1024
INTERMEDIATE = 4096
QKV_DIM = 2560
Q_DIM = 2048
NUM_Q = 16
NUM_KV = 2
HEAD_DIM = 128
RMS_EPS = 1e-6
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _time_iters(fn, warmup=30, iters=500):
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


def _pad_M_to_64(x):
    M = x.size(0); pad = (64 - M % 64) % 64
    return x if pad == 0 else torch.nn.functional.pad(x, (0, 0, 0, pad))


def main(B=2, S=11):
    torch.manual_seed(17)
    N_real = B * S
    g = torch.Generator(device="cuda").manual_seed(17)
    hs_real = (torch.randn(N_real, H, device="cuda", dtype=torch.bfloat16,
                           generator=g) * 0.1).contiguous()
    hs = _pad_M_to_64(hs_real).contiguous()
    positions_real = torch.arange(N_real, device="cuda", dtype=torch.int32) % 1024
    positions = torch.nn.functional.pad(positions_real, (0, hs.size(0) - N_real), value=0)

    g2 = torch.Generator(device="cuda").manual_seed(19)
    w_in_ln = (torch.ones(H, device="cuda", dtype=torch.bfloat16)
               + torch.randn(H, device="cuda", dtype=torch.bfloat16, generator=g2) * 0.02).contiguous()
    w_qkv = (torch.randn(QKV_DIM, H, device="cuda", dtype=torch.bfloat16, generator=g2) * 0.02).contiguous()
    w_o = (torch.randn(H, Q_DIM, device="cuda", dtype=torch.bfloat16, generator=g2) * 0.02).contiguous()
    w_post_ln = (torch.ones(H, device="cuda", dtype=torch.bfloat16)
                 + torch.randn(H, device="cuda", dtype=torch.bfloat16, generator=g2) * 0.02).contiguous()
    w_gu = (torch.randn(2 * INTERMEDIATE, H, device="cuda", dtype=torch.bfloat16, generator=g2) * 0.02).contiguous()
    w_dn = (torch.randn(H, INTERMEDIATE, device="cuda", dtype=torch.bfloat16, generator=g2) * 0.02).contiguous()
    weights = {
        "input_layernorm.weight":          w_in_ln,
        "self_attn.qkv_proj.weight":       w_qkv,
        "self_attn.o_proj.weight":         w_o,
        "post_attention_layernorm.weight": w_post_ln,
        "mlp.gate_up_proj.weight":         w_gu,
        "mlp.down_proj.weight":            w_dn,
    }

    g3 = torch.Generator(device="cuda").manual_seed(7)
    cos = (torch.randn(2048, HEAD_DIM, device="cuda", dtype=torch.float32, generator=g3) * 0.5 + 0.5).contiguous()
    sin = (torch.randn(2048, HEAD_DIM, device="cuda", dtype=torch.float32, generator=g3) * 0.5).contiguous()

    chained = FusedLayer(
        weights=weights, rope_cos_cache=cos, rope_sin_cache=sin,
        hidden=H, intermediate=INTERMEDIATE,
        causal=False, use_rope=True, rms_eps=RMS_EPS)

    def chained_step():
        with torch.inference_mode():
            return chained.forward(hs_real, positions_real, batch_size=B)

    def mk_step():
        with torch.inference_mode():
            return mk_ext.step2b2_full_layer(
                hs, w_in_ln, w_qkv, w_o, w_post_ln, w_gu, w_dn,
                cos, sin, positions, B, S,
                RMS_EPS, ATTN_SCALE)

    print(f"B={B} S={S} N_real={N_real} M_padded={hs.size(0)} H={H}")
    print(f"gpu={torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    print()

    # Eager timings.
    ch_ms = _time_iters(chained_step)
    mk_ms = _time_iters(mk_step)

    # Graph capture for chained: PyTorch CUDAGraph.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(5):
            _ = chained.forward(hs_real, positions_real, batch_size=B)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    chained_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(chained_graph), torch.inference_mode():
        _ = chained.forward(hs_real, positions_real, batch_size=B)
    ch_g_ms = _time_iters(lambda: chained_graph.replay())

    # Graph capture for megakernel. Cooperative launches CANNOT be
    # captured by torch.cuda.CUDAGraph currently (cudaLaunchCooperativeKernel
    # raises cudaErrorStreamCaptureUnsupported during capture). We skip the
    # graphed-megakernel measurement and flag this as a Step 3 follow-up
    # (persistent kernel across layers/Euler-iters eliminates the launch
    # count anyway so graph-capture of the single-layer wrapper isn't the
    # long-term path).
    mk_g_ms = None
    try:
        warmup_stream2 = torch.cuda.Stream()
        warmup_stream2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream2), torch.inference_mode():
            for _ in range(5):
                _ = mk_ext.step2b2_full_layer(
                    hs, w_in_ln, w_qkv, w_o, w_post_ln, w_gu, w_dn,
                    cos, sin, positions, B, S, RMS_EPS, ATTN_SCALE)
        torch.cuda.current_stream().wait_stream(warmup_stream2)
        torch.cuda.synchronize()
        mk_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(mk_graph), torch.inference_mode():
            _ = mk_ext.step2b2_full_layer(
                hs, w_in_ln, w_qkv, w_o, w_post_ln, w_gu, w_dn,
                cos, sin, positions, B, S, RMS_EPS, ATTN_SCALE)
        mk_g_ms = _time_iters(lambda: mk_graph.replay())
    except Exception as e:
        mk_g_ms = None
        mk_graph_err = str(e).split("\n")[0][:120]

    def pct(xs, p): return xs[int(round(p / 100.0 * (len(xs) - 1)))]

    print(f"{'phase':30s}  {'p50':>7s}  {'p95':>7s}  {'p99':>7s}  (µs)")
    print(f"{'-' * 30}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
    for name, xs in [("chained eager",  ch_ms),
                     ("chained graphed", ch_g_ms),
                     ("megakernel eager", mk_ms),
                     *([("megakernel graphed", mk_g_ms)] if mk_g_ms else [])]:
        print(f"{name:30s}  {pct(xs,50)*1000:7.1f}  {pct(xs,95)*1000:7.1f}  {pct(xs,99)*1000:7.1f}")

    if mk_g_ms is None:
        print()
        print(f"[info] megakernel graph-capture skipped: cooperative launches not "
              f"supported inside torch.cuda.CUDAGraph")
        try: print(f"       ({mk_graph_err})")
        except NameError: pass

    print()
    ch50 = pct(ch_ms, 50)
    ch_g50 = pct(ch_g_ms, 50)
    mk50 = pct(mk_ms, 50)
    print(f"Speedup per-layer:")
    print(f"  megakernel eager   vs chained eager    : {ch50/mk50:.2f}x  ({(ch50-mk50)*1000:+.1f} µs)")
    print(f"  megakernel eager   vs chained graphed  : {ch_g50/mk50:.2f}x  ({(ch_g50-mk50)*1000:+.1f} µs)")

    # Per full DiT prefill (12 layers × 9 Euler iters):
    calls_per_prefill = 12 * 9
    print()
    print(f"Extrapolated to full DiT prefill ({calls_per_prefill} layer calls):")
    print(f"  chained eager       : {ch50 * calls_per_prefill:.2f} ms")
    print(f"  chained graphed     : {ch_g50 * calls_per_prefill:.2f} ms")
    print(f"  megakernel eager    : {mk50 * calls_per_prefill:.2f} ms")
    if mk_g_ms is not None:
        mk_g50 = pct(mk_g_ms, 50)
        print(f"  megakernel graphed  : {mk_g50 * calls_per_prefill:.2f} ms")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-B", type=int, default=2)
    ap.add_argument("-S", type=int, default=11)
    args = ap.parse_args()
    main(B=args.B, S=args.S)
