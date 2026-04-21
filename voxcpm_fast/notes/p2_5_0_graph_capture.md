# P2.5.0 — CUDA graph capture of chained base_lm

**Status:** PASS with a critical finding. Graph replay is numerically identical
to eager (0.0 / 0.0 max/mae), but **launch-overhead savings are only 0.58 ms
(2.9 %)** — practically nothing. This is decisive: the 10× gap from our
current chained implementation to the physics floor is **not** launch-overhead
bound. It is architecturally bound. Everything from here is P2.5.1+ work.

## What was built

- `voxcpm_fast/benchmarks/bench_base_lm_graph.py` — wraps the existing
  `FusedCpm4Model.forward` (28 causal layers + final RMSNorm) in a
  `torch.cuda.CUDAGraph`. Uses the standard side-stream warmup recipe, then
  captures and replays. Numerics spot-check vs eager; bench against upstream
  eager for the familiar reference.

## Results (N=100, 28 layers, hidden=2048, 200 iters)

```
phase                             p50      p95      p99     mean  (ms)
----------------------------  -------  -------  -------  -------
ours eager (chained)           19.955   20.016   22.007   20.092
ours graphed                   19.377   19.393   19.403   19.377
upstream eager                 25.875   45.247   56.943   27.801

ours eager    vs upstream     : 1.30×
ours graphed  vs upstream     : 1.34×
launch-overhead saved by graph: 0.578 ms   (2.9 %)

physics floor (HBM-bw)        : 1.93 ms
ours graphed gap to floor     : 10.04×
```

Graph replay numerics vs eager: **max=0.0, mae=0.0**. Bit-exact.

The graph tightens jitter dramatically (p99 − p50: 2.05 ms eager → 0.03 ms
graphed) but the *absolute* savings is the p50 delta, which is tiny.

## Interpretation

Launch overhead is **~0.58 ms total** across ~168 kernel launches (6 per layer
× 28 layers) ≈ **3.4 µs per launch**. That matches the CUDA-graph-replay
minimum-overhead floor measured on modern GPUs. There is no more launch-overhead
to remove.

The remaining 19.38 − 1.93 = **17.45 ms** is real work, not dispatch. Where
it goes:

1. **HBM residual-stream roundtrip between layers.** Each layer writes its
   output to HBM (~0.4 MB activations) and the next layer reads it back. With
   the chained architecture there is no way around this because each kernel
   is an independent launch.
2. **Weight re-reads without pipelining.** Each layer's 67 MB of weights are
   demand-loaded from HBM at the start of its first GEMM, not overlapped
   with the prior layer's compute. L2 is only 128 MB, so we can hold ~2
   layers but not amortize across 28.
3. **Per-GEMM cuBLAS overhead at M=100.** cuBLAS selects a kernel per call;
   at M=100 (padded to 112) the tile utilization is ~40 % of peak TC. 4
   GEMMs × 28 layers = 112 cuBLAS (or our WMMA) calls, each suboptimal.
4. **flash_attn kernel fixed cost.** At N=100, short-sequence attention is
   dominated by launch + setup costs that don't amortize.
5. **Activation alloc/copy churn.** `.contiguous()` on QKV slices, `hs.clone()`
   for residual, `F.pad` for M-padding — each allocates, each is a separate
   kernel or async memcpy.

None of 1-5 is removable by graph capture. All of them are removable by a
persistent megakernel that:
- keeps the residual stream in fp32 registers / SMEM across all 28 layers,
- loads each weight tile once (TMA async-prefetch of layer N+1 during layer
  N compute),
- replaces cuBLAS with custom WGMMA sized for our exact shapes,
- inlines attention with online softmax (no flash_attn library hop),
- never materializes intermediate activations in HBM.

## Bonus: eager jitter disappears

The graph replay's **0.03 ms p99 − p50 window** is ~70× tighter than eager.
For production this matters — graph-captured inference has deterministic
timing, which helps TTFPA SLO enforcement at the edges. But it costs the
same wall time at p50. Useful secondary property, not the headline.

## Upstream p95/p99 weirdness

Upstream eager shows p50=25.9 ms but p95=45.2 ms and p99=57.0 ms. That's
severe jitter — probably cuBLAS reloading its kernel cache on some iterations
(cuBLAS picks a new algo when the workload fingerprint changes slightly).
Our graphed implementation does not have this issue. Not a new result; just
confirms our approach is correct for production stability.

## Decision for P2.5.1+

Start the persistent megakernel now. Every remaining millisecond is architectural
— not dispatch — and the persistent kernel is the only path to close it.

P2.5.0 is a **ceiling measurement**, not a speedup. It tells us the starting
point for P2.5.1 is 19.38 ms graphed (tight, reproducible), gap to floor
10×. The persistent megakernel must close that 10× by fusing compute +
pipelining weight loads + holding state in SMEM.

## Next step (P2.5.1)

Single-layer persistent kernel:
1. One `extern "C" __global__` that owns compute for one decoder layer.
2. Residual stream accumulator in fp32 registers.
3. WGMMA-based GEMM kernels (drop cuBLAS and our WMMA chained form both).
4. Inline attention with online softmax — no flash_attn call.
5. KV-cache writes in the same kernel body (prefill path, slot_mapping
   direct-indexed).
6. Validation: numerics within 1 bf16 ULP of `FusedLayer(causal=True)` at N=100.
7. Perf: target ≤ 500 µs per layer (vs current 693 µs graphed per layer).

Then P2.5.2 scales that to all 28 layers inside one kernel, with TMA-async
weight prefetch across layer boundaries — that's where the 10× lands.
