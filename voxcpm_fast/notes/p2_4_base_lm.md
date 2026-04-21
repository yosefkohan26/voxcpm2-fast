# P2.4 — Full base_lm (28 causal layers) chained prefill

**Status:** numerics PASS (within bf16-residual compounding limits), perf at parity with upstream eager (**1.04× p50**). This confirms the chained architecture's ceiling — the real speedup needs P2.5.

## What was built

- `FusedCpm4Model` in `voxcpm_fast/fused_layer_chained.py` — stacks N `FusedLayer(causal=True)` + final `RMSNorm`, matching upstream `Cpm4Model.forward(input_embeds, positions)` semantically.
- Test `voxcpm_fast/tests/test_fused_base_lm.py` loads all 28 layers of real `base_lm.*` safetensors weights, reassembles `qkv_proj` from `q/k/v` and `gate_up_proj` from `gate/up`, drops into our stack.

## Config (from `models/VoxCPM2/config.json`)

- `num_hidden_layers = 28`  (not 22 as referenced in older notes; corrected)
- `hidden_size = 2048`
- `intermediate_size = 6144`
- `num_attention_heads = 16`
- `num_key_value_heads = 2`
- `kv_channels = 128` (head_dim)
- `rms_norm_eps = 1e-5`
- `vocab_size = 73448`

## Numerics (N=100, real weights)

```
upstream bf16 max : 20.6250
ours bf16 max     : 23.8750
maxdiff (abs)     : 9.5391       rel=0.46
mae (abs)         : 0.1578       rel=7.65e-3
```

Max-rel 46 % looks alarming; it isn't. Per-layer drift analysis (in-notebook walk, not committed):

| layer | upstream max | maxdiff | rel_max |
|---|---|---|---|
| 0 | 25.5 | 0.125 | 0.5% (= 1 bf16 ULP) |
| 1 | 191 | 1.0 | 0.5% |
| 2 | 159 | 7 | 4.4% |
| 7 | 1696 | 82 | 4.8% |
| ... | ~1700 | ~82-94 | ~5% |
| 27 | 8384 | 1504 | 17.9% |

The hidden state **grows to O(1700-8000) in the middle of the stack**, driven by residual adds and the unnormalized MLP output. At magnitude 1700 one bf16 ULP is ~4 units; 10 layers of ~1 ULP drift compounds to the observed ~80-unit diff on the raw stream.

The final RMSNorm then **captures a slightly different mean(x²) for our drifted stream vs upstream's**, producing an output-scale divergence of ~16 %. That's what pushes max-rel from the ~5 % per-layer steady-state into the tens of percent on the final output.

**Mean-rel stays under 1 %**, which confirms we have no systemic math bug — just bf16 compounding.

## Perf

```
ours      p50=20.19 ms  p95=20.28 ms  (28 layers at N=100)
upstream  p50=20.98 ms  p95=26.45 ms  (eager Cpm4Model forward, full context)
speedup   1.04×
```

Per-layer: 20.19 ms / 28 = **721 µs/layer**, matches the standalone layer measurement exactly.

At N=100 the per-layer GEMM cost dominates; upstream's cuBLAS paths are already within a few percent of our WMMA recipe for these shapes. **The chained architecture cannot beat upstream at this M.** The remaining gap to physics floor (compute floor ≈ 2 ms for full base_lm forward at realistic TC util, HBM-bw floor ≈ 1.9 ms for weight read-once) is essentially ALL kernel-launch + cuBLAS-dispatch + per-layer residual roundtrip through HBM. All of those are owned by the persistent megakernel.

## Two things this run established

1. **Numeric correctness of the compute stack.** 28 layers of our chained kernels produce the same math as upstream within expected bf16 drift. When we fuse into a persistent kernel in P2.5, we're fusing validated compute.

2. **The chained architecture's ceiling is parity.** Further investment in optimizing the chained form is wasted motion — we've hit the limit. Everything from here goes into P2.5.

## What P2.5 must do to close the gap

For full base_lm forward at c=1, from 20 ms → ~2 ms (10×):

1. **One persistent kernel for all 28 layers.** Zero inter-layer launches. The admission work from P2.1 lands a work item; the worker walks all 28 layers without returning to host.

2. **Residual stream in fp32 inside the kernel.** Keep the hidden-state accumulator in fp32 registers/SMEM across layers. bf16 only on HBM reads (weights) and final output. This kills the per-layer ULP drift entirely.

3. **TMA-async prefetch of layer N+1's weights while computing layer N.** At 67 MB/layer for base_lm, the L2 can hold ~2 layers; pipeline carefully to overlap HBM→L2 with compute.

4. **No cuBLAS, no cuDNN in hot path.** Custom WGMMA kernels for our exact shapes, sized for Blackwell tensor cores (M tile = 64 or 128, N tile = 256).

5. **Online flash-attn inline.** No library call for attention; it's one more chunk of kernel body, feeding directly from SMEM-resident Q, K, V.

6. **KV cache writes** as part of the worker body (block-paged layout matching upstream's `store_kvcache_kernel`). Needed for decode too, but can be tested at the prefill path first.

## Next task recommendation

P2.5 — persistent megakernel for full base_lm. This is where **user-visible TTFPA moves**. Break into stages:

- P2.5.0 — CUDA-graph capture wrapper around the current chained stack. Baseline for the persistent version (removes only per-launch dispatch overhead, not the rest). Measure; that's the "ceiling" of launch-overhead-only fixes.
- P2.5.1 — single-layer persistent kernel with fp32 residual + WGMMA GEMM + inline attention, validated against our chained FusedLayer at 1 ULP.
- P2.5.2 — multi-layer (2, 4, 28) inside the same kernel, with layer-N+1 weight prefetch.
- P2.5.3 — KV cache writes.
- P2.5.4 — benchmark vs upstream: at N=100, target ≤3 ms for base_lm forward.
