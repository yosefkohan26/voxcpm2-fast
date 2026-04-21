# Physics Floor — concurrency = 1

This is the scoreboard. Every subsequent change must be measured against these floors, not against the upstream baseline.

## Premise

- Target: **c=1 TTFPA ≈ 3 ms**, sustained RTF ≪ 1 afterwards.
- Design rule: at c=1, **all 170 SMs serve the one active stream**. No batching, no sharing.
- c > 1 is a degradation mode, not a design target.

## Hardware ceilings (RTX 5090)

| resource | peak | sustained (85 %) |
|---|---|---|
| bf16 tensor-core throughput | 209 TFLOPS | **178 TFLOPS** |
| HBM3 bandwidth | 1.792 TB/s | **1.52 TB/s** |
| L2 cache | 128 MB | - |
| SM count | 170 | - |
| minimum kernel launch (CUDA graph replay) | ~3 µs | - |

Anything that requires > 178 TFLOPS of compute or > 1.52 TB/s of HBM *per unit time* is infeasible; that's the physics boundary.

## Model constants (VoxCPM2, from `topology.md`)

- Total params: **2.29 B bf16 = 4.58 GB** on-device weights.
- `base_lm`: 22 layers, hidden 2048, intermediate 4096, GQA 16/2, head_dim 128. QKV 2048→2560, O 2048→2048, gate_up 2048→8192, down 4096→2048.
- `residual_lm`: 6 layers, same per-layer shape as base_lm.
- `feat_encoder`: 4 layers, hidden 1024, intermediate 4096, GQA 16/2, head_dim 128 (heads × head_dim = 2048 ≠ hidden, decoupled).
- `feat_decoder`: `VoxCPM2LocDiT` — 4 layers, hidden 1024, same per-layer as feat_encoder. Runs 10 Euler timesteps × CFG ×2, non-causal.
- `AudioVAE.decoder`: causal conv stack, ~20 ops. Generates one patch = 4 latents → `patch_size × chunk_size = 4 × 960 = 3840` samples at 48 kHz = 80 ms of audio.
- Prompt length typical: **~100 tokens** (rough; user will validate).

## Per-op compute cost (FLOPs per token)

### Transformer layer with hidden H, intermediate I, num_heads H_n, num_kv H_kv, head_dim D

- QKV proj: 2 × H × (H_n + 2×H_kv) × D
- O proj: 2 × H_n × D × H
- gate_up: 2 × H × 2I
- down: 2 × I × H
- RMSNorm, RoPE, SiLU, residuals, KV write: bandwidth, not compute (< 1 % of layer FLOPs)

**base_lm per layer**: 10.5 + 8.4 + 33.6 + 16.8 = **69.3 MFLOPs/token**
**residual_lm per layer**: same as base_lm = **69.3 MFLOPs/token**
**feat_encoder per layer**: 5.24 + 4.2 + 16.8 + 8.4 = **34.6 MFLOPs/token**
**DiT per layer**: same shape as feat_encoder = **34.6 MFLOPs/token**

## Compute-bound floors — c=1 prefill of 100 prompt tokens

Assume perfect tensor-core utilization (178 TFLOPS). For small M=100 we will *not* hit this with cuBLAS — our custom kernels need to use WGMMA + cooperative M-dim splitting across SMs to close the tile-utilization gap. The floor below is the "best achievable with custom tiling" number.

| component | layers | FLOPs total | compute-floor ms @ 178 TFLOPS |
|---|---|---|---|
| feat_encoder | 4 | 100 × 4 × 34.6 M = 13.8 G | **0.08** |
| base_lm | 22 | 100 × 22 × 69.3 M = 152 G | **0.86** |
| residual_lm | 6 | 100 × 6 × 69.3 M = 41.5 G | **0.23** |
| DiT prefill first step (1 chunk-0) | 4 × 10 Euler | 22 tok (CFG×2) × 40 × 34.6 M = 30.5 G | **0.17** |
| AudioVAE decode one patch | conv stack | ~0.5 GFLOPs | **0.003** |
| **prefill compute-floor subtotal** | | | **~1.35 ms** |

With realistic M=100 tensor-core utilization of 40 % (WGMMA tile M=64, bf16, well-tuned), the compute-bound floor inflates to:

| component | realistic compute floor |
|---|---|
| feat_encoder | 0.2 ms |
| base_lm | 2.2 ms |
| residual_lm | 0.6 ms |
| DiT | 0.4 ms |
| VAE | 0.01 ms |
| **realistic compute-bound subtotal** | **~3.4 ms** |

## Bandwidth-bound floors — c=1 prefill

Weights must be read at least once. Activations are small at M=100 (~500 KB per full hidden × batch, fits in SMEM/L2 within a layer).

| component | weights | bw-floor ms @ 1.52 TB/s |
|---|---|---|
| feat_encoder | 46 M params × 2 B = 92 MB | **0.06** |
| base_lm | 1.47 G params × 2 B = 2.94 GB | **1.93** |
| residual_lm | 330 M × 2 = 660 MB | **0.43** |
| DiT | 46 M × 2 × 10 timesteps (weights re-read each step with non-persistent design; **1× in persistent design**) | **0.06** |
| AudioVAE | ~100 MB | **0.07** |
| **bw-floor subtotal (weights cold, one pass)** | | **~2.5 ms** |

Key insight: **HBM bandwidth dominates**. `base_lm` alone is 1.93 ms just to read its weights once. That is the single hardest-to-beat floor.

**L2 amortisation:** base_lm's 22 layers × 67 MB ≈ 1.47 GB — we can NOT keep base_lm in L2 (128 MB). However, *one layer at a time* fits (67 MB). With correct prefetching (TMA async loads of layer N+1 weights while compute runs on layer N), compute and HBM read overlap and the floor is `max(compute, bw)` per layer. For base_lm: per-layer compute ~0.04 ms, per-layer bw 0.087 ms ⇒ **bw-bound, floor = 1.93 ms** for full base_lm forward at c=1.

## c=1 TTFPA physics floor — combined

Realistic case with HBM-bw dominating base_lm and realistic TC utilization elsewhere:

| phase | floor (ms) |
|---|---|
| feat_encoder (one pass over 100 tokens) | 0.2 |
| base_lm prefill (100 tokens, HBM-bw-bound) | 1.9 |
| residual_lm prefill | 0.6 |
| DiT first chunk-0 | 0.4 |
| VAE decode one patch | 0.05 |
| persistent-kernel queue pop + output ring | 0.1 |
| **total c=1 TTFPA floor** | **≈ 3.25 ms** |

Target: **≤ 3 ms**. Stretch: close the gap between "realistic TC utilization" and "perfect TC utilization" = another ~0.4 ms. Anything below 2.8 ms will require FP8/FP4 or algorithmic change (fewer DiT steps), both out of scope for now.

## c=1 steady-state decode step floor

Once streaming, each decode step processes 1 new token per stream through base_lm + residual_lm + feat_encoder (on current feat tokens, typically a few) + full DiT (10 Euler × CFG×2) + VAE one patch.

| component | compute FLOPs | compute-floor ms | weights re-read? | bw-floor ms |
|---|---|---|---|---|
| base_lm | 1 × 22 × 69.3 M = 1.52 G | 0.009 | yes | 1.93 |
| residual_lm | 1 × 6 × 69.3 M = 415 M | 0.003 | yes | 0.43 |
| feat_encoder | ~5 tok × 4 × 34.6 M = 0.69 G | 0.004 | yes | 0.06 |
| DiT | 10 × 4 × 22 tok × 34.6 M = 30.5 G | 0.17 | yes | 0.06 |
| VAE one patch | ~0.5 G | 0.003 | yes | 0.07 |
| **per-step** | | ~0.2 ms compute | | **≈ 2.5 ms HBM-bw-bound** |

Per-patch audio duration = 80 ms. **RTF floor = 2.5 / 80 = 0.031 at c=1.** 32× headroom for jitter. Playable immediately after chunk-0 (first-chunk to second-chunk gap 2.5 ms ≪ 80 ms audio duration of chunk-0).

## Current vs floor

| metric | upstream c=1 (measured) | physics floor | gap |
|---|---|---|---|
| TTFPA | **187.3 ms** | **~3 ms** | **60×** |
| prefill wall time | 222.7 ms | ~2.3 ms | 97× |
| decode step | 8.5 ms (CUDA graph) | ~2.5 ms | 3.4× |
| decode step (eager) | 180 ms | ~2.5 ms | 72× |

Decode is already 3.4× from floor with cuBLAS + CUDA graphs. Prefill is 97× from floor because upstream doesn't graph-capture prefill and runs eager matmuls per token. Our megakernel has to close the 60× TTFPA gap, and **the plan to do it is the one in `PROJECT_PLAN.md §P2.0–P2.5`**.

## What the persistent megakernel must win

Each of these is one of the 60× T_first gap's constituent multipliers. The scoreboard below is what we check after every kernel lands.

| source of gap | current overhead | megakernel target | multiplier won |
|---|---|---|---|
| Kernel launches (100s per prefill at eager) | ~50 ms on c=1 prefill | ~0.1 ms (1 persistent launch) | ~500× on launches |
| Hidden-state HBM reads between ops | ~1.5 GB traffic per prefill | 0 (live in SMEM within layer) | HBM freed for weights |
| cuBLAS kernel-selection / unused precomputation | ~10 ms | 0 | 10× on setup |
| Per-op tensor allocation (torch.empty) | ~5 ms | 0 (pre-allocated ring) | - |
| Prefill runs *eager* upstream (no graph) | ~100 ms of dispatch | 0 | ~100× on prefill |
| Python scheduler + IPC + mp.Queue per step | ~50 ms at c=64, ~3 ms at c=1 | ~0.01 ms (GPU-resident queue) | 300× at c=1 |
| D2H sync on stop_flag, latents, waveform | ~5 ms per step | ~0.1 ms (pinned ring, no sync) | 50× |

Total multiplicative budget needed: ~60× on prefill TTFPA.

## Rules of the road — applies to every future kernel

1. **Never re-read a hidden state from HBM within a layer.** QKV → attn → O → MLP must share SMEM.
2. **Weights loaded exactly once per forward.** TMA async load of layer N+1 during layer N compute.
3. **No cuBLAS, no cuDNN** in the persistent megakernel hot path. Custom WGMMA-based GEMM for the exact shapes we run.
4. **No `.cpu()`, no `torch.tensor(...).cuda()`, no `.item()` in the step loop.** Pinned host rings only, memcpy_async.
5. **Attention is online softmax inside the kernel** — not a flash_attn function call.
6. **Stop flag, latents, waveform output go to GPU-resident rings.** Host polls pinned scratch; no sync between steps.
7. **Every kernel is numerically validated** to 1e-2 bf16 / 1e-5 fp32 vs upstream before the next stage starts.

## Open instrumentation debt

- `notes/topology.md` recorded **decode-step eager timings**. We do not yet have **prefill-step per-section timings** at c=1. Need this to validate my compute-vs-bw attribution above; assumption is that cuBLAS+launch overhead accounts for the gap between 2.3 ms floor and 222 ms measured. One quick pass with section-timed prefill (similar to `explore_model.py` but for prefill) will confirm or reshape the P2.4/P2.5 kernel priorities.
- No `ncu` kernel metrics yet. We don't know the *actual* TC utilization of cuBLAS at M=100 for our shapes. Before writing `mk_*_prefill`, one `ncu --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis` pass on the upstream prefill is worth the 10 min it costs.

These two items become small tasks after P2.1 lands (the persistent-kernel PoC).
