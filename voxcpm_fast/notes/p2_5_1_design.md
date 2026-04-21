# P2.5.1 — Single-layer physics-floor design

**Status:** design + attribution complete. Implementation in progress.

This document specifies the target architecture for one causal decoder layer
at physics-floor quality. It is anchored in the per-stage measurement from
`benchmarks/bench_layer_breakdown.py` — no guessing, no estimates. Every
sub-kernel's speedup target is tied to a measured µs.

## 1. The measurement (base_lm.layers.0, N=100, one layer)

```
stage                                 p50 µs    floor µs    gap     limit
--------------------------------  ----------  ----------  ------  --------
rmsnorm (in)                           22.94        0.54   42.4x        bw
qkv_gemm  (2560×2048)                  99.81        7.50   13.3x        bw
rope (Q+K inplace)                     55.55        0.61   91.6x        bw
flash_attn (causal)                   198.30       10.00   19.8x        bw
o_gemm    (2048×2048)                  99.78        6.06   16.5x        bw
residual_add (post-attn)               55.07        0.81   68.1x        bw
rmsnorm (post)                         23.39        0.54   43.2x        bw
gate_up   (12288×2048)                228.99       35.00    6.5x        bw
silu_mul                               21.60        2.43    8.9x        bw
down_gemm (2048×6144)                 117.95       17.63    6.7x        bw
residual_add (post-mlp)                31.78        0.81   39.3x        bw
--------------------------------  ----------  ----------  ------  --------
sum of stages                         955.17       81.93   11.7x
full FusedLayer.forward              1028.38
  (overhead vs stage sum)              73.22                      launch/glue
```

Floors assume 178 TFLOPS bf16 tensor-core sustained and 1.52 TB/s HBM.
Per-stage floor = max(compute-bound, HBM-bw-bound). flash_attn floor is
held to 10 µs minimum to reflect launch-setup dominance at small N.

## 2. Attack surface

Sorted by absolute µs of opportunity per layer:

| target | current µs | realistic µs | saved/layer | × 28 layers |
|---|---|---|---|---|
| GEMMs (4 total) | 546 | ~80 | **~466 µs** | **13.0 ms** |
| flash_attn → inline attn | 198 | ~40 | **~158 µs** | **4.4 ms** |
| tiny ops → epilogue-fused into GEMMs | 210 | ~10 | **~200 µs** | **5.6 ms** |
| launch/glue | 73 | ~5 | **~68 µs** | **1.9 ms** |
| **total** | **1027** | **~135** | **~892 µs** | **~25 ms** |

If realized: 28-layer base_lm drops from 20 ms → **~4 ms** — within striking
distance of the 1.93 ms HBM-bw physics floor, and under the 3 ms c=1 TTFPA
budget with headroom for the other model components (feat_encoder, residual_lm,
DiT, VAE).

## 3. Architecture

**Three kernels per layer**, chained on one stream. Each kernel fuses
multiple upstream ops so that intermediate activations never materialize
to HBM within the kernel; only the layer-boundary residual stream and the
KV cache live in HBM across kernel boundaries. This split avoids
cooperative-grid launch (the P2.2 debugging trap) while eliminating the
tiny-op HBM roundtrips that currently cost 200 µs/layer.

### 3.1 `vcpm_pre_attn_causal<H=2048>` — fused rmsnorm + QKV gemm + RoPE + KV write

- **Inputs:** x[N, H] bf16 (residual stream in), w_ln[H], w_qkv[QKV_DIM, H],
  cos_cache, sin_cache, positions[N], k_cache, v_cache, slot_mapping[N]
- **Outputs:** q[N, Q_DIM] bf16 in HBM; K and V rows written directly to
  k_cache, v_cache at slot_mapping[i] (V untouched by RoPE).
- **Fusion:** normalized row lives in registers between RMSNorm and GEMM.
  QKV output tile lives in SMEM between GEMM and RoPE and the final write.
- **Current cost:** 22.9 + 99.8 + 55.6 + (store_kvcache not integrated) = 178 µs
- **Target:** ≤ 30 µs (5.9× speedup). Floor sum = ~9 µs.

### 3.2 `vcpm_attention_causal<H_n=16, H_kv=2, D=128>` — inline online-softmax causal

- **Inputs:** q[N, Q_DIM], k_cache, v_cache, slot_mapping, cu_seqlens_q, cu_seqlens_k
- **Output:** o[N, Q_DIM] bf16 in HBM
- **Algorithm:** flash-attn 2 style — tile over N_q in outer loop, stream
  over N_k in inner loop with online softmax update. GQA: 8-way share of
  KV cache across heads.
- **Current cost:** 198 µs flash_attn
- **Target:** ≤ 40 µs (5× speedup). Floor ≈ 10 µs.

### 3.3 `vcpm_post_attn_causal<H=2048, I=6144>` — fused o_gemm + residual + rmsnorm + gate_up + silu·mul + down_gemm + residual

- **Inputs:** residual_prev[N, H] (the pre-attn residual x0), attn_out[N, Q_DIM],
  w_o, w_post_ln, w_gate_up, w_down
- **Output:** residual_next[N, H] bf16 in HBM (= attn_output + mlp_output)
- **Fusion sequence (residual stream stays in fp32 registers/SMEM throughout):**
  1. o_gemm: attn_out @ w_o^T → fp32 tile
  2. fuse: add residual_prev → residual_after_attn (fp32 in SMEM)
  3. rmsnorm over residual_after_attn → normalized fp32 tile in registers
  4. gate_up gemm: normalized @ w_gate_up^T → [N, 2I] fp32 (but stream! see below)
  5. fuse silu·mul online: silu(gate[i]) * up[i] → fp32 mid tile
  6. down_gemm: mid @ w_down^T → fp32 output tile
  7. fuse: residual_after_attn + down_out → residual_next (bf16 to HBM)
- **Streaming tip:** gate_up output at [N, 2I=12288] is 4.5 MB at bf16 per
  layer — too big to hold full in SMEM. Stream by columns: process
  intermediate dim I in chunks of I_chunk (e.g., 256) so the [N, I_chunk]
  tile fits comfortably. down_gemm then does a series of partial
  accumulations into the output tile.
- **Current cost:** 99.8 + 55.1 + 23.4 + 229.0 + 21.6 + 118.0 + 31.8 = 578.7 µs
- **Target:** ≤ 80 µs (7× speedup). Floor sum = ~62 µs.

### 3.4 GEMM primitive (the critical path)

All four GEMMs (QKV, O, gate_up, down) share a single templated primitive.
At our M=100 (padded to 112 or 128) the current WMMA 16×16×16 kernel
delivers 8-16× over floor because:

1. Single warp per output tile — serializes K-dim accumulation.
2. No cp.async — weight loads are synchronous, exposing HBM latency.
3. No software pipelining — compute waits on every K step's load.
4. No L2 residency hints — next layer's weights aren't hinted for prefetch.

**Target primitive: `vcpm_gemm_bf16_tuned<TM, TN, TK, STAGES, WARPS>`.**
Design:

- **Tile:** TM=64 or 128 (M-wave covers padded M=128), TN=128, TK=32.
- **Warp group:** WARPS=4 (128 threads), each warp owns a 16x32 sub-tile
  via WMMA; 4 warps cover TM × TN with staggered K iterations.
- **Pipelining:** STAGES=3 — at any time, one stage is loading weights
  (cp.async for A and B), one is computing MMA, one is draining.
- **Memory:** A and B each use STAGES * TM*TK and TN*TK of SMEM. Fits:
  3 × 64 × 32 × 2B = 12 KB + 3 × 128 × 32 × 2B = 24 KB = 36 KB per block,
  well under 228 KB SMEM.
- **L2 prefetch:** on the last MMA of layer N, issue
  `cp.async.bulk.prefetch.L2` for layer N+1's first weight tile (relevant
  for P2.5.2 multi-layer work; no-op in P2.5.1 single-layer).
- **Epilogue:** optional callable that the pre_attn / post_attn kernels
  specialize: `EPILOGUE::apply(tile_fp32, row, col) -> bf16`. Lets
  `vcpm_post_attn` fuse residual_add + RMSNorm + gate input directly in
  SMEM without materializing the GEMM output.

Floor verification at the four shapes:

| GEMM | M | N | K | FLOPs | compute µs | weight MB | bw µs | floor µs |
|---|---|---|---|---|---|---|---|---|
| QKV   | 100 | 2560  | 2048 | 1.05 G | 5.9 | 10.5 | 6.9 | **7.5** |
| O     | 100 | 2048  | 2048 | 0.84 G | 4.7 | 8.4  | 5.5 | **6.1** |
| gate_up | 100 | 12288 | 2048 | 5.03 G | 28.3 | 50.3 | 33.1 | **35.0** |
| down  | 100 | 2048  | 6144 | 2.52 G | 14.1 | 25.2 | 16.6 | **17.6** |
| sum |  |  |  | 9.44 G | 53 | 94.4 | 62 | **66** |

At the 4 shapes, HBM bw is the binding constraint (weight reads dominate).
Our kernel's job is to **saturate HBM**, not the tensor cores. cp.async +
STAGES=3 pipelining is the mechanism: issue weight loads early enough
that by the time MMA needs them, they're arriving from HBM at line rate.

## 4. Numerics contract

Per-layer (single layer at base_lm shape):
- `vcpm_pre_attn_causal` output q and cached K,V ↔ upstream Cpm4Attention's
  `(q, k, v)` after split+rope: **max rel ≤ 1e-2 (bf16), mean rel ≤ 1e-3**.
- `vcpm_attention_causal` output ↔ flash_attn_func output: same gates.
- `vcpm_post_attn_causal` output ↔ upstream layer's hidden_states after MLP
  residual add: same gates.
- Full fused-kernel single-layer output ↔ upstream Cpm4DecoderLayer: same gates.

Full 28-layer stack: bf16 residual-stream compounding gates stay as in P2.4
(max rel ≤ 5e-1, mean rel ≤ 1e-2). The fp32-residual-in-SMEM design WITHIN
each kernel will reduce intra-layer drift; inter-layer boundaries stay bf16
because the residual still crosses kernel boundaries in HBM.

(If we later fuse multiple layers into ONE persistent kernel — P2.5.2 —
the residual can stay fp32 across layer boundaries too, which fixes the
28-layer drift problem measured in P2.4.)

## 5. Implementation order (P2.5.1.a → P2.5.1.d)

Each stage is **independently validated and merged** before moving on.
No step marks "done" until its tests pass with the listed gates AND the
measured µs beats the prior implementation.

- **P2.5.1.a — Tuned GEMM primitive** (`vcpm_gemm_bf16_tuned`)
  - Single new kernel, 4 shape specializations (or one templated)
  - Tests: numerics vs current `vcpm_gemm_bf16` at all 4 shapes, bf16 bit-exact
  - Bench: replaces current GEMM in FusedLayer, measure 28-layer end-to-end
  - **Gate:** 28-layer stack ≤ **12 ms** (vs 20 ms current) at N=100

- **P2.5.1.b — Inline attention kernel** (`vcpm_attention_causal`)
  - Replaces flash_attn_func in FusedLayer(causal=True)
  - Tests: numerics vs flash_attn at the exact shapes used in base_lm (N=100,
    heads 16/2, head_dim 128)
  - **Gate:** 28-layer stack ≤ **8 ms** (vs 12 ms after P2.5.1.a)

- **P2.5.1.c — Fused pre_attn kernel** (`vcpm_pre_attn_causal`)
  - Replaces rmsnorm + qkv_gemm + rope + kv_write in FusedLayer
  - Tests: q output and KV-cache contents match upstream's Cpm4Attention
  - **Gate:** 28-layer stack ≤ **6 ms**

- **P2.5.1.d — Fused post_attn kernel** (`vcpm_post_attn_causal`)
  - Replaces o_gemm + residual + rmsnorm + gate_up + silu_mul + down + residual
  - Tests: residual output matches upstream layer output
  - **Gate:** 28-layer stack ≤ **~4 ms** — approaches physics floor

Then P2.5.2 (multi-layer persistence with TMA prefetch) extracts the last
factor of 2 to reach ≤ 2 ms full base_lm forward.

## 6. Non-goals for P2.5.1

- No cooperative-grid launch. Each of the 3 kernels per layer is a plain
  `<<<grid, block>>>` launch on one stream. Ordering is stream-FIFO.
- No WGMMA PTX / tcgen05 intrinsics. sm_120 supports WMMA 16×16×16 with
  bf16 which is enough to saturate HBM at our M. Blackwell's 5th-gen TC
  native paths are P2.5.2+ territory.
- No quantization. bf16 throughout.
- No LoRA. Reference-audio cloning only (per CLAUDE.md decision).
- No multi-stream concurrency. c=1-optimized path; multi-slot is P2.6.

## 7. What this document is not

A promise that everything lands in one session. These 4 stages are
3-5 sessions of careful CUDA work. Each one delivers a validated,
committed speedup — no half-built kernels, no numerics handwaved, no
"will tune later."

Physics floor is the bar. Every stage's gate above is measured in ms at
N=100, not estimated. If a stage doesn't hit its gate, stop and
diagnose before moving on.
