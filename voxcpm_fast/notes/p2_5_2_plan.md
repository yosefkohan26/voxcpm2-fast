# P2.5.2 — persistent megakernel plan

## Current graphed breakdown at bucket=16 (text-only, fast path)

| phase | graphed ms | floor ms | gap |
|---|---|---|---|
| feat_encoder (12L non-causal, bsz=1, S=5..16) | ~0.5 | 0.2 | 2.5× |
| **base_lm (28L causal, N=16)** | **4.51** measured | 1.93 | **2.34×** |
| residual_lm (8L causal no-rope, N=1) | ~1.2 est | 0.6 | 2.0× |
| projections + fsq + stop_head | ~0.5 | - | - |
| **DiT (9 Euler × 12L × bsz=2, seq=11)** | **9 × 1.28 = 11.5** measured | ~3 | **3.8×** |
| VAE decode + IPC | ~2 | - | - |
| **total** | **~22** | **~7** | **3.1×** |

Physics-floor ceiling (per `notes/physics_floor_c1.md`) says ≤ 3.25 ms is
theoretically achievable for the whole thing. Realistic sm_120a ceiling
(no WGMMA, no tcgen05) is closer to 5-6 ms.

## Biggest chunks

**DiT (11.5 ms)** — biggest, because 9 Euler iterations × 12 layers all
run sequentially on 204 MB of weights that don't fit in L2 (128 MB).

**base_lm (4.5 ms)** — second-biggest, 28 layers × ~89 MB weights each,
graphed chained form. Per-call already at 2.34× of 1.93 ms HBM floor —
graph capture has extracted most of the launch-overhead savings.

## Attack plan

### Phase A — **DiT megakernel** (priority #1, biggest lever)

Target: fuse all 9 Euler iterations × 12 DiT layers into ONE cooperative
kernel launch. Goal: ≤ 4 ms for the whole DiT loop (from 11.5 ms).

Design:
- Single kernel, 170 persistent CTAs (one per SM).
- Input: `x_init [2, 1024, 4] bf16`, `mu [2, 1024]`, `cond [2, 1024, 4]`,
  `cfg_value [2] bf16`, `temperature [2] bf16`, DiT-layer weights, time
  embeddings lookup.
- Output: `pred_feat [1, 64, 4] bf16` (after CFG combine + transpose).

Control flow (inside kernel, all on the same grid):
```
for t in 0..9:
  if t == 0:
    dphi_dt = 0
  else:
    cg::this_grid().sync()     # synchronize before DiT
    dphi_dt = DiT_12_layers(x, mu, t, cond)    # in-kernel
    dphi_dt = CFG_combine(dphi_dt, cfg_value)   # in-kernel
  x = x - dt * dphi_dt
  cg::this_grid().sync()
```

Key wins:
- Kernel launches: 9×12×~12 = 1296 ops → 1 launch. Saves ~1.3 ms graph-
  replay overhead at 1 µs/op.
- Activation handoff between layers stays in SMEM (~45 KB per CTA).
  Eliminates 11 residual HBM roundtrips × 9 Euler = 99 HBM roundtrips.
  At ~5 µs each, ~0.5 ms saved.
- Weight reads: still 12 × 17 MB = 204 MB per Euler × 9 iters = 1836 MB.
  L2 holds 128 MB, so 6/12 layers warm at end of Euler step, cold at
  start of next. HBM load ≈ 9 × 12 × 17 MB = 1.8 GB / 1.52 TB/s = 1.2 ms.
  (Optimistic — current chained may already be close to this.)
- Attention inline vs flash_attn: saves ~0.1 ms per call × 9 × 12 = 1.1 ms.
  **Big, but only if inline attn matches flash_attn's occupancy.** Our
  P2.5.1.b experiment showed it didn't. Needs re-tuning for non-causal
  bsz=2 seq=11 shape.

Realistic savings: 2-4 ms off DiT. Lands around 7.5-9.5 ms for DiT.

### Phase B — **base_lm megakernel** (priority #2)

Target: 28 causal layers fused + TMA weight prefetch pipeline. Goal:
≤ 3 ms for base_lm (from 4.5 ms).

Design:
- 170 persistent CTAs.
- TMA `cp.async.bulk.tensor.2d` prefetches layer N+1's weights into SMEM
  while computing layer N.
- Grid-sync between layers.
- Inline attention (GQA 16/2, causal).
- KV cache writes inline (no store_kvcache triton launch).

### Phase C — **Unified prefill megakernel** (far future)

Everything in one launch: feat_encoder → enc_to_lm_proj → base_lm →
fsq_layer → fusion_concat_proj → residual_lm → lm_to_dit_proj →
res_to_dit_proj → 9×DiT → stop_head.

Requires careful activation layout between phases. Mostly a
composition exercise once Phase A/B land.

## Compile strategy

- Target sm_120a (RTX 5090). No WGMMA (not supported), no tcgen05 (not
  supported), use m16n8k16 MMA.
- Use `cg::grid_group` and `cg::this_grid().sync()` from
  `cooperative_groups.h`. Requires `cudaLaunchCooperativeKernel` launch
  path (not a regular `<<<>>>`).
- TMA via PTX `cp.async.bulk.tensor.2d` — confirmed working on sm_120a
  per AGENT_LOG 2026-04-20 hardware capability survey.

## Correctness budget

- bf16 max-abs-diff ≤ 1e-2 vs chained form reference, per CLAUDE.md R4.
- Run through the existing `test_fused_dit_decoder.py` + end-to-end
  `test_voxcpm2_forward.py` after each phase lands.

## Incremental landing order (each is one PR/commit)

1. Scaffolding: new file `voxcpm_fast/megakernels/mk_dit_prefill.cu`,
   stub kernel `cooperative_dit_prefill_noop` (launches + grid-syncs
   + writes zeros). Prove cooperative launch works on sm_120a.
2. Single DiT layer in-kernel (no Euler loop): fused rmsnorm + qkv_gemm +
   rope + attention + o_gemm + residual + post_ln + gate_up + silu_mul +
   down_gemm inside one CTA block. Validate numerics vs chained single
   layer.
3. Multi-layer (all 12 layers in-kernel, still 1 Euler step). Grid-sync
   between layers. Validate numerics vs chained full DiT decoder.
4. Full Euler loop inside kernel (9 iterations). Validate numerics vs
   chained full `feat_decoder.forward`.
5. Integrate into engine (swap DiT shim for megakernel path gated by
   `VOXCPM_MEGAKERNEL_DIT=1`).

Each step needs its own `test_mk_*.py` and `bench_mk_*.py` pair.
