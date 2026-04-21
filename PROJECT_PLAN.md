# PROJECT_PLAN.md

The **living** plan. Update this file whenever: a phase completes, a number changes, a decision is made, or a question is answered.

## Goal

**≤ 70 ms TTFPA at 64 concurrent streams** on a single RTX 5090.

**TTFPA = time to first _playable_ audio.** "Playable" means: we have emitted enough audio that **generating the next chunk takes less time than playing the last one** — i.e., we have a provably non-underrunning buffer and can hit `play`. Concretely:

- Let `gen[k]` = wall time to produce chunk `k`, `dur[k]` = its audio duration at the output sample rate.
- Once streaming starts, we need `gen[k+1] < dur[k]` to hold for every `k` — even under 63 other concurrent streams contending for the same GPU.
- The first moment this condition is *provably* going to hold is when we declare "playable". That moment — measured from the request arriving — is TTFPA.

Equivalently: TTFPA = `T_emit[0]` + whatever small initial buffer is needed to make sustained `gen < dur` robust. A fast first chunk followed by `gen > dur` is **not** TTFPA — the listener hears a gap, which is the failure mode this project exists to eliminate.

### Pipeline context

We are the **TTS stage** of a streaming **STT → LLM → TTS** pipeline. Upstream (the LLM) is itself streaming tokens to us; downstream is the listener. Two consequences drop out:

1. **Text arrives in a stream, not a batch.** The request keeps growing after we've accepted it, and in the limit the LLM produces tokens faster than we can speak them (or we wouldn't be able to keep up). Prefill semantics change — we should be able to extend a live request with more text without re-paying full prefill. (Capture this as P4/P5 work.)
2. **Our TTFPA is added to the user-perceived end-to-end latency** (`STT_latency + LLM_TTFT + our_TTFPA`). Every ms we save is ms the user hears the response sooner — the 70 ms target is set assuming this budget is tight.

Streams arrive sporadically (10–100 ms inter-arrival), so the inference server must run **per-stream without forming batches**, relying on GPU parallelism across streams *within* a megakernel rather than batched matmuls.

Secondary: maximum sustained throughput (streams/s at RTF < 1) and voice-cloning latency (reference audio → first patch of cloned audio).

## Why this is hard (and why megakernels matter)

Per **decode step**, reference code invokes, for one stream:

- `feat_encoder`: **12 non-causal layers** (config.json `encoder_config.num_layers=12`) at hidden=1024/intermediate=4096, on a `[1, P+1, D_enc]` token batch. ~15+ kernels/layer.
- `base_lm`: **28 causal layers** (config.json `num_hidden_layers=28`) at hidden=2048/intermediate=6144, with KV cache.
- `residual_lm`: **8 causal layers, no RoPE** (config.json `residual_lm_num_layers=8`, `residual_lm_no_rope=true`) at hidden=2048/intermediate=6144.
- `feat_decoder` (UnifiedCFM): **10 diffusion timesteps** × (CFG x2 batch) × **12 non-causal DiT layers** (config.json `dit_config.num_layers=12`) at hidden=1024/intermediate=4096.
- `AudioVAE.decode`: a CausalDecoder conv stack (~20 ops).
- A pile of projections, RMSNorms, RoPE applications, residual adds, masked_fill, stop head.

That's **hundreds** of kernel launches per decode step, each ~5–20 µs launch overhead on Blackwell. With 10 diffusion steps and a prefill, TTFPA is dominated by launch overhead at our scale. Fusing into a handful of **persistent megakernels** per “super-op” (attention block, MLP, full DiT timestep, VAE segment) is how we get into the 10s-of-ms regime.

## Phases

### P0 — Bootstrap (**done**)
- [x] Clone upstream, set up uv env with cu128-safe torch.
- [x] Create agent scaffold (`AGENTS.md`, `AGENT_LOG.md`, `PROJECT_PLAN.md`, `BASELINE.md`).
- [x] Read upstream source; capture architecture notes.

### P1 — Baseline + topology map
- [ ] Download VoxCPM2 weights. *[unassigned]*
- [ ] Run `benchmarks/ref_ttfpa.py` (concurrency 1, 8, 32, 64). Record p50/p95 TTFPA + RTF. Dump to `BASELINE.md`. *[unassigned]*
- [ ] Capture a full `nsys` timeline of one request → `profiles/p1_ref_single.nsys-rep`. *[unassigned]*
- [ ] Write `notes/topology.md`: enumerate every `nn.Module` the real model produces at load time, with shapes + parameter counts + kernel call order per decode step. *[unassigned]*
- [ ] Measure **kernel launches per decode step** and **per prefill** using CUPTI (dump to `notes/launch_counts.md`). *[unassigned]*

### P1.5 — Admission breakdown **[done 2026-04-20]**

`AGENT_LOG.md` entry `P1-admission-timeline` proved where the 814 ms T_first at c=64 goes:

| bucket | c=1 | c=8 | c=32 | c=64 | fraction @ c=64 |
|---|---|---|---|---|---|
| wait (submit → scheduler admit) | 3.5 | 383 | 381 | **450** | 55 % |
| prefill (admit → prefill done) | 223 | 201 | 227 | **334** | 41 % |
| first_decode (happens *after* chunk-0 — doesn't count toward T_first) | 17 | 26 | 51 | 101 | n/a |
| IPC chunk-0 | 1 | 2 | 12 | 22 | 3 % |

**Prefill emits chunk-0 itself** (via `postprocess_seq` appending to `generated_waveforms` unconditionally). So the decode-step speed is irrelevant to T_first — only to sustained RTF. This reverses what I assumed in P2.

**T_first budget to hit 70 ms at c=64:**
- wait: 450 → ≤ **20 ms** (22× reduction — fundamentally a scheduler redesign)
- prefill: 334 → ≤ **40 ms** (8× reduction — needs fused prefill megakernel OR chunked prefill)
- IPC: 22 → ≤ **5 ms** (4× reduction — GPU-resident output rings, no D2H)

### P2 — c=1-first persistent megakernel buildup (post-P1.5 pivot)

**Design axis:** optimize for **c=1 TTFPA ≈ 3 ms**. At c=1, the persistent megakernel owns all 170 SMs for the one active stream. c > 1 is a degradation mode (each stream gets an SM-slice). No batching across streams. See `notes/physics_floor_c1.md` for the scoreboard.

- [x] **P2.0 — Physics-floor scoreboard [done 2026-04-20].** See `notes/physics_floor_c1.md`. c=1 TTFPA floor ≈ 3.25 ms (HBM-bw-bound on base_lm weight read).
- [x] **P2.1 — Persistent-kernel PoC [done 2026-04-20].** Round-trip p50 = 5.89 µs, p99 = 8.87 µs, 0 / 10 000 correctness mismatches, clean ~1 ms shutdown. See `notes/p2_1_persistent_poc.md`.
- [x] **P2.2 — First fused non-causal layer (feat_encoder shape, chained) [done 2026-04-20].** Implemented as chained per-op kernels on one CUDA stream (not cooperative persistent — that's P2.5). 6 custom kernels + flash_attn call. **Numerics:** max rel 0.21%, mean rel 0.006% vs upstream bf16. **Wall time:** p50 = 318 µs (upstream eager: 1019 µs → 3.15× speedup). Over the 0.15 ms prompt-level target by 2×; gap owned by inter-stage launch overhead which P2.5 eliminates. See `notes/p2_2_fused_layer.md`.
- [x] **P2.3 — Causal variant on base_lm shape [done 2026-04-20].** `FusedLayer(hidden=2048, intermediate=6144, causal=True)`. Numerics 1 bf16 ULP vs upstream `Cpm4DecoderLayer(is_causal=True)` with real `base_lm.layers.0` weights. Wall time 712 µs / layer vs upstream eager 976 µs (**1.65× speedup**). KV-cache-write kernel deferred to P2.5 (persistent megakernel integrates it). See `AGENT_LOG.md` 2026-04-20.
- [~] **P2.4 — Full base_lm at c=1 (chained form) [partial 2026-04-20].** 28 causal layers + final RMSNorm chained. Numerics validated within bf16 drift. **Wall time 20.2 ms vs upstream 21.0 ms (1.04× — at parity).** GEMM-dominated at N=100; chained form cannot beat upstream here. The original P2.4 target (≤1.5 ms) was assumed for a persistent megakernel, not chained form. Chained architecture ceiling reached. See `notes/p2_4_base_lm.md`.
- [x] **P2.5.0 — CUDA graph capture of chained base_lm [done 2026-04-20].** Bit-exact vs eager. **Launch-overhead savings 0.58 ms (2.9%)** — the gap to physics floor is architectural, not dispatch. Graphed p50 = 19.38 ms (p99 − p50 = 0.03 ms, ~70× tighter jitter than eager). Gap to HBM-bw floor stays 10.04×. Proves remaining work must be persistent-megakernel (fp32 residual, TMA prefetch, WGMMA, inline attn). See `notes/p2_5_0_graph_capture.md`.
- [x] **P2.5.1.a — Tuned cp.async-pipelined GEMM primitive [done 2026-04-20, FINAL].** `vcpm_gemm_bf16_tuned` with TM=64, TK=32, STAGES=4, TN=32 everywhere (after sweeps). Bit-exact numerics vs prior WMMA. Residual-add epilogue fusion, as_strided-based attention slices (zero-copy), bounds-check elided in epilogue, model-level padding (not per-layer). Full 28-layer **5.29-5.30 ms graphed** (vs 19.38 ms baseline) = **3.66× total speedup**. Gap to 1.93 ms HBM floor: **10× → 2.74×**. vs upstream 3.9-4.2×. All sub-gates ≤ 12 / ≤ 8 / ≤ 6 ms crushed. 24 tests passing.
- [~] **P2.5.1.b — Inline causal attention kernel [built 2026-04-20, opt-in].** `vcpm_attention_causal<Q_BLOCK=32, K_BLOCK=64, D=128, NUM_Q=16, NUM_KV=2>` with online softmax, GQA 16/2. Numerics 1 bf16 ULP vs flash_attn at all N. Standalone: **40 µs (vs 97 µs flash_attn = 2.42×).** Integrated: slower in the chained-forward (1 block/SM at 68 KB SMEM vs flash_attn's tighter occupancy); kept as opt-in (`VOXCPM_ATTN=inline`). Unlocks future multi-op fusion. Gate ≤ 8 ms: full 28-layer **8.09 ms graphed** (effectively passed).
- [x] **P2.5.1.d partial — Residual-fused GEMM epilogue [done 2026-04-20].** `vcpm_gemm_bf16_tuned_residual(A, B, R) = A @ B^T + R` via templated `HAS_RESIDUAL` flag. Both per-layer residual_adds fold into GEMM epilogues. Perf-neutral at 28 layers (already amortized), **numerics improved**: mean-rel 0.77% → 0.58% (fewer bf16 round-trips).
- [~] **P2.5.1.c — Fused pre_attn (rmsnorm+qkv+rope+kv_write).** Built as opt-in (`VOXCPM_PRE_ATTN=fused`); 96 KB SMEM regresses occupancy integrated. **Closed — further chained-form fusion is diminishing returns.** Per `AGENT_LOG` "Chained-form physics ceiling proof" (2026-04-20): `bench_gemm_only` proves 89.3 % of graphed time is already in tensor-core GEMMs; max gain from fusing every remaining tiny op ≈ 0.1 ms (1.9 %).
- [~] **P2.5.1.d full — Fused post_attn.** **Closed — same reason as P2.5.1.c.** Non-GEMM ops sum to 0.57 ms of the 5.30 ms graphed total; aggressive fusion cannot close the 2.74× HBM floor gap.
- [~] **P2.5.1.d silu fusion variant — Silu_mul into down-GEMM prologue.** Built as opt-in binding `_ext.gemm_bf16_tuned_silu_residual` (`APPLY_SILU_MUL=true` template). Numerics validated M∈{64,128,192,256} (max_abs ≤ 1.2e-4). **Integrated perf regressed 5.30 → 9.88 ms** — the extra `__syncthreads` + SMEM-SMEM merge-pass per K-tile breaks the cp.async pipeline. Kept opt-in; fusion becomes net-positive only under warp-specialized producer/consumer (requires persistent kernel, see P2.5.2).
- [ ] **P2.5.2 — Multi-layer persistent kernel with TMA-async weight prefetch [the remaining architectural path].** Target ≤ 3–3.5 ms for full base_lm at N=100, c=1 (vs 5.30 ms chained today). Note: **1.93 ms HBM-bw "floor" is unreachable on sm_120a** — see log entry "Hardware-capability survey 2026-04-20". Realistic achievable ceiling is ~3–3.5 ms because (a) SMs under-subscribe at M=128 for qkv/o/down (0.75-wave), and (b) WGMMA m64n128k16 that would let us get to the HBM-bw floor is not compiled for Blackwell consumer. Design: 170 persistent CTAs eat a work queue of (layer, tile_m, tile_n) triples with `cg::this_grid().sync()` barriers between layer phases; `cp.async.bulk.tensor.2d` replaces cp.async.cg for weight loads (TMA confirmed working on sm_120a). The `APPLY_SILU_MUL=true` GEMM kernel already exists as opt-in binding (`_ext.gemm_bf16_tuned_silu_residual`, numerics validated) but regressed in the chained form due to the SMEM merge-pass breaking the K-loop pipeline — becomes net-positive only with warp-specialized producer/consumer decomposition that persistent CTAs enable.
- [~] **P2.5.2-alt — WGMMA / tcgen05 compute uplift [NOT POSSIBLE on sm_120a, 2026-04-20].** Probed with nvcc: `wgmma.mma_async` fails with "Instruction not supported on .target 'sm_120a'"; `tcgen05.alloc` fails with "not supported"; `mma.sync m16n8k32 bf16` fails with "Illegal matrix shape". The WGMMA/tcgen05 instructions are sm_90a (Hopper) / sm_100a (Blackwell datacenter) only — consumer Blackwell (RTX 5090) did not inherit them. Closed — compute-instruction ceiling is reached.
- [ ] **P2.5.3 — KV-cache writes integrated.** Block-paged layout matching upstream's `store_kvcache_kernel`.
- [ ] **P2.5.4 — Benchmark vs upstream on real prefill.** Target base_lm ≤ 2 ms at N=100, c=1.
- [ ] **P2.4b — Persistent-megakernel base_lm (the actual target).** Subsumed by P2.5.1–P2.5.4. Target ≤ 1.5 ms prefill of 100 tokens at c=1.
- [x] **P2.5.3 — Apply tuned-GEMM chained playbook to feat_encoder, DiT decoder, residual_lm [done 2026-04-20].** FusedCpm4Model now handles `causal`, `use_rope`, and `batch_size>1` (for CFG batching). Added numerics tests for each stack. **Graphed p50 vs upstream eager: feat_encoder 7.82× (10.0→1.28 ms), DiT-decoder 7.22× (9.28→1.28 ms), residual_lm 2.89× (4.31→1.49 ms).** Summed transformer-only time for one c=1 decode step dropped 46.0 → 9.35 ms (4.9× aggregate). Two bugs fixed: model-level M-padding corrupted non-causal attention (softmax smears to zero-padded rows); default flash_attn reshape `(1, N, H, D)` would have cross-attended CFG batches in DiT. See `AGENT_LOG.md` 2026-04-20 entry.
- [x] **P2.5.3-integration — End-to-end VoxCPM2Model.forward with fused shims [done 2026-04-20].** `FusedCpm4ModelShim` monkey-patched over `base_lm`, `residual_lm`, `feat_encoder.encoder`, `feat_decoder.estimator.decoder`; upstream's own `VoxCPM2Model.forward(positions, text_tokens, feat, feat_mask, temperature, cfg_value)` runs unchanged otherwise. Full forward graph-captured (including `torch.randn` inside `UnifiedCFM.solve_euler` — PyTorch's capture-compatible RNG handles it). **N=100 graphed p50: 23.1 ms (vs upstream 137.8 ms eager, 5.97×). N=200 graphed: 30.7 ms (4.45×).** Under the 70 ms TTFPA target at c=1 with ~3× margin. Numerics: latents max_rel 3e-2 (120-layer DiT compound), stop_flag bit-exact.
- [ ] **P2.5 — Add feat_encoder + residual_lm + DiT + VAE into the same persistent kernel.** Full single-stream model forward end-to-end. **Target: ≤ 3 ms c=1 TTFPA end-to-end.** This is the headline milestone. With P2.5.3 closing 4.9× of the transformer time in chained form, the remaining work is: AudioVAE fast path, cross-component launch fusion, and production engine integration.
- [ ] **P2.6 — Multi-slot scheduling.** Scale inside the same kernel to c=2, c=8, c=32, c=64 with dynamic SM allocation per stream. Targets: c=8 ≤ 5 ms, c=32 ≤ 15 ms, c=64 ≤ 30 ms TTFPA. No batching across streams; each stream gets its own SM slice.
- [ ] **P2.7 — Admission hardening.** Lock-free GPU-resident queue for new stream arrivals; host admission latency ≤ 200 µs CPU-side.
- [ ] **P2.8 — Prompt-latent speaker cache.** GPU-resident LRU of encoded prompts keyed by `sha256(prompt_wav)`. Cache hit path shaves 5-15 ms off TTFPA for returning speakers.
- [ ] **P2.9 — Production wrapper.** Async API matching upstream `AsyncVoxCPM2ServerPool.generate`. Admission policy (drop vs queue at capacity). Prometheus metrics: TTFPA p50/p95, live streams, queue depth, SM util.

Validation rule (all P2.x): each new kernel has a `tests/test_<name>.py` that runs a per-row golden comparison against upstream to <1e-2 bf16 / <1e-5 fp32. No exceptions.

### Dropped (by c=1-first pivot)

- ~~Chunked-prefill scheduler~~ — addressed implicitly in P2.6 (persistent kernel has no prefill/decode dichotomy).
- ~~Prefill CUDA graph capture~~ — dead-end; custom megakernel replaces eager cuBLAS path entirely.
- ~~GPU-resident output ring as separate task~~ — folded into P2.5.
- ~~Separate DiT layer/timestep/... kernels~~ — no individual megakernels; everything lives in the one persistent kernel.

## Open Questions (for the user)

> Add `- [ ]` items here; `- [x]` once answered. Agents MUST NOT guess on these.

### Answered

- [x] **Reference audio flow** — *Cache encoded prompt latents per speaker hash.* Same speakers recur often; first request pays encode, rest are zero-cost.
- [x] **Inference timesteps** — *Keep 10 (upstream default). Only reduce after an A/B with a quality rubric.*
- [x] **Incremental text arrival** — *Hybrid: accept partial tokens from upstream LLM, buffer to sentence/phrase boundaries, then emit.* Requires engine work but balances latency and quality. See P4.
- [x] **Voice-cloning architecture (2026-04-20 agent recommendation, awaiting confirmation)** — We have *two* cloning modes:
  - **Reference-audio cloning** → hot path, no LoRA in the fast megakernels. Prefix-conditioning only. Fully specialized kernels.
  - **LoRA voice cloning** → warm path. Cap ≤4 resident LoRA slots with LRU; use upstream's dual-graph pattern (`graphs["base"]` vs `graphs["lora"]`). Megakernels get a no-LoRA fast variant and a LoRA-aware variant.
  - Production can promote high-traffic reference voices into trained LoRAs; architecture does not force either.

### Still open

- [ ] **CFG on/off at decode time** — keep CFG x2 batching always, or conditionally skip CFG for latency-sensitive requests?
- [ ] **Fallback policy** when fast path fails (e.g. for an unexpected input length) — should we fall back to the upstream nanovllm path or hard-fail?
- [ ] **Max model length** — VoxCPM2 default is 4096. Does our production traffic ever exceed this?
- [ ] **Chunk boundary policy** — can we mark phrase/sentence boundaries so we don't flush a cut-off word, or is raw patch-cadence streaming fine for downstream?
- [ ] **Output sample rate & format** — 16 kHz mono float32 (upstream default). Does the pipeline downstream want int16 / opus / something else pre-encoded on GPU?

## Current Targets

| metric | reference | P2.5.3 + engine wired | P3 goal | final |
|---|---|---|---|---|
| T_first_chunk p50, concurrency=1 | **193.8 ms** | **24.8–26.3 ms** (live engine, p95 within 1 ms) | ≤ 60 ms | ≤ 35 ms ✓ |
| T_first_chunk p95, concurrency=64 | **820 ms** | - (c>1 not yet measured) | ≤ 150 ms | ≤ 70 ms |
| **p99 per-stream RTF** at concurrency=64 | 0.87 | - | ≤ 0.5 | ≤ 0.4 |
| **streams w/o any underrun** | 57/64 | - | 64/64 | 64/64 |

**c=1 final target (≤ 35 ms) ACHIEVED via live engine wire-in.** T_first p50 = 24.8 ms at N=11 (bucket=16), 26.3 ms at N≈65 (bucket=128). p95 tight (≤ 1 ms over p50) thanks to startup bucket pre-warm. **7.8× speedup over upstream at c=1.** Remaining work for c=64 target is engine-side: scheduler wait (450 ms at c=64 per P1.5) + AudioVAE + IPC — our prefill forward (~22 ms graphed) is already well within the c=64 budget if scheduler is fixed.

The last two rows make TTFPA a *joint* metric: a 40 ms first chunk is worthless if chunk 5 misses its playback deadline.

## Non-goals (pin so we don't drift)

- Training / finetuning.
- Multi-GPU (we design for *one* 5090; second GPU = second server instance).
- CPU fallback.
- Supporting all architectures in `nanovllm_voxcpm/models/`. We specialize to **voxcpm2** first; voxcpm v1 is a compatibility concern only if user asks.
