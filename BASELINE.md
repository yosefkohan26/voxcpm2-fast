# BASELINE.md

Measured baseline numbers. Add one section per measurement run — **never overwrite** past runs. Format numbers to 1 decimal ms.

## Environment fingerprint format

```
date         : <UTC ISO>
hostname     : <...>
gpu          : NVIDIA GeForce RTX 5090 (sm_120, 32 GB)
driver       : 570.124.06
cuda runtime : 12.8
torch        : 2.10.x
flash-attn   : 2.8.3
git          : <hash or "dirty + <short msg>">
script       : voxcpm_fast/benchmarks/<name>.py <args>
```

## Results template

| concurrency | T_first p50 (ms) | T_first p95 (ms) | sustained RTF p50 | sustained RTF p99 | underruns observed | notes |
|---|---|---|---|---|---|---|
| 1  |   |   |   |   |   |   |
| 8  |   |   |   |   |   |   |
| 32 |   |   |   |   |   |   |
| 64 |   |   |   |   |   |   |

`T_first` is measured as `(first chunk yield ts) − (request submit ts)` outside the TTS server process, to include queue & IPC overhead.

`sustained RTF` is measured per stream over **all chunks after the first** as `sum(gen[k]) / sum(dur[k])`.

An **underrun** is any `k` where `gen[k+1] > dur[k]` (cumulatively). The real TTFPA target requires zero underruns at concurrency = 64.

---

## Run 1 — upstream nanovllm_voxcpm (2026-04-20, commit 47f412b)

**Environment**
```
date         : 2026-04-20T02:39Z
gpu          : NVIDIA GeForce RTX 5090 (sm_120, 32 GB), driver 570.124.06
cuda runtime : 12.8
torch        : 2.10.0+cu128
flash-attn   : 2.8.1  (prebuilt wheel cu12+torch2.10+cp312, no compile)
model        : openbmb/VoxCPM2 (2.29 B params, 48 kHz)
script       : voxcpm_fast/benchmarks/ref_ttfpa.py --iters 1 --max-generate-length 100
flags        : enforce_eager=False (CUDA graphs on), gpu_memory_utilization=0.9-0.92
```

**Headline numbers (upstream as-is, fair reference):**

| concurrency | T_first p50 (ms) | T_first p95 (ms) | RTF p50 | RTF p99 | streams w/ underrun | notes |
|---|---|---|---|---|---|---|
| 1  | **187.3** | 195.7 | 0.107 | 0.107 | 0 / 3  | single stream is healthy |
| 8  | **634.3** | 639.0 | 0.189 | 0.189 | **4 / 8**  | 50% of streams underrun |
| 32 | **680.7** | 690.9 | 0.426 | 0.440 | 4 / 32 | 12.5% underrun, batch saturates |
| 64 | **813.9** | 819.5 | 0.848 | 0.868 | 7 / 64 | 11% underrun, RTF near 1 |

**Raw JSON:**
- `logs/ref_ttfpa_c1.json`
- `logs/ref_ttfpa_c8.json`
- `logs/ref_ttfpa_c32.json`
- `logs/ref_ttfpa_c64.json`

**Key observations**

1. **The single-stream decode is already fast** (RTF 0.107 ≈ 8.5 ms per patch at 80 ms audio). The CUDA-graph hot path gives upstream a ~20× speedup vs the eager `enforce_eager=True` path we measured in `notes/topology.md` (180 ms/step eager → ~8.5 ms/step graph-captured).

2. **TTFPA is dominated by first-chunk latency, not steady-state RTF.** Even at c=64, RTF p99 is 0.87 — the system *could* sustain 64 streams if they all arrived already warmed up. The **814 ms T_first** at c=64 is mostly queuing / sequential admission, not raw compute.

3. **Underruns at c=8 and above are real.** 4/8, 4/32, 7/64. Average RTF is safe, but tail latencies (prefill admission + scheduler round-robin + CUDA graph switching across batch sizes) cause gaps. A production listener hears these.

4. **Target gap:** 70 ms @ c=64 vs 814 ms baseline = **~11.6× faster** required on T_first. RTF at c=64 is already acceptable (0.85) but has zero headroom — we need it comfortably under 0.5 at c=64 to absorb jitter.

**Where the time goes (from `notes/topology.md`, eager mode, single stream)**

| section | ms/step | share |
|---|---|---|
| `feat_decoder` (10-step CFM + CFG×2) | 124.0 | **73.5 %** |
| `base_lm` (22-layer MiniCPM4) | 26.4 | 15.6 % |
| `feat_encoder` | 12.8 | 7.6 % |
| `residual_lm` (6 layers) | 5.0 | 3.0 % |
| everything else (projections, stop, VAE) | < 2 | < 1 % |

Under CUDA graphs this compresses to ~8.5 ms/step but the *proportions* hold: the DiT diffusion loop is where the speedup budget lives.

---

## Run 2 — *(placeholder: first voxcpm_fast megakernel pass)*

Fill in after P2 completes.
