# VoxCPM2 Fast Inference — 5090 Megakernel Project

**Goal:** drive VoxCPM/VoxCPM2 audio inference to the **physics floor** on a single RTX 5090 (Blackwell, sm_120). Target: **≤70 ms TTFPA** with **64 concurrent streams**, where streams arrive sporadically (every 10–100 ms) so no traditional batch formation is possible.

TTFPA = **time to first patch of audio** (from request submit → first waveform chunk yielded). This is not TTFB — it's the full critical path through prefill, one decode step, 10 CFM diffusion sub-steps, and AudioVAE decode.

## Directory Layout

```
VoxCPM2/
  nanovllm-voxcpm/        # upstream reference (do not edit in place — diff against this)
  voxcpm_fast/            # our production inference server & kernels
    csrc/                 # C++ / CUDA sources (host-side glue, .cpp only)
    megakernels/          # persistent megakernel .cu implementations
    benchmarks/           # TTFPA + throughput benchmarks
    experiments/          # ad-hoc scripts (explorations, kept for history)
    profiles/             # nsys / ncu reports (binary — gitignore)
    notes/                # technical notes (one .md per topic)
    scripts/              # helper shell scripts (download model, profile, etc.)
  models/                 # model weight cache (gitignored)
  logs/                   # run logs (gitignored)
  AGENTS.md               # START HERE — rules for agent collaboration
  AGENT_LOG.md            # append-only log of what each agent did
  PROJECT_PLAN.md         # evergreen plan: phases, targets, open questions
  BASELINE.md             # measured baselines (reference vs each iteration)
  README.md               # this file
```

## Quickstart

```bash
# 1. Python env (uv-managed)
cd nanovllm-voxcpm && uv sync --all-packages

# 2. Download model weights
bash ../voxcpm_fast/scripts/download_model.sh

# 3. Reference baseline (upstream nanovllm_voxcpm)
uv run python ../voxcpm_fast/benchmarks/ref_ttfpa.py --model ../models/VoxCPM2 --concurrency 64

# 4. Our fast server (TODO)
uv run python -m voxcpm_fast.server --model ../models/VoxCPM2 --concurrency 64
```

## Hardware

- **GPU:** NVIDIA RTX 5090 (Blackwell, SM 12.0, 32 GB VRAM, cu128)
- **CUDA toolkit:** 12.8 (driver 570.124, max cu12.9)
- **PyTorch:** 2.10.x (cu128 default — do not upgrade to 2.11+, which ships cu130)
- **FlashAttention:** 2.8.3 (built from source; supports Blackwell)

**Do not touch** `torch>=2.11` without first installing a CUDA 13 toolkit; mismatch breaks flash-attn build.

See `AGENTS.md` for collaboration rules.
See `CLAUDE.md` for the orchestrator charter + auto-loaded agent context. **Start there if you are an agent.**
