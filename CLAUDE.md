# CLAUDE.md — VoxCPM2 Fast Inference (Orchestrated)

This is the operating charter for the **orchestrator** and all agents working on this repo.

The mission is the **physics floor** of TTS inference on a single RTX 5090: the moment a single bit-flip or kernel-launch removal can no longer shave a microsecond. We stop when we hit it, not before.

---

## Required reading (auto-loaded — every agent sees this)

@AGENTS.md
@PROJECT_PLAN.md
@BASELINE.md
@voxcpm_fast/notes/reference_architecture.md
@voxcpm_fast/notes/topology.md
@voxcpm_fast/notes/physics_floor_c1.md

If you are an agent and you landed here without reading those files, stop and read them. The answers to 80% of the "should I do X?" questions you are about to ask are already written in them.

---

## Roles

### Orchestrator (parent session)

- Decides what gets built, in what order.
- Writes **exact** prompts for agents: specific files to read, specific code to write, specific tests to run, specific numbers to produce.
- Reviews every agent's work — runs their tests again, spot-reads their code, confirms their measured numbers.
- Updates `AGENT_LOG.md` and `PROJECT_PLAN.md` after every completed agent task.
- Does some coding directly, but delegates the majority. **Preserves context window** by offloading exploration, enumeration, and grinding to agents.
- Asks the user *only* for scope/architecture decisions the user hasn't already answered. Does not ask for micro-decisions.

### Agents (sub-sessions spawned by the orchestrator)

- Execute one bounded task as prompted. Do not scope-creep.
- Read the auto-loaded docs before writing any code.
- **Measure everything.** No "looks faster" claims. No "should work" without running it.
- Report back with: files changed (exact paths), commands run (exact invocations), measured numbers (pasted output), dead ends tried, and the one recommended next step.
- Never mark a task complete if its acceptance criteria aren't met. If you hit a blocker, explain it and hand back; don't improvise scope.

---

## The Physics Floor — how we know we're done

| metric | physics-floor target |
|---|---|
| T_first_chunk p95, concurrency = 64 | ≤ 70 ms |
| sustained RTF p99, concurrency = 64 | ≤ 0.4 |
| underruns at concurrency = 64 | 0 |
| kernel launches per decode step | ≤ 8 (megakernel + VAE + output emit) |
| host↔device syncs per decode step | ≤ 1 (final pinned-host emit) |
| SM occupancy of the persistent megakernel | ≥ 80 % |
| per-step CPU time (Python + IPC) | ≤ 200 µs |

These are not negotiable. If a change gets us closer, it ships. If it doesn't, it doesn't.

---

## Hard rules — agents must follow exactly

### R1. Never be lazy. Measure everything.

- No perf claim without a before/after number.
- No correctness claim without a bf16 diff ≤ 1e-2 **and** fp32 diff ≤ 1e-5 against an upstream reference (use `nanovllm_voxcpm.models.voxcpm2.model` as the reference on CPU or GPU with `torch.set_grad_enabled(False)`).
- No "I didn't check because the shapes matched." Always check.
- No "I'll trust upstream" for numbers you could re-measure. Re-measure.
- If a kernel fails at edge shapes (seq_len=1, seq_len=4096, batch=1, batch=64), that is a real failure — not a future issue.

### R2. Environment is fixed. Don't touch it.

```bash
UV_CACHE_DIR=/workspace/Developments/VoxCPM2/.uv_cache
HF_HOME=/workspace/Developments/VoxCPM2/hf_cache
MODEL=/workspace/Developments/VoxCPM2/models/VoxCPM2
```

- Always invoke Python via `UV_CACHE_DIR=/workspace/Developments/VoxCPM2/.uv_cache uv run python …` from inside `/workspace/Developments/VoxCPM2/nanovllm-voxcpm`.
- **torch must stay `>=2.9.0,<2.11`.** 2.11.0 ships against cuda 13 and breaks the install (our driver caps at cu12.9). See `AGENT_LOG.md` 2026-04-20.
- **flash-attn must stay `==2.8.1` with the prebuilt wheel** pinned via `[tool.uv.sources]`. Do not let `uv lock` pick another version. If you have to bump, find a matching cu12 + torch2.{9,10} + cp312 wheel in the Dao-AILab releases first.
- **Never set `MAX_JOBS` for any CUDA build greater than 4.** `MAX_JOBS=32` OOM'd the host on 2026-04-20.

### R3. Files you may not touch in place

- `nanovllm-voxcpm/` — **upstream reference.** Only `pyproject.toml` is editable (and has been; do not revert). Copy patterns into `voxcpm_fast/` instead of editing upstream files.
- `models/VoxCPM2/` — immutable weights.
- `BASELINE.md` — append new runs, never overwrite past ones.
- `AGENT_LOG.md` — append-only. Newest at the top. Never edit older entries except to add a correction entry.

### R4. Numerics bars

- Megakernels: bf16 max-abs-diff ≤ **1e-2**, fp32 max-abs-diff ≤ **1e-5** vs upstream per-layer reference. Print the shape, the max-abs-diff, and the max-abs-value for context.
- End-to-end audio: mean squared error vs upstream waveform ≤ **1e-3**, spectrograms visually aligned (hand check after the kernel lands).

### R5. Always log

Every agent appends to `AGENT_LOG.md` before finishing. Use the template at the top of that file. Do **not** strip the `Dead ends:` section even if empty — put `- none` explicitly. That section is how the next agent avoids re-trying the same thing.

### R6. Don't introduce new dependencies without asking

Adding a package changes the lockfile and may trip the torch / flash-attn pin. If you think you need one, stop and ask the orchestrator.

---

## Topology digest (fast-lookup, in case auto-loaded context is skipped)

- **Model:** `VoxCPM2Model`, 2.29 B params, 48 kHz output, patch_size = 4, feat_dim = 64.
- **base_lm:** 22-layer MiniCPM4. hidden = 2048, intermediate = 4096, 16 Q heads × 128 head_dim, **GQA 16 / 2** (2 KV heads), qkv_proj out = 2560, causal with KV cache.
- **residual_lm:** 6 layers, same per-layer shape as base_lm, causal with KV cache, uses `fusion_concat_proj` to combine enc_outputs and feat_embeds.
- **feat_encoder:** non-causal Cpm4Model. hidden = 1024 (different!), 4 layers, input `[T, P+1, 64→1024]` with CLS token.
- **feat_decoder = UnifiedCFM:** 10 Euler timesteps × **CFG ×2** batch. First step (≤ `ceil(0.04 × 11) = 1` step) is skipped (dphi_dt = 0). DiT = `VoxCPM2LocDiT`, 4-layer non-causal Cpm4Model at hidden = 1024, input `[2B, 1 + prefix + P, 1024]`.
- **Per-patch audio duration:** 80 ms (one decode step must finish in < 80 ms per stream to keep RTF < 1).
- **Current single-stream decode:** ~8.5 ms with CUDA graphs; 180 ms with `enforce_eager=True`.
- **Section share (eager baseline):** feat_decoder 73.5 %, base_lm 15.6 %, feat_encoder 7.6 %, residual_lm 3.0 %, everything else < 1 %.

The full per-module call trace is in `voxcpm_fast/notes/topology.md` (also auto-loaded above).

---

## File layout

```
VoxCPM2/
  CLAUDE.md              # you are here
  AGENTS.md              # rules + RAM hazards + conventions
  AGENT_LOG.md           # append-only work log
  PROJECT_PLAN.md        # phases, targets, open questions
  BASELINE.md            # measured numbers per run

  nanovllm-voxcpm/       # upstream reference — read only (except pyproject.toml)
    nanovllm_voxcpm/…    # the reference code we're beating
    pyproject.toml       # torch<2.11, flash-attn==2.8.1 prebuilt wheel pin
    uv.lock              # regenerate only when pyproject changes

  models/VoxCPM2/        # weights (4.7 GB, downloaded)
  hf_cache/              # HF download cache

  voxcpm_fast/           # our code — all new work lives here
    csrc/                # host-side C++ glue (.cpp)
    megakernels/         # one .cu file per fused op
    benchmarks/          # end-to-end TTFPA scripts
    experiments/         # one-off runs kept for history
    profiles/            # nsys / ncu reports (binary)
    notes/               # markdown docs (reference_architecture.md, topology.md, ...)
    scripts/             # shell helpers

  logs/                  # stdout captures + benchmark JSON
```

---

## Commands cheat sheet

Run from `/workspace/Developments/VoxCPM2/nanovllm-voxcpm` unless noted.

```bash
# full model exploration (emits voxcpm_fast/notes/topology.md)
UV_CACHE_DIR=/workspace/Developments/VoxCPM2/.uv_cache uv run python \
  /workspace/Developments/VoxCPM2/voxcpm_fast/experiments/explore_model.py \
  --model /workspace/Developments/VoxCPM2/models/VoxCPM2 \
  --iters 10 \
  --output /workspace/Developments/VoxCPM2/voxcpm_fast/notes/topology.md

# baseline TTFPA (upstream reference)
UV_CACHE_DIR=/workspace/Developments/VoxCPM2/.uv_cache uv run python \
  /workspace/Developments/VoxCPM2/voxcpm_fast/benchmarks/ref_ttfpa.py \
  --model /workspace/Developments/VoxCPM2/models/VoxCPM2 \
  --devices 0 \
  --concurrency 64 \
  --iters 1 \
  --max-generate-length 100 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.92 \
  --output-json /workspace/Developments/VoxCPM2/logs/ref_ttfpa_c64.json

# nsys timeline of one decode step
UV_CACHE_DIR=/workspace/Developments/VoxCPM2/.uv_cache \
  /usr/local/cuda-12.8/bin/nsys profile \
    --trace cuda,nvtx,osrt \
    --sample none \
    --output /workspace/Developments/VoxCPM2/voxcpm_fast/profiles/p1_ref_c1 \
    uv run python /workspace/Developments/VoxCPM2/voxcpm_fast/benchmarks/ref_ttfpa.py \
      --model /workspace/Developments/VoxCPM2/models/VoxCPM2 \
      --concurrency 1 --iters 1 --max-generate-length 20

# sanity: does flash-attn still load?
UV_CACHE_DIR=/workspace/Developments/VoxCPM2/.uv_cache uv run python -c \
  "import torch, flash_attn; from flash_attn import flash_attn_func; \
   q = torch.randn(1,8,4,64,device='cuda',dtype=torch.bfloat16); \
   print(flash_attn_func(q,q,q).shape)"
```

---

## Orchestration protocol

The orchestrator works a **one-task-per-agent** model. Each agent task has:

1. **Task name** (e.g., `P1-nsys`, `P2-mk_dit_layer`).
2. **Exact objective** — one sentence, specific outcome.
3. **Files to read** (beyond the auto-loaded set).
4. **Files to write** (new) or modify (existing, with path).
5. **Acceptance criteria** — numbers or test outcomes the orchestrator will re-run.
6. **Forbidden** — things this agent must not touch or do.
7. **Expected output format** — the structure of the report back.

The orchestrator then:

- Spawns the agent (`Plan` / `Explore` / `general-purpose`).
- Receives the report.
- Re-runs the acceptance tests itself (doesn't trust the agent's summary).
- Writes the `AGENT_LOG.md` entry if the task passes (not the agent — the orchestrator, to keep it terse and honest).
- Marks the `PROJECT_PLAN.md` bullet done.
- Picks the next task.

---

## "Never be lazy" — concrete failure modes to reject

These are things agents have tried before (in other projects) that we reject on sight:

- **"I ran it and it worked"** with no pasted output. Reject.
- **"The shapes match so the math is right."** Reject — verify values, not shapes.
- **"I assumed X."** Don't assume. Check.
- **"It's faster in theory."** Reject without wall-clock numbers.
- **"I'll add logging later."** No. Add it now or don't mention the claim.
- **"Tests failed but this unrelated one passes."** Failure blocks completion.
- **"I reduced scope to make it fit."** Come back and say so explicitly; don't silently drop work.
- **"Skipping the numerics check because the file is small."** Numerics check is mandatory. Always.
- **Editing upstream `nanovllm-voxcpm/` files.** Ban.
- **Changing the torch or flash-attn pin.** Ban without orchestrator approval.
- **Installing new packages.** Ban without orchestrator approval.

---

## What is NOT in scope (stop trying)

- Training, finetuning, LoRA training.
- Multi-GPU (one server per GPU; project is intra-GPU).
- CPU fallback or ARM support.
- FP4 / FP6 / FP8 weight quantization (bf16 throughout; we have headroom).
- Supporting VoxCPM v1 (we target v2 only unless explicitly asked).
- Refactoring for "cleanliness" on code that works.
- Writing new tests for code that isn't the task of this turn.

---

## Open scope questions for the user (orchestrator may need answers)

See `PROJECT_PLAN.md § Still open` for the live list. The orchestrator can make calls on CFG-on/off, fallback policy, max_model_len, chunk boundary policy, and output format if production is silent — but **it must write down the call** in `AGENT_LOG.md` so it's revocable.

---

## Getting started (if you are a new agent)

1. Read the auto-loaded files above (they are in your context already).
2. Read the `PROJECT_PLAN.md` phase you were assigned.
3. Read the most recent 2 entries of `AGENT_LOG.md`.
4. Do exactly what your task prompt says.
5. Report back in the format the prompt requires.
6. Stop.
