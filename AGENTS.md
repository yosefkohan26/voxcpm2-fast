# AGENTS.md — Collaboration Rules for VoxCPM2-Fast

This project is built almost entirely by agents. The single biggest failure mode is **rework and drift**: two agents re-derive the same thing, or an agent undoes a decision a prior agent made without realizing why. Follow these rules strictly.

## 0. The three files to read first (every session, before you write any code)

1. `README.md` — project purpose, hardware, quickstart.
2. `PROJECT_PLAN.md` — phase plan, current targets, open questions. **Your work belongs to a phase.** If your task doesn't fit a phase, flag that first.
3. `AGENT_LOG.md` — last ~20 entries. This tells you what other agents already did, tried, and concluded.

Also skim `BASELINE.md` to know current measured numbers.

## 1. Every agent MUST append to `AGENT_LOG.md` when done

Entry format (see `AGENT_LOG.md` for the template and existing entries):

- **Who & when** — agent ID / date
- **Phase & task** — link to phase in `PROJECT_PLAN.md`
- **What changed** — files touched, commands run
- **Results** — measured numbers (TTFPA, RTF, occupancy, launch count…) with command to reproduce
- **Dead ends** — what was tried and abandoned, with one-sentence reason. **Write these honestly** — saving the next agent 30 min is worth a 3-line entry
- **Next step** — what the next agent should pick up

## 2. Never edit `nanovllm-voxcpm/` in place

Upstream is the reference. Only exceptions:

- `nanovllm-voxcpm/pyproject.toml` — already pinned to cu128-safe torch; do not revert.
- You may **read** it freely and **copy patterns** into `voxcpm_fast/`.

If you need to diverge from upstream behavior, copy the file into `voxcpm_fast/`, modify, and add a note in `AGENT_LOG.md` with *why*.

## 3. Always measure before claiming a win

- Every kernel/megakernel change needs a **before/after** TTFPA and concurrent-throughput measurement. One-off “looks faster” claims are not acceptable.
- Save raw numbers into `BASELINE.md`. Include git hash (or timestamp + summary) and the full `nvidia-smi` / `nvcc --version` / `torch.__version__` snapshot in the entry.
- Profile with `nsys profile` (timeline) and `ncu` (kernel metrics); drop reports into `profiles/`. Summarize findings in `notes/`, not in the report binary.

## 4. Ownership & locking

- Before starting a non-trivial task, **claim it** in `PROJECT_PLAN.md` by marking the bullet `[in-progress by <agent-id> on <date>]`.
- If you see something claimed but stale (>24h with no `AGENT_LOG.md` update), you may take it over — and say so in the log.

## 5. Scope discipline

We are building a **production inference server**. Everything is in service of **TTFPA at concurrency**. Reject scope creep:

- No training code.
- No "general" refactors — only those on the critical path.
- No premature abstractions — we can specialize to this exact model shape and GPU.

Explicit non-goals:

- CPU fallback, multi-GPU (single-GPU first class), quantization (for now — may revisit).
- Training/finetuning, LoRA training paths. We keep LoRA *inference* because production needs voice adapters, but not LoRA training.

## 5a. RAM hazards — read before any CUDA/C++ build

nvcc compiling a large fused kernel takes **4–8 GB of RAM per process**. flash-attn, cutlass, and our own megakernels will all trip this. Rules:

- **Set `MAX_JOBS=4`** (or 2–6 depending on box) for *any* source build that invokes nvcc in parallel. Never `MAX_JOBS=$(nproc)` on a 128-core host.
- Keep `nvidia-smi` *and* `free -h` in a second pane (or `watch -n1`) during builds. If RSS climbs past ~80% of RAM, kill the build.
- flash-attn 2.8.x full rebuild takes ~25–40 min at `MAX_JOBS=4` on a 5090. Budget it.
- Prior incident (2026-04-20): `MAX_JOBS=32` during initial bring-up consumed all host RAM, OOM-killed ssh, forced a reboot. See `AGENT_LOG.md` entry.

## 6. Kernel-development conventions

- Target: **sm_120** (Blackwell). Use `-arch=sm_120a` for WGMMA / TMA features where supported.
- Default dtype: **bfloat16** for model, **fp32** for accumulation; match upstream unless a numerics experiment is explicitly approved.
- One megakernel per file in `megakernels/`, named `mk_<what>.cu`. Each file must include a top-comment describing the fused ops, the expected shapes, and the launch parameters.
- Use CUDA Graphs for the generic path; megakernel persistence is for the *concurrent* path where graphs can't batch dynamically.
- Always write a `tests/test_mk_<what>.py` that compares against a PyTorch reference, FP32 and BF16.

## 7. When you get stuck

Prefer asking the user once, clearly, with options — see `PROJECT_PLAN.md § Open Questions`. Don't silently pick a direction on a scope/design decision.

## 8. Git discipline

- This tree is *not* yet a git repo by default. When we start committing, prefer **small atomic commits** with a one-line subject that names the phase: `[P2] add mk_rmsnorm_qkv kernel`.
- Do not `git push` to any remote without explicit user approval.

## 9. Repro envelope

Every benchmark script prints, at the top of its output:

```
hostname     : <...>
gpu          : NVIDIA GeForce RTX 5090 (sm_120, 32 GB)
driver       : 570.124.06
cuda runtime : 12.8
torch        : 2.10.x
flash-attn   : 2.8.3
git hash     : <..>
```

If your script doesn't, add it. Future-you will thank you.
