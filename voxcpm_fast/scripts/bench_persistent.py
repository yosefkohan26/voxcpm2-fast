"""Persistent-pool bench harness: load the model ONCE, run many scenarios.

Usage:
    bench_persistent.py                 # run default scenario matrix
    bench_persistent.py --interactive   # drop to REPL after warm-up

Why this exists: each SyncVoxCPM2ServerPool startup costs ~40s model load +
~20s bucket pre-warm before trials can run. Sweeping 5 scenarios costs
5 × 60s = 5 min. Loading once amortizes that to ~60s + 5 × (trials only).

Env toggles we can A/B without reload (checked at every forward call):
  VOXCPM_PREFETCH       l2 | 0
  VOXCPM_GEMM           tuned | wmma
  VOXCPM_PRE_ATTN       fused | (unset)
  VOXCPM_ATTN           inline | (unset)

Env toggles that MUST be set before load (read once by fast_main_loop):
  VOXCPM_FAST_ENC, _DIT, _BASE, _RES   — pick your stack swaps up-front.

To compare the no-reload-toggleable knobs, we run all scenarios in one
process. Fast-path code reads os.environ on every forward, so flipping
between trials gives a clean A/B without a fresh load.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"
MODEL_DIR = REPO_ROOT / "models" / "VoxCPM2"
VOICES_DIR = REPO_ROOT / "voxcpm_fast" / "voices"

sys.path.insert(0, str(NANOVLLM_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))


def _stats(xs: list[float]) -> tuple[float, float, float, float]:
    xs_s = sorted(xs)
    return (
        statistics.median(xs),
        xs_s[min(len(xs_s) - 1, int(0.95 * len(xs_s)))],
        statistics.mean(xs),
        statistics.stdev(xs) if len(xs) > 1 else 0.0,
    )


def _run_trials(pool, *, target_text: str, prompt_id: str | None,
                trials: int, warmup: int = 2) -> tuple[list[float], list[float]]:
    # Warmup (separate short text so graphs & channels are hot).
    for _ in range(warmup):
        for _ in pool.generate(target_text="Hi there.", prompt_id=prompt_id,
                               max_generate_length=200, temperature=0.7, cfg_value=2.0):
            pass

    tfirst_ms: list[float] = []
    total_ms: list[float] = []
    for _ in range(trials):
        t0 = time.time()
        tfirst = None
        n = 0
        for _ in pool.generate(target_text=target_text, prompt_id=prompt_id,
                               max_generate_length=200, temperature=0.7, cfg_value=2.0):
            if tfirst is None:
                tfirst = (time.time() - t0) * 1000
            n += 1
        total = (time.time() - t0) * 1000
        tfirst_ms.append(tfirst)
        total_ms.append(total)
    return tfirst_ms, total_ms


def _print_scenario(name: str, env_toggle: dict[str, str | None],
                    tfirst_ms: list[float], total_ms: list[float]) -> None:
    p50, p95, mean, std = _stats(tfirst_ms)
    tp50, _, tmean, _ = _stats(total_ms)
    extras = ", ".join(f"{k}={v!r}" for k, v in env_toggle.items()) or "defaults"
    print(f"\n=== {name}   [{extras}] ===")
    print(f"  T_first  p50 = {p50:6.2f}   p95 = {p95:6.2f}   mean = {mean:6.2f}   std = {std:4.2f} ms")
    print(f"  total    p50 = {tp50:6.1f}   mean = {tmean:6.1f} ms")


def _apply_env(env_toggle: dict[str, str | None]) -> dict[str, str | None]:
    """Set env vars in the current process. Returns prior values for restore."""
    prior: dict[str, str | None] = {}
    for k, v in env_toggle.items():
        prior[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return prior


def _restore_env(prior: dict[str, str | None]) -> None:
    for k, v in prior.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def main():
    fast = "--fast" in sys.argv or "--no-fast" not in sys.argv  # default fast

    # Pre-load env toggles that must be set BEFORE model load.
    if fast:
        os.environ.setdefault("VOXCPM_FAST_ENC", "1")
        os.environ.setdefault("VOXCPM_FAST_DIT", "1")
        # VOXCPM_FAST_BASE / _RES stay off by default.

        from voxcpm_fast.fast_main_loop import patch_server_module
        patch_server_module()

    from nanovllm_voxcpm.models.voxcpm2.server import SyncVoxCPM2ServerPool

    print(f"mode: {'FAST' if fast else 'UPSTREAM'}", flush=True)
    t0 = time.time()
    pool = SyncVoxCPM2ServerPool(
        model_path=str(MODEL_DIR),
        inference_timesteps=10,
        max_num_seqs=8,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        devices=[0],
    )
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    # Register one voice so voice-prompted scenarios are available.
    voice_id = None
    try:
        voice_dir = VOICES_DIR / "bobby"
        wav_bytes = (voice_dir / "audio.wav").read_bytes()
        prompt_text = (voice_dir / "transcript.txt").read_text().strip()
        t0 = time.time()
        voice_id = pool.add_prompt(wav_bytes, "wav", prompt_text)
        print(f"voice registered: took={(time.time()-t0)*1000:.0f} ms", flush=True)
    except Exception as e:
        print(f"voice registration skipped: {e}", flush=True)

    target_short = "The quick brown fox jumps over the lazy dog."
    target_long = ("Hello friend, the weather today is absolutely lovely, "
                   "and I hope your day is going well.")
    trials = int(os.environ.get("BENCH_TRIALS", "20"))

    scenarios: list[tuple[str, dict[str, str | None], dict]] = []
    # Baseline text-only with current env defaults.
    scenarios.append(("text-only   PREFETCH=l2 (default)",
                      {"VOXCPM_PREFETCH": "l2"},
                      dict(target_text=target_short, prompt_id=None)))
    scenarios.append(("text-only   PREFETCH=off",
                      {"VOXCPM_PREFETCH": "0"},
                      dict(target_text=target_short, prompt_id=None)))
    scenarios.append(("text-only   PREFETCH=l2  long-target",
                      {"VOXCPM_PREFETCH": "l2"},
                      dict(target_text=target_long, prompt_id=None)))
    if voice_id:
        scenarios.append(("voice-bobby PREFETCH=l2 (default)",
                          {"VOXCPM_PREFETCH": "l2"},
                          dict(target_text=target_short, prompt_id=voice_id)))
        scenarios.append(("voice-bobby long-target",
                          {"VOXCPM_PREFETCH": "l2"},
                          dict(target_text=target_long, prompt_id=voice_id)))

    for name, env, kwargs in scenarios:
        prior = _apply_env(env)
        try:
            tf, tt = _run_trials(pool, trials=trials, **kwargs)
            _print_scenario(name, env, tf, tt)
        finally:
            _restore_env(prior)

    if "--interactive" in sys.argv:
        import code
        banner = ("\nInteractive mode — pool is live. Useful handles:\n"
                  "  pool, voice_id, target_short, target_long\n"
                  "  _run_trials(pool, target_text=..., prompt_id=..., trials=5)\n")
        code.interact(banner=banner,
                      local={"pool": pool, "voice_id": voice_id,
                             "target_short": target_short, "target_long": target_long,
                             "_run_trials": _run_trials})
    pool.stop()


if __name__ == "__main__":
    main()
