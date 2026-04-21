"""Measure T_first with a real voice prompt (cloning path).

Uses upstream's add_prompt (VAE-encodes prompt audio, caches by prompt_id)
so the per-request cost is strictly feat_encoder + base_lm + residual_lm +
DiT + VAE, not VAE-encoding the reference.

Usage:
  bench_voice_ttfpa.py [--fast] [--voice bobby|jackson|jerome|joseph]
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


def main():
    fast = "--fast" in sys.argv
    voice = "bobby"
    for a in sys.argv:
        if a.startswith("--voice"):
            voice = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]

    voice_dir = VOICES_DIR / voice
    assert voice_dir.exists(), f"{voice_dir} missing"
    wav_bytes = (voice_dir / "audio.wav").read_bytes()
    prompt_text = (voice_dir / "transcript.txt").read_text().strip()
    print(f"voice: {voice}  prompt_text={prompt_text!r}  wav_size={len(wav_bytes)}", flush=True)

    if fast:
        os.environ["VOXCPM_FAST_ENC"] = "1"
        os.environ["VOXCPM_FAST_DIT"] = "1"
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

    # Register the prompt once (cached by prompt_id — VAE-encode happens here).
    t0 = time.time()
    prompt_id = pool.add_prompt(wav_bytes, "wav", prompt_text)
    print(f"prompt registered: id={prompt_id}  took={(time.time()-t0)*1000:.1f} ms", flush=True)

    target = "Hello friend, the weather today is absolutely lovely, and I hope your day is going well."

    # Warmup.
    for _ in range(2):
        for _ in pool.generate(target_text="Hi there.", prompt_id=prompt_id,
                               max_generate_length=200, temperature=0.7, cfg_value=2.0):
            pass

    trials = 20
    tfirst_ms, total_ms = [], []
    for i in range(trials):
        t0 = time.time()
        tfirst = None
        n_chunks = 0
        for _ in pool.generate(target_text=target, prompt_id=prompt_id,
                               max_generate_length=200,
                               temperature=0.7, cfg_value=2.0):
            if tfirst is None:
                tfirst = (time.time() - t0) * 1000
            n_chunks += 1
        total = (time.time() - t0) * 1000
        tfirst_ms.append(tfirst)
        total_ms.append(total)
        print(f"  trial {i:2d}: T_first={tfirst:6.1f} ms  total={total:6.1f} ms  chunks={n_chunks}", flush=True)

    def stats(xs):
        xs_s = sorted(xs)
        return (statistics.median(xs), xs_s[int(0.95 * len(xs_s))],
                statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)

    p50, p95, mean, std = stats(tfirst_ms)
    print()
    print(f"T_first over {trials} trials with voice={voice!r}:")
    print(f"  p50  = {p50:.1f} ms")
    print(f"  p95  = {p95:.1f} ms")
    print(f"  mean = {mean:.1f} ms  std = {std:.1f} ms")

    pool.stop()


if __name__ == "__main__":
    main()
