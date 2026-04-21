"""Measure real engine T_first at c=1, upstream vs fast path, N trials each.

Run in two separate invocations (upstream=no hook, fast=install hook) since
the engine caches graphs once.
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

sys.path.insert(0, str(NANOVLLM_ROOT))
sys.path.insert(0, str(REPO_ROOT))  # expose voxcpm_fast as a package
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))


def main():
    fast = "--fast" in sys.argv
    trials = 20

    if fast:
        os.environ["VOXCPM_FAST_ENC"] = "1"
        os.environ["VOXCPM_FAST_DIT"] = "1"
        from voxcpm_fast.fast_main_loop import patch_server_module
        patch_server_module()

    from nanovllm_voxcpm.models.voxcpm2.server import SyncVoxCPM2ServerPool

    print(f"mode: {'FAST (feat_encoder+DiT swapped)' if fast else 'UPSTREAM (control)'}", flush=True)
    print("loading model ...", flush=True)
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

    text = os.environ.get(
        "BENCH_TEXT",
        "The quick brown fox jumps over the lazy dog.",
    )

    # Warmup (2 runs, stable graph pool).
    for _ in range(2):
        for _ in pool.generate(target_text="Hello there friend.",
                               max_generate_length=200,
                               temperature=0.7, cfg_value=2.0):
            pass

    tfirst_ms: list[float] = []
    total_ms: list[float] = []
    for i in range(trials):
        t0 = time.time()
        tfirst = None
        n_chunks = 0
        for chunk in pool.generate(target_text=text, max_generate_length=200,
                                   temperature=0.7, cfg_value=2.0):
            if tfirst is None:
                tfirst = (time.time() - t0) * 1000
            n_chunks += 1
        total = (time.time() - t0) * 1000
        tfirst_ms.append(tfirst)
        total_ms.append(total)
        print(f"  trial {i:2d}: T_first={tfirst:6.1f} ms  total={total:6.1f} ms  "
              f"chunks={n_chunks}", flush=True)

    def stats(xs):
        xs_s = sorted(xs)
        return (statistics.median(xs), xs_s[int(0.95*len(xs_s))], statistics.mean(xs),
                statistics.stdev(xs) if len(xs) > 1 else 0.0)

    p50, p95, mean, std = stats(tfirst_ms)
    print()
    print(f"T_first over {trials} trials:")
    print(f"  p50 = {p50:.1f} ms")
    print(f"  p95 = {p95:.1f} ms")
    print(f"  mean = {mean:.1f} ms  std = {std:.1f} ms")

    p50, p95, mean, std = stats(total_ms)
    print(f"total over {trials} trials:")
    print(f"  p50 = {p50:.1f} ms  mean = {mean:.1f} ms")

    pool.stop()


if __name__ == "__main__":
    main()
