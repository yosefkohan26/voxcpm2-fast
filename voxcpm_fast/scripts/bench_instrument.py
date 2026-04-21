"""Time inside the engine. Wraps VoxCPM2Runner.run to report prep + forward + VAE + IPC.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"
MODEL_DIR = REPO_ROOT / "models" / "VoxCPM2"

sys.path.insert(0, str(NANOVLLM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))


def _install_timing_probe():
    from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Runner
    _orig_run_model = VoxCPM2Runner.run_model

    def run_model(self, inputs, is_prefill):
        if is_prefill:
            # Time the forward separately.
            torch.cuda.synchronize()
            t0 = time.time()
            out = _orig_run_model(self, inputs, is_prefill)
            torch.cuda.synchronize()
            t1 = time.time()
            out.setdefault("_timing", {})["forward_ms"] = (t1 - t0) * 1000
            out["_timing"]["N"] = inputs["positions"].size(0)
        else:
            out = _orig_run_model(self, inputs, is_prefill)
        return out

    VoxCPM2Runner.run_model = run_model


def main():
    fast = "--fast" in sys.argv

    if fast:
        from engine_hook import install_fast_path
        install_fast_path(enable_feat_encoder=True, enable_dit=True)

    _install_timing_probe()

    from nanovllm_voxcpm.models.voxcpm2.server import SyncVoxCPM2ServerPool

    print(f"mode: {'FAST' if fast else 'UPSTREAM'}", flush=True)
    pool = SyncVoxCPM2ServerPool(
        model_path=str(MODEL_DIR),
        inference_timesteps=10,
        max_num_seqs=8,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        devices=[0],
    )

    text = "The quick brown fox jumps over the lazy dog."

    # Warmup
    for _ in range(2):
        for _ in pool.generate(target_text="Hi.", max_generate_length=50,
                               temperature=0.7, cfg_value=2.0):
            pass

    # Timed
    tfirst_ms, fwd_ms, others_ms = [], [], []
    for i in range(10):
        t0 = time.time()
        tfirst = None
        for chunk in pool.generate(target_text=text, max_generate_length=200,
                                   temperature=0.7, cfg_value=2.0):
            if tfirst is None:
                tfirst = (time.time() - t0) * 1000
        tfirst_ms.append(tfirst)
        # The forward timing was recorded inside run_model on the runner's side
        # but it's not propagated through IPC. Instead log from runner via print.

    def stats(xs):
        return statistics.median(xs), statistics.mean(xs)

    p50, mean = stats(tfirst_ms)
    print(f"T_first (n=10): p50={p50:.1f} ms  mean={mean:.1f} ms", flush=True)
    pool.stop()


if __name__ == "__main__":
    main()
