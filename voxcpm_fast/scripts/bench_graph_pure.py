"""Measure pure graph-replay time — strips every engine overhead. Shows
how close we are to the physics floor on the captured forward.
"""

import os, sys, time, statistics
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "nanovllm-voxcpm"))
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))

os.environ["VOXCPM_FAST_ENC"] = "1"
os.environ["VOXCPM_FAST_DIT"] = "1"
os.environ["VOXCPM_PREFILL_GRAPH"] = "1"
from voxcpm_fast.fast_main_loop import patch_server_module
patch_server_module()

from nanovllm_voxcpm.models.voxcpm2.server import SyncVoxCPM2ServerPool

pool = SyncVoxCPM2ServerPool(
    model_path=str(REPO_ROOT / "models" / "VoxCPM2"),
    inference_timesteps=10, max_num_seqs=8, max_model_len=4096,
    gpu_memory_utilization=0.85, enforce_eager=False, devices=[0],
)
# Warmup so the N=16 bucket is captured
for _ in pool.generate(target_text="Hi.", max_generate_length=20,
                       temperature=0.7, cfg_value=2.0):
    pass

# Now measure the pool.generate pipeline carefully.
text = "The quick brown fox jumps over the lazy dog."
N = 20  # trials
per_call_ms = []
for i in range(N):
    t0 = time.time()
    # Just exhaust until first chunk.
    for chunk in pool.generate(target_text=text, max_generate_length=1,
                               temperature=0.7, cfg_value=2.0):
        t1 = time.time()
        per_call_ms.append((t1 - t0) * 1000)
        break

print(f"T_first (max_gen=1 → single prefill): "
      f"p50={statistics.median(per_call_ms):.2f} ms  "
      f"mean={statistics.mean(per_call_ms):.2f} ms  "
      f"std={statistics.stdev(per_call_ms):.2f} ms")
print(f"min={min(per_call_ms):.2f}  max={max(per_call_ms):.2f}")

pool.stop()
