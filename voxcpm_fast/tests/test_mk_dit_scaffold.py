"""P2.5.2 Step 1 — validate the cooperative-launch scaffolding.

Checks:
1. `mk_dit_prefill_ext.noop(num_blocks)` returns a (num_blocks * 2,)
   int32 tensor.
2. First half = blockIdx.x values written BEFORE grid.sync (0..N-1).
3. Second half = 2 * blockIdx.x values written AFTER grid.sync
   (0, 2, 4, ..., 2*(N-1)).

If any block skipped the grid.sync or the launch config fell back to
non-cooperative, the second half will be mismatched. So this is also
the smoke test for sm_120a cooperative launch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))

import mk_dit_prefill_ext as ext


def test_scaffold():
    # Keep it modest; cooperative launch is gated by
    # cudaOccupancyMaxActiveBlocksPerMultiprocessor * sm_count. On sm_120
    # a 32-thread trivial kernel easily fits ~170 blocks concurrently.
    N = 128
    out = ext.noop(N)
    assert out.shape == (N * 2,), out.shape
    assert out.dtype == torch.int32

    cpu = out.cpu().tolist()

    # First half: block id.
    for i in range(N):
        assert cpu[i] == i, f"pre-sync[{i}] = {cpu[i]}, expected {i}"

    # Second half: 2 * block id.
    for i in range(N):
        assert cpu[N + i] == 2 * i, f"post-sync[{i}] = {cpu[N + i]}, expected {2 * i}"

    print(f"mk_dit_prefill scaffold OK (num_blocks={N}, cooperative grid.sync verified)")


if __name__ == "__main__":
    test_scaffold()
