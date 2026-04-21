"""Profile megakernel overhead: isolate grid.sync + cooperative-launch cost."""
from __future__ import annotations
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))

import mk_dit_prefill_ext as mk_ext


def _time(fn, warmup=30, iters=1000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))


def main():
    def pct(xs, p): return xs[int(round(p / 100.0 * (len(xs) - 1)))]

    for nb in [64, 128, 170, 256, 340]:
        def noop(nb=nb): return mk_ext.noop(nb)
        t = _time(noop)
        print(f"coop launch + 1 grid.sync, num_blocks={nb:3d} : "
              f"p50={pct(t,50)*1000:6.1f} µs  p99={pct(t,99)*1000:6.1f}")


if __name__ == "__main__":
    main()
