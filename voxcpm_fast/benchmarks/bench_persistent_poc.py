"""P2.1 persistent-kernel PoC benchmark.

Four phases:

  1. Warm-up        — 1 000 items serially, discard timings.
  2. Round-trip     — 10 000 items serially, one at a time, measure per-item
                      wall time in ``time.perf_counter()``. Print p50/p95/p99/
                      max.
  3. Throughput     — pre-fill 100 000 items into the queue, measure wall
                      clock, compute items/s.
  4. Correctness    — 10 000 items with random (a, b, c), verify every out
                      equals a + b*c. Zero mismatches required.

Run as:

    UV_CACHE_DIR=/workspace/Developments/VoxCPM2/.uv_cache uv run python \\
      /workspace/Developments/VoxCPM2/voxcpm_fast/benchmarks/bench_persistent_poc.py

Pass ``--iters N`` to override the round-trip sample count (useful under
``ncu``, which multiplies wall time).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Make voxcpm_fast/ importable regardless of where we're invoked from.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402  (must precede the C extension)

from persistent_kernel import PersistentKernel  # noqa: E402


def _print_env(args: argparse.Namespace) -> None:
    try:
        import flash_attn
        fa_v = flash_attn.__version__
    except Exception:
        fa_v = "n/a"
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = "n/a"
    try:
        smi = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"]
        ).decode().strip()
    except Exception:
        smi = "n/a"
    try:
        nvcc_v = subprocess.check_output(
            ["/usr/local/cuda-12.8/bin/nvcc", "--version"]
        ).decode().strip().splitlines()[-1]
    except Exception:
        nvcc_v = "n/a"
    try:
        props = torch.cuda.get_device_properties(0)
        gpu_props = f"{props.name} sm{props.major}{props.minor} "\
                    f"{props.total_memory//(1024**2)} MB {props.multi_processor_count} SM"
    except Exception:
        gpu_props = "n/a"

    print("=" * 78)
    print(f"date         : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"hostname     : {platform.node()}")
    print(f"gpu          : {smi}")
    print(f"gpu_props    : {gpu_props}")
    print(f"nvcc         : {nvcc_v}")
    print(f"torch        : {torch_v}")
    print(f"flash-attn   : {fa_v}")
    print(f"python       : {sys.version.split()[0]}")
    print(f"script       : bench_persistent_poc.py num_sms={args.num_sms} "
          f"iters={args.iters} throughput_n={args.throughput_n} "
          f"correct_n={args.correct_n}")
    print("=" * 78, flush=True)


def _percentiles(xs, ps):
    xs2 = sorted(xs)
    n = len(xs2)
    return [xs2[min(n - 1, int(round(p * (n - 1))))] for p in ps]


def phase_warmup(kernel: PersistentKernel, n: int) -> None:
    print(f"[warmup] {n} items serial...", flush=True)
    for i in range(n):
        evt, out = kernel.submit(a=i, b=i + 1, c=2)
        evt.wait(timeout=1.0)
    print("[warmup] done", flush=True)


def phase_round_trip(kernel: PersistentKernel, n: int) -> dict:
    print(f"[phase 2] serial round-trip, n={n}", flush=True)
    times_us = [0.0] * n
    mismatches = 0
    # Use fresh random input each iter so we also double-check the echo.
    # Keep values small to stay in int32 domain comfortably.
    rng = random.Random(0xBEEF)
    t_overall0 = time.perf_counter()
    for i in range(n):
        a = rng.randrange(-1000, 1000)
        b = rng.randrange(-1000, 1000)
        c = rng.randrange(-1000, 1000)
        t0 = time.perf_counter()
        evt, out = kernel.submit(a, b, c)
        evt.wait(timeout=1.0)
        res = out[0]
        t1 = time.perf_counter()
        times_us[i] = (t1 - t0) * 1e6
        if res != a + b * c:
            mismatches += 1
    t_overall1 = time.perf_counter()

    p50, p95, p99, p999 = _percentiles(times_us, [0.5, 0.95, 0.99, 0.999])
    maxv = max(times_us)
    minv = min(times_us)
    mean = statistics.mean(times_us)
    stdev = statistics.pstdev(times_us)

    print(f"[phase 2] n={n} wall={1e3*(t_overall1-t_overall0):.2f} ms")
    print(f"[phase 2] round-trip µs: "
          f"p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} p99.9={p999:.2f} "
          f"min={minv:.2f} max={maxv:.2f} mean={mean:.2f} stdev={stdev:.2f}")
    if mismatches:
        print(f"[phase 2] !!! mismatches={mismatches}")
    return dict(n=n, p50=p50, p95=p95, p99=p99, p999=p999,
                min=minv, max=maxv, mean=mean, stdev=stdev,
                mismatches=mismatches,
                wall_ms=(t_overall1 - t_overall0) * 1e3)


def phase_throughput(kernel: PersistentKernel, n: int) -> dict:
    """Measure GPU-side sustained throughput.

    We use the C-loop ``burst_submit`` entrypoint so the host submit path
    doesn't cap the measurement at Python/ctypes speed. The throughput we
    report is the rate at which the dispatcher drains the pinned-ring and
    workers publish completions — the target-relevant number for the real
    megakernel.
    """
    print(f"[phase 3] sustained throughput, n={n}", flush=True)
    ring_capacity = kernel._doorbell_ring_capacity
    queue_capacity = kernel._queue_capacity

    # Pre-fill per-item args on the host.
    a_all = torch.arange(n, dtype=torch.int32)
    b_all = a_all + 1
    c_all = torch.full((n,), 2, dtype=torch.int32)

    # We push in bursts then drain completions before we'd overwrite a
    # still-in-flight slot. The done[] array is the *same size* as the HBM
    # queue (4096 slots); once the host tries to resubmit an item whose
    # slot hasn't yet fired its done flag, the burst_submit routine zeros
    # that flag as part of publishing the next lap, so we lose the signal
    # for the pre-wrap item. The window we enforce is therefore
    # ``queue_capacity - burst`` so the *next* burst never overlaps
    # still-outstanding slots.
    burst = 256
    window = queue_capacity - burst

    # Work in the global sequence space. The first item in this phase has
    # sequence == `start_seq`, the n-th has `start_seq + n - 1`.
    start_seq = kernel._next_tail
    t0 = time.perf_counter()
    pos = 0           # count of items we've pushed in this phase
    completed = 0     # count of items we've seen done for in this phase
    while pos < n:
        m = min(burst, n - pos)

        # Drain enough completions BEFORE submitting the next burst to
        # leave a clean window.
        while pos + m - completed > window:
            oldest_seq = start_seq + completed
            oldest_idx = oldest_seq - 1
            oldest_slot = oldest_idx & (queue_capacity - 1)
            flag_addr = (kernel._done_addr
                         + oldest_slot * kernel._done_stride
                         + kernel._done_flag_off)
            expected = oldest_seq & 0xFFFFFFFF
            deadline = time.perf_counter() + 5.0
            while ctypes.c_uint32.from_address(flag_addr).value != expected:
                if time.perf_counter() > deadline:
                    raise TimeoutError(
                        f"drain stall: phase-item={completed} "
                        f"seq={oldest_seq} slot={oldest_slot} "
                        f"expected={expected} "
                        f"got={ctypes.c_uint32.from_address(flag_addr).value} "
                        f"pushed={pos}"
                    )
            completed += 1

        kernel.burst_submit(
            a_all[pos:pos + m], b_all[pos:pos + m], c_all[pos:pos + m]
        )
        pos += m

    # Final drain.
    while completed < pos:
        oldest_seq = start_seq + completed
        oldest_idx = oldest_seq - 1
        oldest_slot = oldest_idx & (queue_capacity - 1)
        flag_addr = (kernel._done_addr
                     + oldest_slot * kernel._done_stride
                     + kernel._done_flag_off)
        expected = oldest_seq & 0xFFFFFFFF
        deadline = time.perf_counter() + 5.0
        while ctypes.c_uint32.from_address(flag_addr).value != expected:
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"final drain stall: phase-item={completed} "
                    f"seq={oldest_seq} slot={oldest_slot} expected={expected}"
                )
        completed += 1
    t1 = time.perf_counter()

    wall = t1 - t0
    rate = n / wall
    print(f"[phase 3] wall={wall*1e3:.2f} ms  throughput={rate/1e6:.3f} M items/s "
          f"({rate:.0f}/s)")
    return dict(n=n, wall_ms=wall * 1e3, rate_per_s=rate,
                rate_m_per_s=rate / 1e6)


def phase_correctness(kernel: PersistentKernel, n: int) -> dict:
    print(f"[phase 4] correctness, n={n}", flush=True)
    rng = random.Random(42)
    mismatches = 0
    max_abs = 0
    for i in range(n):
        a = rng.randrange(-2000, 2000)
        b = rng.randrange(-2000, 2000)
        c = rng.randrange(-2000, 2000)
        evt, out = kernel.submit(a, b, c)
        evt.wait(timeout=1.0)
        res = out[0]
        exp = a + b * c
        if res != exp:
            mismatches += 1
            max_abs = max(max_abs, abs(res - exp))
    if mismatches == 0:
        print(f"[phase 4] all {n} ops produced exact expected output")
    else:
        print(f"[phase 4] !!! {mismatches}/{n} mismatches (max diff {max_abs})")
    return dict(n=n, mismatches=mismatches)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-sms", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=10000,
                    help="phase 2 round-trip samples")
    ap.add_argument("--throughput-n", type=int, default=100_000)
    ap.add_argument("--correct-n", type=int, default=10_000)
    args = ap.parse_args()

    _print_env(args)

    kernel = PersistentKernel(num_sms=args.num_sms)
    kernel.start()
    try:
        phase_warmup(kernel, args.warmup)
        rt = phase_round_trip(kernel, args.iters)
        tp = phase_throughput(kernel, args.throughput_n)
        cc = phase_correctness(kernel, args.correct_n)
    finally:
        kernel.stop()

    print("=" * 78)
    print("SUMMARY")
    print(f"round_trip_p50 = {rt['p50']:.2f} us")
    print(f"round_trip_p95 = {rt['p95']:.2f} us")
    print(f"round_trip_p99 = {rt['p99']:.2f} us")
    print(f"round_trip_max = {rt['max']:.2f} us")
    print(f"round_trip_mean = {rt['mean']:.2f} us (stdev {rt['stdev']:.2f})")
    print(f"sustained_throughput = {tp['rate_m_per_s']:.3f} M items/s")
    print(f"correctness_mismatches = {cc['mismatches']} / {cc['n']}")
    print("=" * 78)

    if cc['mismatches'] != 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
