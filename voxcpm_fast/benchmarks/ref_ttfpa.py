"""Baseline TTFPA benchmark against the *upstream* nanovllm_voxcpm stack.

TTFPA = time to first playable audio. We measure two things per stream:

1. ``T_first`` — wall time from ``submit`` to the first yielded chunk.
2. ``sustained_rtf`` — ``sum(gen[k] for k>0) / sum(dur[k] for k>0)``.
   Any underrun (``gen[k+1] > dur[k]``) is fatal and printed.

At concurrency > 1 the streams share one server process (one per GPU) — exactly
the production shape we need to beat. This is the *reference* TTFPA; our fast
server must match or beat it at every concurrency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np


DEFAULT_TEXT = (
    "In a quiet village at the edge of a great dark forest, there lived an old clockmaker "
    "whose hands, though wrinkled with age, could coax song from gears and silence from bells. "
    "Every morning before the sun climbed over the pine tops he walked to his workshop "
    "and opened the shutters to let the birds in. They sang to him while he worked; he pretended not to listen."
)


@dataclass
class StreamResult:
    stream_idx: int
    t_submit: float
    t_first_chunk: float | None = None
    t_last_chunk: float | None = None
    chunk_gens: list[float] = field(default_factory=list)  # ms
    chunk_durs: list[float] = field(default_factory=list)  # ms
    n_chunks: int = 0
    underruns: list[tuple[int, float, float]] = field(default_factory=list)
    error: str | None = None

    @property
    def t_first_chunk_ms(self) -> float | None:
        if self.t_first_chunk is None:
            return None
        return (self.t_first_chunk - self.t_submit) * 1000.0

    @property
    def sustained_rtf(self) -> float | None:
        if self.n_chunks <= 1:
            return None
        gen_total = sum(self.chunk_gens[1:])
        dur_total = sum(self.chunk_durs[1:])
        if dur_total <= 0:
            return None
        return gen_total / dur_total


def print_env(args: argparse.Namespace) -> None:
    try:
        git = subprocess.check_output(["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        git = "unknown"
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = "n/a"
    try:
        import flash_attn
        fa_v = flash_attn.__version__
    except Exception:
        fa_v = "n/a"
    print("=" * 72)
    print(f"date         : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"hostname     : {platform.node()}")
    print(f"script       : ref_ttfpa.py model={args.model} concurrency={args.concurrency} iters={args.iters}")
    print(f"torch        : {torch_v}")
    print(f"flash-attn   : {fa_v}")
    print(f"git          : {git}")
    try:
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]).decode().strip()
        print(f"gpu          : {smi}")
    except Exception:
        pass
    print("=" * 72, flush=True)


async def run_stream(pool, stream_idx: int, target_text: str, sample_rate: int, max_generate_length: int) -> StreamResult:
    res = StreamResult(stream_idx=stream_idx, t_submit=time.perf_counter())
    last_chunk_emit = None
    try:
        gen = pool.generate(target_text=target_text, max_generate_length=max_generate_length)
        async for chunk in gen:
            now = time.perf_counter()
            gen_ms = (now - (last_chunk_emit or res.t_submit)) * 1000.0
            dur_ms = (chunk.shape[0] / sample_rate) * 1000.0
            res.chunk_gens.append(gen_ms)
            res.chunk_durs.append(dur_ms)
            res.n_chunks += 1
            if res.t_first_chunk is None:
                res.t_first_chunk = now
            res.t_last_chunk = now
            if last_chunk_emit is not None and gen_ms > dur_ms:
                res.underruns.append((res.n_chunks - 1, gen_ms, dur_ms))
            last_chunk_emit = now
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
    return res


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to VoxCPM2 weights dir (config.json + safetensors + audiovae.pth)")
    ap.add_argument("--devices", type=int, nargs="+", default=[0])
    ap.add_argument("--concurrency", type=int, default=1, help="Concurrent streams to launch")
    ap.add_argument("--iters", type=int, default=1, help="Waves of streams (each wave = concurrency streams)")
    ap.add_argument("--arrival-jitter-ms", type=float, default=50.0, help="Randomize submit times within 0..N ms per wave to mimic sporadic arrivals")
    ap.add_argument("--max-generate-length", type=int, default=600)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--target-text-file", type=str, default=None, help="Optional path to a text file to speak")
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    print_env(args)

    if args.target_text_file:
        target_text = open(args.target_text_file).read().strip()
    else:
        target_text = DEFAULT_TEXT

    # Upstream import deferred so we print env info before any lazy CUDA init.
    from nanovllm_voxcpm import VoxCPM

    print(f"[ref_ttfpa] loading model from {args.model}", flush=True)
    pool = VoxCPM.from_pretrained(
        model=args.model,
        devices=args.devices,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    await pool.wait_for_ready()

    model_info = await pool.get_model_info()
    sample_rate = int(model_info["sample_rate"])
    print(f"[ref_ttfpa] ready; sample_rate={sample_rate}", flush=True)

    # Warmup — first request compiles triton/JIT paths and fills prefix cache.
    print("[ref_ttfpa] warmup run…", flush=True)
    warm = await run_stream(pool, -1, target_text, sample_rate, max_generate_length=min(args.max_generate_length, 50))
    print(f"[ref_ttfpa] warmup: T_first={warm.t_first_chunk_ms:.1f} ms, chunks={warm.n_chunks}", flush=True)

    all_results: list[StreamResult] = []
    for wave in range(args.iters):
        async def scheduled(idx: int) -> StreamResult:
            # sporadic arrivals: uniform 0..jitter
            if args.arrival_jitter_ms > 0:
                await asyncio.sleep(random.uniform(0, args.arrival_jitter_ms / 1000.0))
            return await run_stream(pool, idx, target_text, sample_rate, args.max_generate_length)

        tasks = [asyncio.create_task(scheduled(wave * args.concurrency + i)) for i in range(args.concurrency)]
        wave_results = await asyncio.gather(*tasks)
        all_results.extend(wave_results)
        print(f"[ref_ttfpa] wave {wave + 1}/{args.iters} done ({len(wave_results)} streams).", flush=True)

    await pool.stop()

    # Summarize
    ok = [r for r in all_results if r.error is None and r.t_first_chunk_ms is not None]
    failed = [r for r in all_results if r.error is not None]

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return float("nan")
        xs = sorted(xs)
        k = int(round(p / 100.0 * (len(xs) - 1)))
        return xs[k]

    t_first = [r.t_first_chunk_ms for r in ok if r.t_first_chunk_ms is not None]
    rtfs = [r.sustained_rtf for r in ok if r.sustained_rtf is not None]
    underruns = sum(len(r.underruns) for r in ok)
    streams_with_underrun = sum(1 for r in ok if r.underruns)

    print()
    print("=" * 72)
    print(f"concurrency          : {args.concurrency}  (×{args.iters} waves, {len(all_results)} streams total)")
    print(f"failures             : {len(failed)}")
    if t_first:
        print(f"T_first_chunk ms     : p50={pct(t_first, 50):.1f}  p95={pct(t_first, 95):.1f}  p99={pct(t_first, 99):.1f}  max={max(t_first):.1f}")
    if rtfs:
        print(f"sustained RTF (>0)   : p50={pct(rtfs, 50):.3f}  p95={pct(rtfs, 95):.3f}  p99={pct(rtfs, 99):.3f}  max={max(rtfs):.3f}")
    print(f"streams w/ underrun  : {streams_with_underrun} / {len(ok)}")
    print(f"total underruns      : {underruns}")
    print("=" * 72, flush=True)

    if args.output_json:
        payload = {
            "args": vars(args),
            "sample_rate": sample_rate,
            "streams": [
                {
                    "idx": r.stream_idx,
                    "t_first_chunk_ms": r.t_first_chunk_ms,
                    "n_chunks": r.n_chunks,
                    "sustained_rtf": r.sustained_rtf,
                    "underruns": r.underruns,
                    "error": r.error,
                }
                for r in all_results
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[ref_ttfpa] wrote {args.output_json}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
