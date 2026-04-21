"""P1-admission-timeline — bucketed breakdown of T_first at concurrency.

Records, per stream, the following CPU-side wall-clock timestamps
(all via ``time.perf_counter()``):

  * ``t_submit``                — client → engine (before add_request cmd).
  * ``t_add_request``           — engine acknowledged the add_request call.
  * ``t_first_scheduler_admit`` — entry of the engine.step() that first included
                                  this seq_id (the prefill step).
  * ``t_prefill_done``          — end of that prefill step on the server side,
                                  after the chunk-0 stream message has been put.
  * ``t_first_decode_done``     — end of the first decode step that included
                                  this seq_id (i.e. chunk-1 produced).
  * ``t_first_chunk_yielded``   — client received the first stream chunk.

Derived buckets (ms):
  * ``wait``         = t_first_scheduler_admit − t_submit
  * ``prefill``      = t_prefill_done           − t_first_scheduler_admit
  * ``first_decode`` = t_first_decode_done      − t_prefill_done
  * ``ipc_out``      = t_first_chunk_yielded    − t_first_decode_done
  * ``total_t_first``= t_first_chunk_yielded    − t_submit  (matches ref_ttfpa)

NOTE: In VoxCPM2, the prefill step also emits a waveform chunk (the "chunk-0"
that the client sees first). So in practice ``t_first_chunk_yielded`` lands
*shortly after* ``t_prefill_done`` (reaching the client through the mp queue)
and therefore ``first_decode`` + ``ipc_out`` sum to the time from chunk-0
reaching the client to the first decode step finishing — i.e. ``ipc_out`` is
negative for the chunk-0 path, and the four buckets do NOT sum to
``total_t_first``. This is reported honestly: we also expose
``ipc_out_chunk0 = t_first_chunk_yielded − t_prefill_done`` so that
``wait + prefill + ipc_out_chunk0 ≈ total_t_first``.

Not editable: ``nanovllm-voxcpm/``. We spawn our own server process with an
instrumented copy of ``VoxCPM2ServerImpl.main_loop`` that emits timestamp
messages through the same ``queue_out`` the client is already reading from.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from queue import Empty
from typing import Any

import numpy as np
import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# Benchmark text — same as ref_ttfpa.py so totals are comparable.
# ---------------------------------------------------------------------------
DEFAULT_TEXT = (
    "In a quiet village at the edge of a great dark forest, there lived an old clockmaker "
    "whose hands, though wrinkled with age, could coax song from gears and silence from bells. "
    "Every morning before the sun climbed over the pine tops he walked to his workshop "
    "and opened the shutters to let the birds in. They sang to him while he worked; he pretended not to listen."
)


# ---------------------------------------------------------------------------
# Server-side main loop — copy of upstream main_loop with step() instrumented.
# This function runs inside the spawned server process. It imports upstream
# classes but does NOT modify them.
# ---------------------------------------------------------------------------
def instrumented_main_loop(queue_in: mp.Queue, queue_out: mp.Queue, args: tuple, kwargs: dict) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        coalesce_ms = float(os.environ.get("NANOVLLM_QUEUE_COALESCE_MS", "2"))
    except ValueError:
        coalesce_ms = 2.0
    if coalesce_ms > 0:
        coalesce_ms = min(coalesce_ms, 50.0)

    try:
        # Imports deferred so CUDA init happens inside the subprocess.
        from nanovllm_voxcpm.models.voxcpm2.server import VoxCPM2ServerImpl
        srv = VoxCPM2ServerImpl(*args, **kwargs)
    except Exception:
        queue_out.put({"type": "init_error", "error": traceback.format_exc()})
        return
    else:
        queue_out.put({"type": "init_ok"})

    # Per-seq book-keeping — how many times we've already seen this seq_id in a
    # step() output. 0 = not yet, 1 = just finished prefill, 2+ = decode step.
    steps_seen: dict[str, int] = {}
    states = {"is_stoped": False}

    def method_call(cmd: dict) -> dict:
        opid = cmd.get("id", "")
        try:
            method_name = cmd["type"]
            fn_args = cmd["args"]
            fn_kwargs = cmd["kwargs"]
            if method_name == "stop":
                states["is_stoped"] = True
                return {"type": "response", "id": opid, "data": None}
            ret = getattr(srv, method_name)(*fn_args, **fn_kwargs)
            return {"type": "response", "id": opid, "data": ret}
        except Exception:
            return {"type": "error", "id": opid, "error": traceback.format_exc()}

    def do_one_instrumented_step() -> None:
        """Mirror of the inner body of upstream main_loop's step path, with
        per-seq timestamps emitted into queue_out alongside the stream chunk."""
        t_step_enter = time.perf_counter()
        # Call engine.step() directly so we can tell which seqs were scheduled
        # and whether this was a prefill step.
        seqs, is_prefill = srv.llm.scheduler.schedule()
        if not seqs:
            return

        # Build tasks + run, exactly as LLMEngineBase.step() does, but
        # re-implemented so we can time the call itself.
        runner_tasks = [srv.llm.preprocess_seq(seq, is_prefill) for seq in seqs]
        outputs = srv.llm.model_runner.call("run", runner_tasks, is_prefill)

        for seq, output in zip(seqs, outputs):
            srv.llm.postprocess_seq(seq, output, is_prefill)

        # Finalise stoped seqs so seq.is_finished is True when we read it below
        # — match upstream LLMEngineBase.step() ordering.
        for seq in seqs:
            if seq.stoped:
                srv.llm.scheduler.finish(seq)

        # At this point chunks are ready to be emitted.
        t_step_done = time.perf_counter()

        for seq in seqs:
            sid = seq.seq_id
            seen = steps_seen.get(sid, 0)
            latest_waveform = seq.custom_payload.generated_waveforms[-1]

            if seen == 0:
                # This step is the first time the seq was scheduled → prefill.
                queue_out.put(
                    {
                        "type": "timestamp",
                        "id": sid,
                        "key": "first_scheduler_admit",
                        "ts": t_step_enter,
                    }
                )
                queue_out.put(
                    {
                        "type": "timestamp",
                        "id": sid,
                        "key": "prefill_done",
                        "ts": t_step_done,
                    }
                )
            elif seen == 1:
                # First decode step for this seq.
                queue_out.put(
                    {
                        "type": "timestamp",
                        "id": sid,
                        "key": "first_decode_done",
                        "ts": t_step_done,
                    }
                )

            steps_seen[sid] = seen + 1

            queue_out.put({"type": "stream", "id": sid, "data": latest_waveform})
            if seq.is_finished:
                queue_out.put({"type": "stream", "id": sid, "data": None})
                steps_seen.pop(sid, None)

    while not states["is_stoped"]:
        cmd = queue_in.get()
        queue_out.put(method_call(cmd))

        if coalesce_ms > 0:
            deadline = time.perf_counter() + (coalesce_ms / 1000.0)
            while not states["is_stoped"]:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    cmd = queue_in.get(timeout=remaining)
                except Empty:
                    break
                queue_out.put(method_call(cmd))

        while not srv.is_finished() and not states["is_stoped"]:
            while not states["is_stoped"]:
                try:
                    cmd = queue_in.get_nowait()
                    queue_out.put(method_call(cmd))
                except Empty:
                    break
            if states["is_stoped"]:
                break
            do_one_instrumented_step()


# ---------------------------------------------------------------------------
# Client-side wrapper. Spawns a single server process and drives requests.
# ---------------------------------------------------------------------------
@dataclass
class StreamTimeline:
    stream_idx: int
    seq_id: str
    t_submit: float
    t_add_request: float | None = None
    t_first_scheduler_admit: float | None = None
    t_prefill_done: float | None = None
    t_first_decode_done: float | None = None
    t_first_chunk_yielded: float | None = None
    n_chunks: int = 0
    error: str | None = None

    def buckets_ms(self) -> dict[str, float | None]:
        def _ms(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return (a - b) * 1000.0

        total = _ms(self.t_first_chunk_yielded, self.t_submit)
        return {
            "wait": _ms(self.t_first_scheduler_admit, self.t_submit),
            "prefill": _ms(self.t_prefill_done, self.t_first_scheduler_admit),
            "first_decode": _ms(self.t_first_decode_done, self.t_prefill_done),
            "ipc_out": _ms(self.t_first_chunk_yielded, self.t_first_decode_done),
            "ipc_out_chunk0": _ms(self.t_first_chunk_yielded, self.t_prefill_done),
            "total_t_first": total,
        }


class InstrumentedServer:
    """Thin re-implementation of AsyncVoxCPM2Server that uses our own main_loop
    and exposes raw timestamp messages to the client."""

    def __init__(
        self,
        *,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: list[int] | None = None,
        lora_config: Any | None = None,
    ) -> None:
        devices = devices or [0]
        ctx = mp.get_context("spawn")
        self.queue_in: mp.Queue = ctx.Queue()
        self.queue_out: mp.Queue = ctx.Queue()
        self.process = ctx.Process(
            target=instrumented_main_loop,
            args=(
                self.queue_in,
                self.queue_out,
                (
                    model_path,
                    inference_timesteps,
                    max_num_batched_tokens,
                    max_num_seqs,
                    max_model_len,
                    gpu_memory_utilization,
                    enforce_eager,
                    devices,
                    lora_config,
                ),
                {},
            ),
            daemon=True,
        )
        self.process.start()

        loop = asyncio.get_running_loop()
        self._init_fut: asyncio.Future[None] = loop.create_future()
        self.recv_task = asyncio.create_task(self._recv_loop())
        self.op_table: dict[str, asyncio.Future[Any]] = {}
        self.stream_table: dict[str, asyncio.Queue] = {}
        self.timestamp_callbacks: dict[str, "asyncio.Future[None] | asyncio.Queue"] = {}
        # Per-seq dict of pending timestamps. Keyed by seq_id → {key: ts}.
        self.seq_timestamps: dict[str, dict[str, float]] = {}

    async def _recv_loop(self) -> None:
        try:
            while True:
                try:
                    res = await asyncio.to_thread(self.queue_out.get, timeout=1.0)
                except Empty:
                    continue

                t = res.get("type")
                if t == "init_ok":
                    if not self._init_fut.done():
                        self._init_fut.set_result(None)
                    continue
                if t == "init_error":
                    if not self._init_fut.done():
                        self._init_fut.set_exception(RuntimeError(res.get("error", "init error")))
                    continue

                if t == "stream":
                    sid = res["id"]
                    if sid in self.stream_table:
                        await self.stream_table[sid].put(res["data"])
                elif t == "timestamp":
                    sid = res["id"]
                    self.seq_timestamps.setdefault(sid, {})[res["key"]] = res["ts"]
                elif t == "response":
                    opid = res["id"]
                    if opid in self.op_table:
                        self.op_table[opid].set_result(res.get("data"))
                        del self.op_table[opid]
                elif t == "error":
                    opid = res["id"]
                    if opid in self.op_table:
                        self.op_table[opid].set_exception(RuntimeError(res.get("error", "error")))
                        del self.op_table[opid]
        except asyncio.CancelledError:
            return

    async def wait_for_ready(self) -> None:
        while not self._init_fut.done():
            if self.process.exitcode is not None:
                if not self._init_fut.done():
                    self._init_fut.set_exception(
                        RuntimeError(f"server process exited: exitcode={self.process.exitcode}")
                    )
                break
            await asyncio.sleep(0.05)
        await self._init_fut

    async def submit(self, cmd: str, *args: object, **kwargs: object) -> Any:
        op_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self.op_table[op_id] = fut
        await asyncio.to_thread(
            self.queue_in.put,
            {"id": op_id, "type": cmd, "args": list(args), "kwargs": dict(kwargs)},
        )
        return await fut

    async def get_model_info(self) -> dict:
        return await self.submit("get_model_info")

    async def stop(self) -> None:
        if self.process.exitcode is None and self.process.is_alive():
            try:
                await asyncio.wait_for(self.submit("stop"), timeout=2.0)
            except Exception:
                pass
        self.recv_task.cancel()
        try:
            await self.recv_task
        except asyncio.CancelledError:
            pass
        if self.process.is_alive():
            await asyncio.to_thread(self.process.join, 5.0)
        if self.process.is_alive():
            self.process.terminate()
            await asyncio.to_thread(self.process.join, 2.0)
        if self.process.is_alive():
            kill = getattr(self.process, "kill", None)
            if callable(kill):
                kill()
                await asyncio.to_thread(self.process.join, 2.0)

    async def run_stream(self, stream_idx: int, target_text: str, max_generate_length: int) -> StreamTimeline:
        seq_id = uuid.uuid4().hex
        self.stream_table[seq_id] = asyncio.Queue()
        tl = StreamTimeline(stream_idx=stream_idx, seq_id=seq_id, t_submit=time.perf_counter())
        try:
            await self.submit(
                "add_request",
                seq_id,
                target_text,
                None,  # prompt_latents
                "",    # prompt_text
                max_generate_length,
                1.0,   # temperature
                2.0,   # cfg_value
                None,  # ref_audio_latents
                None,  # lora_name
            )
            tl.t_add_request = time.perf_counter()
            while True:
                data = await self.stream_table[seq_id].get()
                if data is None:
                    break
                tl.n_chunks += 1
                if tl.t_first_chunk_yielded is None:
                    tl.t_first_chunk_yielded = time.perf_counter()
        except Exception as e:
            tl.error = f"{type(e).__name__}: {e}"
        finally:
            # Pull per-seq timestamps the server emitted.
            ts = self.seq_timestamps.get(seq_id, {})
            tl.t_first_scheduler_admit = ts.get("first_scheduler_admit")
            tl.t_prefill_done = ts.get("prefill_done")
            tl.t_first_decode_done = ts.get("first_decode_done")
            self.stream_table.pop(seq_id, None)
            self.seq_timestamps.pop(seq_id, None)
        return tl


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------
def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(round(p / 100.0 * (len(xs) - 1)))
    return xs[k]


def print_env(args: argparse.Namespace) -> None:
    try:
        git = subprocess.check_output(
            ["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
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
    print(f"script       : admission_timeline.py model={args.model} concurrency={args.concurrency} iters={args.iters}")
    print(f"torch        : {torch_v}")
    print(f"flash-attn   : {fa_v}")
    print(f"git          : {git}")
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ).decode().strip()
        print(f"gpu          : {smi}")
    except Exception:
        pass
    print("=" * 72, flush=True)


async def amain() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--devices", type=int, nargs="+", default=[0])
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--arrival-jitter-ms", type=float, default=50.0)
    ap.add_argument("--max-generate-length", type=int, default=20)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--target-text-file", type=str, default=None)
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    print_env(args)

    if args.target_text_file:
        target_text = open(args.target_text_file).read().strip()
    else:
        target_text = DEFAULT_TEXT

    print(f"[admission_timeline] loading model from {args.model}", flush=True)
    server = InstrumentedServer(
        model_path=args.model,
        inference_timesteps=10,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        devices=args.devices,
    )
    await server.wait_for_ready()

    info = await server.get_model_info()
    sample_rate = int(info["sample_rate"])
    print(f"[admission_timeline] ready; sample_rate={sample_rate}", flush=True)

    # Warmup — same as ref_ttfpa.py so first-call JIT/compile costs don't
    # contaminate the measurement.
    print("[admission_timeline] warmup run...", flush=True)
    warm = await server.run_stream(-1, target_text, max_generate_length=min(args.max_generate_length, 20))
    if warm.error is not None:
        print(f"[admission_timeline] warmup FAILED: {warm.error}", flush=True)
    else:
        b = warm.buckets_ms()
        print(
            f"[admission_timeline] warmup: total_t_first={b['total_t_first']:.1f} ms "
            f"(chunks={warm.n_chunks})",
            flush=True,
        )

    all_results: list[StreamTimeline] = []
    for wave in range(args.iters):
        async def scheduled(idx: int) -> StreamTimeline:
            if args.arrival_jitter_ms > 0:
                await asyncio.sleep(random.uniform(0, args.arrival_jitter_ms / 1000.0))
            return await server.run_stream(idx, target_text, args.max_generate_length)

        tasks = [
            asyncio.create_task(scheduled(wave * args.concurrency + i))
            for i in range(args.concurrency)
        ]
        wave_results = await asyncio.gather(*tasks)
        all_results.extend(wave_results)
        print(f"[admission_timeline] wave {wave + 1}/{args.iters} done.", flush=True)

    await server.stop()

    ok = [r for r in all_results if r.error is None and r.t_first_chunk_yielded is not None]
    failed = [r for r in all_results if r.error is not None]

    # Completeness check (interior timestamps).
    interior_keys = ("t_first_scheduler_admit", "t_prefill_done", "t_first_decode_done")
    complete = [r for r in ok if all(getattr(r, k) is not None for k in interior_keys)]

    # Per-bucket medians.
    by_bucket: dict[str, list[float]] = {
        "wait": [], "prefill": [], "first_decode": [], "ipc_out": [],
        "ipc_out_chunk0": [], "total_t_first": [],
    }
    for r in ok:
        bk = r.buckets_ms()
        for k, v in bk.items():
            if v is not None:
                by_bucket[k].append(v)

    def med(xs: list[float]) -> float:
        return statistics.median(xs) if xs else float("nan")

    print()
    print("=" * 72)
    print(f"concurrency           : {args.concurrency}  (×{args.iters} waves, {len(all_results)} streams total)")
    print(f"failures              : {len(failed)}")
    print(f"streams w/ full interior timestamps: {len(complete)} / {len(ok)}")
    if by_bucket["total_t_first"]:
        print(
            f"wait (p50)            : {med(by_bucket['wait']):.1f} ms"
        )
        print(
            f"prefill (p50)         : {med(by_bucket['prefill']):.1f} ms"
        )
        print(
            f"first_decode (p50)    : {med(by_bucket['first_decode']):.1f} ms"
        )
        print(
            f"ipc_out (p50)         : {med(by_bucket['ipc_out']):.1f} ms  (spec; negative when chunk-0 arrives before 1st decode finishes)"
        )
        print(
            f"ipc_out_chunk0 (p50)  : {med(by_bucket['ipc_out_chunk0']):.1f} ms  (real chunk-0 IPC)"
        )
        t = by_bucket["total_t_first"]
        print(
            f"total_t_first         : p50={pct(t, 50):.1f}  p95={pct(t, 95):.1f}  p99={pct(t, 99):.1f}  max={max(t):.1f}"
        )
    print("=" * 72, flush=True)

    if args.output_json:
        payload = {
            "args": vars(args),
            "sample_rate": sample_rate,
            "n_streams": len(all_results),
            "n_failed": len(failed),
            "n_complete_interior": len(complete),
            "streams": [
                {
                    "idx": r.stream_idx,
                    "seq_id": r.seq_id,
                    "t_submit": r.t_submit,
                    "t_add_request": r.t_add_request,
                    "t_first_scheduler_admit": r.t_first_scheduler_admit,
                    "t_prefill_done": r.t_prefill_done,
                    "t_first_decode_done": r.t_first_decode_done,
                    "t_first_chunk_yielded": r.t_first_chunk_yielded,
                    "n_chunks": r.n_chunks,
                    "buckets_ms": r.buckets_ms(),
                    "error": r.error,
                }
                for r in all_results
            ],
            "summary": {
                k: {
                    "p50": pct(v, 50) if v else None,
                    "p95": pct(v, 95) if v else None,
                    "p99": pct(v, 99) if v else None,
                    "max": max(v) if v else None,
                    "n": len(v),
                }
                for k, v in by_bucket.items()
            },
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[admission_timeline] wrote {args.output_json}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
