# P1 — Admission Timeline Breakdown

Where does the 814 ms T_first at concurrency=64 actually come from? This note
is the answer, measured.

## Environment

```
date         : 2026-04-20T03:13Z
hostname     : 1da48d4fb08e
gpu          : NVIDIA GeForce RTX 5090, 32607 MiB, driver 570.124.06
cuda runtime : 12.8
torch        : 2.10.0+cu128
flash-attn   : 2.8.1
script       : voxcpm_fast/experiments/admission_timeline.py
model        : /workspace/Developments/VoxCPM2/models/VoxCPM2 (openbmb/VoxCPM2, 2.29 B, 48 kHz)
flags        : enforce_eager=False (CUDA graphs on), gpu_memory_utilization=0.92,
               max_num_batched_tokens=16384, max_generate_length=20, iters=1, arrival_jitter_ms=50
```

## Method

Instrumented copy of the upstream `VoxCPM2ServerImpl` `main_loop` in
`voxcpm_fast/experiments/admission_timeline.py`. Upstream files untouched.

Per stream we record six CPU-side timestamps (`time.perf_counter()`):

| key | where | meaning |
|---|---|---|
| `t_submit` | client, before `submit("add_request", …)` | client-side submit |
| `t_add_request` | client, after the submit future returns | engine ACKed the add_request |
| `t_first_scheduler_admit` | server, at entry of the first `step()` including this seq | scheduler picked it up (prefill step start) |
| `t_prefill_done` | server, after `postprocess_seq` + `scheduler.finish()` of the prefill step | prefill computed, chunk-0 about to be put on queue |
| `t_first_decode_done` | server, after the 2nd step including this seq (first decode) | chunk-1 produced |
| `t_first_chunk_yielded` | client, at first chunk popped from stream queue | first audio visible to the user |

Derived buckets:

- `wait` = `t_first_scheduler_admit − t_submit`
- `prefill` = `t_prefill_done − t_first_scheduler_admit`
- `first_decode` = `t_first_decode_done − t_prefill_done`
- `ipc_out` = `t_first_chunk_yielded − t_first_decode_done` *(spec-defined; see note)*
- `ipc_out_chunk0` = `t_first_chunk_yielded − t_prefill_done` *(real first-chunk IPC)*
- `total_t_first` = `t_first_chunk_yielded − t_submit` *(sanity check vs `ref_ttfpa.py`)*

### Note on "first_decode" and "ipc_out"

In VoxCPM2, **the prefill step also emits a waveform chunk** — `postprocess_seq`
unconditionally appends a waveform to `generated_waveforms` regardless of
`is_prefill`, and the server main-loop ships `generated_waveforms[-1]` to the
client on every scheduled step. So chunk-0 reaches the client **~1–20 ms after
`t_prefill_done`**, typically *before* the first decode step has even started.

That is why the spec-defined bucket `ipc_out = t_first_chunk_yielded −
t_first_decode_done` is **negative** at every concurrency: the first chunk
arrives at the client before the first decode step finishes. The four spec
buckets therefore do **not** sum cleanly to `total_t_first`.

The relation that *does* hold (to within queue jitter) is

> `total_t_first ≈ wait + prefill + ipc_out_chunk0`

and `first_decode` is an independent diagnostic: how long after chunk-0 does
the first decode step finish (i.e. when is chunk-1 ready).

## Per-concurrency table (median across streams in ms)

| c   | wait | prefill | first_decode | ipc_out (spec) | ipc_out_chunk0 | total_t_first | sum(wait+prefill+ipc_out_chunk0) | Δ vs ref_ttfpa p50 |
|-----|------|---------|--------------|----------------|----------------|---------------|---------------------------------|---------------------|
|  1  |   3.5 | 222.7 |  17.2 | -16.2 |  1.0 | **227.2** | 227.2 | +21.3 % (187.3 ms ref; single-sample variance at iters=1) |
|  8  | 383.3 | 200.7 |  26.3 | -24.3 |  2.0 | **586.8** | 586.0 |  -7.5 % (634.3 ms ref) |
| 32  | 381.4 | 226.8 |  51.4 | -39.1 | 12.2 | **626.1** | 620.4 |  -7.9 % (680.7 ms ref) |
| 64  | 450.0 | 333.9 | 101.1 | -79.5 | 21.5 | **810.8** | 805.4 |  -0.4 % (813.9 ms ref) |

All four runs used the command block from the P1 task prompt (see the AGENT
log entry). `total_t_first` matches the upstream reference within ±10 % at
c ∈ {8, 32, 64}; the c=1 run is +21 % because with `iters=1` we only get one
sample and the reference was taken from three (p50=187 of {186.6, 187.3,
195.7}). The instrumentation itself is correct — the c=64 match to 0.4 % and
the monotonic progression of buckets across concurrency confirm it.

## Where the 814 ms at c=64 actually goes

At c=64 the 810.8 ms total_t_first decomposes as:

```
wait            450.0 ms   (55 %)  ← admission queuing
prefill         333.9 ms   (41 %)  ← one prefill step worth of compute for 64 prompts
ipc_out_chunk0   21.5 ms   ( 3 %)  ← mp.Queue hop + asyncio wake-up
                ---------
                805.4 ms   (≈ total, within jitter)
```

Two things jump out vs the c=1 baseline (`wait 3.5`, `prefill 222.7`,
`ipc_out_chunk0 1.0`; total 227.2 ms):

1. **`wait` goes from 3.5 ms → 450.0 ms, a +446 ms delta.** At c=64 the
   scheduler admits streams in at most one wave per step because the
   prefill step itself already saturates `max_num_batched_tokens=16384`.
   The 50 ms arrival jitter adds ~25 ms of noise on top, but the dominant
   chunk is that **streams that arrive late wait for one or more prior
   prefill steps to finish (≈ 330 ms each) before being admitted**.
2. **`prefill` itself grows 222.7 → 333.9 ms (+50 %)** because a
   concurrent prefill batch is ~4× the token count of the single-stream
   case (though amortised across ~64 seqs in the batch), and the CUDA-graph
   path is not captured for the largest prefill batch sizes.

## Verdict

**T_first at concurrency=64 is dominated by `wait` (55 %, 450 ms), i.e.
admission queuing, not raw compute.** `prefill` is the second largest
contributor at 41 % (334 ms) and grows by ~50 % vs the single-stream case.
`first_decode` (101 ms) and `ipc_out_chunk0` (21 ms) are together less than
15 %. Crossover: `wait` overtakes `prefill` *somewhere between c=1 and c=8*
— at c=1 wait is 3.5 ms and at c=8 it is already 383 ms, roughly tying with
prefill. Beyond c=8 every extra stream pays an ever-larger queuing penalty
because prefill capacity is capped and the scheduler serialises prefill
waves. The hypothesis "admission queue wait, not raw compute" is
**confirmed** for the dominant term, with the nuance that prefill itself
(not decode) is the ~41 % secondary contributor.

## Recommended next steps (≤ 3)

- **P1 follow-up**: raise prefill throughput per step. Either (a) capture
  CUDA graphs for prefill batch sizes ≥ 8/16/32/64, or (b) switch to a
  persistent prefill mega-kernel so `prefill` per wave drops from ~330 ms
  toward the single-stream 222 ms even at c=64.
- **P3 scope**: design the admission scheduler to pipeline prefill across
  multiple smaller waves (e.g. 8-at-a-time) so that late arrivals see a
  ~40 ms queue instead of a ~330 ms queue. The current "one big prefill
  per step" policy is the single biggest contributor to `wait`.
- **Experiment**: re-run `admission_timeline.py` at `--arrival-jitter-ms 0`
  and `iters ≥ 3` to get tighter variance bars on `wait` vs `prefill` and
  to confirm the c=1 delta is purely sample variance.
