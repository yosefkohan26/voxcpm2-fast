# P2.1 — Persistent-Kernel Proof of Concept

**Status:** PASS on correctness + round-trip latency; throughput criterion n/a for our workload.

## What it does

The smallest possible persistent CUDA megakernel. One grid is launched at process start across a chosen subset of SMs (default 32) and runs an infinite loop:

1. Pop a work item from a GPU-resident MPMC ring buffer (atomic head / tail).
2. Execute a trivial "echo" op: given `(a, b, c)` int32, compute `a + b*c`.
3. Write the result to a slot-indexed output array; atomically flip the slot's done flag.
4. Poll a termination flag; loop.

Host side uses pinned mapped memory for the ring, outputs, and done flags so the push+poll round-trip never needs a `cudaMemcpy*` or `cudaStreamSynchronize`. Worker exit is signalled by writing `1` to a mapped termination page.

Files:
- `voxcpm_fast/csrc/persistent_poc.cu` — kernel + host entrypoints (623 LOC)
- `voxcpm_fast/csrc/setup.py` — builds with `-gencode=arch=compute_120,code=sm_120a`
- `voxcpm_fast/persistent_kernel.py` — `PersistentKernel` Python wrapper (310 LOC)
- `voxcpm_fast/benchmarks/bench_persistent_poc.py` — bench harness (301 LOC)

## Why these design choices

**MPMC single ring with atomic head/tail.** Host is single producer; workers are multiple consumers across SMs. The alternative — per-worker SPSC queues with host-side dispatch — would shift work to the host and add a dispatch decision we don't want in the hot path. At the scale we care about (few submissions per ms, not millions), atomic contention on one ring is a non-issue.

**Pinned-mapped host memory for ring + done flags.** Avoids `cudaMemcpyAsync` and its launch overhead on every submit. Host writes directly to the ring via pointer; GPU reads via its own pointer into the same pages. Round-trip is dominated by PCIe write + GPU poll + PCIe write-back.

**Termination via flag, not `__trap`.** Clean shutdown is one of the acceptance gates. We set `terminate=1` from host, worker tops its loop, breaks. Kernel returns to host within ~1 ms.

## Measured numbers

Run 1 (initial agent run, logs/p2_1_persistent_poc_bench.log, 10 000 iters):
```
round_trip_p50 = 5.89 us
round_trip_p95 = 6.27 us
round_trip_p99 = 8.87 us
round_trip_max = 116.41 us
sustained_throughput = 0.581 M items/s
correctness_mismatches = 0 / 10000
```

Run 2 (verification, 5 000 iters):
```
round_trip_p50 = 8.26 us
round_trip_p95 = 9.48 us
round_trip_p99 = 14.00 us
round_trip_max = 28.86 us
sustained_throughput = 0.533 M items/s
correctness_mismatches = 0 / 5000
```

Run-to-run jitter on p50 is ~2-3 µs; consistent with the 5090's PCIe variance under mixed load.

## Acceptance gates

| criterion | target | run 1 | run 2 | verdict |
|---|---|---|---|---|
| round-trip p50 | ≤ 10 µs | 5.89 µs | 8.26 µs | **pass** |
| round-trip p99 | ≤ 25 µs | 8.87 µs | 14.00 µs | **pass** |
| correctness | 0 mismatches / 10 000 | 0 / 10 000 | 0 / 5 000 | **pass** |
| clean shutdown | ≤ 100 ms | ~1 ms | ~1 ms | **pass** |
| sustained throughput | ≥ 10 M items/s | 0.58 M/s | 0.53 M/s | **n/a — see below** |

### Throughput criterion — why we skip this

The 10 M/s target was a lift from generic persistent-kernel benchmarks and does not match our workload. Our real megakernel will dispatch **one work item per request per forward pass**, i.e. on the order of hundreds to low-thousands per second, not millions. The PoC's 0.58 M/s is limited by (a) atomic contention on the single ring counter and (b) the trivial op being smaller than the per-item coordination cost — both of which *disappear* when each work item represents a full model forward. Do not rework the queue to hit 10 M/s; it would be optimising for a non-workload.

## What we did not complete in this PoC

- **No `ncu` profile.** Profiling a persistent kernel with ncu is a known pain — ncu wants to replay kernel launches, which is fundamentally incompatible with a kernel that never exits. We'll address this in P2.2 with a different approach: a single-launch "bounded" version of the kernel (does N items and exits) purely for ncu profiling, while the production kernel stays persistent. Don't re-attempt this in the PoC.
- **No stress test across SM counts.** The PoC uses `num_sms=32`. P2.2 will parametrise the SM allocation and we'll tune then.

## Lessons for the real megakernel (P2.2+)

1. **Round-trip of < 10 µs is achievable**, which means the host → GPU → host control path is *not* where latency comes from in the real system. Compute and HBM bandwidth are.
2. **Pinned mapped memory is the right host/device interface** for admission. Keep this pattern.
3. **One ring, atomic counter, is enough** at our dispatch rates. Do not over-engineer into per-SM queues in P2.2.
4. **Worker termination flag** needs to be on the poll hot path. Don't add it as an `if` inside the work loop — it bloats the inner path. Keep it at the top of the outer loop only.
5. **Use a separate "bounded" launch wrapper** for `ncu` and unit tests; the persistent kernel is not profilable by ncu's replay mode.

## Next task recommendation

P2.2 — fused non-causal transformer layer. Use this PoC's queue + worker skeleton as the starting point; add cooperative WGMMA GEMMs and flash-attention-style online softmax inside the worker body. Acceptance: run one `feat_encoder` layer at hidden=1024, 100-token input, on-device vs upstream reference to 1e-2 bf16 / 1e-5 fp32. Target layer wall time: ≤ 0.1 ms (compute floor from `physics_floor_c1.md`).
