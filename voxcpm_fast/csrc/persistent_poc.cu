// voxcpm_fast/csrc/persistent_poc.cu
//
// P2.1 persistent-kernel PoC.
//
// Design
// ------
// The persistent kernel launches ``num_sms + 1`` blocks:
//
//   * Block 0 is the **dispatcher** — it owns the single zero-copy doorbell
//     in pinned host memory. It polls the doorbell over PCIe and enqueues
//     published work into an **HBM-resident** ring buffer.
//   * Blocks 1..num_sms are **workers** — they poll the HBM ring (fast HBM
//     reads, no PCIe), execute the echo op, and write results + flags back
//     to mapped host memory (one PCIe write per completion, unavoidable).
//
// Why split dispatcher vs workers?
//   The "naive" design put the whole queue in mapped pinned memory so
//   workers could see host writes directly. On a PCIe-attached 5090 that
//   path saturated the PCIe root complex: N workers × 2 PCIe reads per
//   poll-iteration blew up the transaction rate and pushed p50 round-trip
//   to ~60 µs at num_sms=32 (vs 7 µs at num_sms=1). Moving the seq/head
//   reads onto HBM by funneling them through one dispatcher block brings
//   the round-trip back to the ~single-worker floor regardless of worker
//   count.
//
// Zero-copy push path (host -> dispatcher):
//   A single ``Doorbell`` in mapped pinned memory. Host writes
//   ``{a, b, c, idx}`` then releases ``doorbell.seq = idx + 1``. The
//   dispatcher block's thread 0 does ``ld.global.acquire.sys.u32`` on
//   ``doorbell.seq`` and wakes on the new value, copies the payload into
//   the HBM ring, stores ``doorbell.ack = idx + 1`` (so the host can tell
//   the dispatcher picked it up, though the round-trip here is just
//   "done flag flipped"), then writes ``hbm_queue.seq[pos] = idx + 1``.
//
// HBM queue (dispatcher -> workers):
//   Standard Vyukov MPMC with a 4096-slot ring. Head is an HBM atomic that
//   all workers atomicCAS against; per-slot seq disambiguates empty/full/
//   freed. HBM reads are ~80 ns local, not PCIe latency, so workers can
//   spin without saturating anything.
//
// Completion (worker -> host):
//   Each worker writes ``done[slot].out`` then ``st.global.release.sys.u32
//   done[slot].flag = idx + 1`` into mapped pinned memory. Host polls
//   ``done[slot].flag``.
//
// Target: host-timed round-trip p50 ≤ 10 µs, p99 ≤ 25 µs, at num_sms=32.
//
// Compiled for sm_120a (RTX 5090 Blackwell). This PoC doesn't exercise
// WGMMA/TMA, but the compile target matches the real megakernels.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace voxcpm_fast {

static constexpr int QUEUE_CAPACITY = 4096;   // must be power of two
static constexpr int THREADS_PER_BLOCK = 32;  // one warp per block

struct alignas(16) WorkItem {
    int32_t a;
    int32_t b;
    int32_t c;
    int32_t idx;
};

struct alignas(16) DoneSlot {
    // flag == idx + 1 when the worker has published the result.
    uint32_t flag;
    int32_t  out;
    uint32_t _pad0;
    uint32_t _pad1;
};

// Host-visible ring of doorbell slots. The host ("producer") owns `tail`;
// the dispatcher ("consumer") owns `head`. Host writes payloads to
// ``slots[tail & mask]`` then release-stores the new tail. Dispatcher spins
// on tail > head, drains every new slot, bumps head.
//
// Keeping a ring (not a single slot) means the host can enqueue multiple
// items without waiting for the dispatcher to ack each one — throughput
// stops being gated by single-doorbell round-trip.
static constexpr int DOORBELL_RING_CAPACITY = 4096;  // power of two

struct alignas(64) Doorbell {
    WorkItem slots[DOORBELL_RING_CAPACITY];  // 64 KB
    alignas(64) uint32_t tail;    // host-only writer
    uint32_t _pad0[15];
    alignas(64) uint32_t head;    // dispatcher-only writer (diagnostic)
    uint32_t _pad1[15];
};

// HBM-resident ring buffer. Workers use this; dispatcher fills it.
struct alignas(128) HBMQueue {
    uint32_t seq[QUEUE_CAPACITY];   // per-slot seq (Vyukov)
    WorkItem slot[QUEUE_CAPACITY];
    // `turn` is the next slot index to be served. Workers each act only when
    // `turn % num_workers == worker_id`. The chosen worker advances `turn`.
    // Dispatcher only inspects it indirectly (via the seq lap values).
    alignas(128) uint32_t turn;
    alignas(128) uint32_t head;     // diagnostic
    alignas(128) uint32_t tail;     // diagnostic
    // Workers read this (HBM) instead of the pinned `ctrl->terminate` so
    // that 32 workers don't each issue a PCIe read per iteration. The
    // dispatcher mirrors the pinned terminate into here when it observes
    // the flip.
    alignas(128) uint32_t terminate_hbm;
};

struct alignas(64) ControlBlock {
    uint32_t terminate;
    uint32_t _pad[15];
};


// --------------------------------------------------------------------------
// Device helpers.
// --------------------------------------------------------------------------

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(p), "r"(v));
}

__device__ __forceinline__ uint32_t ld_acquire_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
    asm volatile("st.global.release.gpu.u32 [%0], %1;" :: "l"(p), "r"(v));
}


// --------------------------------------------------------------------------
// Dispatcher block: one block, one thread, owns the PCIe doorbell.
// --------------------------------------------------------------------------
//
// Responsibility: read mapped-pinned doorbell, publish into HBM queue.
//
// Host contract: host picks a monotonically increasing 32-bit `idx`, writes
// the payload, then releases `doorbell.seq = idx + 1`. Dispatcher:
//   1. acquire-loads doorbell.seq; if == expected, proceed.
//   2. reads doorbell.payload (also in mapped pinned memory — one PCIe read).
//   3. writes into HBM slot[idx % CAP], then release-stores seq[idx%CAP]=idx+1.
//   4. bumps `tail` (diagnostic).
//   5. writes `doorbell.ack = idx + 1` (not strictly required; useful for
//      host-side diagnostics).

__device__ void dispatcher_loop(Doorbell* doorbell,
                                HBMQueue* q,
                                const ControlBlock* ctrl) {
    if (threadIdx.x != 0) {
        while (ld_acquire_gpu(&q->terminate_hbm) == 0u) {
            __nanosleep(1000);
        }
        return;
    }

    uint32_t my_head = 0;  // dispatcher-local copy of head; host never touches

    while (true) {
        // Check pinned terminate (one PCIe read, owned by this thread).
        if (ld_acquire_sys(&ctrl->terminate) != 0u) {
            st_release_gpu(&q->terminate_hbm, 1u);
            return;
        }

        // Read host-published tail. If new items are available, drain them.
        uint32_t tail = ld_acquire_sys(&doorbell->tail);
        if (tail == my_head) {
            // No new items. Continue spinning on tail.
            continue;
        }

        // Drain all newly-published items in one dispatcher pass. Each item
        // is one PCIe read of the payload + one HBM write. Batching here is
        // important for throughput: the PCIe tail-read cost is amortized
        // across however many items the host has enqueued.
        while (my_head != tail) {
            uint32_t ring_pos = my_head & (DOORBELL_RING_CAPACITY - 1);
            WorkItem w = doorbell->slots[ring_pos];
            uint32_t idx = my_head;  // dispatcher assigns sequential HBM idxs
            uint32_t pos = idx & (QUEUE_CAPACITY - 1);

            // Wait until HBM slot is free for this lap (backstop; the
            // producer paces itself via `tail < head + CAPACITY`).
            for (;;) {
                uint32_t cs = ld_acquire_gpu(&q->seq[pos]);
                if (cs == idx) break;
                __nanosleep(100);
                if (ld_acquire_gpu(&q->terminate_hbm) != 0u) return;
            }

            q->slot[pos] = w;
            st_release_gpu(&q->seq[pos], idx + 1u);

            my_head += 1u;
        }

        // Publish dispatcher's drained position so host can pace.
        st_release_sys(&doorbell->head, my_head);
    }
}


// --------------------------------------------------------------------------
// Worker blocks: pop from HBM queue, execute op, write result to pinned
// memory.
// --------------------------------------------------------------------------

__device__ void worker_loop(HBMQueue* q,
                            DoneSlot* done) {
    if (threadIdx.x != 0) {
        // Inactive threads watch the HBM terminate mirror (not the pinned
        // one) — keeps PCIe clear for the active dispatcher path.
        while (ld_acquire_gpu(&q->terminate_hbm) == 0u) {
            __nanosleep(1000);
        }
        return;
    }

    // Token-passing turn lock. Only the worker whose id matches the current
    // turn is allowed to claim. Wraps mod num_workers. Each completion
    // hands the turn to the next worker. This makes single-item-in-flight
    // workloads behave exactly like num_sms=1 (no CAS contention, no wasted
    // HBM reads), while keeping throughput at full num_sms for burst
    // workloads because successive items find the next worker already
    // spinning on its turn.
    const uint32_t worker_id = (uint32_t)(blockIdx.x - 1u);  // 0..num_workers-1
    const uint32_t num_workers = (uint32_t)(gridDim.x - 1u);

    while (true) {
        // Pure HBM poll on turn. All reads come out of L2; no PCIe traffic.
        uint32_t turn = ld_acquire_gpu(&q->turn);
        uint32_t my_turn = turn % num_workers;
        if (my_turn != worker_id) {
            if (ld_acquire_gpu(&q->terminate_hbm) != 0u) return;
            continue;
        }

        // It's our turn. Claim the slot matching `turn`.
        uint32_t cur_head = turn;
        uint32_t pos = cur_head & (QUEUE_CAPACITY - 1);
        uint32_t expected = cur_head + 1u;

        // Spin on slot seq. Because turn advanced, the dispatcher will have
        // published exactly this slot (or will very shortly).
        for (;;) {
            uint32_t s = ld_acquire_gpu(&q->seq[pos]);
            if (s == expected) break;
            if (ld_acquire_gpu(&q->terminate_hbm) != 0u) return;
        }

        WorkItem w = q->slot[pos];
        int32_t res = w.a + w.b * w.c;

        // Publish done (one PCIe write).
        done[pos].out = res;
        st_release_sys(&done[pos].flag, cur_head + 1u);

        // Free the HBM slot for the next lap.
        st_release_gpu(&q->seq[pos], cur_head + (uint32_t)QUEUE_CAPACITY);

        // Hand the turn to the next worker.
        st_release_gpu(&q->turn, cur_head + 1u);

        // Bump head for diagnostics (atomic-add so tail watchers see it).
        atomicAdd(&q->head, 1u);
    }
}


// --------------------------------------------------------------------------
// Unified persistent kernel: blockIdx.x == 0 is dispatcher, rest are workers.
// --------------------------------------------------------------------------

extern "C" __global__ void persistent_worker_kernel(Doorbell* doorbell,
                                                    HBMQueue* q,
                                                    DoneSlot* done,
                                                    ControlBlock* ctrl) {
    if (blockIdx.x == 0) {
        dispatcher_loop(doorbell, q, ctrl);
    } else {
        worker_loop(q, done);
    }
}


// --------------------------------------------------------------------------
// Host-side launch / shutdown.
// --------------------------------------------------------------------------

struct LaunchHandles {
    Doorbell*    doorbell_host;   // mapped pinned, host addr
    Doorbell*    doorbell_dev;    // device-mapped pointer
    HBMQueue*    q_dev;           // cudaMalloc
    DoneSlot*    done_host;       // mapped pinned, host addr
    DoneSlot*    done_dev;        // device-mapped pointer
    ControlBlock* ctrl_host;      // mapped pinned, host addr
    ControlBlock* ctrl_dev;       // device-mapped pointer
    cudaStream_t stream;
};

static LaunchHandles g_handles;  // singleton for the PoC


extern "C" int voxcpm_launch_persistent(int num_sms,
                                        Doorbell** doorbell_out,
                                        ControlBlock** ctrl_out,
                                        DoneSlot** done_out,
                                        void** queue_dev_out,
                                        void** stream_out) {
    cudaError_t err;

    // Doorbell (host-mapped pinned). We deliberately do *not* use WC —
    // WC would require an explicit SFENCE on the host between the payload
    // store and the seq store, and the 16-byte payload is small enough
    // that cacheable TSO-ordered writes are fine.
    err = cudaHostAlloc((void**)&g_handles.doorbell_host, sizeof(Doorbell),
                        cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) return 1;
    memset(g_handles.doorbell_host, 0, sizeof(Doorbell));
    err = cudaHostGetDevicePointer((void**)&g_handles.doorbell_dev,
                                    g_handles.doorbell_host, 0);
    if (err != cudaSuccess) return 2;

    // Done slots (host-mapped pinned, default mapping — host reads from them).
    err = cudaHostAlloc((void**)&g_handles.done_host,
                        sizeof(DoneSlot) * QUEUE_CAPACITY,
                        cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) return 3;
    memset(g_handles.done_host, 0, sizeof(DoneSlot) * QUEUE_CAPACITY);
    err = cudaHostGetDevicePointer((void**)&g_handles.done_dev,
                                    g_handles.done_host, 0);
    if (err != cudaSuccess) return 4;

    // Control block (host-mapped pinned).
    err = cudaHostAlloc((void**)&g_handles.ctrl_host, sizeof(ControlBlock),
                        cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) return 5;
    memset(g_handles.ctrl_host, 0, sizeof(ControlBlock));
    err = cudaHostGetDevicePointer((void**)&g_handles.ctrl_dev,
                                    g_handles.ctrl_host, 0);
    if (err != cudaSuccess) return 6;

    // HBM queue.
    err = cudaMalloc((void**)&g_handles.q_dev, sizeof(HBMQueue));
    if (err != cudaSuccess) return 7;
    // Initialize on host, memcpy across.
    HBMQueue* init = (HBMQueue*)malloc(sizeof(HBMQueue));
    memset(init, 0, sizeof(HBMQueue));
    for (int i = 0; i < QUEUE_CAPACITY; ++i) {
        init->seq[i] = (uint32_t)i;
    }
    err = cudaMemcpy(g_handles.q_dev, init, sizeof(HBMQueue),
                     cudaMemcpyHostToDevice);
    free(init);
    if (err != cudaSuccess) return 8;

    err = cudaStreamCreateWithFlags(&g_handles.stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) return 9;

    // Launch: 1 dispatcher block + num_sms worker blocks, 32 threads/block
    // (one warp — minimum for reasonable occupancy). All blocks read
    // blockIdx.x to decide their role; only thread 0 of each block does
    // work, the other 31 spin on terminate and retire on shutdown.
    int total_blocks = num_sms + 1;
    dim3 grid(total_blocks, 1, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    persistent_worker_kernel<<<grid, block, 0, g_handles.stream>>>(
        g_handles.doorbell_dev, g_handles.q_dev, g_handles.done_dev,
        g_handles.ctrl_dev);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "persistent_worker_kernel launch failed: %s\n",
                cudaGetErrorString(launch_err));
        return 10;
    }

    *doorbell_out = g_handles.doorbell_host;
    *ctrl_out = g_handles.ctrl_host;
    *done_out = g_handles.done_host;
    *queue_dev_out = g_handles.q_dev;
    *stream_out = (void*)g_handles.stream;
    return 0;
}


extern "C" int voxcpm_shutdown_persistent(void) {
    if (g_handles.ctrl_host == nullptr) return 1;
    __atomic_store_n(&g_handles.ctrl_host->terminate, 1u, __ATOMIC_RELEASE);

    if (g_handles.stream) {
        cudaStreamSynchronize(g_handles.stream);
        cudaStreamDestroy(g_handles.stream);
        g_handles.stream = nullptr;
    }
    if (g_handles.q_dev) {
        cudaFree(g_handles.q_dev);
        g_handles.q_dev = nullptr;
    }
    if (g_handles.doorbell_host) {
        cudaFreeHost(g_handles.doorbell_host);
        g_handles.doorbell_host = nullptr;
    }
    if (g_handles.done_host) {
        cudaFreeHost(g_handles.done_host);
        g_handles.done_host = nullptr;
    }
    if (g_handles.ctrl_host) {
        cudaFreeHost(g_handles.ctrl_host);
        g_handles.ctrl_host = nullptr;
    }
    return 0;
}


extern "C" int voxcpm_queue_info(size_t* doorbell_slots_offset,
                                 size_t* doorbell_tail_offset,
                                 size_t* doorbell_head_offset,
                                 size_t* doorbell_slot_stride,
                                 size_t* done_flag_offset,
                                 size_t* done_out_offset,
                                 size_t* done_stride,
                                 size_t* ctrl_terminate_offset,
                                 int* queue_capacity,
                                 int* doorbell_ring_capacity) {
    *doorbell_slots_offset = offsetof(Doorbell, slots);
    *doorbell_tail_offset = offsetof(Doorbell, tail);
    *doorbell_head_offset = offsetof(Doorbell, head);
    *doorbell_slot_stride = sizeof(WorkItem);
    *done_flag_offset = offsetof(DoneSlot, flag);
    *done_out_offset = offsetof(DoneSlot, out);
    *done_stride = sizeof(DoneSlot);
    *ctrl_terminate_offset = offsetof(ControlBlock, terminate);
    *queue_capacity = QUEUE_CAPACITY;
    *doorbell_ring_capacity = DOORBELL_RING_CAPACITY;
    return 0;
}

}  // namespace voxcpm_fast


// --------------------------------------------------------------------------
// PyTorch extension glue. All pointers are returned to Python through an
// int64 tensor; Python owns the lifetime via explicit shutdown().
// --------------------------------------------------------------------------

#include <torch/extension.h>

static torch::Tensor py_launch_persistent(int num_sms) {
    voxcpm_fast::Doorbell* doorbell = nullptr;
    voxcpm_fast::ControlBlock* ctrl = nullptr;
    voxcpm_fast::DoneSlot* done = nullptr;
    void* q_dev = nullptr;
    void* stream = nullptr;
    int rc = voxcpm_fast::voxcpm_launch_persistent(num_sms, &doorbell, &ctrl,
                                                    &done, &q_dev, &stream);
    if (rc != 0) {
        throw std::runtime_error("voxcpm_launch_persistent rc=" + std::to_string(rc));
    }
    auto t = torch::empty({5}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    int64_t* p = t.data_ptr<int64_t>();
    p[0] = (int64_t)doorbell;
    p[1] = (int64_t)ctrl;
    p[2] = (int64_t)done;
    p[3] = (int64_t)q_dev;
    p[4] = (int64_t)stream;
    return t;
}

static int py_shutdown_persistent() {
    return voxcpm_fast::voxcpm_shutdown_persistent();
}

static std::vector<int64_t> py_queue_info() {
    size_t doorbell_slots_offset, doorbell_tail_offset, doorbell_head_offset;
    size_t doorbell_slot_stride;
    size_t done_flag_offset, done_out_offset, done_stride;
    size_t ctrl_terminate_offset;
    int queue_capacity, doorbell_ring_capacity;
    voxcpm_fast::voxcpm_queue_info(&doorbell_slots_offset,
                                   &doorbell_tail_offset,
                                   &doorbell_head_offset,
                                   &doorbell_slot_stride,
                                   &done_flag_offset,
                                   &done_out_offset, &done_stride,
                                   &ctrl_terminate_offset,
                                   &queue_capacity,
                                   &doorbell_ring_capacity);
    return {(int64_t)doorbell_slots_offset,
            (int64_t)doorbell_tail_offset,
            (int64_t)doorbell_head_offset,
            (int64_t)doorbell_slot_stride,
            (int64_t)done_flag_offset,
            (int64_t)done_out_offset,
            (int64_t)done_stride,
            (int64_t)ctrl_terminate_offset,
            (int64_t)queue_capacity,
            (int64_t)doorbell_ring_capacity};
}

#include <emmintrin.h>  // _mm_sfence, _mm_mfence

// Host-side release fence. Python calls this between the payload write and
// the seq store. We explicitly issue an SFENCE (or MFENCE) — neither
// __atomic_store_n(__ATOMIC_RELEASE) nor plain TSO stores are sufficient
// for PCIe-coherent observers: the CPU store buffer may hold payload bytes
// when the GPU snoops the doorbell cache line, so without a store fence
// the GPU reads stale bytes and observes the new seq afterwards.
static void py_store_release_seq(int64_t seq_addr, uint32_t value) {
    _mm_sfence();
    *((volatile uint32_t*)seq_addr) = value;
    _mm_sfence();
}

// Batched burst-submit from C. Used by the throughput benchmark to bypass
// per-item Python/ctypes overhead. Pushes `n` items with inputs provided in
// flat int32 arrays, publishing `tail_base + n` at the end. Caller must
// ensure the doorbell ring has enough capacity (host-side backpressure,
// same as PersistentKernel.submit()).
//
// Does NOT wait for completion — caller polls done flags afterwards.
static int64_t py_burst_submit(int64_t doorbell_slots_addr,
                               int64_t doorbell_tail_addr,
                               int64_t doorbell_head_addr,
                               int64_t done_addr,
                               int64_t slot_stride,
                               int64_t done_stride,
                               int64_t done_flag_off,
                               int64_t ring_capacity,
                               int64_t starting_tail,
                               int64_t n,
                               torch::Tensor a_arr,
                               torch::Tensor b_arr,
                               torch::Tensor c_arr) {
    TORCH_CHECK(a_arr.scalar_type() == torch::kInt32);
    TORCH_CHECK(b_arr.scalar_type() == torch::kInt32);
    TORCH_CHECK(c_arr.scalar_type() == torch::kInt32);
    TORCH_CHECK(a_arr.is_cpu() && a_arr.is_contiguous());
    TORCH_CHECK(b_arr.is_cpu() && b_arr.is_contiguous());
    TORCH_CHECK(c_arr.is_cpu() && c_arr.is_contiguous());
    TORCH_CHECK(a_arr.numel() >= n && b_arr.numel() >= n && c_arr.numel() >= n);

    const int32_t* a_ptr = a_arr.data_ptr<int32_t>();
    const int32_t* b_ptr = b_arr.data_ptr<int32_t>();
    const int32_t* c_ptr = c_arr.data_ptr<int32_t>();

    uint8_t* slots_base = (uint8_t*)doorbell_slots_addr;
    uint32_t* tail_ptr = (uint32_t*)doorbell_tail_addr;
    const uint32_t* head_ptr = (const uint32_t*)doorbell_head_addr;
    uint8_t* done_base = (uint8_t*)done_addr;

    const int64_t ring_mask = ring_capacity - 1;
    // `starting_tail` is the seq of the first item in this burst (= first
    // item's `seq` == first `doorbell.tail` value). Items have seqs
    // `starting_tail .. starting_tail + n - 1`. idx = seq - 1.
    const uint32_t seq_base = (uint32_t)starting_tail;

    for (int64_t k = 0; k < n; ++k) {
        uint32_t seq_k = seq_base + (uint32_t)k;
        uint32_t idx = seq_k - 1u;
        int64_t ring_pos = idx & ring_mask;

        // Backpressure vs. ring fill.
        int64_t need = (int64_t)idx - (int64_t)ring_capacity + 1;
        if (need > 0) {
            while ((int64_t)__atomic_load_n(head_ptr, __ATOMIC_ACQUIRE) < need) {
                // spin
            }
        }

        // Reset done flag for this lap.
        int64_t slot_pos = idx & (voxcpm_fast::QUEUE_CAPACITY - 1);
        uint8_t* done_slot = done_base + slot_pos * done_stride;
        *((uint32_t*)(done_slot + done_flag_off)) = 0u;

        // Write payload.
        int32_t* payload = (int32_t*)(slots_base + ring_pos * slot_stride);
        payload[0] = a_ptr[k];
        payload[1] = b_ptr[k];
        payload[2] = c_ptr[k];
        payload[3] = (int32_t)(idx & 0x7FFFFFFF);
    }
    // doorbell.tail matches the seq of the LAST item in this burst. The
    // dispatcher drains items `my_head .. tail - 1`... but in the single-
    // submit path we write `tail = seq`, not `seq + 1`. So here we also
    // publish `tail = seq_base + n - 1`. The dispatcher then processes
    // items with idx < tail, which is idx 0..tail-1 = seq_base-1..seq_base+n-2.
    //
    // Wait — both paths use the same ``tail == seq`` convention: when the
    // Nth item (seq=N, idx=N-1) is published, tail becomes N. The dispatcher
    // drains while `my_head < tail`, processing idx = my_head, so it drains
    // idx 0..N-1 which is correct.
    uint32_t final_tail = seq_base + (uint32_t)n - 1u;  // seq of the last item
    _mm_sfence();
    *((volatile uint32_t*)tail_ptr) = final_tail;
    _mm_sfence();
    return (int64_t)final_tail;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_persistent", &py_launch_persistent,
          "Launch persistent kernel; returns int64[5]="
          "{doorbell, ctrl, done, q_dev, stream}");
    m.def("shutdown_persistent", &py_shutdown_persistent,
          "Signal terminate, join, and free all buffers");
    m.def("queue_info", &py_queue_info,
          "Return offsets/capacity for direct mapped-memory access");
    m.def("store_release_seq", &py_store_release_seq,
          "Release-ordered 32-bit store at *(uint32_t*)addr = value");
    m.def("burst_submit", &py_burst_submit,
          "Push N items in a C loop; returns the new tail after the burst");
}
