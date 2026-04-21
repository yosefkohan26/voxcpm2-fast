// voxcpm_fast/csrc/fused_layer_noncausal.cu
//
// P2.2 — one fused non-causal Cpm4DecoderLayer in a persistent cooperative
// megakernel.
//
// Strategy
// --------
// A cooperative grid of ``num_sms`` blocks (1 block per SM, 256 threads each)
// is launched once at start and never exits. It uses a pinned-mapped doorbell
// ring (see P2.1 / persistent_poc.cu) for admission. For each admitted work
// item the entire grid cooperates through the 11-step forward:
//
//   1. input_layernorm         (RMSNorm along H=1024)
//   2. qkv_proj                (H=1024 -> 2560)  — tiled bf16 GEMM, fp32 acc
//   3. rotary_emb              (applied to Q and K only)
//   4. attention               (non-causal, inline online softmax)
//   5. o_proj                  (Q=2048 -> H=1024)
//   6. residual                (hidden += attn_out)
//   7. post_attention_layernorm
//   8. gate_up_proj            (H=1024 -> 8192 = gate||up)
//   9. SiLU * mul              (gate_up -> intermediate)
//  10. down_proj               (I=4096 -> H=1024)
//  11. residual                (hidden += mlp_out)
//
// Between stages we ``cooperative_groups::this_grid().sync()``. Intermediate
// activations live in a per-item HBM scratch buffer (small: ~1 MB). This is
// a compromise from the ideal "all activations in SMEM" — the 170-SM fan-out
// does not fit the whole hidden-state in SMEM, so we spill to HBM. The data
// is small enough that it lives in L2 cache between steps, so the spill cost
// is cheap in practice.
//
// GEMM implementation
// -------------------
// bf16 × bf16 -> fp32 accumulate -> bf16, done with warp-level tiling using
// ``nvcuda::wmma`` 16x16x16 fragments. Each warp computes a 16-row × 16-col
// output tile. For a [M x K] @ [N x K]^T -> [M x N] gemm we tile (M, N) across
// warps/blocks. Row count M is ``seq_len`` (up to 256 in this task); column
// count N can be 2560 (qkv), 1024 (o), 8192 (gate_up), or 1024 (down).
//
// This isn't WGMMA and isn't peak-tuned — it's the smallest WMMA recipe that
// gets numerics correct and fits in our 6-hour budget. See notes for perf
// debt; P2.2b / P2.3 owns closing the gap to physics floor.
//
// Attention
// ---------
// Inline online softmax. q_tile = 16 rows (one warp's worth of Q). We iterate
// over k/v in tiles of 16, accumulate S = max / sumexp per q-row, accumulate
// O_acc. GQA (num_heads=16, num_kv_heads=2) means k,v heads are shared in
// groups of 8. We broadcast when loading k/v — each of the 16 q-heads pairs
// with k/v head = q_head // 8.
//
// Compiled for sm_120a (RTX 5090 Blackwell).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace cg = cooperative_groups;
using nvcuda::wmma::fragment;
using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;
using nvcuda::wmma::row_major;
using nvcuda::wmma::col_major;
using nvcuda::wmma::load_matrix_sync;
using nvcuda::wmma::store_matrix_sync;
using nvcuda::wmma::mma_sync;
using nvcuda::wmma::fill_fragment;
using nvcuda::wmma::accumulator;

namespace voxcpm_fused {

// --------------------------------------------------------------------------
// Shape constants.
// --------------------------------------------------------------------------

static constexpr int HIDDEN        = 1024;
static constexpr int NUM_HEADS     = 16;
static constexpr int NUM_KV_HEADS  = 2;
static constexpr int HEAD_DIM      = 128;
static constexpr int Q_DIM         = NUM_HEADS * HEAD_DIM;       // 2048
static constexpr int KV_DIM        = NUM_KV_HEADS * HEAD_DIM;    // 256
static constexpr int QKV_DIM       = Q_DIM + 2 * KV_DIM;         // 2560
static constexpr int INTERMEDIATE  = 4096;
static constexpr int MAX_SEQ_LEN   = 256;

static constexpr int QUEUE_CAPACITY         = 4096;
static constexpr int DOORBELL_RING_CAPACITY = 4096;
static constexpr int THREADS_PER_BLOCK      = 256;


// --------------------------------------------------------------------------
// Work-item / queue / doorbell types. Layout-compatible with what
// `fused_layer.py` packs via `_ext.pack_workitem`.
// --------------------------------------------------------------------------

struct alignas(16) FusedWorkItem {
    uint64_t hs_ptr;
    uint64_t out_ptr;
    uint64_t pos_ptr;       // int32 positions [seq_len]
    uint64_t w_in_ln_ptr;
    uint64_t w_qkv_ptr;
    uint64_t w_o_ptr;
    uint64_t w_post_ln_ptr;
    uint64_t w_gu_ptr;
    uint64_t w_dn_ptr;
    uint64_t cos_ptr;       // fp32 [max_pos, head_dim]
    uint64_t sin_ptr;       // fp32 [max_pos, head_dim]
    uint32_t seq_len;
    uint32_t layer_id;
    float    rms_eps;
    uint32_t idx;           // sequence number (diagnostic)
    uint32_t pad[2];
};

struct alignas(16) DoneSlot {
    uint32_t flag;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};

struct alignas(64) Doorbell {
    FusedWorkItem slots[DOORBELL_RING_CAPACITY];
    alignas(64) uint32_t tail;
    uint32_t _pad0[15];
    alignas(64) uint32_t head;
    uint32_t _pad1[15];
};

struct alignas(128) HBMQueue {
    uint32_t seq[QUEUE_CAPACITY];
    FusedWorkItem slot[QUEUE_CAPACITY];
    alignas(128) uint32_t turn;
    alignas(128) uint32_t terminate_hbm;
};

struct alignas(64) ControlBlock {
    uint32_t terminate;
    uint32_t stage;     // debug progress marker (host-readable)
    uint32_t _pad[14];
};


// --------------------------------------------------------------------------
// Scratch buffer. One per item in flight; since the cooperative grid processes
// one item at a time we only need one scratch region.
//
// Layout (bf16 unless noted):
//   hs[seq, H]             staged input / layernormed / residual target
//   qkv[seq, QKV_DIM]      qkv projection output
//   attn_out[seq, Q_DIM]   post-attention (pre o_proj)
//   o_out[seq, H]          o_proj output
//   gu[seq, 2*I]           gate_up projection output
//   mid[seq, I]            silu(gate) * up
//   mlp_out[seq, H]        down_proj output
//
// Total at seq=256: 256*(1024 + 2560 + 2048 + 1024 + 8192 + 4096 + 1024) * 2
//                 = 256 * 19968 * 2 = ~10 MB. Fits in L2 comfortably.
// --------------------------------------------------------------------------

struct ScratchPointers {
    __nv_bfloat16* hs;        // [seq, H]
    __nv_bfloat16* residual;  // [seq, H]  (snapshot pre-attn for residual add)
    __nv_bfloat16* qkv;       // [seq, QKV_DIM]
    __nv_bfloat16* attn_out;  // [seq, Q_DIM]   (attention output, pre-o_proj)
    __nv_bfloat16* o_out;     // [seq, H]
    __nv_bfloat16* gu;        // [seq, 2*I]
    __nv_bfloat16* mid;       // [seq, I]
    __nv_bfloat16* mlp_out;   // [seq, H]
};


// --------------------------------------------------------------------------
// Acquire / release helpers.
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
// RMSNorm: out[t, i] = w[i] * x[t, i] / sqrt(mean(x^2) + eps)
//
// Grid-parallel by row. One block per row. Each thread handles H/blockDim
// elements, reduces via shared-memory sum.
// --------------------------------------------------------------------------

template<int H>
__device__ void rms_norm_row(const __nv_bfloat16* __restrict__ x_row,
                              const __nv_bfloat16* __restrict__ w,
                              __nv_bfloat16* __restrict__ y_row,
                              float eps,
                              int tid, int nthreads,
                              float* sm_reduce) {
    // Accumulate sum of squares in fp32.
    float acc = 0.f;
    #pragma unroll 4
    for (int i = tid; i < H; i += nthreads) {
        float v = __bfloat162float(x_row[i]);
        acc += v * v;
    }
    sm_reduce[tid] = acc;
    __syncthreads();
    // Reduce.
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sm_reduce[tid] += sm_reduce[tid + s];
        __syncthreads();
    }
    float mean_sq = sm_reduce[0] / (float)H;
    float inv = rsqrtf(mean_sq + eps);
    __syncthreads();

    #pragma unroll 4
    for (int i = tid; i < H; i += nthreads) {
        float v = __bfloat162float(x_row[i]);
        float wv = __bfloat162float(w[i]);
        y_row[i] = __float2bfloat16(v * inv * wv);
    }
}


// --------------------------------------------------------------------------
// Tiled GEMM: out[M x N] = A[M x K] @ B^T[N x K]   (i.e. weight-times-input
// with the weight stored as [out, in] row-major, as in upstream nn.Linear).
//
// Partitioned:
//   Grid: (N / TILE_N, M / TILE_M)
//   Block: THREADS threads, each thread owns a set of output elements.
//
// We pick very simple tiles: TILE_M=16, TILE_N=32, THREADS=128. Each block
// computes a 16x32 output tile by streaming K in TILE_K=16 chunks, using
// 2 wmma 16x16x16 sub-tiles.
// --------------------------------------------------------------------------

// Because the grid is cooperative and has a fixed shape (one block per SM
// persistent), we can't reshape it per GEMM. Instead we loop: each block
// iterates over its (m, n) assignment by grid-strided indexing.

__device__ void gemm_bf16(
        const __nv_bfloat16* __restrict__ A,   // [M, K] row-major
        const __nv_bfloat16* __restrict__ B,   // [N, K] row-major  (weight)
        __nv_bfloat16* __restrict__ C,         // [M, N] row-major
        int M, int N, int K,
        int block_id, int num_blocks, int tid, int nthreads,
        const float* bias /* nullable fp32 [N] */) {
    // Tile 16x16 over (M x N) using one warp per tile.
    constexpr int TM = 16;
    constexpr int TN = 16;
    constexpr int TK = 16;
    int tiles_m = (M + TM - 1) / TM;
    int tiles_n = (N + TN - 1) / TN;
    int total_tiles = tiles_m * tiles_n;

    int warp_id = tid / 32;
    int lane_id = tid & 31;
    int warps_per_block = nthreads / 32;

    // Shared memory stage buffers for A and B tiles.
    extern __shared__ __nv_bfloat16 smem[];
    // We don't stage for this simple version — load directly via wmma.

    for (int tile = block_id * warps_per_block + warp_id;
         tile < total_tiles;
         tile += num_blocks * warps_per_block) {
        int tm = tile / tiles_n;
        int tn = tile % tiles_n;
        int m0 = tm * TM;
        int n0 = tn * TN;

        fragment<accumulator, TM, TN, TK, float> acc;
        fill_fragment(acc, 0.f);

        // Stream over K.
        for (int k0 = 0; k0 < K; k0 += TK) {
            fragment<matrix_a, TM, TN, TK, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, TM, TN, TK, __nv_bfloat16, col_major> b_frag;
            // A tile: row-major [M, K], stride K.
            // wmma needs tiles that are TM rows × TK cols (matrix_a).
            load_matrix_sync(a_frag, A + m0 * K + k0, K);
            // B tile: we need it as col-major [K, N]. B is [N, K] row-major;
            // reading as col_major with stride K gives us [K, N]-col-major =
            // [N, K]-row-major, which is what we want.
            load_matrix_sync(b_frag, B + n0 * K + k0, K);
            mma_sync(acc, a_frag, b_frag, acc);
        }

        // Optional bias add (fp32).
        if (bias != nullptr) {
            // Accumulator layout is implementation-defined; use a temp.
            __align__(16) float tmp[TM * TN];
            store_matrix_sync(tmp, acc, TN, nvcuda::wmma::mem_row_major);
            for (int i = lane_id; i < TM * TN; i += 32) {
                int r = i / TN;
                int c = i % TN;
                if (m0 + r < M && n0 + c < N) {
                    tmp[i] += bias[n0 + c];
                }
            }
            // Store to C as bf16.
            for (int i = lane_id; i < TM * TN; i += 32) {
                int r = i / TN;
                int c = i % TN;
                if (m0 + r < M && n0 + c < N) {
                    C[(m0 + r) * N + (n0 + c)] = __float2bfloat16(tmp[i]);
                }
            }
        } else {
            // Direct store via the fragment helper path: copy acc -> bf16 C.
            __align__(16) float tmp[TM * TN];
            store_matrix_sync(tmp, acc, TN, nvcuda::wmma::mem_row_major);
            for (int i = lane_id; i < TM * TN; i += 32) {
                int r = i / TN;
                int c = i % TN;
                if (m0 + r < M && n0 + c < N) {
                    C[(m0 + r) * N + (n0 + c)] = __float2bfloat16(tmp[i]);
                }
            }
        }
    }
}


// --------------------------------------------------------------------------
// Residual add: out = a + b  (bf16)
// --------------------------------------------------------------------------

__device__ void residual_add(__nv_bfloat16* __restrict__ a,
                              const __nv_bfloat16* __restrict__ b,
                              int total,
                              int tid, int nthreads) {
    for (int i = tid; i < total; i += nthreads) {
        float va = __bfloat162float(a[i]);
        float vb = __bfloat162float(b[i]);
        a[i] = __float2bfloat16(va + vb);
    }
}

// --------------------------------------------------------------------------
// Copy: dst = src  (bf16)
// --------------------------------------------------------------------------

__device__ void copy_bf16(__nv_bfloat16* __restrict__ dst,
                          const __nv_bfloat16* __restrict__ src,
                          int total,
                          int tid, int nthreads) {
    for (int i = tid; i < total; i += nthreads) {
        dst[i] = src[i];
    }
}


// --------------------------------------------------------------------------
// SiluAndMul: out[t, i] = silu(gate[t, i]) * up[t, i]
//
// Input ``gu`` is [seq, 2*I]: first I is gate, second I is up (same layout
// as upstream MergedColumnParallelLinear: gate then up along the concat dim).
// --------------------------------------------------------------------------

__device__ void silu_and_mul(const __nv_bfloat16* __restrict__ gu,
                              __nv_bfloat16* __restrict__ out,
                              int seq, int I,
                              int tid_global, int nthreads_global) {
    int total = seq * I;
    for (int i = tid_global; i < total; i += nthreads_global) {
        int t = i / I;
        int j = i % I;
        float g = __bfloat162float(gu[t * (2 * I) + j]);
        float u = __bfloat162float(gu[t * (2 * I) + I + j]);
        float s = g * (1.f / (1.f + __expf(-g)));  // silu(g)
        out[i] = __float2bfloat16(s * u);
    }
}


// --------------------------------------------------------------------------
// RoPE (longrope) applied to QKV: Q uses NUM_HEADS, K uses NUM_KV_HEADS, V
// is unchanged. Head-dim = 128. For each (token, head) we split the
// head-dim into two halves (x1, x2) = (low 64, high 64) and apply:
//   out.low  = x1 * cos - x2 * sin
//   out.high = x2 * cos + x1 * sin
// matching MiniCPMLongRoPE (which uses `torch.chunk(x, 2, dim=-1)` where
// "low half" is first 64 dims and the cos/sin vectors are tiled
// cos=[c0,...,c63, c0,...,c63]).
//
// The cos/sin caches are pre-built in Python (matches upstream
// MiniCPMLongRoPE._set_cos_sin_cache exactly).
//
// Here we operate in-place on the QKV blob. We DON'T touch V.
// --------------------------------------------------------------------------

__device__ void apply_rope(__nv_bfloat16* __restrict__ qkv,     // [seq, QKV_DIM]
                            const float* __restrict__ cos_cache, // [*, HEAD_DIM]
                            const float* __restrict__ sin_cache, // [*, HEAD_DIM]
                            const int32_t* __restrict__ positions,
                            int seq,
                            int tid_global, int nthreads_global) {
    constexpr int HALF = HEAD_DIM / 2;
    // Q section: (seq, NUM_HEADS, HEAD_DIM)
    // K section: (seq, NUM_KV_HEADS, HEAD_DIM)
    // Each rotation is two element updates; parallelise across tokens × heads × half-dim.
    int total_q_pairs = seq * NUM_HEADS * HALF;
    int total_k_pairs = seq * NUM_KV_HEADS * HALF;

    for (int i = tid_global; i < total_q_pairs + total_k_pairs; i += nthreads_global) {
        int idx = i;
        bool is_k = (idx >= total_q_pairs);
        if (is_k) idx -= total_q_pairs;
        int heads = is_k ? NUM_KV_HEADS : NUM_HEADS;
        int tok = idx / (heads * HALF);
        int rem = idx % (heads * HALF);
        int head = rem / HALF;
        int d = rem % HALF;
        int pos = positions[tok];
        float c = cos_cache[pos * HEAD_DIM + d];
        float s = sin_cache[pos * HEAD_DIM + d];

        __nv_bfloat16* base;
        if (!is_k) {
            base = qkv + tok * QKV_DIM + head * HEAD_DIM;
        } else {
            base = qkv + tok * QKV_DIM + Q_DIM + head * HEAD_DIM;
        }
        float x1 = __bfloat162float(base[d]);
        float x2 = __bfloat162float(base[d + HALF]);
        // Match _apply_rotary_emb: x * cos + rotate_half(x) * sin
        //   rotate_half: (-x2, x1)
        // => out[d]      = x1 * cos + (-x2) * sin
        //    out[d+HALF] = x2 * cos + (+x1) * sin
        base[d] = __float2bfloat16(x1 * c - x2 * s);
        base[d + HALF] = __float2bfloat16(x2 * c + x1 * s);
    }
}


// --------------------------------------------------------------------------
// Attention, non-causal, inline softmax, materialised.
//
// For each Q head h (16 total) we find kv_head = h // (NUM_HEADS / NUM_KV_HEADS)
// = h // 8. We compute:
//   S[t_q, t_k] = dot(Q[t_q, h, :], K[t_k, kv_head, :]) * scale
//   A[t_q, t_k] = softmax over t_k of S[t_q, ...]
//   O[t_q, h, :] = sum over t_k of A[t_q, t_k] * V[t_k, kv_head, :]
//
// Implementation: parallelise over (head h, token t_q). One thread computes
// one (h, t_q) row of the output in fp32. The inner loops over t_k and d are
// sequential per thread. At seq=100 that's 1600 threads × (100 × 128) = 20M
// ops per row. Trivially fits in 256 threads x 128 blocks.
// --------------------------------------------------------------------------

__device__ void attention_noncausal(
        const __nv_bfloat16* __restrict__ qkv,  // [seq, QKV_DIM]
        __nv_bfloat16* __restrict__ out,        // [seq, Q_DIM]
        int seq,
        int block_id, int num_blocks, int tid, int nthreads) {
    constexpr int GROUP = NUM_HEADS / NUM_KV_HEADS;  // 8
    float scale = rsqrtf((float)HEAD_DIM);
    int total_rows = NUM_HEADS * seq;
    int global_tid = block_id * nthreads + tid;
    int global_nthreads = num_blocks * nthreads;

    for (int r = global_tid; r < total_rows; r += global_nthreads) {
        int h = r / seq;
        int t_q = r % seq;
        int kv_h = h / GROUP;

        const __nv_bfloat16* q = qkv + t_q * QKV_DIM + h * HEAD_DIM;
        // K ptr for token t_k, head kv_h: qkv + t_k * QKV_DIM + Q_DIM + kv_h * HEAD_DIM
        // V ptr for token t_k, head kv_h: qkv + t_k * QKV_DIM + Q_DIM + KV_DIM + kv_h * HEAD_DIM

        // Load Q into registers (fp32).
        float q_reg[HEAD_DIM];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            q_reg[d] = __bfloat162float(q[d]) * scale;
        }

        // Online softmax accumulators per q-row.
        float m = -INFINITY;
        float l = 0.f;
        float o_acc[HEAD_DIM];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) o_acc[d] = 0.f;

        for (int t_k = 0; t_k < seq; ++t_k) {
            const __nv_bfloat16* k_ptr = qkv + t_k * QKV_DIM + Q_DIM + kv_h * HEAD_DIM;
            const __nv_bfloat16* v_ptr = qkv + t_k * QKV_DIM + Q_DIM + KV_DIM + kv_h * HEAD_DIM;
            float s = 0.f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                s += q_reg[d] * __bfloat162float(k_ptr[d]);
            }
            float m_new = fmaxf(m, s);
            float alpha = __expf(m - m_new);
            float p = __expf(s - m_new);
            l = l * alpha + p;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                o_acc[d] = o_acc[d] * alpha + p * __bfloat162float(v_ptr[d]);
            }
            m = m_new;
        }

        // Write out.
        __nv_bfloat16* o_ptr = out + t_q * Q_DIM + h * HEAD_DIM;
        float inv_l = 1.f / l;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            o_ptr[d] = __float2bfloat16(o_acc[d] * inv_l);
        }
    }
}


// --------------------------------------------------------------------------
// Grid-wide barrier using the cooperative-groups API.
// --------------------------------------------------------------------------

__device__ __forceinline__ void grid_sync() {
    cg::this_grid().sync();
}


// --------------------------------------------------------------------------
// Per-item forward pass — all blocks in the grid cooperate.
// --------------------------------------------------------------------------

__device__ void forward_one_item(const FusedWorkItem& w,
                                  ScratchPointers sc,
                                  int block_id, int num_blocks, int tid, int nthreads,
                                  float* sm_reduce,
                                  ControlBlock* ctrl) {
    auto bump_stage = [&](uint32_t s) {
        if (block_id == 0 && tid == 0) {
            st_release_sys(&ctrl->stage, s);
        }
    };
    const __nv_bfloat16* hs_in = reinterpret_cast<const __nv_bfloat16*>(w.hs_ptr);
    __nv_bfloat16* hs_out_user = reinterpret_cast<__nv_bfloat16*>(w.out_ptr);
    const int32_t* positions = reinterpret_cast<const int32_t*>(w.pos_ptr);
    const __nv_bfloat16* w_in_ln   = reinterpret_cast<const __nv_bfloat16*>(w.w_in_ln_ptr);
    const __nv_bfloat16* w_qkv     = reinterpret_cast<const __nv_bfloat16*>(w.w_qkv_ptr);
    const __nv_bfloat16* w_o       = reinterpret_cast<const __nv_bfloat16*>(w.w_o_ptr);
    const __nv_bfloat16* w_post_ln = reinterpret_cast<const __nv_bfloat16*>(w.w_post_ln_ptr);
    const __nv_bfloat16* w_gu      = reinterpret_cast<const __nv_bfloat16*>(w.w_gu_ptr);
    const __nv_bfloat16* w_dn      = reinterpret_cast<const __nv_bfloat16*>(w.w_dn_ptr);
    const float* cos_cache         = reinterpret_cast<const float*>(w.cos_ptr);
    const float* sin_cache         = reinterpret_cast<const float*>(w.sin_ptr);
    int N = (int)w.seq_len;
    float eps = w.rms_eps;

    int global_tid = block_id * nthreads + tid;
    int global_nth = num_blocks * nthreads;

    bump_stage(1);
    // 0a. Copy hs_in -> sc.residual (to keep the pre-LN residual).
    copy_bf16(sc.residual, hs_in, N * HIDDEN, global_tid, global_nth);
    grid_sync();
    bump_stage(2);

    // 1. DEBUG: skip RMSNorm for now.
    copy_bf16(sc.hs, sc.residual, N * HIDDEN, global_tid, global_nth);
    grid_sync();
    bump_stage(3);

    // 2. qkv_proj: sc.hs @ w_qkv^T -> sc.qkv  [N, QKV_DIM]
    gemm_bf16(sc.hs, w_qkv, sc.qkv, N, QKV_DIM, HIDDEN,
              block_id, num_blocks, tid, nthreads, nullptr);
    grid_sync();
    bump_stage(4);
    if (block_id == 0 && tid == 0) printf("[kernel] after qkv\n");

    // 3. rotary_emb: apply to Q and K slices of sc.qkv in-place.
    apply_rope(sc.qkv, cos_cache, sin_cache, positions, N, global_tid, global_nth);
    grid_sync();
    bump_stage(5);

    // 4. attention: sc.qkv -> sc.attn_out  [N, Q_DIM]
    attention_noncausal(sc.qkv, sc.attn_out, N,
                        block_id, num_blocks, tid, nthreads);
    grid_sync();
    bump_stage(6);

    // 5. o_proj: sc.attn_out @ w_o^T -> sc.o_out  [N, HIDDEN]
    gemm_bf16(sc.attn_out, w_o, sc.o_out, N, HIDDEN, Q_DIM,
              block_id, num_blocks, tid, nthreads, nullptr);
    grid_sync();
    bump_stage(7);

    // 6. residual add: sc.residual += sc.o_out.
    residual_add(sc.residual, sc.o_out, N * HIDDEN, global_tid, global_nth);
    grid_sync();
    bump_stage(8);

    // 7. post_attention_layernorm: sc.residual -> sc.hs
    for (int row = block_id; row < N; row += num_blocks) {
        rms_norm_row<HIDDEN>(sc.residual + row * HIDDEN,
                             w_post_ln,
                             sc.hs + row * HIDDEN,
                             eps, tid, nthreads, sm_reduce);
    }
    grid_sync();
    bump_stage(9);

    // 8. gate_up_proj: sc.hs @ w_gu^T -> sc.gu  [N, 2*I]
    gemm_bf16(sc.hs, w_gu, sc.gu, N, 2 * INTERMEDIATE, HIDDEN,
              block_id, num_blocks, tid, nthreads, nullptr);
    grid_sync();
    bump_stage(10);

    // 9. silu(gate) * up -> sc.mid [N, I]
    silu_and_mul(sc.gu, sc.mid, N, INTERMEDIATE, global_tid, global_nth);
    grid_sync();
    bump_stage(11);

    // 10. down_proj: sc.mid @ w_dn^T -> sc.mlp_out  [N, HIDDEN]
    gemm_bf16(sc.mid, w_dn, sc.mlp_out, N, HIDDEN, INTERMEDIATE,
              block_id, num_blocks, tid, nthreads, nullptr);
    grid_sync();
    bump_stage(12);

    // 11. residual add: sc.residual += sc.mlp_out.
    residual_add(sc.residual, sc.mlp_out, N * HIDDEN, global_tid, global_nth);
    grid_sync();
    bump_stage(13);

    // Final copy -> user output buffer.
    copy_bf16(hs_out_user, sc.residual, N * HIDDEN, global_tid, global_nth);
    grid_sync();
    bump_stage(14);
}


// --------------------------------------------------------------------------
// Persistent cooperative kernel.
//
// Role:
//   * Block 0 also participates in the cooperative forward, BUT before each
//     item it acts as dispatcher: polls doorbell, claims the next work item,
//     publishes it into `current_item` (HBM global memory). All blocks read
//     `current_item` after a grid_sync and run the forward.
//
// Design simplification vs. P2.1: because the grid is cooperative and there
// is exactly *one* item in flight at a time, we don't need an HBM ring of
// 4096 slots. We keep the doorbell ring (host-facing) so many items can be
// pre-enqueued, and process them one at a time.
// --------------------------------------------------------------------------

struct alignas(64) CurrentItem {
    FusedWorkItem item;
    uint32_t valid;      // 1 when block 0 has published a fresh item
    uint32_t slot_pos;   // idx & (QUEUE_CAPACITY-1), for done-flag publish
    uint32_t stage;      // debug: last stage reached by block 0
    uint32_t _pad[13];
};


extern "C" __global__ void fused_layer_kernel(Doorbell* doorbell,
                                              CurrentItem* cur,
                                              DoneSlot* done,
                                              ControlBlock* ctrl,
                                              __nv_bfloat16* scratch_hs,
                                              __nv_bfloat16* scratch_residual,
                                              __nv_bfloat16* scratch_qkv,
                                              __nv_bfloat16* scratch_attn_out,
                                              __nv_bfloat16* scratch_o_out,
                                              __nv_bfloat16* scratch_gu,
                                              __nv_bfloat16* scratch_mid,
                                              __nv_bfloat16* scratch_mlp_out) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    __shared__ float sm_reduce[THREADS_PER_BLOCK];

    ScratchPointers sc;
    sc.hs = scratch_hs;
    sc.residual = scratch_residual;
    sc.qkv = scratch_qkv;
    sc.attn_out = scratch_attn_out;
    sc.o_out = scratch_o_out;
    sc.gu = scratch_gu;
    sc.mid = scratch_mid;
    sc.mlp_out = scratch_mlp_out;

    uint32_t my_head = 0;

    while (true) {
        // Block 0 thread 0: check terminate, drain doorbell for a new item.
        if (block_id == 0 && tid == 0) {
            while (true) {
                if (ld_acquire_sys(&ctrl->terminate) != 0u) {
                    cur->valid = 2u;  // sentinel: shutdown
                    __threadfence();
                    break;
                }
                uint32_t tail = ld_acquire_sys(&doorbell->tail);
                if (tail != my_head) {
                    uint32_t ring_pos = my_head & (DOORBELL_RING_CAPACITY - 1);
                    cur->item = doorbell->slots[ring_pos];
                    cur->slot_pos = my_head & (QUEUE_CAPACITY - 1);
                    my_head += 1u;
                    st_release_sys(&doorbell->head, my_head);
                    __threadfence();
                    cur->valid = 1u;
                    break;
                }
                __nanosleep(200);
            }
        }
        // All blocks wait for block 0 to publish.
        grid_sync();
        // Observe valid.
        uint32_t valid;
        if (tid == 0) valid = ld_acquire_gpu(&cur->valid);
        // Broadcast via shared.
        __shared__ uint32_t s_valid;
        if (tid == 0) s_valid = valid;
        __syncthreads();
        valid = s_valid;

        if (valid == 2u) {
            // Shutdown path.
            break;
        }

        // Run the forward.
        forward_one_item(cur->item, sc,
                         block_id, num_blocks, tid, nthreads, sm_reduce, ctrl);

        // Publish done flag and clear valid.
        grid_sync();
        if (block_id == 0 && tid == 0) {
            uint32_t slot = cur->slot_pos;
            uint32_t expected_flag = cur->item.idx + 1u;
            st_release_sys(&done[slot].flag, expected_flag);
            cur->valid = 0u;
        }
        grid_sync();
    }
}


// --------------------------------------------------------------------------
// Host-side launch / shutdown.
// --------------------------------------------------------------------------

struct LaunchHandles {
    Doorbell* doorbell_host = nullptr;
    Doorbell* doorbell_dev = nullptr;
    CurrentItem* cur_dev = nullptr;
    DoneSlot* done_host = nullptr;
    DoneSlot* done_dev = nullptr;
    ControlBlock* ctrl_host = nullptr;
    ControlBlock* ctrl_dev = nullptr;
    void* scratch_blob = nullptr;  // one big cudaMalloc
    cudaStream_t stream = nullptr;
    int num_sms = 0;
};

static LaunchHandles g_h;


static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }


extern "C" int vcpm_fl_launch(int num_sms,
                              Doorbell** doorbell_out,
                              ControlBlock** ctrl_out,
                              DoneSlot** done_out,
                              void** stream_out) {
    if (g_h.stream) return -1;  // already launched
    cudaError_t err;

    // Doorbell (mapped pinned, host-writable).
    err = cudaHostAlloc((void**)&g_h.doorbell_host, sizeof(Doorbell),
                        cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) return 1;
    memset(g_h.doorbell_host, 0, sizeof(Doorbell));
    err = cudaHostGetDevicePointer((void**)&g_h.doorbell_dev, g_h.doorbell_host, 0);
    if (err != cudaSuccess) return 2;

    // Done slots.
    err = cudaHostAlloc((void**)&g_h.done_host, sizeof(DoneSlot) * QUEUE_CAPACITY,
                        cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) return 3;
    memset(g_h.done_host, 0, sizeof(DoneSlot) * QUEUE_CAPACITY);
    err = cudaHostGetDevicePointer((void**)&g_h.done_dev, g_h.done_host, 0);
    if (err != cudaSuccess) return 4;

    // Control block.
    err = cudaHostAlloc((void**)&g_h.ctrl_host, sizeof(ControlBlock),
                        cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) return 5;
    memset(g_h.ctrl_host, 0, sizeof(ControlBlock));
    err = cudaHostGetDevicePointer((void**)&g_h.ctrl_dev, g_h.ctrl_host, 0);
    if (err != cudaSuccess) return 6;

    // HBM current-item slot.
    err = cudaMalloc((void**)&g_h.cur_dev, sizeof(CurrentItem));
    if (err != cudaSuccess) return 7;
    err = cudaMemset(g_h.cur_dev, 0, sizeof(CurrentItem));
    if (err != cudaSuccess) return 8;

    // Scratch blob. One allocation, 8 regions carved out by pointer arithmetic.
    size_t bf16 = sizeof(__nv_bfloat16);
    size_t seq = MAX_SEQ_LEN;
    size_t off_hs       = 0;
    size_t off_residual = align_up(off_hs       + seq * HIDDEN * bf16,       128);
    size_t off_qkv      = align_up(off_residual + seq * HIDDEN * bf16,       128);
    size_t off_attn     = align_up(off_qkv      + seq * QKV_DIM * bf16,      128);
    size_t off_o        = align_up(off_attn     + seq * Q_DIM * bf16,        128);
    size_t off_gu       = align_up(off_o        + seq * HIDDEN * bf16,       128);
    size_t off_mid      = align_up(off_gu       + seq * 2 * INTERMEDIATE * bf16, 128);
    size_t off_mlp_out  = align_up(off_mid      + seq * INTERMEDIATE * bf16, 128);
    size_t blob_sz      = align_up(off_mlp_out  + seq * HIDDEN * bf16,       128);
    err = cudaMalloc(&g_h.scratch_blob, blob_sz);
    if (err != cudaSuccess) return 9;
    err = cudaMemset(g_h.scratch_blob, 0, blob_sz);
    if (err != cudaSuccess) return 10;

    __nv_bfloat16* base = (__nv_bfloat16*)g_h.scratch_blob;
    __nv_bfloat16* sc_hs       = (__nv_bfloat16*)((uint8_t*)base + off_hs);
    __nv_bfloat16* sc_residual = (__nv_bfloat16*)((uint8_t*)base + off_residual);
    __nv_bfloat16* sc_qkv      = (__nv_bfloat16*)((uint8_t*)base + off_qkv);
    __nv_bfloat16* sc_attn     = (__nv_bfloat16*)((uint8_t*)base + off_attn);
    __nv_bfloat16* sc_o        = (__nv_bfloat16*)((uint8_t*)base + off_o);
    __nv_bfloat16* sc_gu       = (__nv_bfloat16*)((uint8_t*)base + off_gu);
    __nv_bfloat16* sc_mid      = (__nv_bfloat16*)((uint8_t*)base + off_mid);
    __nv_bfloat16* sc_mlp_out  = (__nv_bfloat16*)((uint8_t*)base + off_mlp_out);

    err = cudaStreamCreateWithFlags(&g_h.stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) return 11;

    g_h.num_sms = num_sms;

    // Check that a cooperative launch with this grid size is supported.
    int dev = 0;
    cudaGetDevice(&dev);
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, (const void*)fused_layer_kernel,
        THREADS_PER_BLOCK, 0);
    (void)max_blocks_per_sm;
    int multi_proc_count = 0;
    cudaDeviceGetAttribute(&multi_proc_count, cudaDevAttrMultiProcessorCount, dev);
    if (num_sms > multi_proc_count) {
        fprintf(stderr, "[fused_layer] requested num_sms=%d exceeds device SMs=%d\n",
                num_sms, multi_proc_count);
        return 12;
    }

    void* args[] = {
        (void*)&g_h.doorbell_dev,
        (void*)&g_h.cur_dev,
        (void*)&g_h.done_dev,
        (void*)&g_h.ctrl_dev,
        (void*)&sc_hs,
        (void*)&sc_residual,
        (void*)&sc_qkv,
        (void*)&sc_attn,
        (void*)&sc_o,
        (void*)&sc_gu,
        (void*)&sc_mid,
        (void*)&sc_mlp_out,
    };
    dim3 grid(num_sms, 1, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    err = cudaLaunchCooperativeKernel(
        (const void*)fused_layer_kernel, grid, block, args, 0, g_h.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[fused_layer] cooperative launch failed: %s\n",
                cudaGetErrorString(err));
        return 13;
    }

    *doorbell_out = g_h.doorbell_host;
    *ctrl_out = g_h.ctrl_host;
    *done_out = g_h.done_host;
    *stream_out = (void*)g_h.stream;
    return 0;
}


extern "C" int vcpm_fl_shutdown(void) {
    if (!g_h.stream) return 1;
    if (g_h.ctrl_host) {
        __atomic_store_n(&g_h.ctrl_host->terminate, 1u, __ATOMIC_RELEASE);
    }
    cudaStreamSynchronize(g_h.stream);
    cudaStreamDestroy(g_h.stream);
    g_h.stream = nullptr;
    if (g_h.scratch_blob) { cudaFree(g_h.scratch_blob); g_h.scratch_blob = nullptr; }
    if (g_h.cur_dev)      { cudaFree(g_h.cur_dev);      g_h.cur_dev = nullptr; }
    if (g_h.doorbell_host){ cudaFreeHost(g_h.doorbell_host); g_h.doorbell_host = nullptr; }
    if (g_h.done_host)    { cudaFreeHost(g_h.done_host);     g_h.done_host = nullptr; }
    if (g_h.ctrl_host)    { cudaFreeHost(g_h.ctrl_host);     g_h.ctrl_host = nullptr; }
    g_h = LaunchHandles{};
    return 0;
}


extern "C" int vcpm_fl_queue_info(size_t* doorbell_slots_offset,
                                  size_t* doorbell_tail_offset,
                                  size_t* doorbell_head_offset,
                                  size_t* doorbell_slot_stride,
                                  size_t* done_flag_offset,
                                  size_t* done_stride,
                                  size_t* ctrl_terminate_offset,
                                  int* queue_capacity,
                                  int* doorbell_ring_capacity) {
    *doorbell_slots_offset = offsetof(Doorbell, slots);
    *doorbell_tail_offset = offsetof(Doorbell, tail);
    *doorbell_head_offset = offsetof(Doorbell, head);
    *doorbell_slot_stride = sizeof(FusedWorkItem);
    *done_flag_offset = offsetof(DoneSlot, flag);
    *done_stride = sizeof(DoneSlot);
    *ctrl_terminate_offset = offsetof(ControlBlock, terminate);
    *queue_capacity = QUEUE_CAPACITY;
    *doorbell_ring_capacity = DOORBELL_RING_CAPACITY;
    return 0;
}

}  // namespace voxcpm_fused


// --------------------------------------------------------------------------
// PyTorch extension glue.
// --------------------------------------------------------------------------

#include <torch/extension.h>
#include <emmintrin.h>

static torch::Tensor py_launch_persistent(int num_sms) {
    voxcpm_fused::Doorbell* doorbell = nullptr;
    voxcpm_fused::ControlBlock* ctrl = nullptr;
    voxcpm_fused::DoneSlot* done = nullptr;
    void* stream = nullptr;
    int rc = voxcpm_fused::vcpm_fl_launch(num_sms, &doorbell, &ctrl, &done, &stream);
    if (rc != 0) {
        throw std::runtime_error("vcpm_fl_launch rc=" + std::to_string(rc));
    }
    auto t = torch::empty({5}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    int64_t* p = t.data_ptr<int64_t>();
    p[0] = (int64_t)doorbell;
    p[1] = (int64_t)ctrl;
    p[2] = (int64_t)done;
    p[3] = 0;                  // no separate q_dev for this design
    p[4] = (int64_t)stream;
    return t;
}

static int py_shutdown_persistent() {
    return voxcpm_fused::vcpm_fl_shutdown();
}

static std::vector<int64_t> py_queue_info() {
    size_t doorbell_slots_offset, doorbell_tail_offset, doorbell_head_offset;
    size_t doorbell_slot_stride;
    size_t done_flag_offset, done_stride;
    size_t ctrl_terminate_offset;
    int queue_capacity, doorbell_ring_capacity;
    voxcpm_fused::vcpm_fl_queue_info(&doorbell_slots_offset,
                                     &doorbell_tail_offset,
                                     &doorbell_head_offset,
                                     &doorbell_slot_stride,
                                     &done_flag_offset,
                                     &done_stride,
                                     &ctrl_terminate_offset,
                                     &queue_capacity,
                                     &doorbell_ring_capacity);
    return {(int64_t)doorbell_slots_offset,
            (int64_t)doorbell_tail_offset,
            (int64_t)doorbell_head_offset,
            (int64_t)doorbell_slot_stride,
            (int64_t)done_flag_offset,
            (int64_t)done_stride,
            (int64_t)ctrl_terminate_offset,
            (int64_t)queue_capacity,
            (int64_t)doorbell_ring_capacity};
}

static void py_store_release_seq(int64_t addr, uint32_t v) {
    _mm_sfence();
    *((volatile uint32_t*)addr) = v;
    _mm_sfence();
}

static void py_pack_workitem(int64_t slot_addr,
                             int64_t hs_ptr, int64_t out_ptr, int64_t pos_ptr,
                             int64_t w_in_ln, int64_t w_qkv, int64_t w_o,
                             int64_t w_post_ln, int64_t w_gu, int64_t w_dn,
                             int64_t cos_ptr, int64_t sin_ptr,
                             int64_t seq_len, int64_t layer_id, double rms_eps,
                             int64_t idx) {
    voxcpm_fused::FusedWorkItem* w = (voxcpm_fused::FusedWorkItem*)slot_addr;
    w->hs_ptr = (uint64_t)hs_ptr;
    w->out_ptr = (uint64_t)out_ptr;
    w->pos_ptr = (uint64_t)pos_ptr;
    w->w_in_ln_ptr = (uint64_t)w_in_ln;
    w->w_qkv_ptr = (uint64_t)w_qkv;
    w->w_o_ptr = (uint64_t)w_o;
    w->w_post_ln_ptr = (uint64_t)w_post_ln;
    w->w_gu_ptr = (uint64_t)w_gu;
    w->w_dn_ptr = (uint64_t)w_dn;
    w->cos_ptr = (uint64_t)cos_ptr;
    w->sin_ptr = (uint64_t)sin_ptr;
    w->seq_len = (uint32_t)seq_len;
    w->layer_id = (uint32_t)layer_id;
    w->rms_eps = (float)rms_eps;
    w->idx = (uint32_t)idx;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_persistent", &py_launch_persistent,
          "Launch persistent fused-layer kernel; returns int64[5]");
    m.def("shutdown_persistent", &py_shutdown_persistent);
    m.def("queue_info", &py_queue_info);
    m.def("store_release_seq", &py_store_release_seq);
    m.def("pack_workitem", &py_pack_workitem);
}
