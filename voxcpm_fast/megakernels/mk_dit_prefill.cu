// voxcpm_fast/megakernels/mk_dit_prefill.cu
//
// P2.5.2 — DiT persistent megakernel (incremental buildup).
//
//   Step 1 (DONE): cooperative-launch scaffolding (mk_dit_prefill_noop).
//   Step 2a (THIS FILE, current head): non-causal batched attention
//     kernel `vcpm_attention_noncausal_batched`. Validated vs flash_attn
//     at the DiT shape (B=2, S=11, Hq=16, Hk=2, D=128). Used standalone
//     today and as a phase inside Step 2b's cooperative megakernel later.
//   Step 2b (NEXT): fused single-DiT-layer cooperative kernel.
//   Step 3 (LATER): 12 layers + grid.sync between layers.
//   Step 4 (LATER): Euler loop (9 iters).
//   Step 5 (LATER): engine wire-in (FusedDiTMegakernelShim).
//
// Compiled for sm_120a (RTX 5090 Blackwell). No WGMMA/tcgen05.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <cstdio>

namespace cg = cooperative_groups;

using nvcuda::wmma::fragment;
using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;
using nvcuda::wmma::accumulator;
using nvcuda::wmma::row_major;
using nvcuda::wmma::col_major;
using nvcuda::wmma::load_matrix_sync;
using nvcuda::wmma::store_matrix_sync;
using nvcuda::wmma::mma_sync;
using nvcuda::wmma::fill_fragment;
using nvcuda::wmma::mem_row_major;

namespace voxcpm_fast {

// -------------------------------------------------------------------------
// Step 1 — cooperative-launch scaffolding (retained).
// Each block writes its blockIdx.x into OUT[blockIdx.x], then grid.sync(),
// then writes 2*blockIdx.x into OUT[blockIdx.x + num_blocks]. Proves that
// cudaLaunchCooperativeKernel + cg::this_grid().sync() works on sm_120a.
// -------------------------------------------------------------------------

extern "C" __global__ void mk_dit_prefill_noop_kernel(
        int32_t* __restrict__ OUT,
        int num_blocks) {
    cg::grid_group grid = cg::this_grid();

    if (threadIdx.x == 0) {
        OUT[blockIdx.x] = blockIdx.x;
    }

    grid.sync();

    if (threadIdx.x == 0) {
        OUT[num_blocks + blockIdx.x] = 2 * blockIdx.x;
    }
}

torch::Tensor mk_dit_prefill_noop(int64_t num_blocks) {
    TORCH_CHECK(num_blocks > 0 && num_blocks <= 4096,
                "num_blocks must be in (0, 4096]");

    auto opts = torch::TensorOptions()
                    .dtype(torch::kInt32)
                    .device(torch::kCUDA);
    auto out = torch::empty({num_blocks * 2}, opts);

    int device = 0;
    cudaGetDevice(&device);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    int num_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        (const void*)mk_dit_prefill_noop_kernel,
        /*block_size=*/32,
        /*dynamic_smem_bytes=*/0);

    int64_t max_concurrent = (int64_t)sm_count * num_blocks_per_sm;
    TORCH_CHECK(num_blocks <= max_concurrent,
                "cooperative launch limit exceeded: num_blocks=",
                num_blocks, " > sm_count(", sm_count, ")*per_sm(",
                num_blocks_per_sm, ")=", max_concurrent);

    int32_t* out_ptr = out.data_ptr<int32_t>();
    int num_blocks_i = static_cast<int>(num_blocks);
    void* args[] = {
        (void*)&out_ptr,
        (void*)&num_blocks_i,
    };

    dim3 grid(num_blocks_i);
    dim3 block(32);
    auto stream = at::cuda::getCurrentCUDAStream();

    cudaError_t err = cudaLaunchCooperativeKernel(
        (const void*)mk_dit_prefill_noop_kernel,
        grid, block, args, /*dynamic_smem_bytes=*/0, stream.stream());
    TORCH_CHECK(err == cudaSuccess, "cudaLaunchCooperativeKernel failed: ",
                cudaGetErrorString(err));

    return out;
}


// =========================================================================
// Step 2a — non-causal batched attention kernel.
//
// This is the DiT-shape self-attention primitive that the future megakernel
// will call as a __device__ sub-phase. We ship it standalone first so we
// can numerics-validate it in isolation against flash_attn_func(...,
// causal=False) before folding it into the cooperative kernel.
//
// Semantics: for each (batch, head_q, q_token), compute softmax over all
// (k_token) IN THE SAME BATCH of (Q[batch,q] · K[batch,k] * scale), weighted
// sum of V[batch,k]. No causal mask. GQA 16/2: head_q groups of size 8
// share head_kv = head_q / 8.
//
// Input shapes (all bf16 cuda, any stride, inner dim stride == 1):
//   Q: [B, S, Hq=16, D=128]
//   K: [B, S, Hkv=2, D=128]
//   V: [B, S, Hkv=2, D=128]
// Output: [B, S, Hq, D] contiguous (row-stride Q_DIM = Hq*D = 2048).
// The caller will typically view it as [N, Q_DIM] where N = B*S.
//
// Algorithm (flash-attention-2 style with online softmax, but with S small
// enough that one K-block covers all keys — so the K-tile loop is one iter
// at DiT shape and the softmax becomes single-pass):
//   1. Load Q tile [Q_BLOCK, D] for (batch, q_tile, head_q) into SMEM.
//   2. For each K-tile:
//        a. Load K, V tiles [K_BLOCK, D] for (batch, k_tile, head_kv) into SMEM.
//        b. S = Q @ K^T * scale  (WMMA bf16→fp32)
//        c. Mask padded K rows (k_global >= S_batch) to -INF.
//        d. Online softmax update: m, l, O.
//   3. O /= l, bf16, write.
//
// Warp layout: 4 warps per block (128 threads). At Q_BLOCK=16, K_BLOCK=32
// we partition (M=16, N=32) for QK^T as 1 M-tile × 2 N-tiles = 2 warps
// active (2 idle). For PV (M=16, D=128) we use all 4 warps for 1 M-tile ×
// 4 N-strips of 32 cols = 2 N_FRAGS each.
//
// SMEM budget (Q_BLOCK=16, K_BLOCK=32, D=128):
//   Q:   16×128×2  = 4 KB
//   K:   32×128×2  = 8 KB
//   V:   32×128×2  = 8 KB
//   S/P: 16×32×4   = 2 KB (reused for P bf16 = 1 KB)
//   O:   16×128×4  = 8 KB
//   m,l: 16×4×2    = 128 B
//   Total: ~30 KB — fits under 48 KB carveout, so we don't need
//   cudaFuncSetAttribute for this one.
// =========================================================================

template<int Q_BLOCK, int K_BLOCK, int D, int NUM_Q, int NUM_KV>
__global__ void vcpm_attention_noncausal_batched_kernel(
        const __nv_bfloat16* __restrict__ Q,
        int64_t q_b_stride, int64_t q_s_stride, int64_t q_h_stride,
        const __nv_bfloat16* __restrict__ K,
        int64_t k_b_stride, int64_t k_s_stride, int64_t k_h_stride,
        const __nv_bfloat16* __restrict__ V,
        int64_t v_b_stride, int64_t v_s_stride, int64_t v_h_stride,
        __nv_bfloat16* __restrict__ O,        // [B, S, Hq*D] row-major contiguous
        int o_b_stride, int o_s_stride,
        float scale, int B, int S) {
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WARPS = 4;
    constexpr int Q_PER_KV = NUM_Q / NUM_KV;  // 8

    static_assert(Q_BLOCK % WMMA_M == 0, "Q_BLOCK must be multiple of 16");
    static_assert(K_BLOCK % WMMA_N == 0, "K_BLOCK must be multiple of 16");
    static_assert(D % WMMA_K == 0,       "D must be multiple of 16");

    int q_tile  = blockIdx.x;
    int head_q  = blockIdx.y;
    int batch   = blockIdx.z;
    int head_kv = head_q / Q_PER_KV;
    int q_base  = q_tile * Q_BLOCK;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    // Early-exit: if this Q-tile is entirely beyond S, skip. (Still OK in
    // cooperative-launch sense because we're not using coop here — this
    // kernel is launched with plain <<<>>>.)
    if (q_base >= S) return;

    // Base pointers into this (batch, head_q) / (batch, head_kv) slice.
    const __nv_bfloat16* Q_bh = Q + batch * q_b_stride + head_q  * q_h_stride;
    const __nv_bfloat16* K_bh = K + batch * k_b_stride + head_kv * k_h_stride;
    const __nv_bfloat16* V_bh = V + batch * v_b_stride + head_kv * v_h_stride;
    __nv_bfloat16*       O_bh = O + batch * o_b_stride + head_q  * D;

    // SMEM layout.
    extern __shared__ __align__(16) char smem_raw[];
    char* smem_ptr = smem_raw;
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_ptr);
    smem_ptr += Q_BLOCK * D * sizeof(__nv_bfloat16);
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_ptr);
    smem_ptr += K_BLOCK * D * sizeof(__nv_bfloat16);
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_ptr);
    smem_ptr += K_BLOCK * D * sizeof(__nv_bfloat16);
    float* smem_s = reinterpret_cast<float*>(smem_ptr);       // [Q_BLOCK, K_BLOCK] fp32
    smem_ptr += Q_BLOCK * K_BLOCK * sizeof(float);
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_ptr);  // [Q_BLOCK, K_BLOCK] bf16
    smem_ptr += Q_BLOCK * K_BLOCK * sizeof(__nv_bfloat16);
    float* smem_o = reinterpret_cast<float*>(smem_ptr);       // [Q_BLOCK, D] fp32
    smem_ptr += Q_BLOCK * D * sizeof(float);
    float* smem_m = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += Q_BLOCK * sizeof(float);
    float* smem_l = reinterpret_cast<float*>(smem_ptr);

    // --------- Load Q tile [Q_BLOCK, D] into smem_q ---------
    #pragma unroll
    for (int i = tid; i < Q_BLOCK * D; i += 128) {
        int qi = i / D;
        int di = i % D;
        int n  = q_base + qi;
        if (n < S) {
            smem_q[qi * D + di] = Q_bh[n * q_s_stride + di];
        } else {
            smem_q[qi * D + di] = __float2bfloat16(0.f);
        }
    }

    // Init m, l, O.
    if (tid < Q_BLOCK) {
        smem_m[tid] = -INFINITY;
        smem_l[tid] = 0.f;
    }
    #pragma unroll
    for (int i = tid; i < Q_BLOCK * D; i += 128) {
        smem_o[i] = 0.f;
    }
    __syncthreads();

    int num_k_tiles = (S + K_BLOCK - 1) / K_BLOCK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k_base = kt * K_BLOCK;

        // --------- Load K, V tiles [K_BLOCK, D] into smem_k, smem_v ---------
        #pragma unroll
        for (int i = tid; i < K_BLOCK * D; i += 128) {
            int ki = i / D;
            int di = i % D;
            int n  = k_base + ki;
            if (n < S) {
                smem_k[ki * D + di] = K_bh[n * k_s_stride + di];
                smem_v[ki * D + di] = V_bh[n * v_s_stride + di];
            } else {
                smem_k[ki * D + di] = __float2bfloat16(0.f);
                smem_v[ki * D + di] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // --------- S = Q @ K^T * scale  (WMMA bf16 → fp32) ---------
        // Q is [Q_BLOCK=16, D=128] row-major.
        // K is [K_BLOCK, D] row-major in SMEM, viewed as col-major [D, K_BLOCK]
        //   with leading dim D for WMMA matrix_b.
        // Output S[Q_BLOCK, K_BLOCK] fp32 row-major.
        //
        // Warp partition: at Q_BLOCK=16 we have 1 M-tile. K_BLOCK=32 gives
        // 2 N-tiles. Use warps 0 and 1 for the two N-tiles; warps 2..3 idle.
        {
            constexpr int QK_N_TILES = K_BLOCK / WMMA_N;     // 2 at K_BLOCK=32
            constexpr int K_SUBITERS = D / WMMA_K;           // 8 at D=128

            if (warp_id < QK_N_TILES) {
                int warp_n = warp_id * WMMA_N;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_acc;
                fill_fragment(s_acc, 0.f);

                #pragma unroll
                for (int kk = 0; kk < K_SUBITERS; ++kk) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a;
                    load_matrix_sync(a, smem_q + kk * WMMA_K, D);

                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b;
                    // SMEM is [K_BLOCK, D] row-major ≡ [D, K_BLOCK] col-major,
                    // leading dim D. Sub-tile (k_off=kk*WMMA_K, n_off=warp_n):
                    //   ptr = smem_k + kk*WMMA_K + warp_n * D
                    load_matrix_sync(b, smem_k + kk * WMMA_K + warp_n * D, D);
                    mma_sync(s_acc, a, b, s_acc);
                }

                // Scale.
                #pragma unroll
                for (int e = 0; e < s_acc.num_elements; ++e) {
                    s_acc.x[e] *= scale;
                }
                store_matrix_sync(smem_s + warp_n, s_acc, K_BLOCK, mem_row_major);
            }
        }
        __syncthreads();

        // --------- K-padding mask + online softmax + P (bf16) ---------
        // One thread per row (tid < Q_BLOCK handles row tid).
        if (tid < Q_BLOCK) {
            int q_global = q_base + tid;
            float* s_row = smem_s + tid * K_BLOCK;

            float row_max = -INFINITY;
            #pragma unroll
            for (int kj = 0; kj < K_BLOCK; ++kj) {
                int k_global = k_base + kj;
                if (k_global >= S || q_global >= S) {
                    s_row[kj] = -INFINITY;
                } else {
                    if (s_row[kj] > row_max) row_max = s_row[kj];
                }
            }

            float old_m = smem_m[tid];
            float new_m = fmaxf(old_m, row_max);
            float rescale = (old_m == -INFINITY) ? 0.f : __expf(old_m - new_m);
            float row_sum = 0.f;

            #pragma unroll
            for (int kj = 0; kj < K_BLOCK; ++kj) {
                float e = (s_row[kj] == -INFINITY) ? 0.f : __expf(s_row[kj] - new_m);
                row_sum += e;
                smem_p[tid * K_BLOCK + kj] = __float2bfloat16(e);
            }

            smem_l[tid] = rescale * smem_l[tid] + row_sum;
            smem_m[tid] = new_m;

            // Rescale O[tid, :] in fp32.
            #pragma unroll
            for (int di = 0; di < D; ++di) {
                smem_o[tid * D + di] *= rescale;
            }
        }
        __syncthreads();

        // --------- O += P @ V  (WMMA bf16 → fp32, accumulate) ---------
        // P is [Q_BLOCK=16, K_BLOCK=32] bf16 in smem_p.
        // V is [K_BLOCK=32, D=128] bf16 in smem_v (row-major).
        // O: [Q_BLOCK=16, D=128] fp32 in smem_o (accumulate).
        //
        // Warp partition: 4 warps split the D=128 output into 4 N-strips of
        // 32 cols each. Each warp owns 2 N-frags of WMMA_N=16. M-tile = 1.
        {
            constexpr int K_SUBITERS = K_BLOCK / WMMA_K;     // 2 at K_BLOCK=32
            constexpr int PV_N_FRAGS = 2;                    // per warp
            constexpr int PV_STRIP   = D / WARPS;            // 32

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_acc[PV_N_FRAGS];
            int warp_n = warp_id * PV_STRIP;

            // Load current O accumulator from smem_o (fp32 row-major, stride D).
            #pragma unroll
            for (int ni = 0; ni < PV_N_FRAGS; ++ni) {
                load_matrix_sync(o_acc[ni],
                                 smem_o + warp_n + ni * WMMA_N,
                                 D, mem_row_major);
            }

            #pragma unroll
            for (int kk = 0; kk < K_SUBITERS; ++kk) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a;
                load_matrix_sync(a, smem_p + kk * WMMA_K, K_BLOCK);

                #pragma unroll
                for (int ni = 0; ni < PV_N_FRAGS; ++ni) {
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b;
                    int n_off = warp_n + ni * WMMA_N;
                    load_matrix_sync(b, smem_v + kk * WMMA_K * D + n_off, D);
                    mma_sync(o_acc[ni], a, b, o_acc[ni]);
                }
            }

            #pragma unroll
            for (int ni = 0; ni < PV_N_FRAGS; ++ni) {
                store_matrix_sync(smem_o + warp_n + ni * WMMA_N,
                                  o_acc[ni], D, mem_row_major);
            }
        }
        __syncthreads();
    }

    // --------- Final: O / l, write bf16 to global ---------
    #pragma unroll
    for (int i = tid; i < Q_BLOCK * D; i += 128) {
        int qi = i / D;
        int di = i % D;
        int n  = q_base + qi;
        if (n >= S) continue;
        float l_val = smem_l[qi];
        float val = (l_val > 0.f) ? (smem_o[i] / l_val) : 0.f;
        O_bh[n * o_s_stride + di] = __float2bfloat16(val);
    }
}


torch::Tensor vcpm_attention_noncausal_batched(
        const torch::Tensor& q,   // [B, S, Hq, D] bf16
        const torch::Tensor& k,   // [B, S, Hkv, D] bf16
        const torch::Tensor& v,   // [B, S, Hkv, D] bf16
        double scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q,k,v must be cuda");
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && k.dtype() == torch::kBFloat16
                && v.dtype() == torch::kBFloat16, "q,k,v must be bf16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q,k,v must be 4D (B, S, H, D)");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1,
                "inner dim stride must be 1");

    int B  = (int)q.size(0);
    int S  = (int)q.size(1);
    int Hq = (int)q.size(2);
    int D_ = (int)q.size(3);
    TORCH_CHECK((int)k.size(0) == B && (int)v.size(0) == B, "B mismatch");
    TORCH_CHECK((int)k.size(1) == S && (int)v.size(1) == S, "S mismatch");
    TORCH_CHECK((int)k.size(2) == (int)v.size(2), "Hkv mismatch");
    TORCH_CHECK((int)k.size(3) == D_ && (int)v.size(3) == D_, "D mismatch");
    int Hkv = (int)k.size(2);

    constexpr int NUM_Q = 16, NUM_KV = 2, D = 128;
    constexpr int Q_BLOCK = 16, K_BLOCK = 32;
    TORCH_CHECK(Hq == NUM_Q && Hkv == NUM_KV && D_ == D,
                "only supports Hq=", NUM_Q, " Hkv=", NUM_KV, " D=", D);

    // Output: (B, S, Hq, D) contiguous. Expose as (B, S, Hq*D) for the flat
    // [N, Q_DIM] view that the layer wrapper wants.
    auto o = torch::empty({B, S, Hq, D}, q.options());
    int o_b_stride = (int)o.stride(0);
    int o_s_stride = (int)o.stride(1);

    int num_q_tiles = (S + Q_BLOCK - 1) / Q_BLOCK;
    dim3 grid(num_q_tiles, NUM_Q, B);
    dim3 block(128, 1, 1);

    size_t smem_bytes =
          Q_BLOCK * D * sizeof(__nv_bfloat16)
        + K_BLOCK * D * sizeof(__nv_bfloat16)
        + K_BLOCK * D * sizeof(__nv_bfloat16)
        + Q_BLOCK * K_BLOCK * sizeof(float)
        + Q_BLOCK * K_BLOCK * sizeof(__nv_bfloat16)
        + Q_BLOCK * D * sizeof(float)
        + Q_BLOCK * sizeof(float) * 2;

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            vcpm_attention_noncausal_batched_kernel<Q_BLOCK, K_BLOCK, D, NUM_Q, NUM_KV>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    vcpm_attention_noncausal_batched_kernel<Q_BLOCK, K_BLOCK, D, NUM_Q, NUM_KV>
        <<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            q.stride(0), q.stride(1), q.stride(2),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            k.stride(0), k.stride(1), k.stride(2),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            v.stride(0), v.stride(1), v.stride(2),
            reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
            o_b_stride, o_s_stride,
            (float)scale, B, S);
    return o;
}


// =========================================================================
// Step 2b.0 — two-phase cooperative megakernel: RMSNorm + QKV GEMM.
//
// Proves the plumbing for the full 9-phase DiT megakernel (Step 2b.1+):
//   - Cooperative launch with fixed block count.
//   - grid.sync() between compute phases.
//   - Per-phase work-stealing tile partitioning (a block may be busy in
//     one phase and idle in another).
//   - SMEM region reused across phases via `extern __shared__`.
//
// Work:
//     ln_out = RMSNorm(hs,  w_in_ln)   [M, H]
//     qkv    = ln_out @ w_qkv^T         [M, QKV_DIM]
//
// Output:
//     scratch_a == ln_out     (useful for debugging; later phases overwrite)
//     scratch_b == qkv        (what the next phase, RoPE, will consume)
//
// Shapes (DiT): M_padded=64 (real=22, padded with zeros), H=1024,
//               QKV_DIM=2560. Tile layout: GEMM TM=64, TN=128, TK=32,
//               STAGES=4 → tile grid = (20, 1) = 20 tiles.
//
// We launch `MK_COOP_GRID = 64` cooperative CTAs, 128 threads each. Phase 1
// (RMSNorm) uses all 64 (one row per block). Phase 2 (GEMM) uses 20 (others
// idle at grid.sync). Future phases (gate_up GEMM has 64 tiles) saturate
// the 64-block grid.
//
// SMEM = the tuned GEMM's budget: 4×64×32×2 (A stages) + 4×128×32×2 (B
// stages) + 64×128×4 (C fp32) = 16384 + 32768 + 32768 = 81920 B. Requires
// cudaFuncSetAttribute(MaxDynamicSharedMemorySize) at setup.
// =========================================================================

static constexpr int MK_COOP_GRID = 64;      // fixed for Step 2b.x
static constexpr int MK_COOP_BLOCK = 128;    // 4 warps

// ---- SMEM helpers (duplicated from fused_layer_chained.cu intentionally
// so this translation unit is self-contained; the helpers are ~30 lines
// and re-validating them in-place is cheaper than cross-TU coupling). ----

__device__ __forceinline__ uint32_t mk_smem_addr_u32(const void* ptr) {
    uint32_t addr;
    asm("{ .reg .u64 u64_addr; cvta.to.shared.u64 u64_addr, %1; cvt.u32.u64 %0, u64_addr; }\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void mk_cp_async_16B(void* smem_dst, const void* gmem_src) {
    uint32_t smem = mk_smem_addr_u32(smem_dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem), "l"(gmem_src));
}

__device__ __forceinline__ void mk_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void mk_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mk_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}


template<int TM, int TK>
__device__ __forceinline__ void mk_load_A_tile(
        __nv_bfloat16* smem_A,
        const __nv_bfloat16* __restrict__ A,
        int block_m, int k0, int K) {
    static_assert(TM * TK % (128 * 8) == 0, "TM*TK must be multiple of 128*8");
    constexpr int ITERS = (TM * TK) / (128 * 8);
    constexpr int COLS_PER_ROW = TK / 8;
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        int linear = i * 128 + tid;
        int row = linear / COLS_PER_ROW;
        int col = (linear % COLS_PER_ROW) * 8;
        mk_cp_async_16B(
            smem_A + row * TK + col,
            A + (block_m + row) * K + (k0 + col));
    }
}

template<int TN, int TK>
__device__ __forceinline__ void mk_load_B_tile(
        __nv_bfloat16* smem_B,
        const __nv_bfloat16* __restrict__ B,
        int block_n, int k0, int K) {
    static_assert(TN * TK % (128 * 8) == 0, "TN*TK must be multiple of 128*8");
    constexpr int ITERS = (TN * TK) / (128 * 8);
    constexpr int COLS_PER_ROW = TK / 8;
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        int linear = i * 128 + tid;
        int row = linear / COLS_PER_ROW;
        int col = (linear % COLS_PER_ROW) * 8;
        mk_cp_async_16B(
            smem_B + row * TK + col,
            B + (block_n + row) * K + (k0 + col));
    }
}


// ---- Phase helpers (called from the megakernel with grid.sync between) ----

// Phase: RMSNorm over [N, H] rows. Each block handles one row (blockIdx.x).
// Blocks with blockIdx.x >= N simply return. Writes y = w * x / sqrt(mean(x^2)+eps).
// THREADS is the block dim (we always launch 128 in this kernel, even though
// the stride-over-H pattern only cares that THREADS divides H).
template<int H, int THREADS>
__device__ __forceinline__ void mk_phase_rmsnorm(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ w,
        __nv_bfloat16* __restrict__ y,
        int N, float eps) {
    int tid = threadIdx.x;
    __shared__ float sm[THREADS];

    // Work-stealing over rows: each block handles rows [blockIdx.x, +gridDim.x, +2*gridDim.x, ...].
    // Required because M can exceed gridDim.x (e.g. M=256 with grid=64).
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        const __nv_bfloat16* xr = x + row * H;
        __nv_bfloat16*       yr = y + row * H;

        float acc = 0.f;
        #pragma unroll
        for (int i = tid; i < H; i += THREADS) {
            float v = __bfloat162float(xr[i]);
            acc += v * v;
        }
        sm[tid] = acc;
        __syncthreads();
        for (int s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) sm[tid] += sm[tid + s];
            __syncthreads();
        }
        float mean_sq = sm[0] / (float)H;
        float inv = rsqrtf(mean_sq + eps);

        #pragma unroll
        for (int i = tid; i < H; i += THREADS) {
            float v  = __bfloat162float(xr[i]);
            float wv = __bfloat162float(w[i]);
            yr[i] = __float2bfloat16(v * inv * wv);
        }
        __syncthreads();  // before reusing `sm` for the next row
    }
}


// Phase: tuned bf16 GEMM tile. Each block processes one (tile_m, tile_n).
// Work-stealing over the total tile count so blocks can pick up multiple
// tiles if tile_count > gridDim.x. smem_raw must be the full megakernel
// SMEM region; the GEMM aliases it as A/B stages + C fp32 staging.
template<int TM, int TN, int TK, int STAGES>
__device__ __forceinline__ void mk_phase_gemm(
        const __nv_bfloat16* __restrict__ A,  // [M, K]
        const __nv_bfloat16* __restrict__ B,  // [N, K]
        __nv_bfloat16* __restrict__ C,        // [M, N]
        int M, int N, int K,
        char* smem_raw) {
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WARPS = 4;
    constexpr int N_FRAGS = TN / WMMA_N;
    constexpr int K_SUBITERS = TK / WMMA_K;

    static_assert(TM == WMMA_M * WARPS, "TM must equal WMMA_M * WARPS");
    static_assert(TN % WMMA_N == 0, "TN must be multiple of WMMA_N");
    static_assert(TK % WMMA_K == 0, "TK must be multiple of WMMA_K");

    int tiles_m = M / TM;
    int tiles_n = N / TN;
    int total_tiles = tiles_m * tiles_n;

    // SMEM layout (identical to fused_layer_chained tuned kernel).
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_B = smem_A + STAGES * TM * TK;
    float*         smem_C = reinterpret_cast<float*>(smem_B + STAGES * TN * TK);

    int warp_id = threadIdx.x >> 5;
    int warp_m  = warp_id * WMMA_M;

    // Work-stealing: each block grabs tiles with stride = gridDim.x.
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        int tm = tile / tiles_n;
        int tn = tile % tiles_n;
        int block_m = tm * TM;
        int block_n = tn * TN;

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[N_FRAGS];
        #pragma unroll
        for (int i = 0; i < N_FRAGS; ++i) fill_fragment(acc[i], 0.f);

        int num_k_tiles = K / TK;
        int prologue = (STAGES - 1) < num_k_tiles ? (STAGES - 1) : num_k_tiles;

        #pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < prologue) {
                int k0 = s * TK;
                mk_load_A_tile<TM, TK>(smem_A + s * TM * TK, A, block_m, k0, K);
                mk_load_B_tile<TN, TK>(smem_B + s * TN * TK, B, block_n, k0, K);
            }
            mk_cp_async_commit();
        }

        int stage = 0;
        for (int kt = 0; kt < num_k_tiles; ++kt) {
            mk_cp_async_wait_group<STAGES - 2>();
            __syncthreads();

            int kt_next = kt + (STAGES - 1);
            if (kt_next < num_k_tiles) {
                int next_stage = (stage + STAGES - 1) % STAGES;
                int k0_next = kt_next * TK;
                mk_load_A_tile<TM, TK>(smem_A + next_stage * TM * TK, A, block_m, k0_next, K);
                mk_load_B_tile<TN, TK>(smem_B + next_stage * TN * TK, B, block_n, k0_next, K);
            }
            mk_cp_async_commit();

            __nv_bfloat16* a_base = smem_A + stage * TM * TK + warp_m * TK;
            __nv_bfloat16* b_base = smem_B + stage * TN * TK;

            #pragma unroll
            for (int kk = 0; kk < K_SUBITERS; ++kk) {
                int k_off = kk * WMMA_K;
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                load_matrix_sync(a_frag, a_base + k_off, TK);

                #pragma unroll
                for (int ni = 0; ni < N_FRAGS; ++ni) {
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
                    load_matrix_sync(b_frag, b_base + k_off + ni * WMMA_N * TK, TK);
                    mma_sync(acc[ni], a_frag, b_frag, acc[ni]);
                }
            }

            stage = (stage + 1) % STAGES;
        }

        mk_cp_async_wait_all();
        __syncthreads();

        #pragma unroll
        for (int ni = 0; ni < N_FRAGS; ++ni) {
            store_matrix_sync(smem_C + warp_m * TN + ni * WMMA_N,
                              acc[ni], TN, mem_row_major);
        }
        __syncthreads();

        constexpr int ELEMS_PER_THREAD = (TM * TN) / 128;
        int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            int linear = i * 128 + tid;
            int r = linear / TN;
            int c = linear % TN;
            int gr = block_m + r;
            int gc = block_n + c;
            C[gr * N + gc] = __float2bfloat16(smem_C[linear]);
        }
        __syncthreads();  // before next tile reuses smem_C
    }
}


// ---- Step 2b.0 megakernel: RMSNorm + QKV_GEMM ----
//
// Shapes locked to DiT layer: H=1024, QKV_DIM=2560. M and K_gemm passed in
// so the same kernel works for base_lm (H=2048) later.
extern "C" __global__ __launch_bounds__(MK_COOP_BLOCK, 2)
void mk_dit_step2b0_kernel(
        const __nv_bfloat16* __restrict__ hs,        // [M, H]
        const __nv_bfloat16* __restrict__ w_in_ln,   // [H]
        const __nv_bfloat16* __restrict__ w_qkv,     // [QKV_DIM, H]
        __nv_bfloat16* __restrict__ scratch_a,       // [M, H]  — ln_out
        __nv_bfloat16* __restrict__ scratch_b,       // [M, QKV_DIM] — qkv
        int M, int H, int QKV_DIM, float rms_eps) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ __align__(16) char smem[];

    // Phase 1: ln_out = RMSNorm(hs, w_in_ln).
    // We locked H=1024 here; a future build will template on H for base_lm.
    mk_phase_rmsnorm<1024, MK_COOP_BLOCK>(
        hs, w_in_ln, scratch_a, M, rms_eps);
    grid.sync();

    // Phase 2: qkv = ln_out @ w_qkv^T.
    // (TM=64, TN=128, TK=32, STAGES=4) identical to the chained tuned GEMM.
    mk_phase_gemm<64, 128, 32, 4>(
        scratch_a, w_qkv, scratch_b,
        M, QKV_DIM, H, smem);
}


// Launcher. Caller provides padded hs/scratch (M padded to TM=64 multiple)
// and bf16 weights. Returns `scratch_b` = qkv_out (caller slices to real
// rows if M has padding).
static void mk_dit_step2b0_launch(
        const torch::Tensor& hs,         // [M, H] bf16
        const torch::Tensor& w_in_ln,    // [H] bf16
        const torch::Tensor& w_qkv,      // [QKV_DIM, H] bf16
        torch::Tensor& scratch_a,        // [M, H] bf16 (ln_out output)
        torch::Tensor& scratch_b,        // [M, QKV_DIM] bf16 (qkv output)
        double rms_eps) {
    TORCH_CHECK(hs.is_cuda() && hs.dtype() == torch::kBFloat16
                && hs.is_contiguous() && hs.dim() == 2, "hs [M,H] bf16 cuda contig");
    TORCH_CHECK(w_in_ln.is_cuda() && w_in_ln.dtype() == torch::kBFloat16
                && w_in_ln.is_contiguous() && w_in_ln.dim() == 1, "w_in_ln [H] bf16");
    TORCH_CHECK(w_qkv.is_cuda() && w_qkv.dtype() == torch::kBFloat16
                && w_qkv.is_contiguous() && w_qkv.dim() == 2, "w_qkv [QKV_DIM,H] bf16");
    int M = (int)hs.size(0);
    int H = (int)hs.size(1);
    int QKV_DIM = (int)w_qkv.size(0);
    TORCH_CHECK((int)w_qkv.size(1) == H, "w_qkv H mismatch");
    TORCH_CHECK((int)w_in_ln.numel() == H, "w_in_ln H mismatch");
    TORCH_CHECK(H == 1024, "Step 2b.0: only DiT H=1024 supported");
    TORCH_CHECK(M % 64 == 0, "M must be multiple of 64");
    TORCH_CHECK(H % 32 == 0 && QKV_DIM % 128 == 0, "H%32 QKV%128");

    TORCH_CHECK(scratch_a.is_cuda() && scratch_a.dtype() == torch::kBFloat16
                && scratch_a.is_contiguous()
                && (int)scratch_a.size(0) == M && (int)scratch_a.size(1) == H,
                "scratch_a [M,H] bf16 cuda contig");
    TORCH_CHECK(scratch_b.is_cuda() && scratch_b.dtype() == torch::kBFloat16
                && scratch_b.is_contiguous()
                && (int)scratch_b.size(0) == M && (int)scratch_b.size(1) == QKV_DIM,
                "scratch_b [M,QKV_DIM] bf16 cuda contig");

    // SMEM sizing: driven by the GEMM (RMSNorm's tiny reduction buffer is a
    // static __shared__ inside the __device__ function, not counted in
    // dynamic SMEM).
    constexpr int TM = 64, TN = 128, TK = 32, STAGES = 4;
    size_t smem_bytes =
          STAGES * TM * TK * sizeof(__nv_bfloat16)
        + STAGES * TN * TK * sizeof(__nv_bfloat16)
        + TM * TN * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            mk_dit_step2b0_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }

    // Cooperative launch check: total blocks ≤ sm_count * blocks_per_sm.
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, (const void*)mk_dit_step2b0_kernel,
        MK_COOP_BLOCK, smem_bytes);
    int64_t max_concurrent = (int64_t)sm_count * blocks_per_sm;
    TORCH_CHECK(MK_COOP_GRID <= max_concurrent,
                "cooperative limit exceeded: grid=", (int)MK_COOP_GRID,
                " > sm_count(", sm_count, ")*per_sm(",
                blocks_per_sm, ")=", max_concurrent,
                ". Reduce grid or SMEM.");

    // Launch args (pointers to the args must live until the launch returns;
    // taking addresses of locals is fine here because cudaLaunchCooperative
    // copies them synchronously into the command stream).
    const __nv_bfloat16* hs_p     = reinterpret_cast<const __nv_bfloat16*>(hs.data_ptr());
    const __nv_bfloat16* w_ln_p   = reinterpret_cast<const __nv_bfloat16*>(w_in_ln.data_ptr());
    const __nv_bfloat16* w_qkv_p  = reinterpret_cast<const __nv_bfloat16*>(w_qkv.data_ptr());
    __nv_bfloat16*       sa_p     = reinterpret_cast<__nv_bfloat16*>(scratch_a.data_ptr());
    __nv_bfloat16*       sb_p     = reinterpret_cast<__nv_bfloat16*>(scratch_b.data_ptr());
    float                eps      = (float)rms_eps;

    void* args[] = {
        (void*)&hs_p, (void*)&w_ln_p, (void*)&w_qkv_p,
        (void*)&sa_p, (void*)&sb_p,
        (void*)&M,    (void*)&H,     (void*)&QKV_DIM,
        (void*)&eps
    };

    dim3 grid(MK_COOP_GRID);
    dim3 block(MK_COOP_BLOCK);
    auto stream = at::cuda::getCurrentCUDAStream();

    cudaError_t err = cudaLaunchCooperativeKernel(
        (const void*)mk_dit_step2b0_kernel,
        grid, block, args, (size_t)smem_bytes, stream.stream());
    TORCH_CHECK(err == cudaSuccess, "cudaLaunchCooperativeKernel failed: ",
                cudaGetErrorString(err));
}


// Python-facing wrapper. Takes M-padded inputs; caller is responsible for
// any pre/post-padding (the chained FusedLayer's _pad_M_to(64) convention).
// Returns (ln_out, qkv) so the Python test can verify both phases.
std::tuple<torch::Tensor, torch::Tensor> mk_dit_step2b0_rmsnorm_qkv(
        const torch::Tensor& hs,
        const torch::Tensor& w_in_ln,
        const torch::Tensor& w_qkv,
        double rms_eps) {
    int M = (int)hs.size(0);
    int H = (int)hs.size(1);
    int QKV_DIM = (int)w_qkv.size(0);
    auto opts = hs.options();
    auto scratch_a = torch::empty({M, H},       opts);
    auto scratch_b = torch::empty({M, QKV_DIM}, opts);
    mk_dit_step2b0_launch(hs, w_in_ln, w_qkv, scratch_a, scratch_b, rms_eps);
    return std::make_tuple(scratch_a, scratch_b);
}

}  // namespace voxcpm_fast


// =========================================================================
// PyBind11 module.
// =========================================================================

PYBIND11_MODULE(mk_dit_prefill_ext, m) {
    m.def("noop", &voxcpm_fast::mk_dit_prefill_noop,
          "P2.5.2 scaffolding: cooperative-launch no-op that writes block "
          "indices before and after a grid.sync().",
          pybind11::arg("num_blocks"));

    m.def("attention_noncausal_batched",
          &voxcpm_fast::vcpm_attention_noncausal_batched,
          "P2.5.2 Step 2a: batched non-causal self-attention. Inputs are 4D "
          "strided bf16 (B, S, H, D) — same shape flash_attn_func accepts. "
          "GQA 16/2, head_dim 128. Each Q only attends to K/V IN THE SAME "
          "BATCH (no cross-batch attention). Numerics match flash_attn_func"
          "(..., causal=False) within bf16 rounding.",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("scale"));

    m.def("step2b0_rmsnorm_qkv",
          &voxcpm_fast::mk_dit_step2b0_rmsnorm_qkv,
          "P2.5.2 Step 2b.0: cooperative 2-phase megakernel — RMSNorm + QKV "
          "GEMM with cg::this_grid().sync() between phases. Proves the "
          "plumbing for the full 9-phase DiT megakernel. Returns "
          "(ln_out, qkv) both bf16 [M, H] and [M, QKV_DIM]. Caller must "
          "pre-pad M to multiple of 64.",
          pybind11::arg("hs"),
          pybind11::arg("w_in_ln"),
          pybind11::arg("w_qkv"),
          pybind11::arg("rms_eps") = 1e-6);
}
