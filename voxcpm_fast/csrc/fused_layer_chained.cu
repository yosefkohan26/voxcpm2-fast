// voxcpm_fast/csrc/fused_layer_chained.cu
//
// P2.2 (rewrite, orchestrator hands-on) — non-causal transformer layer split
// into individual __global__ kernels that the Python side chains in one CUDA
// stream. NO cooperative launch, NO grid_sync; inter-stage ordering comes
// from CUDA stream FIFO semantics. Each kernel is independently testable
// and compute-sanitizer attachable.
//
// Layer composition (feat_encoder shape):
//   hidden=1024, heads=16, kv_heads=2, head_dim=128, intermediate=4096.
//
//   1. input_layernorm:          RMSNorm
//   2. qkv_proj:                 GEMM 1024 -> 2560 (Q=2048 || K=256 || V=256)
//   3. rotary_emb:               RoPE on Q and K halves of qkv
//   4. attention:                non-causal online softmax
//   5. o_proj:                   GEMM 2048 -> 1024
//   6. residual:                 x += attn
//   7. post_attention_layernorm: RMSNorm
//   8. gate_up_proj:             GEMM 1024 -> 8192 (gate || up)
//   9. SiLU·mul:                 silu(gate) * up -> 4096
//  10. down_proj:                GEMM 4096 -> 1024
//  11. residual:                 x += mlp
//
// Built for sm_120 (RTX 5090 Blackwell). GEMM uses nvcuda::wmma 16x16x16
// bf16 fragments with fp32 accumulation — the "fallback" from the P2.2
// prompt. WGMMA upgrade is P2.5 fuel.
//
// Binding model: pybind11 via torch.utils.cpp_extension. Each host-side
// launcher takes torch::Tensor args, extracts CUDA pointers, launches on
// the tensor's default stream.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <cstdio>

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


// ===========================================================================
// Sanity kernel — multiplies bf16 input by 2. Used to prove the build
// pipeline + torch binding + stream handoff work before we add any real
// compute. Will be removed once real kernels are validated, but keep it
// during bringup.
// ===========================================================================

extern "C" __global__ void vcpm_times_two_kernel(
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ y,
        int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = __bfloat162float(x[i]);
    y[i] = __float2bfloat16(v * 2.f);
}


torch::Tensor vcpm_times_two(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be cuda");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    auto y = torch::empty_like(x);
    int total = (int)x.numel();
    int block = 256;
    int grid = (total + block - 1) / block;
    auto stream = at::cuda::getCurrentCUDAStream();
    vcpm_times_two_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        total);
    return y;
}


// ===========================================================================
// RMSNorm: y[t, i] = w[i] * x[t, i] / sqrt(mean(x^2, axis=-1) + eps)
//
// One block per row. Block has THREADS threads; each thread strides over
// the hidden dim. Reduction in shared memory. Hidden dim H is a kernel
// template parameter so the compiler can fully unroll the inner loops.
// ===========================================================================

template<int H, int THREADS>
__global__ void vcpm_rmsnorm_kernel(
        const __nv_bfloat16* __restrict__ x,   // [N, H]
        const __nv_bfloat16* __restrict__ w,   // [H]
        __nv_bfloat16* __restrict__ y,         // [N, H]
        float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const __nv_bfloat16* xr = x + row * H;
    __nv_bfloat16*       yr = y + row * H;

    float acc = 0.f;
    #pragma unroll
    for (int i = tid; i < H; i += THREADS) {
        float v = __bfloat162float(xr[i]);
        acc += v * v;
    }
    __shared__ float sm[THREADS];
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
}


torch::Tensor vcpm_rmsnorm(
        const torch::Tensor& x,
        const torch::Tensor& w,
        double eps) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x,w must be cuda");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(w.dtype() == torch::kBFloat16, "w must be bf16");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be [N, H]");
    int N = (int)x.size(0);
    int H = (int)x.size(1);
    TORCH_CHECK(w.numel() == H, "w must match H");

    auto y = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int THREADS = 256;
    // 1 block per row is faster integrated than multi-row-per-block (tried
    // 16 rows/block = 8 blocks for M=128 — regressed 28-layer wall by 0.5 ms
    // because the fewer blocks interfere with stream-level pipelining of
    // neighboring kernels). Keep the simple per-row form.
    if (H == 1024) {
        vcpm_rmsnorm_kernel<1024, THREADS><<<N, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            (float)eps);
    } else if (H == 2048) {
        vcpm_rmsnorm_kernel<2048, THREADS><<<N, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            (float)eps);
    } else {
        TORCH_CHECK(false, "unsupported H for rmsnorm: ", H);
    }
    return y;
}


// ===========================================================================
// GEMM bf16: C[M, N] = A[M, K] @ B[N, K]^T
//
// Weights are stored as [out, in] row-major (upstream nn.Linear convention),
// so we compute A @ W^T where W is loaded as col-major with stride K.
//
// Tile: TM=16, TN=16, TK=16. One warp per 16x16 output tile. One block =
// 4 warps, each covering a different tile. Grid = (ceil(N/TN), ceil(M/TM))
// then collapsed into a 1D grid-strided loop for padding robustness at the
// M/N edges.
// ===========================================================================

template<int WARPS_PER_BLOCK>
__global__ void vcpm_gemm_bf16_kernel(
        const __nv_bfloat16* __restrict__ A,   // [M, K] row-major
        const __nv_bfloat16* __restrict__ B,   // [N, K] row-major (weight)
        __nv_bfloat16* __restrict__ C,         // [M, N] row-major
        int M, int N, int K) {
    constexpr int TM = 16;
    constexpr int TN = 16;
    constexpr int TK = 16;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;

    int tiles_n = (N + TN - 1) / TN;
    int tiles_m = (M + TM - 1) / TM;
    int total_tiles = tiles_m * tiles_n;

    int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int stride = gridDim.x * WARPS_PER_BLOCK;

    // Per-warp SMEM output staging. WMMA store_matrix_sync requires
    // shared or global memory as destination; local/stack memory is UB.
    __shared__ float smem_tile[WARPS_PER_BLOCK * TM * TN];
    float* my_tile = smem_tile + warp_id * (TM * TN);

    for (int tile = global_warp; tile < total_tiles; tile += stride) {
        int tm = tile / tiles_n;
        int tn = tile % tiles_n;
        int m0 = tm * TM;
        int n0 = tn * TN;

        fragment<accumulator, TM, TN, TK, float> acc;
        fill_fragment(acc, 0.f);

        // Full tiles only (caller is responsible for padding M and N to
        // multiples of 16). We TORCH_CHECK on the host side.
        for (int k0 = 0; k0 < K; k0 += TK) {
            fragment<matrix_a, TM, TN, TK, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, TM, TN, TK, __nv_bfloat16, col_major> b_frag;
            load_matrix_sync(a_frag, A + m0 * K + k0, K);
            // B is [N, K] row-major; reading as col_major with stride K
            // treats it as [K, N]-col-major, i.e. effectively B^T.
            load_matrix_sync(b_frag, B + n0 * K + k0, K);
            mma_sync(acc, a_frag, b_frag, acc);
        }

        // Store into shared memory (valid target for WMMA), then convert
        // to bf16 and stream to global with per-tile bounds checks so a
        // ragged M or N is still safe.
        store_matrix_sync(my_tile, acc, TN, mem_row_major);
        // __syncwarp would be needed if we shared this tile across warps;
        // here it's warp-private, so lane-local reads right after the
        // store are already ordered within the warp by mma's lane fence.
        for (int i = lane_id; i < TM * TN; i += 32) {
            int r = i / TN;
            int c = i % TN;
            int gr = m0 + r;
            int gc = n0 + c;
            if (gr < M && gc < N) {
                C[gr * N + gc] = __float2bfloat16(my_tile[i]);
            }
        }
    }
}


torch::Tensor vcpm_gemm_bf16(
        const torch::Tensor& A,   // [M, K]
        const torch::Tensor& B) { // [N, K]  (weight, [out, in])
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A,B must be cuda");
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16,
                "A,B must be bf16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A,B must be 2D");
    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);
    TORCH_CHECK((int)B.size(1) == K, "B.size(1) must equal A.size(1)");
    TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16 for WMMA");

    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int tiles_m = (M + 15) / 16;
    int tiles_n = (N + 15) / 16;
    int total_tiles = tiles_m * tiles_n;
    int num_blocks = (total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    // Cap grid so we don't over-launch tiny matmuls.
    if (num_blocks > 2048) num_blocks = 2048;

    vcpm_gemm_bf16_kernel<WARPS_PER_BLOCK><<<num_blocks, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, N, K);
    return C;
}


// ===========================================================================
// SiLU·mul: out[t, i] = silu(gate[t, i]) * up[t, i]
// Input `gu` is [N, 2*I], first I cols = gate, second I cols = up (matches
// upstream MergedColumnParallelLinear concat order).
// ===========================================================================

// silu_mul: simple 1-element-per-thread form. Tried 16-elements-per-thread
// vectorization — isolated faster (~14 µs vs ~22) but regressed 28-layer
// wall by ~0.2 ms because the smaller grid interacted poorly with the
// adjacent gate_up/down_gemm kernels' SM scheduling. Keep the simple form.
__global__ void vcpm_silu_mul_kernel(
        const __nv_bfloat16* __restrict__ gu,  // [N, 2*I]
        __nv_bfloat16* __restrict__ out,       // [N, I]
        int N, int I) {
    int total = N * I;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int t = i / I;
    int j = i % I;
    float g = __bfloat162float(gu[t * (2 * I) + j]);
    float u = __bfloat162float(gu[t * (2 * I) + I + j]);
    float s = g / (1.f + __expf(-g));
    out[i] = __float2bfloat16(s * u);
}


torch::Tensor vcpm_silu_mul(const torch::Tensor& gu) {
    TORCH_CHECK(gu.is_cuda(), "gu must be cuda");
    TORCH_CHECK(gu.dtype() == torch::kBFloat16, "gu must be bf16");
    TORCH_CHECK(gu.is_contiguous(), "gu must be contiguous");
    TORCH_CHECK(gu.dim() == 2, "gu must be [N, 2*I]");
    int N = (int)gu.size(0);
    int G = (int)gu.size(1);
    TORCH_CHECK(G % 2 == 0, "gu last dim must be even");
    int I = G / 2;
    auto out = torch::empty({N, I}, gu.options());
    int total = N * I;
    int block = 256;
    int grid = (total + block - 1) / block;
    auto stream = at::cuda::getCurrentCUDAStream();
    vcpm_silu_mul_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gu.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        N, I);
    return out;
}


// ===========================================================================
// Residual add in-place: a += b (bf16).
// ===========================================================================

__global__ void vcpm_residual_add_kernel(
        __nv_bfloat16* __restrict__ a,
        const __nv_bfloat16* __restrict__ b,
        int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float va = __bfloat162float(a[i]);
    float vb = __bfloat162float(b[i]);
    a[i] = __float2bfloat16(va + vb);
}


void vcpm_residual_add(torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a,b must be cuda");
    TORCH_CHECK(a.dtype() == torch::kBFloat16 && b.dtype() == torch::kBFloat16,
                "a,b must be bf16");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "must be contiguous");
    TORCH_CHECK(a.numel() == b.numel(), "a,b size mismatch");
    int total = (int)a.numel();
    int block = 256;
    int grid = (total + block - 1) / block;
    auto stream = at::cuda::getCurrentCUDAStream();
    vcpm_residual_add_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        total);
}


// ===========================================================================
// RoPE (MiniCPM LongRoPE) applied to Q and K slices of a packed [N, QKV_DIM]
// tensor in-place. V is left untouched.
//
// Layout: qkv[token, :] = [Q (num_heads * head_dim) || K (num_kv_heads * head_dim) || V (...)]
//
// cos/sin caches are fp32 [max_pos, head_dim] and already include the
// "repeat" doubling of the half-freq arrays (upstream pattern).
//
// Per-token: for each head, split into two halves and apply:
//   x1, x2 = chunk(x, 2, dim=-1)
//   rot = cat((-x2, x1), dim=-1)
//   y = x * cos + rot * sin
// with fp32 compute, bf16 round-trip.
// ===========================================================================

__global__ void vcpm_rope_kernel(
        __nv_bfloat16* __restrict__ qkv,       // [N, QKV_DIM], inplace
        const float*    __restrict__ cos_cache, // [max_pos, head_dim]
        const float*    __restrict__ sin_cache, // [max_pos, head_dim]
        const int32_t*  __restrict__ positions, // [N]
        int N,
        int num_heads, int num_kv_heads, int head_dim,
        int q_dim, int kv_dim, int qkv_dim) {
    // One warp handles (token, head). Two kinds of heads: Q heads and K heads.
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    int heads_per_token = num_heads + num_kv_heads;
    int total_warps = N * heads_per_token;
    if (warp_id_global >= total_warps) return;

    int tok = warp_id_global / heads_per_token;
    int head_in_tok = warp_id_global % heads_per_token;

    int pos = positions[tok];
    const float* cos_row = cos_cache + pos * head_dim;
    const float* sin_row = sin_cache + pos * head_dim;

    // Compute base offset within qkv for this (token, head).
    __nv_bfloat16* base;
    if (head_in_tok < num_heads) {
        // Q head.
        int h = head_in_tok;
        base = qkv + tok * qkv_dim + h * head_dim;
    } else {
        // K head.
        int h = head_in_tok - num_heads;
        base = qkv + tok * qkv_dim + q_dim + h * head_dim;
    }

    int half = head_dim / 2;
    // Each lane handles pairs (idx, idx+half). For head_dim=128, each lane
    // owns 2 pairs (4 elements total). Read ALL pairs into registers first,
    // then write — avoids the iteration-to-iteration RMW race where a later
    // lane iter reads base[idx-half] that a prior iter has already written.
    constexpr int MAX_PAIRS_PER_LANE = 4;  // head_dim <= 256 supported
    float x_lo[MAX_PAIRS_PER_LANE];
    float x_hi[MAX_PAIRS_PER_LANE];
    int   lo_idx[MAX_PAIRS_PER_LANE];
    int n_pairs = 0;
    // Pair iteration: lane owns lo_idx = lane, lane+32, lane+64, ... while
    // lo_idx < half.
    for (int p = lane; p < half; p += 32) {
        x_lo[n_pairs] = __bfloat162float(base[p]);
        x_hi[n_pairs] = __bfloat162float(base[p + half]);
        lo_idx[n_pairs] = p;
        n_pairs++;
    }
    // Write phase: compute both outputs per pair, then store both.
    for (int i = 0; i < n_pairs; ++i) {
        int lo = lo_idx[i];
        int hi = lo + half;
        float c_lo = cos_row[lo];
        float s_lo = sin_row[lo];
        float c_hi = cos_row[hi];
        float s_hi = sin_row[hi];
        // rot at idx<half uses paired = -x_hi; at idx>=half uses paired = +x_lo.
        float y_lo = x_lo[i] * c_lo + (-x_hi[i]) * s_lo;
        float y_hi = x_hi[i] * c_hi + ( x_lo[i]) * s_hi;
        base[lo] = __float2bfloat16(y_lo);
        base[hi] = __float2bfloat16(y_hi);
    }
}


void vcpm_rope_inplace(
        torch::Tensor& qkv,              // [N, QKV_DIM] bf16
        const torch::Tensor& cos_cache,  // [max_pos, head_dim] fp32
        const torch::Tensor& sin_cache,  // [max_pos, head_dim] fp32
        const torch::Tensor& positions,  // [N] int32 or int64 -> we convert
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t head_dim) {
    TORCH_CHECK(qkv.is_cuda() && cos_cache.is_cuda() && sin_cache.is_cuda()
                && positions.is_cuda(), "tensors must be cuda");
    TORCH_CHECK(qkv.dtype() == torch::kBFloat16, "qkv must be bf16");
    TORCH_CHECK(cos_cache.dtype() == torch::kFloat32
                && sin_cache.dtype() == torch::kFloat32,
                "cos/sin must be fp32");
    TORCH_CHECK(qkv.is_contiguous() && cos_cache.is_contiguous()
                && sin_cache.is_contiguous() && positions.is_contiguous(),
                "inputs must be contiguous");

    auto pos32 = positions.dtype() == torch::kInt32
                 ? positions
                 : positions.to(torch::kInt32);

    int N = (int)qkv.size(0);
    int q_dim  = (int)(num_heads * head_dim);
    int kv_dim = (int)(num_kv_heads * head_dim);
    int qkv_dim = q_dim + 2 * kv_dim;
    TORCH_CHECK((int)qkv.size(1) == qkv_dim, "qkv last dim mismatch");
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");

    int heads_per_token = (int)(num_heads + num_kv_heads);
    int total_warps = N * heads_per_token;
    int block = 256;             // 8 warps/block — halves block count
    int warps_per_block = block / 32;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    vcpm_rope_kernel<<<blocks, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr()),
        cos_cache.data_ptr<float>(),
        sin_cache.data_ptr<float>(),
        pos32.data_ptr<int32_t>(),
        N, (int)num_heads, (int)num_kv_heads, (int)head_dim,
        q_dim, kv_dim, qkv_dim);
}


// ===========================================================================
// P2.5.1.a — Tuned bf16 GEMM with cp.async pipelining and warp-group tiling.
//
// Replaces vcpm_gemm_bf16 for our 4 target shapes:
//   (M=112, N=2560,  K=2048) — qkv_proj
//   (M=112, N=2048,  K=2048) — o_proj
//   (M=112, N=12288, K=2048) — gate_up_proj
//   (M=112, N=2048,  K=6144) — down_proj
// All four have N divisible by TN=128. Caller pads M to TM=64 multiple.
//
// Architecture:
//   - Block tile: TM=64, TN=128, TK=32
//   - 4 warps per block (128 threads), each warp owns 16 rows × 128 cols
//   - 8 WMMA 16x16x16 N-fragments per warp, 2 K-sub-iterations per stage
//   - 3-stage cp.async software pipeline over K
//   - fp32 accumulation in WMMA fragments (in registers, 256B/thread)
//   - Output staged through SMEM, converted to bf16 on the final epilogue
//
// Per-block SMEM budget:
//   A tiles: 3 × 64 × 32 × 2 = 12 KB
//   B tiles: 3 × 128 × 32 × 2 = 24 KB
//   C tile:  64 × 128 × 4 = 32 KB (fp32)
//   Total:  68 KB (of 228 KB available on sm_120)
//
// Grid: (ceil(N/TN), ceil(M/TM)). For our shapes: qkv=(20,2)=40 blocks,
// o=(16,2)=32, gate_up=(96,2)=192, down=(16,2)=32. All fit one wave on
// 170 SMs except gate_up which is 1.13 waves.
// ===========================================================================

__device__ __forceinline__ uint32_t smem_addr_u32(const void* ptr) {
    uint32_t addr;
    asm("{ .reg .u64 u64_addr; cvta.to.shared.u64 u64_addr, %1; cvt.u32.u64 %0, u64_addr; }\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
    uint32_t smem = smem_addr_u32(smem_dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}


// Load a [TM, TK] bf16 tile from A at (block_m, k0) into SMEM [TM, TK] row-major.
// Each thread issues ceil(TM*TK/(128*8)) cp.async 16-byte loads.
template<int TM, int TK>
__device__ __forceinline__ void load_A_tile(
        __nv_bfloat16* smem_A,           // destination [TM, TK] row-major
        const __nv_bfloat16* __restrict__ A,
        int block_m, int k0, int K) {
    static_assert(TM * TK % (128 * 8) == 0, "TM*TK must be multiple of 128*8");
    constexpr int ITERS = (TM * TK) / (128 * 8);
    constexpr int COLS_PER_ROW = TK / 8;   // # of 8-element groups per row

    int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        int linear = i * 128 + tid;
        int row = linear / COLS_PER_ROW;
        int col = (linear % COLS_PER_ROW) * 8;
        const __nv_bfloat16* src = A + (block_m + row) * K + (k0 + col);
        __nv_bfloat16*       dst = smem_A + row * TK + col;
        cp_async_16B(dst, src);
    }
}




// Load a [TN, TK] bf16 tile from B at (block_n, k0) into SMEM [TN, TK] row-major.
// (SMEM row-major [TN, TK] is equivalent to col-major [TK, TN] with leading
// dim TK, which is what WMMA matrix_b col-major wants.)
template<int TN, int TK>
__device__ __forceinline__ void load_B_tile(
        __nv_bfloat16* smem_B,           // destination [TN, TK] row-major
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
        const __nv_bfloat16* src = B + (block_n + row) * K + (k0 + col);
        __nv_bfloat16*       dst = smem_B + row * TK + col;
        cp_async_16B(dst, src);
    }
}


// Template param HAS_RESIDUAL controls whether the epilogue adds a bf16
// residual tensor to the output before writeback (fuses residual_add into
// the GEMM). Compile-time branch eliminates runtime cost when unused.
//
// Tried __launch_bounds__(128, 2) — forced register-spill regression on all
// four shapes (gate_up 92.8 → 125.7 µs). Compiler's default register
// allocation beats the forced-occupancy hint at our shapes.
// Silu-mul A-tile helpers (used when APPLY_SILU_MUL is true in the kernel).
// These load the gate and up halves of [M, 2K] into scratch SMEM via cp.async,
// then merge silu(gate)*up into the consumer A tile. Split as two helpers so
// the cp.async commit/wait groups stay aligned with the non-fused path.
template<int TM, int TK>
__device__ __forceinline__ void load_A_tile_silu_mul_raw(
        __nv_bfloat16* smem_A_scratch,   // destination [2, TM, TK] = gate||up
        const __nv_bfloat16* __restrict__ A,
        int block_m, int k0, int K_stride) {
    static_assert(TM * TK % (128 * 8) == 0, "TM*TK must be multiple of 128*8");
    constexpr int ITERS = (TM * TK) / (128 * 8);
    constexpr int COLS_PER_ROW = TK / 8;
    int K_out = K_stride >> 1;
    int tid = threadIdx.x;
    __nv_bfloat16* smem_gate = smem_A_scratch;
    __nv_bfloat16* smem_up   = smem_A_scratch + TM * TK;
    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        int linear = i * 128 + tid;
        int row = linear / COLS_PER_ROW;
        int col = (linear % COLS_PER_ROW) * 8;
        const __nv_bfloat16* gate_src = A + (block_m + row) * K_stride + (k0 + col);
        const __nv_bfloat16* up_src   = A + (block_m + row) * K_stride + K_out + (k0 + col);
        cp_async_16B(smem_gate + row * TK + col, gate_src);
        cp_async_16B(smem_up   + row * TK + col, up_src);
    }
}

template<int TM, int TK>
__device__ __forceinline__ void merge_silu_mul_A_tile(
        __nv_bfloat16* smem_A_dst,
        const __nv_bfloat16* smem_A_scratch) {
    constexpr int ITERS = (TM * TK) / (128 * 8);
    constexpr int COLS_PER_ROW = TK / 8;
    const __nv_bfloat16* smem_gate = smem_A_scratch;
    const __nv_bfloat16* smem_up   = smem_A_scratch + TM * TK;
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        int linear = i * 128 + tid;
        int row = linear / COLS_PER_ROW;
        int col = (linear % COLS_PER_ROW) * 8;
        uint4 g4 = *reinterpret_cast<const uint4*>(smem_gate + row * TK + col);
        uint4 u4 = *reinterpret_cast<const uint4*>(smem_up   + row * TK + col);
        __nv_bfloat16* gate8 = reinterpret_cast<__nv_bfloat16*>(&g4);
        __nv_bfloat16* up8   = reinterpret_cast<__nv_bfloat16*>(&u4);
        __nv_bfloat16 out8[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float g = __bfloat162float(gate8[j]);
            float u = __bfloat162float(up8[j]);
            float silu = g * __frcp_rn(1.0f + __expf(-g));
            out8[j] = __float2bfloat16(silu * u);
        }
        *reinterpret_cast<uint4*>(smem_A_dst + row * TK + col) = *reinterpret_cast<uint4*>(out8);
    }
}


template<int TM, int TN, int TK, int STAGES, bool HAS_RESIDUAL, bool APPLY_SILU_MUL = false>
__global__ void vcpm_gemm_bf16_tuned_kernel(
        const __nv_bfloat16* __restrict__ A,   // [M, K] or [M, 2K] row-major
        const __nv_bfloat16* __restrict__ B,   // [N, K] row-major (weight [out, in])
        __nv_bfloat16* __restrict__ C,         // [M, N] row-major
        const __nv_bfloat16* __restrict__ R,   // [M, N] residual (ignored if !HAS_RESIDUAL)
        int M, int N, int K) {
    // When APPLY_SILU_MUL is true:
    //   - A is [M, 2K] (gate||up). Kernel consumes silu(A[:, :K]) * A[:, K:2K].
    //   - K is the CONSUMED K dim; A's row stride is 2K (handled via K_a).
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WARPS = 4;
    constexpr int N_FRAGS = TN / WMMA_N;          // 8 per warp
    constexpr int K_SUBITERS = TK / WMMA_K;       // 2 per stage

    static_assert(TM == WMMA_M * WARPS, "TM must equal WMMA_M * WARPS");
    static_assert(TN % WMMA_N == 0, "TN must be multiple of WMMA_N");
    static_assert(TK % WMMA_K == 0, "TK must be multiple of WMMA_K");

    int warp_id = threadIdx.x >> 5;
    int block_m = blockIdx.y * TM;
    int block_n = blockIdx.x * TN;
    int warp_m  = warp_id * WMMA_M;  // row offset within block
    int K_a = APPLY_SILU_MUL ? (K << 1) : K;  // A-matrix row stride

    // SMEM layout (non-silu): A stages | B stages | C fp32
    //   + if APPLY_SILU_MUL: trailing STAGES*2*TM*TK bf16 scratch (gate||up halves)
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_B = smem_A + STAGES * TM * TK;
    float*         smem_C = reinterpret_cast<float*>(smem_B + STAGES * TN * TK);
    __nv_bfloat16* smem_A_scratch = APPLY_SILU_MUL
        ? reinterpret_cast<__nv_bfloat16*>(smem_C + TM * TN)
        : nullptr;

    // Accumulators: one fragment per N-tile, fp32.
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[N_FRAGS];
    #pragma unroll
    for (int i = 0; i < N_FRAGS; ++i) fill_fragment(acc[i], 0.f);

    // Prologue: issue loads for the first (STAGES-1) stages.
    int num_k_tiles = K / TK;  // caller ensures K % TK == 0
    int prologue = (STAGES - 1) < num_k_tiles ? (STAGES - 1) : num_k_tiles;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < prologue) {
            int k0 = s * TK;
            if constexpr (APPLY_SILU_MUL) {
                load_A_tile_silu_mul_raw<TM, TK>(
                    smem_A_scratch + s * 2 * TM * TK, A, block_m, k0, K_a);
            } else {
                load_A_tile<TM, TK>(smem_A + s * TM * TK, A, block_m, k0, K_a);
            }
            load_B_tile<TN, TK>(smem_B + s * TN * TK, B, block_n, k0, K);
        }
        cp_async_commit();
    }

    // Main K loop: for each k-tile, wait on the earliest issued stage, run MMAs,
    // then issue the next load to keep the pipeline full.
    int stage = 0;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Wait for the current stage's loads.
        cp_async_wait_group<STAGES - 2>();  // keep STAGES-2 groups in flight
        __syncthreads();

        // Silu fusion: merge gate||up scratch -> consumer A tile (then __syncthreads).
        if constexpr (APPLY_SILU_MUL) {
            merge_silu_mul_A_tile<TM, TK>(
                smem_A + stage * TM * TK,
                smem_A_scratch + stage * 2 * TM * TK);
            __syncthreads();
        }

        // Issue load for the stage STAGES-1 ahead (if any k-tiles remain).
        int kt_next = kt + (STAGES - 1);
        if (kt_next < num_k_tiles) {
            int next_stage = (stage + STAGES - 1) % STAGES;
            int k0_next = kt_next * TK;
            if constexpr (APPLY_SILU_MUL) {
                load_A_tile_silu_mul_raw<TM, TK>(
                    smem_A_scratch + next_stage * 2 * TM * TK, A, block_m, k0_next, K_a);
            } else {
                load_A_tile<TM, TK>(smem_A + next_stage * TM * TK, A, block_m, k0_next, K_a);
            }
            load_B_tile<TN, TK>(smem_B + next_stage * TN * TK, B, block_n, k0_next, K);
        }
        cp_async_commit();

        // MMA over this stage's (TK) k-range, in K_SUBITERS steps of WMMA_K.
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
                // SMEM is [TN, TK] row-major ≡ [TK, TN] col-major, leading dim TK.
                // Sub-tile (k_off, ni*WMMA_N) in col-major coords:
                //   ptr = smem_B_stage + k_off + (ni*WMMA_N) * TK
                load_matrix_sync(b_frag, b_base + k_off + ni * WMMA_N * TK, TK);
                mma_sync(acc[ni], a_frag, b_frag, acc[ni]);
            }
        }

        stage = (stage + 1) % STAGES;
    }

    // Drain remaining loads (harmless if none).
    cp_async_wait_all();
    __syncthreads();

    // Write accumulators to SMEM C tile (fp32).
    #pragma unroll
    for (int ni = 0; ni < N_FRAGS; ++ni) {
        store_matrix_sync(smem_C + warp_m * TN + ni * WMMA_N, acc[ni], TN, mem_row_major);
    }
    __syncthreads();

    // Cooperatively convert fp32 → bf16 (+ optional residual) and write to C.
    // Caller guarantees M % TM == 0 and N % TN == 0 (TORCH_CHECK'd), so all
    // (gr, gc) are in-bounds — no per-element bounds check needed.
    constexpr int ELEMS_PER_THREAD = (TM * TN) / 128;
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int linear = i * 128 + tid;
        int r = linear / TN;
        int c = linear % TN;
        int gr = block_m + r;
        int gc = block_n + c;
        float val = smem_C[linear];
        if constexpr (HAS_RESIDUAL) {
            val += __bfloat162float(R[gr * N + gc]);
        }
        C[gr * N + gc] = __float2bfloat16(val);
    }
}


// Internal launcher used by both vcpm_gemm_bf16_tuned and
// vcpm_gemm_bf16_tuned_residual. R may be nullopt; HAS_RESIDUAL templated at
// compile time via the residual.has_value() branch.
static torch::Tensor vcpm_gemm_bf16_tuned_impl(
        const torch::Tensor& A,
        const torch::Tensor& B,
        const torch::Tensor* R_opt) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A,B must be cuda");
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16,
                "A,B must be bf16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A,B must be 2D");

    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);
    TORCH_CHECK((int)B.size(1) == K, "B.size(1) must equal A.size(1)");

    if (R_opt != nullptr) {
        TORCH_CHECK(R_opt->is_cuda() && R_opt->dtype() == torch::kBFloat16
                    && R_opt->is_contiguous() && R_opt->dim() == 2
                    && (int)R_opt->size(0) == M && (int)R_opt->size(1) == N,
                    "residual must be [M, N] bf16 cuda contiguous");
    }

    constexpr int TM = 64, TK = 32, STAGES = 4;
    TORCH_CHECK(M % TM == 0, "M (", M, ") must be multiple of TM=", TM,
                " — caller should pad");
    TORCH_CHECK(K % TK == 0, "K (", K, ") must be multiple of TK=", TK);

    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto TN_v) {
        constexpr int TN = decltype(TN_v)::value;
        TORCH_CHECK(N % TN == 0, "N (", N, ") must be multiple of TN=", TN);

        int grid_x = N / TN;
        int grid_y = M / TM;
        dim3 grid(grid_x, grid_y, 1);
        dim3 block(128, 1, 1);

        size_t smem_bytes = STAGES * TM * TK * sizeof(__nv_bfloat16)
                          + STAGES * TN * TK * sizeof(__nv_bfloat16)
                          + TM * TN * sizeof(float);

        const __nv_bfloat16* A_p = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
        const __nv_bfloat16* B_p = reinterpret_cast<const __nv_bfloat16*>(B.data_ptr());
        __nv_bfloat16*       C_p = reinterpret_cast<__nv_bfloat16*>(C.data_ptr());
        const __nv_bfloat16* R_p = (R_opt ? reinterpret_cast<const __nv_bfloat16*>(R_opt->data_ptr()) : nullptr);

        if (R_opt != nullptr) {
            if (smem_bytes > 48 * 1024) {
                cudaFuncSetAttribute(
                    vcpm_gemm_bf16_tuned_kernel<TM, TN, TK, STAGES, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
            }
            vcpm_gemm_bf16_tuned_kernel<TM, TN, TK, STAGES, true>
                <<<grid, block, smem_bytes, stream>>>(A_p, B_p, C_p, R_p, M, N, K);
        } else {
            if (smem_bytes > 48 * 1024) {
                cudaFuncSetAttribute(
                    vcpm_gemm_bf16_tuned_kernel<TM, TN, TK, STAGES, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
            }
            vcpm_gemm_bf16_tuned_kernel<TM, TN, TK, STAGES, false>
                <<<grid, block, smem_bytes, stream>>>(A_p, B_p, C_p, nullptr, M, N, K);
        }
    };

    if (N % 32 == 0) {
        launch(std::integral_constant<int, 32>{});
    } else if (N % 64 == 0) {
        launch(std::integral_constant<int, 64>{});
    } else if (N % 128 == 0) {
        launch(std::integral_constant<int, 128>{});
    } else {
        TORCH_CHECK(false, "N (", N, ") must be multiple of 32");
    }
    return C;
}


torch::Tensor vcpm_gemm_bf16_tuned(const torch::Tensor& A, const torch::Tensor& B) {
    return vcpm_gemm_bf16_tuned_impl(A, B, nullptr);
}


// Debug/tuning entry: force a specific TN (64 or 128), bypassing the
// automatic dispatch. Used by test_gemm_tn_sweep.py.
static torch::Tensor vcpm_gemm_bf16_tuned_impl_forced_tn(
        const torch::Tensor& A,
        const torch::Tensor& B,
        int force_tn) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A,B must be cuda");
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16, "A,B bf16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && A.dim() == 2 && B.dim() == 2, "contig 2D");
    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);
    TORCH_CHECK((int)B.size(1) == K, "K mismatch");
    constexpr int TM = 64, TK = 32, STAGES = 4;
    TORCH_CHECK(M % TM == 0 && K % TK == 0, "M%64 K%32 required");

    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto TN_v) {
        constexpr int TN = decltype(TN_v)::value;
        TORCH_CHECK(N % TN == 0, "N must be multiple of TN=", TN);
        int grid_x = N / TN;
        int grid_y = M / TM;
        dim3 grid(grid_x, grid_y, 1);
        dim3 block(128, 1, 1);
        size_t smem_bytes = STAGES * TM * TK * sizeof(__nv_bfloat16)
                          + STAGES * TN * TK * sizeof(__nv_bfloat16)
                          + TM * TN * sizeof(float);
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(
                vcpm_gemm_bf16_tuned_kernel<TM, TN, TK, STAGES, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        }
        vcpm_gemm_bf16_tuned_kernel<TM, TN, TK, STAGES, false>
            <<<grid, block, smem_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                nullptr, M, N, K);
    };

    if (force_tn == 32) {
        launch(std::integral_constant<int, 32>{});
    } else if (force_tn == 64) {
        launch(std::integral_constant<int, 64>{});
    } else if (force_tn == 128) {
        launch(std::integral_constant<int, 128>{});
    } else {
        TORCH_CHECK(false, "force_tn must be 32, 64, or 128");
    }
    return C;
}

torch::Tensor vcpm_gemm_bf16_tuned_tn(
        const torch::Tensor& A, const torch::Tensor& B, int64_t tn) {
    return vcpm_gemm_bf16_tuned_impl_forced_tn(A, B, (int)tn);
}

// Fused GEMM + residual: returns A @ B^T + residual.  Residual is [M, N] bf16.
torch::Tensor vcpm_gemm_bf16_tuned_residual(
        const torch::Tensor& A,
        const torch::Tensor& B,
        const torch::Tensor& residual) {
    return vcpm_gemm_bf16_tuned_impl(A, B, &residual);
}


// Fused silu_mul + GEMM (+ optional residual) for the MLP down path:
//   C = silu(A[:, :K]) * A[:, K:2K]  @  B^T   (+ residual if provided)
// A is [M, 2K] bf16 (gate||up). Fuses silu_mul kernel into down-GEMM's A-load
// prologue. Saves 28 kernel launches/forward + eliminates silu_mul intermediate
// HBM (~45 MB/forward at M=128, I=6144).
static torch::Tensor vcpm_gemm_bf16_tuned_silu_impl(
        const torch::Tensor& A_gu,
        const torch::Tensor& B,
        const torch::Tensor* R_opt) {
    TORCH_CHECK(A_gu.is_cuda() && B.is_cuda(), "cuda");
    TORCH_CHECK(A_gu.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16, "bf16");
    TORCH_CHECK(A_gu.is_contiguous() && B.is_contiguous(), "contig");
    TORCH_CHECK(A_gu.dim() == 2 && B.dim() == 2, "2D");

    int M    = (int)A_gu.size(0);
    int twoK = (int)A_gu.size(1);
    TORCH_CHECK(twoK % 2 == 0, "A.size(1) must be even");
    int K = twoK / 2;
    int N = (int)B.size(0);
    TORCH_CHECK((int)B.size(1) == K, "B.size(1) must equal A.size(1)/2");

    if (R_opt != nullptr) {
        TORCH_CHECK(R_opt->is_cuda() && R_opt->dtype() == torch::kBFloat16
                    && R_opt->is_contiguous() && R_opt->dim() == 2
                    && (int)R_opt->size(0) == M && (int)R_opt->size(1) == N,
                    "residual must be [M, N] bf16 cuda contiguous");
    }

    constexpr int TM = 64, TK = 32, STAGES = 4;
    TORCH_CHECK(M % TM == 0, "M must be multiple of 64");
    TORCH_CHECK(K % TK == 0, "K must be multiple of 32");

    auto C = torch::empty({M, N}, A_gu.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto TN_v) {
        constexpr int TN = decltype(TN_v)::value;
        TORCH_CHECK(N % TN == 0, "N must be multiple of TN");
        int grid_x = N / TN;
        int grid_y = M / TM;
        dim3 grid(grid_x, grid_y, 1);
        dim3 block(128, 1, 1);

        // +STAGES*2*TM*TK bf16 scratch for gate||up halves
        size_t smem_bytes = STAGES * TM * TK * sizeof(__nv_bfloat16)
                          + STAGES * TN * TK * sizeof(__nv_bfloat16)
                          + TM * TN * sizeof(float)
                          + STAGES * 2 * TM * TK * sizeof(__nv_bfloat16);

        const __nv_bfloat16* A_p = reinterpret_cast<const __nv_bfloat16*>(A_gu.data_ptr());
        const __nv_bfloat16* B_p = reinterpret_cast<const __nv_bfloat16*>(B.data_ptr());
        __nv_bfloat16*       C_p = reinterpret_cast<__nv_bfloat16*>(C.data_ptr());
        const __nv_bfloat16* R_p = (R_opt ? reinterpret_cast<const __nv_bfloat16*>(R_opt->data_ptr()) : nullptr);

        if (R_opt != nullptr) {
            if (smem_bytes > 48 * 1024) {
                cudaFuncSetAttribute(
                    vcpm_gemm_bf16_tuned_kernel<TM, TN, TK, STAGES, true, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
            }
            vcpm_gemm_bf16_tuned_kernel<TM, TN, TK, STAGES, true, true>
                <<<grid, block, smem_bytes, stream>>>(A_p, B_p, C_p, R_p, M, N, K);
        } else {
            if (smem_bytes > 48 * 1024) {
                cudaFuncSetAttribute(
                    vcpm_gemm_bf16_tuned_kernel<TM, TN, TK, STAGES, false, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
            }
            vcpm_gemm_bf16_tuned_kernel<TM, TN, TK, STAGES, false, true>
                <<<grid, block, smem_bytes, stream>>>(A_p, B_p, C_p, nullptr, M, N, K);
        }
    };

    if (N % 32 == 0) launch(std::integral_constant<int, 32>{});
    else if (N % 64 == 0) launch(std::integral_constant<int, 64>{});
    else if (N % 128 == 0) launch(std::integral_constant<int, 128>{});
    else TORCH_CHECK(false, "N must be multiple of 32");

    return C;
}

torch::Tensor vcpm_gemm_bf16_tuned_silu(
        const torch::Tensor& A_gu, const torch::Tensor& B) {
    return vcpm_gemm_bf16_tuned_silu_impl(A_gu, B, nullptr);
}

torch::Tensor vcpm_gemm_bf16_tuned_silu_residual(
        const torch::Tensor& A_gu, const torch::Tensor& B, const torch::Tensor& residual) {
    return vcpm_gemm_bf16_tuned_silu_impl(A_gu, B, &residual);
}


// ===========================================================================
// P2.5.1.b — Inline causal attention kernel (flash-attn-2 style, prefill).
//
// Replaces flash_attn_func in FusedLayer's causal path. Supports GQA with
// NUM_HEADS_Q=16, NUM_HEADS_KV=2, HEAD_DIM=128 (base_lm shape).
//
// Layout — one block per (q_tile, head_q):
//   Grid: (ceil(N/Q_BLOCK), NUM_HEADS_Q) where Q_BLOCK=32
//   Block: 128 threads = 4 warps arranged 2×2 in (M, N) for S = Q @ K^T tiles
//
// Algorithm:
//   1. Load Q tile for this (q_tile, head_q) into SMEM  [Q_BLOCK × D]
//   2. Init m=-INF, l=0, O=0 (all per-row, fp32)
//   3. For each K tile (stepping by K_BLOCK=64, skipping fully-masked tiles):
//        a. Load K, V tiles for head_kv = head_q / (NUM_Q / NUM_KV)
//        b. Compute S = Q @ K^T * scale via WMMA bf16→fp32
//        c. Apply causal mask
//        d. Online softmax: new_m = max(m, rowmax(S))
//           rescale = exp(m - new_m)
//           P = exp(S - new_m)  (bf16 for next MMA)
//           l = rescale*l + rowsum(P)
//           O = rescale*O + P @ V  (fp32 accumulate)
//           m = new_m
//   4. Write O / l to output[q_tile_rows, head_q, :]
//
// SMEM budget (Q_BLOCK=32, K_BLOCK=64, D=128):
//   Q tile: 32 × 128 × 2    =  8 KB
//   K tile: 64 × 128 × 2    = 16 KB
//   V tile: 64 × 128 × 2    = 16 KB
//   S/P tile: 32 × 64 × 4   =  8 KB (fp32; reinterpreted as bf16 half for P@V)
//   O: 32 × 128 × 4          = 16 KB (fp32 running accumulator)
//   m, l: 32 × 4 × 2         = 256 B
//   Total: ~64 KB. Fits on sm_120 (228 KB max).
// ===========================================================================

template<int Q_BLOCK, int K_BLOCK, int D, int NUM_Q, int NUM_KV>
__global__ void vcpm_attention_causal_kernel(
        const __nv_bfloat16* __restrict__ Q, int q_row_stride,  // [N, *] with Q_DIM contig cols
        const __nv_bfloat16* __restrict__ K, int k_row_stride,
        const __nv_bfloat16* __restrict__ V, int v_row_stride,
        __nv_bfloat16* __restrict__ O,        int o_row_stride,
        float scale, int N) {
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WARPS = 4;
    constexpr int Q_DIM = NUM_Q * D;
    constexpr int KV_DIM = NUM_KV * D;
    constexpr int Q_PER_KV = NUM_Q / NUM_KV;  // 8

    static_assert(Q_BLOCK % WMMA_M == 0, "Q_BLOCK must be multiple of 16");
    static_assert(K_BLOCK % WMMA_N == 0, "K_BLOCK must be multiple of 16");
    static_assert(D % WMMA_K == 0, "D must be multiple of 16");

    int head_q   = blockIdx.y;
    int head_kv  = head_q / Q_PER_KV;
    int q_tile   = blockIdx.x;
    int q_base   = q_tile * Q_BLOCK;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    // Warp 2D partition for S = Q @ K^T (M=Q_BLOCK, N=K_BLOCK).
    // With Q_BLOCK=32 (2 M-tiles) and 4 warps we pick a 2×2 layout:
    // each warp owns (1 M-tile=16 rows) × (K_BLOCK/2 cols = K_BLOCK/32 N-frags).
    // For K_BLOCK=64 → 2 N-frags per warp. For K_BLOCK=32 → 1 N-frag per warp.
    constexpr int QK_N_FRAGS_PER_WARP = (K_BLOCK / WMMA_N) / 2;  // 2 M-warps
    constexpr int QK_WARP_N_STRIDE    = K_BLOCK / 2;
    int warp_m_tile = warp_id >> 1;   // 0..1 (maps to M=0 or 16)
    int warp_n_tile = warp_id & 1;    // 0..1 (maps to N=0 or K_BLOCK/2)

    // SMEM allocations.
    extern __shared__ __align__(16) char smem_raw[];
    char* smem_ptr = smem_raw;
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_ptr);
    smem_ptr += Q_BLOCK * D * sizeof(__nv_bfloat16);
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_ptr);
    smem_ptr += K_BLOCK * D * sizeof(__nv_bfloat16);
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_ptr);
    smem_ptr += K_BLOCK * D * sizeof(__nv_bfloat16);
    float* smem_s = reinterpret_cast<float*>(smem_ptr);       // also reused as P
    smem_ptr += Q_BLOCK * K_BLOCK * sizeof(float);
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_ptr);
    smem_ptr += Q_BLOCK * K_BLOCK * sizeof(__nv_bfloat16);
    float* smem_o = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += Q_BLOCK * D * sizeof(float);
    float* smem_m = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += Q_BLOCK * sizeof(float);
    float* smem_l = reinterpret_cast<float*>(smem_ptr);

    // -------- Load Q tile for (q_tile, head_q) --------
    #pragma unroll
    for (int i = tid; i < Q_BLOCK * D; i += 128) {
        int qi = i / D;
        int di = i % D;
        int n  = q_base + qi;
        if (n < N) {
            smem_q[qi * D + di] = Q[n * q_row_stride + head_q * D + di];
        } else {
            smem_q[qi * D + di] = __float2bfloat16(0.f);
        }
    }

    // Init m, l, O
    if (tid < Q_BLOCK) {
        smem_m[tid] = -INFINITY;
        smem_l[tid] = 0.f;
    }
    #pragma unroll
    for (int i = tid; i < Q_BLOCK * D; i += 128) {
        smem_o[i] = 0.f;
    }
    __syncthreads();

    // -------- K/V tile loop --------
    // Causal: K position k_global must satisfy k_global <= q_global (max q in tile).
    // q_max = q_base + Q_BLOCK - 1 (with clamping at N). We iterate k_tile over
    // [0, num_k_tiles) where num_k_tiles = ceil((q_base + Q_BLOCK) / K_BLOCK),
    // clamped by the actual N. Tiles fully beyond the causal horizon are skipped.
    int q_end = q_base + Q_BLOCK;
    if (q_end > N) q_end = N;
    int num_k_tiles = (q_end + K_BLOCK - 1) / K_BLOCK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k_base = kt * K_BLOCK;
        int k_end  = k_base + K_BLOCK;
        if (k_end > N) k_end = N;

        // Load K, V tiles for head_kv.
        #pragma unroll
        for (int i = tid; i < K_BLOCK * D; i += 128) {
            int ki = i / D;
            int di = i % D;
            int n  = k_base + ki;
            if (n < N) {
                smem_k[ki * D + di] = K[n * k_row_stride + head_kv * D + di];
                smem_v[ki * D + di] = V[n * v_row_stride + head_kv * D + di];
            } else {
                smem_k[ki * D + di] = __float2bfloat16(0.f);
                smem_v[ki * D + di] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // -------- S = Q @ K^T  (WMMA bf16→fp32, with scale) --------
        // Q is [Q_BLOCK=32, D=128] row-major in SMEM.
        // K is [K_BLOCK=64, D=128] row-major in SMEM (viewed as K^T = [D, K_BLOCK]
        //   col-major, leading dim D, for WMMA matrix_b).
        // Output S[Q_BLOCK, K_BLOCK] row-major in smem_s (fp32).
        //
        // Warp partition: each warp handles (1 M-tile=16 rows) × (2 N-tiles=32 cols).
        //   warp_m = warp_m_tile * WMMA_M     (0 or 16)
        //   warp_n = warp_n_tile * 32         (0 or 32)
        {
            constexpr int K_SUBITERS = D / WMMA_K;  // 8
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_acc[QK_N_FRAGS_PER_WARP];
            #pragma unroll
            for (int ni = 0; ni < QK_N_FRAGS_PER_WARP; ++ni) {
                fill_fragment(s_acc[ni], 0.f);
            }

            int warp_m = warp_m_tile * WMMA_M;
            int warp_n = warp_n_tile * QK_WARP_N_STRIDE;

            #pragma unroll
            for (int kk = 0; kk < K_SUBITERS; ++kk) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a;
                load_matrix_sync(a, smem_q + warp_m * D + kk * WMMA_K, D);

                #pragma unroll
                for (int ni = 0; ni < QK_N_FRAGS_PER_WARP; ++ni) {
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b;
                    int n_off = warp_n + ni * WMMA_N;
                    load_matrix_sync(b, smem_k + kk * WMMA_K + n_off * D, D);
                    mma_sync(s_acc[ni], a, b, s_acc[ni]);
                }
            }

            #pragma unroll
            for (int ni = 0; ni < QK_N_FRAGS_PER_WARP; ++ni) {
                #pragma unroll
                for (int e = 0; e < s_acc[ni].num_elements; ++e) {
                    s_acc[ni].x[e] *= scale;
                }
                int n_off = warp_n + ni * WMMA_N;
                store_matrix_sync(smem_s + warp_m * K_BLOCK + n_off,
                                  s_acc[ni], K_BLOCK, mem_row_major);
            }
        }
        __syncthreads();

        // -------- Causal mask + online softmax + P (bf16 for next MMA) --------
        // One warp per row is the cleanest pattern; we have 4 warps and Q_BLOCK=32
        // rows, so each warp handles 8 rows serially.
        if (tid < Q_BLOCK) {
            // One thread per row for softmax reductions. (K_BLOCK=64 so it's fine.)
            int q_global = q_base + tid;
            float* s_row = smem_s + tid * K_BLOCK;

            // Apply causal mask element-wise and find rowmax.
            float row_max = -INFINITY;
            #pragma unroll
            for (int kj = 0; kj < K_BLOCK; ++kj) {
                int k_global = k_base + kj;
                if (k_global > q_global || k_global >= N || q_global >= N) {
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
                // Store P in bf16 for next MMA.
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

        // -------- O += P @ V  (WMMA bf16→fp32, accumulate into smem_o) --------
        // P is [Q_BLOCK=32, K_BLOCK=64] bf16 in smem_p.
        // V is [K_BLOCK=64, D=128] bf16 in smem_v (row-major).
        // Output: smem_o[Q_BLOCK=32, D=128] fp32 (accumulate).
        //
        // Warp partition for the [32, 128] output: 2 M-tiles × 2 N-strips of 64 cols.
        // Each warp handles (1 M-tile=16 rows) × (4 N-frags of WMMA_N=16 = 64 cols).
        //   warp 0: M=0,  N=0..64
        //   warp 1: M=0,  N=64..128
        //   warp 2: M=16, N=0..64
        //   warp 3: M=16, N=64..128
        {
            constexpr int K_SUBITERS = K_BLOCK / WMMA_K;  // 4
            constexpr int PV_N_FRAGS = 4;                  // per warp
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_acc[PV_N_FRAGS];

            int warp_m = warp_m_tile * WMMA_M;   // 0 or 16
            int warp_n = warp_n_tile * 64;       // 0 or 64

            // Load current O accumulator from smem_o (fp32, row-major stride D).
            #pragma unroll
            for (int ni = 0; ni < PV_N_FRAGS; ++ni) {
                load_matrix_sync(o_acc[ni],
                                 smem_o + warp_m * D + warp_n + ni * WMMA_N,
                                 D, mem_row_major);
            }

            #pragma unroll
            for (int kk = 0; kk < K_SUBITERS; ++kk) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a;
                load_matrix_sync(a, smem_p + warp_m * K_BLOCK + kk * WMMA_K, K_BLOCK);

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
                store_matrix_sync(smem_o + warp_m * D + warp_n + ni * WMMA_N,
                                  o_acc[ni], D, mem_row_major);
            }
        }
        __syncthreads();
    }

    // -------- Final: O / l, write bf16 to global --------
    #pragma unroll
    for (int i = tid; i < Q_BLOCK * D; i += 128) {
        int qi = i / D;
        int di = i % D;
        int n  = q_base + qi;
        if (n >= N) continue;
        float l_val = smem_l[qi];
        float val = (l_val > 0.f) ? (smem_o[i] / l_val) : 0.f;
        O[n * o_row_stride + head_q * D + di] = __float2bfloat16(val);
    }
}


torch::Tensor vcpm_attention_causal(
        const torch::Tensor& q,  // [N, Q_DIM]  bf16 — may be strided
        const torch::Tensor& k,  // [N, KV_DIM] bf16
        const torch::Tensor& v,  // [N, KV_DIM] bf16
        double scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q,k,v cuda");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q bf16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16 && v.dtype() == torch::kBFloat16, "k,v bf16");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "q,k,v 2D");
    TORCH_CHECK(q.stride(1) == 1 && k.stride(1) == 1 && v.stride(1) == 1,
                "q,k,v must have inner-dim stride 1");

    int N = (int)q.size(0);
    TORCH_CHECK((int)k.size(0) == N && (int)v.size(0) == N, "N mismatch");

    constexpr int NUM_Q = 16, NUM_KV = 2, D = 128;
    // K_BLOCK=32 keeps SMEM under 48 KB → 2 blocks/SM at default carveout,
    // which wins in-context over K_BLOCK=64 (1 block/SM) despite more K-tile
    // iterations. More K-iters are cheap; SM concurrency is the bottleneck.
    constexpr int Q_BLOCK = 32, K_BLOCK = 32;
    TORCH_CHECK((int)q.size(1) == NUM_Q * D, "q last dim");
    TORCH_CHECK((int)k.size(1) == NUM_KV * D, "k last dim");
    TORCH_CHECK((int)v.size(1) == NUM_KV * D, "v last dim");

    auto o = torch::empty({N, NUM_Q * D}, q.options());
    int num_q_tiles = (N + Q_BLOCK - 1) / Q_BLOCK;

    dim3 grid(num_q_tiles, NUM_Q, 1);
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
            vcpm_attention_causal_kernel<Q_BLOCK, K_BLOCK, D, NUM_Q, NUM_KV>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    vcpm_attention_causal_kernel<Q_BLOCK, K_BLOCK, D, NUM_Q, NUM_KV>
        <<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), (int)q.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()), (int)k.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), (int)v.stride(0),
            reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),       (int)o.stride(0),
            (float)scale, N);
    return o;
}


// ===========================================================================
// P2.5.1.c — Fused pre_attn kernel: rmsnorm + qkv_gemm + rope in one kernel.
//
// Replaces three separate launches per layer on the causal path:
//   hs → rmsnorm → qkv_gemm → rope_inplace(Q, K)
// with one kernel that:
//   1. Loads a [TM, H] A-tile into SMEM via cp.async (entire row group for
//      this block's M-stripe).
//   2. Computes per-row RMSNorm (warp-reduction over H, one row per lane).
//   3. Scales A in-place in SMEM: A[r, c] *= inv_rms[r] * w_ln[c].
//   4. Streams the (scaled) A tile × W_qkv^T GEMM with cp.async-pipelined
//      B-tile loads from HBM (identical math to vcpm_gemm_bf16_tuned).
//   5. If the block's N-strip lies in the Q or K region of QKV_DIM, apply
//      RoPE to the fp32 output tile before the bf16 store.
//
// Block tile: TM=16 rows × TN=128 cols. Each TN-block covers exactly ONE
// D=128 head's worth of cols — RoPE pairs stay within a block's SMEM.
//
// SMEM budget (TM=16, H=2048, TN=128, TK=32, STAGES=3):
//   A tile (bf16):   TM × H × 2           = 16 × 2048 × 2       =  64 KB
//   B stages (bf16): STAGES × TN × TK × 2 =  3 × 128 × 32 × 2   =  24 KB
//   C tile (fp32):   TM × TN × 4          = 16 × 128 × 4        =   8 KB
//   inv_rms (fp32):  TM × 4                                     =  64 B
//   Total: ~96 KB per block (opt-in past the 48 KB default).
//
// Grid: (QKV_DIM / TN, M / TM). For base_lm shape (QKV_DIM=2560, M=128):
// 20 × 8 = 160 blocks. ≈1 block/SM on 170 SMs, good saturation.
// ===========================================================================

template<int TM, int TN, int TK, int STAGES, int H, int NUM_Q, int NUM_KV, int D>
__global__ void vcpm_fused_pre_attn_kernel(
        const __nv_bfloat16* __restrict__ X,      // [M, H]
        const __nv_bfloat16* __restrict__ W_LN,   // [H]
        const __nv_bfloat16* __restrict__ W_QKV,  // [QKV_DIM, H]
        const float*         __restrict__ COS,    // [max_pos, D]
        const float*         __restrict__ SIN,    // [max_pos, D]
        const int32_t*       __restrict__ POS,    // [M]
        __nv_bfloat16*       __restrict__ OUT,    // [M, QKV_DIM]
        float rms_eps, int M) {
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WARPS = 4;
    constexpr int Q_DIM  = NUM_Q * D;
    constexpr int KV_DIM = NUM_KV * D;
    constexpr int QKV_DIM = Q_DIM + 2 * KV_DIM;
    constexpr int K_SUBITERS = TK / WMMA_K;                // 2
    constexpr int N_FRAGS_PER_WARP = (TN / WMMA_N) / WARPS; // 2 for TN=128, 4 warps
    constexpr int WARP_N_STRIDE = TN / WARPS;              // 32 for TN=128, 4 warps
    static_assert(TM == WMMA_M, "TM must equal WMMA_M (single M-tile per block)");
    static_assert(TN % (WARPS * WMMA_N) == 0, "TN must be multiple of WARPS*WMMA_N");

    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int tid     = threadIdx.x;
    int block_m = blockIdx.y * TM;
    int block_n = blockIdx.x * TN;

    // ----- SMEM -----
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* smem_A   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_B   = smem_A + TM * H;
    float*         smem_C   = reinterpret_cast<float*>(smem_B + STAGES * TN * TK);
    float*         smem_inv = smem_C + TM * TN;

    // ----- 1. Load A[block_m:block_m+TM, :H] via cp.async -----
    // Total elements = TM * H. Per thread = TM*H / 128 elements = TM*H/128/8 cp.async ops of 16B.
    constexpr int A_CP_ASYNC_OPS_PER_THREAD = (TM * H) / (128 * 8);  // 32 for TM=16, H=2048
    constexpr int COLS_PER_CACHELINE = 8;
    constexpr int CL_PER_ROW = H / COLS_PER_CACHELINE;
    #pragma unroll
    for (int i = 0; i < A_CP_ASYNC_OPS_PER_THREAD; ++i) {
        int linear = i * 128 + tid;
        int row = linear / CL_PER_ROW;
        int col = (linear % CL_PER_ROW) * COLS_PER_CACHELINE;
        int g_row = block_m + row;
        if (g_row < M) {
            cp_async_16B(smem_A + row * H + col, X + g_row * H + col);
        } else {
            // Pad row with zeros via a direct store (cp.async zero-pad
            // variant is awkward; 8 bf16 stores is fine since padding
            // happens on a tiny fraction of rows).
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                smem_A[row * H + col + j] = __float2bfloat16(0.f);
            }
        }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // ----- 2. RMSNorm: one row per warp-lane set -----
    // With TM=16 and 4 warps, each warp handles 4 rows (one per 8 lanes).
    // Simpler: each warp handles TM/WARPS rows sequentially with full warp reduction per row.
    constexpr int ROWS_PER_WARP = TM / WARPS;  // 4
    int warp_row_base = warp_id * ROWS_PER_WARP;
    #pragma unroll
    for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
        int row = warp_row_base + rr;
        if (block_m + row >= M) {
            if (lane == 0) smem_inv[row] = 0.f;
            continue;
        }
        float sum_sq = 0.f;
        #pragma unroll 8
        for (int i = lane; i < H; i += 32) {
            float v = __bfloat162float(smem_A[row * H + i]);
            sum_sq += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);
        }
        float mean_sq = sum_sq / (float)H;
        float inv = rsqrtf(mean_sq + rms_eps);
        if (lane == 0) smem_inv[row] = inv;
    }
    __syncthreads();

    // ----- 3. Scale A in place: A[r, c] *= inv[r] * w_ln[c] -----
    // Total elements = TM * H = 16 * 2048 = 32 768. Per thread = 256 elements.
    constexpr int SCALE_ELEMS_PER_THREAD = (TM * H) / 128;
    #pragma unroll 16
    for (int i = 0; i < SCALE_ELEMS_PER_THREAD; ++i) {
        int linear = i * 128 + tid;
        int row = linear / H;
        int col = linear % H;
        float v  = __bfloat162float(smem_A[linear]);
        float wv = __bfloat162float(W_LN[col]);
        float inv = smem_inv[row];
        smem_A[linear] = __float2bfloat16(v * inv * wv);
    }
    __syncthreads();

    // ----- 4. Tiled GEMM: A_scaled @ W_QKV^T -----
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[N_FRAGS_PER_WARP];
    #pragma unroll
    for (int ni = 0; ni < N_FRAGS_PER_WARP; ++ni) fill_fragment(acc[ni], 0.f);

    int warp_n_base = warp_id * WARP_N_STRIDE;  // col offset within block's TN

    // Prologue: load first STAGES-1 B tiles
    constexpr int NUM_K_TILES = H / TK;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < NUM_K_TILES) {
            load_B_tile<TN, TK>(smem_B + s * TN * TK, W_QKV, block_n, s * TK, H);
        }
        cp_async_commit();
    }

    int stage = 0;
    for (int kt = 0; kt < NUM_K_TILES; ++kt) {
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();

        int kt_next = kt + STAGES - 1;
        if (kt_next < NUM_K_TILES) {
            int next_stage = (stage + STAGES - 1) % STAGES;
            load_B_tile<TN, TK>(smem_B + next_stage * TN * TK, W_QKV, block_n, kt_next * TK, H);
        }
        cp_async_commit();

        int k0 = kt * TK;
        __nv_bfloat16* b_base = smem_B + stage * TN * TK;

        #pragma unroll
        for (int kk = 0; kk < K_SUBITERS; ++kk) {
            int k_off = kk * WMMA_K;
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a;
            // A is [TM, H] row-major; WMMA load from offset 0 of M-dim and k0+k_off of K-dim
            load_matrix_sync(a, smem_A + (k0 + k_off), H);

            #pragma unroll
            for (int ni = 0; ni < N_FRAGS_PER_WARP; ++ni) {
                int n_in_block = warp_n_base + ni * WMMA_N;  // col within TN
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b;
                // smem_B is [TN, TK] row-major ≡ [TK, TN] col-major lead dim TK.
                load_matrix_sync(b, b_base + k_off + n_in_block * TK, TK);
                mma_sync(acc[ni], a, b, acc[ni]);
            }
        }
        stage = (stage + 1) % STAGES;
    }
    cp_async_wait_all();
    __syncthreads();

    // ----- 5. Store accumulators to smem_C (fp32) -----
    #pragma unroll
    for (int ni = 0; ni < N_FRAGS_PER_WARP; ++ni) {
        int n_in_block = warp_n_base + ni * WMMA_N;
        store_matrix_sync(smem_C + n_in_block, acc[ni], TN, mem_row_major);
    }
    __syncthreads();

    // ----- 6. RoPE for Q or K regions (not V) -----
    bool is_q = block_n < Q_DIM;
    bool is_k = block_n >= Q_DIM && block_n < (Q_DIM + KV_DIM);
    bool apply_rope = is_q || is_k;

    if (apply_rope) {
        // Each block's TN=128 cols map to exactly one head's D=128 dim.
        // RoPE pairs: (pair_idx, pair_idx + D/2) for pair_idx ∈ [0, D/2).
        // Per thread: (TM * D/2) / 128 = 16 * 64 / 128 = 8 pairs.
        constexpr int HALF_D = D / 2;
        constexpr int PAIRS_PER_THREAD = (TM * HALF_D) / 128;  // 8
        #pragma unroll
        for (int i = 0; i < PAIRS_PER_THREAD; ++i) {
            int linear = i * 128 + tid;
            int row = linear / HALF_D;
            int pair_idx = linear % HALF_D;
            int g_row = block_m + row;
            if (g_row >= M) continue;
            int pos = POS[g_row];
            float c_lo = COS[pos * D + pair_idx];
            float s_lo = SIN[pos * D + pair_idx];
            float c_hi = COS[pos * D + pair_idx + HALF_D];
            float s_hi = SIN[pos * D + pair_idx + HALF_D];
            float x_lo = smem_C[row * TN + pair_idx];
            float x_hi = smem_C[row * TN + pair_idx + HALF_D];
            float y_lo = x_lo * c_lo + (-x_hi) * s_lo;
            float y_hi = x_hi * c_hi + ( x_lo) * s_hi;
            smem_C[row * TN + pair_idx]            = y_lo;
            smem_C[row * TN + pair_idx + HALF_D] = y_hi;
        }
        __syncthreads();
    }

    // ----- 7. Write bf16 output to [M, QKV_DIM] -----
    constexpr int STORE_ELEMS_PER_THREAD = (TM * TN) / 128;
    #pragma unroll
    for (int i = 0; i < STORE_ELEMS_PER_THREAD; ++i) {
        int linear = i * 128 + tid;
        int r = linear / TN;
        int c = linear % TN;
        int gr = block_m + r;
        int gc = block_n + c;
        if (gr < M) {
            OUT[gr * QKV_DIM + gc] = __float2bfloat16(smem_C[linear]);
        }
    }
}


torch::Tensor vcpm_fused_pre_attn(
        const torch::Tensor& X,         // [M, H] bf16
        const torch::Tensor& W_LN,      // [H]
        const torch::Tensor& W_QKV,     // [QKV_DIM, H]
        const torch::Tensor& COS,       // [max_pos, D] fp32
        const torch::Tensor& SIN,       // [max_pos, D] fp32
        const torch::Tensor& POS,       // [M] int32
        double rms_eps) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16 && X.is_contiguous() && X.dim() == 2,
                "X: bf16 cuda contiguous 2D");
    TORCH_CHECK(W_LN.is_cuda() && W_LN.dtype() == torch::kBFloat16 && W_LN.is_contiguous(),
                "W_LN: bf16 cuda contiguous");
    TORCH_CHECK(W_QKV.is_cuda() && W_QKV.dtype() == torch::kBFloat16 && W_QKV.is_contiguous()
                && W_QKV.dim() == 2, "W_QKV: bf16 cuda contiguous 2D");
    TORCH_CHECK(COS.is_cuda() && COS.dtype() == torch::kFloat32 && COS.is_contiguous(), "COS fp32");
    TORCH_CHECK(SIN.is_cuda() && SIN.dtype() == torch::kFloat32 && SIN.is_contiguous(), "SIN fp32");
    TORCH_CHECK(POS.is_cuda() && POS.dtype() == torch::kInt32 && POS.is_contiguous(), "POS int32");

    // Base_lm shape hard-coded.
    constexpr int H = 2048, NUM_Q = 16, NUM_KV = 2, D = 128;
    constexpr int QKV_DIM = NUM_Q * D + 2 * NUM_KV * D;  // 2560
    constexpr int TM = 16, TN = 128, TK = 32, STAGES = 3;

    int M = (int)X.size(0);
    TORCH_CHECK((int)X.size(1) == H, "X hidden dim must be ", H);
    TORCH_CHECK((int)W_LN.numel() == H, "W_LN dim must be ", H);
    TORCH_CHECK((int)W_QKV.size(0) == QKV_DIM && (int)W_QKV.size(1) == H,
                "W_QKV must be [", QKV_DIM, ", ", H, "]");
    TORCH_CHECK((int)COS.size(1) == D && (int)SIN.size(1) == D, "cos/sin last dim must be ", D);
    TORCH_CHECK((int)POS.numel() == M, "POS len must match M");
    TORCH_CHECK(M % TM == 0, "M (", M, ") must be multiple of ", TM);

    auto OUT = torch::empty({M, QKV_DIM}, X.options());

    int grid_x = QKV_DIM / TN;        // 20
    int grid_y = M / TM;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(128, 1, 1);

    size_t smem_bytes = TM * H * sizeof(__nv_bfloat16)
                      + STAGES * TN * TK * sizeof(__nv_bfloat16)
                      + TM * TN * sizeof(float)
                      + TM * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            vcpm_fused_pre_attn_kernel<TM, TN, TK, STAGES, H, NUM_Q, NUM_KV, D>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    vcpm_fused_pre_attn_kernel<TM, TN, TK, STAGES, H, NUM_Q, NUM_KV, D>
        <<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(W_LN.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(W_QKV.data_ptr()),
            COS.data_ptr<float>(),
            SIN.data_ptr<float>(),
            POS.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(OUT.data_ptr()),
            (float)rms_eps, M);
    return OUT;
}


// ===========================================================================
// P2.5.2 bootstrap — L2 prefetch helper.
//
// Issues dummy `volatile` loads of the given tensor's bytes; the loads pass
// through L2 (the default cache policy for global loads), warming L2 for a
// subsequent "real" kernel that reads the same bytes. Launched on a side
// stream so it can run concurrently with compute on the main stream.
//
// Use case: during layer N's compute, prefetch layer N+1's weights so that
// when layer N+1's first GEMM runs, its weights are already resident in L2
// (128 MB L2 > one base_lm layer's 67 MB of weights — fits).
// ===========================================================================

__global__ void vcpm_l2_warm_kernel(
        const __nv_bfloat16* __restrict__ ptr, size_t num_elems) {
    // Each thread touches every 128th element (32 threads × 4 = 128, cacheline).
    // We only need to hit each 128B cache line once.
    constexpr size_t STRIDE = 64;  // 64 × 2 bytes = 128B cache line
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t start = tid * STRIDE;
    if (start < num_elems) {
        // A single volatile load per cache line populates L2.
        volatile __nv_bfloat16 v = ptr[start];
        (void)v;
    }
}


void vcpm_l2_warm(const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kBFloat16, "bf16 cuda");
    TORCH_CHECK(t.is_contiguous(), "contig");
    size_t num_elems = (size_t)t.numel();
    constexpr int THREADS = 256;
    // We touch one element per 128B cache line (stride 64 bf16 elements).
    size_t num_touches = (num_elems + 63) / 64;
    int blocks = (int)((num_touches + THREADS - 1) / THREADS);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    auto stream = at::cuda::getCurrentCUDAStream();
    vcpm_l2_warm_kernel<<<blocks, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()), num_elems);
}


// ===========================================================================
// Binding table.
// ===========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("times_two", &vcpm_times_two,
          "Sanity kernel: returns 2*x for bf16 cuda input.");
    m.def("rmsnorm", &vcpm_rmsnorm,
          "RMSNorm: y = w * x / sqrt(mean(x**2) + eps). Args: x[N,H] bf16, w[H] bf16, eps float. Returns y[N,H] bf16.");
    m.def("gemm_bf16", &vcpm_gemm_bf16,
          "bf16 GEMM: C = A @ B^T. Args: A[M,K] bf16, B[N,K] bf16 (weight [out, in] row-major). Returns C[M,N] bf16. K must be multiple of 16.");
    m.def("gemm_bf16_tuned_tn", &vcpm_gemm_bf16_tuned_tn,
          "Debug/tuning entry for vcpm_gemm_bf16_tuned with forced TN ∈ {64, 128}.");
    m.def("gemm_bf16_tuned", &vcpm_gemm_bf16_tuned,
          "bf16 GEMM (tuned): C = A @ B^T with 3-stage cp.async pipelining + warp-group tiling (TM=64, TN=128, TK=32). Caller pads M to 64, K to 32, N to 128.");
    m.def("gemm_bf16_tuned_residual", &vcpm_gemm_bf16_tuned_residual,
          "bf16 GEMM + residual (tuned): C = A @ B^T + residual. residual [M, N] bf16.");
    m.def("gemm_bf16_tuned_silu", &vcpm_gemm_bf16_tuned_silu,
          "Fused silu_mul+GEMM (tuned): C = silu(A[:, :K]) * A[:, K:2K] @ B^T. A is [M, 2K].");
    m.def("gemm_bf16_tuned_silu_residual", &vcpm_gemm_bf16_tuned_silu_residual,
          "Fused silu_mul+GEMM+residual (tuned): C = silu(A[:, :K]) * A[:, K:2K] @ B^T + residual.");
    m.def("silu_mul", &vcpm_silu_mul,
          "SiLU*mul: in [N, 2*I] (gate||up), out [N, I] bf16.");
    m.def("residual_add", &vcpm_residual_add,
          "Residual add in-place: a += b. Both bf16 cuda contiguous.");
    m.def("rope_inplace", &vcpm_rope_inplace,
          "MiniCPM-LongRoPE in-place on Q and K slices of [N, QKV_DIM] bf16.");
    m.def("attention_causal", &vcpm_attention_causal,
          "Inline causal attention (flash-attn-2 style, prefill). GQA 16/2, "
          "head_dim=128, Q_BLOCK=32, K_BLOCK=32. Args: q[N, Q_DIM] bf16, "
          "k[N, KV_DIM] bf16, v[N, KV_DIM] bf16, scale float. Returns o[N, Q_DIM] bf16.");
    m.def("l2_warm", &vcpm_l2_warm,
          "Issue dummy volatile loads of the tensor to populate L2 cache. "
          "Meant for running on a side stream concurrently with compute; the "
          "target tensor's bytes will be L2-resident when next read from a GEMM.");
    m.def("fused_pre_attn", &vcpm_fused_pre_attn,
          "P2.5.1.c fused rmsnorm + qkv_gemm + rope for base_lm shape "
          "(H=2048, QKV_DIM=2560, NUM_Q=16, NUM_KV=2, D=128). "
          "Args: X[M,H] bf16, W_LN[H] bf16, W_QKV[QKV_DIM,H] bf16, "
          "COS/SIN[max_pos,D] fp32, POS[M] int32, rms_eps float. "
          "Returns QKV[M, QKV_DIM] bf16 with RoPE applied to Q and K regions.");
}
