// Probe kernel: does wgmma compile / run on sm_120a?
// Tests a single m64n16k16.f32.bf16.bf16 warpgroup MMA.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

__device__ __forceinline__ uint64_t make_desc(
        const void* smem_ptr,
        uint32_t leading_dim_bytes,
        uint32_t stride_bytes,
        uint32_t base_offset,
        uint32_t swizzle) {
    uint32_t smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= ((uint64_t)(smem_int_ptr >> 4) & 0x3FFF);
    desc |= ((uint64_t)(leading_dim_bytes >> 4) & 0x3FFF) << 16;
    desc |= ((uint64_t)(stride_bytes >> 4) & 0x3FFF) << 32;
    desc |= ((uint64_t)(base_offset & 0x7)) << 49;
    desc |= ((uint64_t)(swizzle & 0x3)) << 62;
    return desc;
}

extern "C" __global__ void wgmma_probe(__nv_bfloat16* out) {
    // SMEM: [64, 16] bf16 A, [16, 16] bf16 B
    __shared__ __align__(16) __nv_bfloat16 sA[64 * 16];
    __shared__ __align__(16) __nv_bfloat16 sB[16 * 16];

    int tid = threadIdx.x;
    // Fill A and B with simple patterns
    if (tid < 16) {
        for (int r = 0; r < 64; ++r) sA[r * 16 + tid] = __float2bfloat16(1.0f);
        for (int r = 0; r < 16; ++r) sB[r * 16 + tid] = __float2bfloat16(1.0f);
    }
    __syncthreads();

    // Accumulators: m64n16 = 64*16 fp32 = 1024 values, distributed across warpgroup (128 threads).
    // Each thread holds 1024/128 = 8 fp32 accumulators.
    float d[8] = {0,0,0,0,0,0,0,0};

    uint64_t desc_a = make_desc(sA, 16, 16 * 64, 0, 0);  // no swizzle
    uint64_t desc_b = make_desc(sB, 16, 16 * 16, 0, 0);

    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(desc_a), "l"(desc_b));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    // Each element of output matrix C[m][n] should equal K = 16 (sum of 16 ones).
    // Write d[0] to out[tid] just to verify non-zero output.
    out[tid] = __float2bfloat16(d[0]);
}

int main() {
    __nv_bfloat16* d_out;
    cudaMalloc(&d_out, 128 * sizeof(__nv_bfloat16));
    cudaMemset(d_out, 0, 128 * sizeof(__nv_bfloat16));
    wgmma_probe<<<1, 128>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    __nv_bfloat16 h_out[128];
    cudaMemcpy(h_out, d_out, 128 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    printf("WGMMA probe: first 4 outputs = %.1f %.1f %.1f %.1f (expected ~16.0)\n",
           __bfloat162float(h_out[0]), __bfloat162float(h_out[1]),
           __bfloat162float(h_out[2]), __bfloat162float(h_out[3]));
    cudaFree(d_out);
    return 0;
}
