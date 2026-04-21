// voxcpm_fast/megakernels/mk_dit_prefill.cu
//
// P2.5.2 Phase A, Step 1 — cooperative-launch scaffolding.
//
// Goal of this file: prove that we can launch a cooperative kernel on
// sm_120a, run cg::this_grid().sync() phase barriers, and write results
// to HBM — before we add any real compute. If this compiles, launches,
// and the tensor comes back filled, we have the foundation the rest of
// the DiT megakernel will live in.
//
// Subsequent commits will grow this kernel into:
//   Step 2: one DiT layer fused in-kernel (rmsnorm+qkv+rope+attn+o+ln+gu+silu+dn)
//   Step 3: all 12 DiT layers in-kernel with grid.sync between layers
//   Step 4: the Euler loop (9 iterations) wrapped around step 3
//   Step 5: engine wire-in via FusedDiTMegakernelShim
//
// Compiled for sm_120a (RTX 5090 Blackwell). No WGMMA/tcgen05 per the
// hardware-capability survey in AGENT_LOG 2026-04-20.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <cstdio>

namespace cg = cooperative_groups;

namespace voxcpm_fast {

// -------------------------------------------------------------------------
// Step 1 — no-op cooperative kernel.
// Each block writes its blockIdx.x into OUT[blockIdx.x], then
// grid.sync(), then writes 2*blockIdx.x into OUT[blockIdx.x + num_blocks]
// so the test can distinguish "before sync" from "after sync".
// -------------------------------------------------------------------------

extern "C" __global__ void mk_dit_prefill_noop_kernel(
        int32_t* __restrict__ OUT,   // [num_blocks * 2]
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

// Launcher. Uses cudaLaunchCooperativeKernel so cg::this_grid().sync()
// is legal (regular <<<>>> launches forbid grid.sync).
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

    // The launch is cooperative → all blocks must be resident concurrently.
    // Cap num_blocks by (sm_count * num_blocks_per_sm).
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

}  // namespace voxcpm_fast

// PyBind11 module — this builds alongside (or separately from) the chained
// fused_layer extension. Keeping it in its own module avoids shared-state
// surprise while we iterate.
PYBIND11_MODULE(mk_dit_prefill_ext, m) {
    m.def("noop", &voxcpm_fast::mk_dit_prefill_noop,
          "P2.5.2 scaffolding: cooperative-launch no-op that writes block "
          "indices before and after a grid.sync().",
          pybind11::arg("num_blocks"));
}
