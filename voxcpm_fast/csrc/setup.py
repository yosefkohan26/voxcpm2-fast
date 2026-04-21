"""Build script for the P2.1 persistent-kernel PoC extension.

Usage (from any cwd):

    UV_CACHE_DIR=/workspace/Developments/VoxCPM2/.uv_cache \\
      MAX_JOBS=4 \\
      uv run python \\
      /workspace/Developments/VoxCPM2/voxcpm_fast/csrc/setup.py build_ext --inplace

The .so lands next to voxcpm_fast/persistent_kernel.py as
`voxcpm_fast/persistent_poc_ext.*.so` so the wrapper can just
`import persistent_poc_ext`.

MAX_JOBS is hard-capped at 4 (see AGENTS.md §5a) — larger values OOM'd the
host in the prior incident.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent
VOXCPM_FAST = HERE.parent  # …/voxcpm_fast

# Always land the built .so inside voxcpm_fast/ so the Python wrapper can find
# it without futzing with sys.path at call sites.
os.chdir(VOXCPM_FAST)

# Clamp MAX_JOBS defensively. We're not going to let a stale env var from
# a parent shell blow up the host.
jobs = int(os.environ.get("MAX_JOBS", "4"))
if jobs > 4:
    print(f"[setup] MAX_JOBS={jobs} > 4 — clamping to 4 (AGENTS.md §5a).",
          file=sys.stderr)
    jobs = 4
os.environ["MAX_JOBS"] = str(jobs)


NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    # RTX 5090 Blackwell — sm_120a enables the WGMMA/TMA dialect the real
    # megakernels will use. This PoC doesn't exercise them but compiles for
    # the same target on purpose.
    "-gencode=arch=compute_120,code=sm_120",
    "-gencode=arch=compute_120a,code=sm_120a",
    # Enable the volta+ async mbarrier / acquire-release inline asm we use.
    "-DNDEBUG",
    "-lineinfo",
]

CXX_FLAGS = ["-O3", "-std=c++17"]

ext = CUDAExtension(
    # Package-relative name: the .so gets copied to the directory we cd'd into
    # above, i.e. voxcpm_fast/persistent_poc_ext*.so.
    name="persistent_poc_ext",
    sources=[str(HERE / "persistent_poc.cu")],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)

# P2.2 — fused non-causal transformer layer (cooperative persistent megakernel).
fused_ext = CUDAExtension(
    name="fused_layer_noncausal_ext",
    sources=[str(HERE / "fused_layer_noncausal.cu")],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        # mma.h / wmma requires fragment-aware codegen; add the relaxed
        # constexpr + rdc flag only where it helps.
        "nvcc": NVCC_FLAGS + [
            # Cooperative launch needs rdc=true on many toolchains; on 12.8
            # the runtime API supports __device__ cg without RDC for
            # this_grid().sync(), so we don't set it here. Keep commented
            # in case a future toolchain regresses.
            # "-rdc=true",
        ],
    },
)

# P2.2 rev2 — chained (non-cooperative) fused-layer extension. Each op is a
# standalone __global__ kernel launched on a CUDA stream; sequencing comes
# from stream FIFO semantics, not cooperative grid sync. Debuggable with
# compute-sanitizer.
chained_ext = CUDAExtension(
    name="fused_layer_chained_ext",
    sources=[str(HERE / "fused_layer_chained.cu")],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)

# P2.5.2 — DiT persistent megakernel. Cooperative launch with
# cg::this_grid().sync() between phases. First commit is a no-op
# scaffolding kernel; subsequent commits grow it into the full fused DiT.
mk_dit_ext = CUDAExtension(
    name="mk_dit_prefill_ext",
    sources=[str(VOXCPM_FAST / "megakernels" / "mk_dit_prefill.cu")],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)

setup(
    name="voxcpm_fast_exts",
    version="0.0.4",
    description="VoxCPM2-Fast CUDA extensions (P2.1 PoC + P2.2 coop + P2.2 chained + P2.5.2 DiT megakernel)",
    ext_modules=[ext, fused_ext, chained_ext, mk_dit_ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
