"""Wrapped spawn-target that installs the fast-path hook inside the child
process *before* the upstream main_loop constructs VoxCPM2ServerImpl.

The ``mp.get_context("spawn")`` child imports modules from scratch, so a
monkey-patch applied in the parent process doesn't transfer. Instead we
swap the `main_loop` reference that ``AsyncVoxCPM2Server.__init__`` passes
as ``target=...`` with this wrapper. The wrapper is picklable-by-qualified-
name, so spawn re-imports it in the child, at which point it installs the
patch and then delegates to the original ``main_loop``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_sys_path():
    """Add voxcpm_fast's parent to sys.path so the child can re-import
    `voxcpm_fast.*` after spawn."""
    repo_root = Path("/workspace/Developments/VoxCPM2")
    for p in (str(repo_root / "voxcpm_fast"), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def fast_main_loop(queue_in, queue_out, args, kwargs):
    _ensure_sys_path()

    # Default enables; flip off via env if needed.
    enc = os.environ.get("VOXCPM_FAST_ENC", "1") != "0"
    dit = os.environ.get("VOXCPM_FAST_DIT", "1") != "0"
    # base_lm + residual_lm swap: currently net-negative at small N because
    # the tuned GEMM (TM=64) forces M=16→64 padding, computing 4× waste per
    # GEMM; at bucket=16 that's ≈+1 ms over upstream's small-M cuBLAS call.
    # Keep the code in place (works end-to-end) but disable by default.
    # Re-evaluate once we ship a smaller-tile (TM=16 or TM=32) GEMM variant
    # or once we bucket up to N≥64 where padding cost disappears.
    base = os.environ.get("VOXCPM_FAST_BASE", "0") != "0"
    res = os.environ.get("VOXCPM_FAST_RES", "0") != "0"

    # L2 prefetch of next-layer weights via a side-stream _ext.l2_warm.
    # Measured +1.4 ms T_first win at bucket=16 (23.5 vs 24.9). Enabled by
    # default; set VOXCPM_PREFETCH=0 to disable for A/B.
    os.environ.setdefault("VOXCPM_PREFETCH", "l2")

    # NOTE: upstream's kvcache_block_size = 256 is a flash_attn paged-KV
    # hard requirement ("Paged KV cache block size must be divisible by
    # 256"), not user-paranoia. Shorter voice prompts (~80-200 tokens)
    # therefore cannot use the block-hash prefix cache as-is. A proper
    # speaker-prompt cache (feat_encoder output + base_lm K/V for the
    # prompt prefix) is tracked as PROJECT_PLAN P2.8.

    from voxcpm_fast.engine_hook import (
        install_fast_path, install_prefill_graph_capture, install_timing_probe,
        install_model_forward_probe, install_graphed_phase_probe,
    )
    install_fast_path(enable_feat_encoder=enc, enable_dit=dit,
                      enable_base_lm=base, enable_residual_lm=res)
    # Install forward probe BEFORE graph capture so the probe's wrapper
    # becomes what gets captured (and thus skipped at replay).
    install_model_forward_probe()
    # Inline-event probe of the captured graph: records events INSIDE the
    # graph body so their timestamps update each replay.
    install_graphed_phase_probe()
    if os.environ.get("VOXCPM_PREFILL_GRAPH", "1") != "0":
        install_prefill_graph_capture()
    if os.environ.get("VOXCPM_TIMING") == "1":
        install_timing_probe(log_every=1)

    # Delegate to the original upstream main_loop.
    from nanovllm_voxcpm.models.voxcpm2.server import main_loop as _upstream
    return _upstream(queue_in, queue_out, args, kwargs)


def patch_server_module() -> None:
    """Replace `server.main_loop` with our wrapper so future
    `AsyncVoxCPM2Server` constructions spawn into `fast_main_loop`.
    """
    import nanovllm_voxcpm.models.voxcpm2.server as _srv
    _srv.main_loop = fast_main_loop
