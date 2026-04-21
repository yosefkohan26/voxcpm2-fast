"""P2.5.1.c — numerics + perf for the fused pre_attn kernel.

Compares `vcpm_fused_pre_attn(x, w_ln, w_qkv, cos, sin, pos, eps)` against
the 3-kernel chained reference at base_lm.layers.0 weights:
    ln_out = rmsnorm(x, w_ln, eps)
    qkv    = gemm_bf16_tuned(ln_out, w_qkv)
    rope_inplace(qkv, cos, sin, pos, NUM_Q=16, NUM_KV=2, D=128)

Gates:
  - max rel ≤ 1e-2  (a couple bf16 ULPs at layer output magnitudes)
  - mean rel ≤ 1e-3
"""

from __future__ import annotations

import os
import sys
import time
import statistics
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"
MODEL_DIR = REPO_ROOT / "models" / "VoxCPM2"

sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))
sys.path.insert(0, str(NANOVLLM_ROOT))


H = 2048
NUM_Q = 16
NUM_KV = 2
D = 128
Q_DIM = NUM_Q * D
KV_DIM = NUM_KV * D
QKV_DIM = Q_DIM + 2 * KV_DIM


def _init_dist():
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("gloo", rank=0, world_size=1)


def _load_base_lm_layer0():
    _init_dist()
    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.model import Cpm4DecoderLayer
    from safetensors.torch import safe_open

    cfg = VoxCPM2Config.model_validate_json((MODEL_DIR / "config.json").read_text())
    lm_cfg = cfg.lm_config.model_copy(deep=True)
    lm_cfg.use_mup = False

    torch.set_default_dtype(torch.bfloat16)
    layer = Cpm4DecoderLayer(lm_cfg, is_causal=True).cuda().eval()
    torch.set_default_dtype(torch.float32)

    prefix = "base_lm.layers.0."
    loaded = {}
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            if not k.startswith(prefix):
                continue
            loaded[k[len(prefix):]] = f.get_tensor(k)
    q = loaded["self_attn.q_proj.weight"]
    k_ = loaded["self_attn.k_proj.weight"]
    v = loaded["self_attn.v_proj.weight"]
    loaded["self_attn.qkv_proj.weight"] = torch.cat([q, k_, v], dim=0)

    rope = layer.self_attn.rotary_emb
    cos_cache = rope.cos_cached.to(torch.float32).contiguous()
    sin_cache = rope.sin_cached.to(torch.float32).contiguous()

    w_ln  = loaded["input_layernorm.weight"].to(torch.bfloat16).cuda().contiguous()
    w_qkv = loaded["self_attn.qkv_proj.weight"].to(torch.bfloat16).cuda().contiguous()
    return w_ln, w_qkv, cos_cache, sin_cache, lm_cfg.rms_norm_eps


@pytest.mark.parametrize("M_unpad", [16, 32, 64, 100, 128])
def test_fused_pre_attn_numerics(M_unpad):
    torch.manual_seed(11)
    w_ln, w_qkv, cos, sin, eps = _load_base_lm_layer0()
    import fused_layer_chained_ext as _ext

    # Pad M to multiple of TM=16 (kernel requirement); also TM_GEMM=64 for chained ref.
    # Use the same M_padded for both paths so comparisons are apples-to-apples.
    M_padded_for_tuned = ((M_unpad + 63) // 64) * 64
    M_padded_for_fused = ((M_unpad + 15) // 16) * 16
    M = max(M_padded_for_tuned, M_padded_for_fused)

    x = (torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    pos = torch.arange(M, device="cuda", dtype=torch.int32)

    # Chained reference: rmsnorm → gemm → rope
    ref_ln = _ext.rmsnorm(x, w_ln, eps)
    ref_qkv = _ext.gemm_bf16_tuned(ref_ln, w_qkv)
    _ext.rope_inplace(ref_qkv, cos, sin, pos, NUM_Q, NUM_KV, D)

    # Fused kernel
    got = _ext.fused_pre_attn(x, w_ln, w_qkv, cos, sin, pos, eps)

    # Only compare valid rows.
    ref_valid = ref_qkv[:M_unpad]
    got_valid = got[:M_unpad]

    diff = (got_valid.float() - ref_valid.float()).abs()
    up_max = ref_valid.abs().max().item()
    max_abs = diff.max().item()
    mae = diff.mean().item()
    rel_max = max_abs / max(up_max, 1e-9)
    rel_mae = mae / max(up_max, 1e-9)

    print()
    print(f"=== pre_attn fused M={M_unpad} (padded to {M}) ===")
    print(f"  up_max={up_max:.4f}  got_max={got_valid.abs().max().item():.4f}")
    print(f"  max={max_abs:.4e}  mae={mae:.4e}  rel_max={rel_max:.4e}  rel_mae={rel_mae:.4e}")

    assert rel_max <= 1e-2, f"M={M_unpad}: rel_max too high: {rel_max:.4e}"
    assert rel_mae <= 1e-3, f"M={M_unpad}: rel_mae too high: {rel_mae:.4e}"


def test_fused_pre_attn_perf():
    w_ln, w_qkv, cos, sin, eps = _load_base_lm_layer0()
    import fused_layer_chained_ext as _ext

    M = 128  # padded from 100
    x = (torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    pos = torch.arange(M, device="cuda", dtype=torch.int32)

    # Chained reference steps
    def chained():
        ln = _ext.rmsnorm(x, w_ln, eps)
        qkv = _ext.gemm_bf16_tuned(ln, w_qkv)
        _ext.rope_inplace(qkv, cos, sin, pos, NUM_Q, NUM_KV, D)
        return qkv

    def fused():
        return _ext.fused_pre_attn(x, w_ln, w_qkv, cos, sin, pos, eps)

    for _ in range(20):
        chained(); fused()
    torch.cuda.synchronize()

    iters = 300

    def bench(fn):
        e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            e0[i].record()
            fn()
            e1[i].record()
        torch.cuda.synchronize()
        xs = sorted(e0[i].elapsed_time(e1[i]) * 1000.0 for i in range(iters))
        return xs[len(xs) // 2]

    p50_chained = bench(chained)
    p50_fused   = bench(fused)

    print()
    print(f"=== pre_attn perf M={M} ===")
    print(f"  chained (rmsnorm+gemm+rope): {p50_chained:7.2f} µs")
    print(f"  fused single-kernel       : {p50_fused:7.2f} µs")
    print(f"  speedup                    : {p50_chained/p50_fused:.2f}×")


if __name__ == "__main__":
    for m in [16, 32, 64, 100, 128]:
        test_fused_pre_attn_numerics(m)
    test_fused_pre_attn_perf()
