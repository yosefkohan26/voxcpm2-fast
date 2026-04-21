"""P2.5.2 Step 2b.2 — validate the 9-phase cooperative megakernel
(full DiT layer in ONE kernel launch) against the chained FusedLayer.

This is the capstone of Step 2b: RMSNorm + QKV GEMM + RoPE + non-causal
attention + O GEMM+residual + RMSNorm2 + gate_up GEMM + silu_mul +
down GEMM+residual — nine phases, one cooperative launch, one
cudaLaunchCooperativeKernel call, `cg::this_grid().sync()` between each.

Reference: `fused_layer_chained.FusedLayer(hidden=1024, intermediate=4096,
causal=False)` forward at DiT shape.

Numerics bar (CLAUDE.md R4): bf16 max-abs ≤ 1e-2.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))

import mk_dit_prefill_ext as mk_ext

from fused_layer_chained import FusedLayer


H = 1024
INTERMEDIATE = 4096
QKV_DIM = 2560
Q_DIM = 2048
KV_DIM = 256
NUM_Q = 16
NUM_KV = 2
HEAD_DIM = 128
RMS_EPS = 1e-6
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _pad_M_to_64(x: torch.Tensor) -> torch.Tensor:
    M = x.size(0)
    pad = (64 - M % 64) % 64
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, pad))


def _make_inputs(B: int, S: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    N_real = B * S
    hs_real = (torch.randn(N_real, H, device="cuda", dtype=torch.bfloat16,
                           generator=g) * 0.1).contiguous()
    hs = _pad_M_to_64(hs_real).contiguous()
    positions_real = torch.arange(N_real, device="cuda", dtype=torch.int32) % 1024
    positions = torch.nn.functional.pad(positions_real, (0, hs.size(0) - N_real), value=0)
    return hs, positions, hs_real


def _make_weights(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w_in_ln = (torch.ones(H, device="cuda", dtype=torch.bfloat16)
               + torch.randn(H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
               ).contiguous()
    w_qkv = (torch.randn(QKV_DIM, H, device="cuda", dtype=torch.bfloat16,
                         generator=g) * 0.02).contiguous()
    w_o = (torch.randn(H, Q_DIM, device="cuda", dtype=torch.bfloat16,
                       generator=g) * 0.02).contiguous()
    w_post_ln = (torch.ones(H, device="cuda", dtype=torch.bfloat16)
                 + torch.randn(H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
                 ).contiguous()
    w_gu = (torch.randn(2 * INTERMEDIATE, H, device="cuda", dtype=torch.bfloat16,
                        generator=g) * 0.02).contiguous()
    w_dn = (torch.randn(H, INTERMEDIATE, device="cuda", dtype=torch.bfloat16,
                        generator=g) * 0.02).contiguous()
    return {
        "input_layernorm.weight":          w_in_ln,
        "self_attn.qkv_proj.weight":       w_qkv,
        "self_attn.o_proj.weight":         w_o,
        "post_attention_layernorm.weight": w_post_ln,
        "mlp.gate_up_proj.weight":         w_gu,
        "mlp.down_proj.weight":            w_dn,
    }


def _make_rope_cache(max_pos=2048, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    cos = (torch.randn(max_pos, HEAD_DIM, device="cuda", dtype=torch.float32,
                       generator=g) * 0.5 + 0.5).contiguous()
    sin = (torch.randn(max_pos, HEAD_DIM, device="cuda", dtype=torch.float32,
                       generator=g) * 0.5).contiguous()
    return cos, sin


def _chained_ref(hs_real, weights, cos, sin, positions_real, B, S):
    """FusedLayer forward on the REAL rows (no M-padding). FusedLayer's
    `batch_size` argument assumes N == batch_size*seq exactly, so we must
    pass N_real = B*S rows in, not the padded M."""
    layer = FusedLayer(
        weights=weights,
        rope_cos_cache=cos, rope_sin_cache=sin,
        hidden=H, intermediate=INTERMEDIATE,
        causal=False, use_rope=True, rms_eps=RMS_EPS)
    with torch.inference_mode():
        return layer.forward(hs_real, positions_real, batch_size=B)


def _check(label, ref, ours, N_real, bar=1e-2):
    diff = (ref[:N_real].float() - ours[:N_real].float()).abs()
    max_abs = diff.max().item()
    mae = diff.mean().item()
    max_val = ref[:N_real].float().abs().max().item()
    print(f"  [{label}] real_rows={N_real}  "
          f"max_abs={max_abs:.3e}  mae={mae:.3e}  max_val={max_val:.3e}")
    assert max_abs <= bar, f"[{label}] max_abs={max_abs:.3e} > {bar}"


def test_dit_shape():
    """DiT inference: B=2, S=11 → N_real=22, padded M=64."""
    hs, positions, hs_real = _make_inputs(B=2, S=11, seed=17)
    weights = _make_weights(seed=19)
    cos, sin = _make_rope_cache()
    B, S = 2, 11
    N_real = B * S
    positions_real = positions[:N_real].contiguous()

    ref  = _chained_ref(hs_real, weights, cos, sin, positions_real, B, S)
    ours = mk_ext.step2b2_full_layer(
        hs,
        weights["input_layernorm.weight"],
        weights["self_attn.qkv_proj.weight"],
        weights["self_attn.o_proj.weight"],
        weights["post_attention_layernorm.weight"],
        weights["mlp.gate_up_proj.weight"],
        weights["mlp.down_proj.weight"],
        cos, sin, positions, B, S,
        RMS_EPS, ATTN_SCALE)

    print("[dit_shape] B=2 S=11 (full 9-phase layer)")
    _check("step2b2_layer_out", ref, ours, N_real)


def test_larger_batch():
    """B=4, S=32 → N_real=128 (64-aligned)."""
    hs, positions, hs_real = _make_inputs(B=4, S=32, seed=29)
    weights = _make_weights(seed=31)
    cos, sin = _make_rope_cache()
    B, S = 4, 32
    N_real = B * S
    positions_real = positions[:N_real].contiguous()

    ref  = _chained_ref(hs_real, weights, cos, sin, positions_real, B, S)
    ours = mk_ext.step2b2_full_layer(
        hs,
        weights["input_layernorm.weight"],
        weights["self_attn.qkv_proj.weight"],
        weights["self_attn.o_proj.weight"],
        weights["post_attention_layernorm.weight"],
        weights["mlp.gate_up_proj.weight"],
        weights["mlp.down_proj.weight"],
        cos, sin, positions, B, S,
        RMS_EPS, ATTN_SCALE)

    print("[larger_batch] B=4 S=32 (full 9-phase layer)")
    _check("step2b2_layer_out", ref, ours, N_real)


def main():
    test_dit_shape()
    test_larger_batch()
    print()
    print("All Step 2b.2 FULL-LAYER cooperative-megakernel numerics tests PASSED.")
    print("One cudaLaunchCooperativeKernel = one complete DiT layer forward.")


if __name__ == "__main__":
    main()
