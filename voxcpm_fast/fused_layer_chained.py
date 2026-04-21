"""Python wrapper for the P2.2 chained fused non-causal transformer layer.

Implements one ``Cpm4DecoderLayer(is_causal=False)``-equivalent forward by
chaining our custom CUDA kernels in a single CUDA stream. Each kernel is
a plain ``<<<grid,block>>>`` launch; ordering is guaranteed by stream FIFO
semantics. This is the validation vehicle for P2.2 (physics-floor is P2.5
once we fold everything into one persistent kernel).

Attention is currently ``flash_attn.flash_attn_func`` (non-causal, GQA).
A native implementation lives inside the future persistent megakernel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fused_layer_chained_ext as _ext  # noqa: E402


# Shape constants shared by feat_encoder and base_lm. Heads and head_dim are
# the same across the whole model; hidden + intermediate vary per-component.
NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
Q_DIM = NUM_HEADS * HEAD_DIM          # 2048
KV_DIM = NUM_KV_HEADS * HEAD_DIM      # 256
QKV_DIM = Q_DIM + 2 * KV_DIM          # 2560
INTERMEDIATE = 4096


def _pad_M_to(x: torch.Tensor, pad_to: int) -> tuple[torch.Tensor, int]:
    """Pad the M (first) dim to a multiple of pad_to with zeros. Returns (padded, pad).

    Must zero-fill (can't use uninitialized): flash_attn-2 computes Q·K^T for
    masked elements and the masked values participate in the rowmax before
    being -INF'd, so NaN/Inf in junk Q/K rows can propagate into real rows'
    softmax. Zero-fill is safe (Q·K on zeros stays finite).
    """
    M = x.size(0)
    pad = (pad_to - M % pad_to) % pad_to
    if pad == 0:
        return x, 0
    return torch.nn.functional.pad(x, (0,) * (2 * (x.dim() - 1)) + (0, pad)), pad


# Back-compat: prior callers used _pad_M_to_16.
def _pad_M_to_16(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    return _pad_M_to(x, 16)


# Switch between the prior WMMA kernel and the P2.5.1.a tuned GEMM via env.
# Default: tuned. Set VOXCPM_GEMM=wmma to fall back.
import os
_GEMM_KIND = os.environ.get("VOXCPM_GEMM", "tuned")


def _gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """bf16 GEMM with auto M-padding.

    VOXCPM_GEMM=tuned   uses vcpm_gemm_bf16_tuned (cp.async pipelined, TM=64)
    VOXCPM_GEMM=wmma    uses vcpm_gemm_bf16        (WMMA 16x16 tiles)
    """
    orig_M = A.size(0)
    if _GEMM_KIND == "tuned":
        A_p, pad = _pad_M_to(A, 64)
        C_p = _ext.gemm_bf16_tuned(A_p.contiguous(), B)
    else:
        A_p, pad = _pad_M_to(A, 16)
        C_p = _ext.gemm_bf16(A_p.contiguous(), B)
    if pad == 0:
        return C_p
    return C_p[:orig_M].contiguous()


def _gemm_silu_residual(
    gu: torch.Tensor, B: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Fused silu_mul + GEMM + residual:
       C = silu(gu[:, :K]) * gu[:, K:2K] @ B^T + residual.
    Saves 28 kernel launches + silu_mul intermediate HBM (~45 MB) per forward.
    gu is [M, 2K] bf16 (gate||up). B is [N, K] bf16. residual is [M, N] bf16.
    """
    orig_M = gu.size(0)
    if _GEMM_KIND != "tuned":
        raise NotImplementedError("silu-fused path requires VOXCPM_GEMM=tuned")
    A_p, pad = _pad_M_to(gu, 64)
    R_p, pad_r = _pad_M_to(residual, 64)
    assert pad == pad_r
    C_p = _ext.gemm_bf16_tuned_silu_residual(A_p.contiguous(), B, R_p.contiguous())
    if pad == 0:
        return C_p
    return C_p[:orig_M].contiguous()


def _gemm_residual(A: torch.Tensor, B: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """C = A @ B^T + residual, with auto M-padding. Fuses residual_add into
    the tuned GEMM's epilogue (single kernel launch).

    Only available when VOXCPM_GEMM=tuned (default). Falls back to the
    separate gemm+residual_add path for VOXCPM_GEMM=wmma.
    """
    orig_M = A.size(0)
    if _GEMM_KIND == "tuned":
        A_p, pad = _pad_M_to(A, 64)
        R_p, pad_r = _pad_M_to(residual, 64)
        assert pad == pad_r
        C_p = _ext.gemm_bf16_tuned_residual(A_p.contiguous(), B, R_p.contiguous())
        if pad == 0:
            return C_p
        return C_p[:orig_M].contiguous()
    # Fallback: GEMM + residual_add.
    out = _gemm(A, B)
    out_clone = out.clone()  # add_ would alias the GEMM output otherwise
    _ext.residual_add(out_clone, residual)
    return out_clone




class FusedLayer:
    """Chained kernel implementation of one Cpm4DecoderLayer.

    Handles both ``is_causal=True`` (base_lm / residual_lm shape) and
    ``is_causal=False`` (feat_encoder / DiT shape). Hidden size is
    parametric; feat_encoder uses 1024, base_lm uses 2048.

    Weights dict keys must match upstream's ``Cpm4DecoderLayer.state_dict()``:
        input_layernorm.weight                  [hidden]
        self_attn.qkv_proj.weight               [QKV_DIM, hidden]
        self_attn.o_proj.weight                 [hidden, Q_DIM]
        post_attention_layernorm.weight         [hidden]
        mlp.gate_up_proj.weight                 [2*intermediate, hidden]
        mlp.down_proj.weight                    [hidden, intermediate]
    """

    def __init__(
        self,
        weights: Mapping[str, torch.Tensor],
        rope_cos_cache: torch.Tensor | None,
        rope_sin_cache: torch.Tensor | None,
        *,
        hidden: int,
        intermediate: int = INTERMEDIATE,
        causal: bool,
        use_rope: bool = True,
        rms_eps: float = 1e-6,
    ):
        self.hidden = hidden
        self.intermediate = intermediate
        self.causal = bool(causal)
        self.use_rope = bool(use_rope)

        def _get(name: str, shape: tuple[int, ...]) -> torch.Tensor:
            w = weights[name]
            assert w.is_cuda and w.dtype == torch.bfloat16, name
            assert tuple(w.shape) == shape, f"{name}: {tuple(w.shape)} != {shape}"
            assert w.is_contiguous(), f"{name} must be contiguous"
            return w

        self.w_in_ln   = _get("input_layernorm.weight",           (hidden,))
        self.w_qkv     = _get("self_attn.qkv_proj.weight",        (QKV_DIM, hidden))
        self.w_o       = _get("self_attn.o_proj.weight",          (hidden, Q_DIM))
        self.w_post_ln = _get("post_attention_layernorm.weight",  (hidden,))
        self.w_gu      = _get("mlp.gate_up_proj.weight",          (2 * intermediate, hidden))
        self.w_dn      = _get("mlp.down_proj.weight",             (hidden, intermediate))

        if self.use_rope:
            assert rope_cos_cache is not None and rope_sin_cache is not None
            assert rope_cos_cache.is_cuda and rope_cos_cache.dtype == torch.float32
            assert rope_sin_cache.is_cuda and rope_sin_cache.dtype == torch.float32
            assert rope_cos_cache.shape[-1] == HEAD_DIM
            self.cos = rope_cos_cache.contiguous()
            self.sin = rope_sin_cache.contiguous()
        else:
            self.cos = self.sin = None
        self.rms_eps = float(rms_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int = 1,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one decoder layer.

        Args:
            hidden_states: [N, hidden] bf16 cuda contiguous (N = batch_size*seq).
            positions:     [N] int32/int64 cuda contiguous.
            batch_size:    >1 routes attention as (batch_size, N/batch_size, H, D);
                           required for non-causal DiT where CFG batches must not
                           cross-attend.
            k_cache, v_cache, slot_mapping: if all three are given, writes the
                layer's K/V into kv_cache at slot_mapping positions so decode
                can later read them. Matches upstream's `store_kvcache` contract.
        Returns:
            [N, hidden] bf16 cuda (new tensor).
        """
        assert hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
        assert hidden_states.dim() == 2 and hidden_states.size(1) == self.hidden
        hs = hidden_states.contiguous()
        N = hs.size(0)
        assert N % batch_size == 0, f"N={N} not divisible by batch_size={batch_size}"

        # Fused pre_attn (rmsnorm + qkv_gemm + rope) via `VOXCPM_PRE_ATTN=fused`.
        # Default off: the fused kernel is ~10% faster standalone (35.5 µs vs
        # 39.4 µs chained) but regresses 28-layer graphed wall by ~0.2 ms
        # because (a) its ~96 KB SMEM keeps it at 1 block/SM on 160 blocks
        # vs the chained flow's 3 smaller-SMEM kernels that pipeline better
        # across the stream, and (b) the Python pad/slice shim around the
        # kernel adds per-call overhead. Kept as a validated building block
        # for the future persistent-megakernel P2.5.2 where the same fusion
        # becomes fundamental rather than optional.
        if self.causal and self.hidden == 2048 and os.environ.get("VOXCPM_PRE_ATTN") == "fused":
            hs_p, pad = _pad_M_to(hs, 64)
            if pad > 0:
                # Positions are the same length as hs; pad with any valid index.
                pos_p = torch.nn.functional.pad(positions.to(torch.int32), (0, pad))
            else:
                pos_p = positions.to(torch.int32) if positions.dtype != torch.int32 else positions
            qkv_p = _ext.fused_pre_attn(
                hs_p, self.w_in_ln, self.w_qkv,
                self.cos, self.sin, pos_p, self.rms_eps)
            qkv = qkv_p[:N].contiguous() if pad > 0 else qkv_p
        else:
            ln_out = _ext.rmsnorm(hs, self.w_in_ln, self.rms_eps)
            qkv = _gemm(ln_out, self.w_qkv)
            if self.use_rope:
                _ext.rope_inplace(
                    qkv, self.cos, self.sin, positions,
                    NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
                )

        # KV-cache write (upstream's store_kvcache). Views into qkv for K and V.
        if (k_cache is not None and v_cache is not None and slot_mapping is not None
                and k_cache.numel() > 0 and v_cache.numel() > 0):
            from nanovllm_voxcpm.layers.attention import store_kvcache
            QKV_STRIDE = qkv.stride(0)
            k_write = qkv.as_strided((N, NUM_KV_HEADS, HEAD_DIM),
                                     (QKV_STRIDE, HEAD_DIM, 1),
                                     storage_offset=Q_DIM)
            v_write = qkv.as_strided((N, NUM_KV_HEADS, HEAD_DIM),
                                     (QKV_STRIDE, HEAD_DIM, 1),
                                     storage_offset=Q_DIM + KV_DIM)
            store_kvcache(k_write, v_write, k_cache, v_cache, slot_mapping)

        # Causal path via inline kernel if VOXCPM_ATTN=inline, else flash_attn.
        # Default flash_attn because at the c=1 chained-kernel-forward it wins
        # in-context despite being 2× slower standalone — smaller SMEM, higher
        # per-SM occupancy, better overlap with adjacent GEMMs. The inline
        # kernel becomes valuable when folded into a fused pre_attn/post_attn
        # kernel that collapses launches (P2.5.1.c/d) — standalone it loses
        # to flash_attn.
        if self.causal and self.hidden == 2048 and os.environ.get("VOXCPM_ATTN") == "inline":
            q_view = qkv[:, :Q_DIM]
            k_view = qkv[:, Q_DIM:Q_DIM + KV_DIM]
            v_view = qkv[:, Q_DIM + KV_DIM:]
            attn_out = _ext.attention_causal(q_view, k_view, v_view, HEAD_DIM ** -0.5)
        else:
            # Use as_strided to build 3D (N, heads, head_dim) views directly
            # from qkv's contiguous storage — no extra alloc, no .contiguous()
            # copies. flash_attn_func accepts 3D input for varlen-style calls
            # when stride(-1)==1 which holds here.
            QKV_STRIDE = qkv.stride(0)  # == QKV_DIM for contiguous qkv
            seq = N // batch_size
            q3 = qkv.as_strided((batch_size, seq, NUM_HEADS, HEAD_DIM),
                                (seq * QKV_STRIDE, QKV_STRIDE, HEAD_DIM, 1),
                                storage_offset=0)
            k3 = qkv.as_strided((batch_size, seq, NUM_KV_HEADS, HEAD_DIM),
                                (seq * QKV_STRIDE, QKV_STRIDE, HEAD_DIM, 1),
                                storage_offset=Q_DIM)
            v3 = qkv.as_strided((batch_size, seq, NUM_KV_HEADS, HEAD_DIM),
                                (seq * QKV_STRIDE, QKV_STRIDE, HEAD_DIM, 1),
                                storage_offset=Q_DIM + KV_DIM)
            from flash_attn import flash_attn_func
            attn_out = flash_attn_func(
                q3, k3, v3,
                causal=self.causal,
                softmax_scale=HEAD_DIM ** -0.5,
            )
            # flash_attn returns contiguous (B, seq, H, D); .view to (N, Q_DIM) is a
            # zero-copy reshape since last 3 dims are already contiguous.
            attn_out = attn_out.view(N, Q_DIM)

        # Fuse o_proj @ attn_out + hs into one kernel epilogue.
        residual = _gemm_residual(attn_out, self.w_o, hs)

        ln2_out = _ext.rmsnorm(residual, self.w_post_ln, self.rms_eps)
        gu = _gemm(ln2_out, self.w_gu)
        mid = _ext.silu_mul(gu)
        # Fuse down_proj @ mid + residual into one kernel epilogue.
        # Note: a silu_mul-into-down-prologue fusion was tried and REGRESSED
        # integrated perf (5.30 → 9.88 ms, +86%) — the extra __syncthreads +
        # SMEM-to-SMEM merge pass between scratch cp.async and consumer MMA
        # breaks the K-loop pipeline. Kept as opt-in via `_gemm_silu_residual`
        # for future re-evaluation under a different (persistent / TMA)
        # scheduling regime. See AGENT_LOG 2026-04-20 "silu+down regression".
        residual = _gemm_residual(mid, self.w_dn, residual)
        return residual


# ---------------------------------------------------------------------------
# Back-compat alias for P2.2 call sites.
# ---------------------------------------------------------------------------

class FusedNonCausalLayer(FusedLayer):
    def __init__(self, weights, rope_cos_cache, rope_sin_cache,
                 hidden: int = 1024, rms_eps: float = 1e-6):
        super().__init__(
            weights=weights,
            rope_cos_cache=rope_cos_cache,
            rope_sin_cache=rope_sin_cache,
            hidden=hidden,
            intermediate=INTERMEDIATE,
            causal=False,
            rms_eps=rms_eps,
        )


# ---------------------------------------------------------------------------
# P2.4 — Full Cpm4Model forward (N causal layers + final RMSNorm).
# ---------------------------------------------------------------------------

class FusedCpm4Model:
    """Stack of FusedLayer(causal=True) emulating upstream Cpm4Model.forward.

    Matches ``base_lm`` semantically: embed tokens → N decoder layers → final
    RMSNorm. The embed_tokens step is NOT owned by this class — callers
    pass in ``input_embeds`` as [N, hidden] bf16 (same contract as upstream
    ``Cpm4Model.forward(input_embeds, positions)``).

    Weights dict contract:
        layers.{i}.input_layernorm.weight               [hidden]
        layers.{i}.self_attn.qkv_proj.weight            [QKV_DIM, hidden]
        layers.{i}.self_attn.o_proj.weight              [hidden, Q_DIM]
        layers.{i}.post_attention_layernorm.weight      [hidden]
        layers.{i}.mlp.gate_up_proj.weight              [2*intermediate, hidden]
        layers.{i}.mlp.down_proj.weight                 [hidden, intermediate]
        norm.weight                                     [hidden]
    for i in range(num_layers).
    """

    def __init__(
        self,
        weights: Mapping[str, torch.Tensor],
        rope_cos_cache: torch.Tensor | None,
        rope_sin_cache: torch.Tensor | None,
        *,
        hidden: int,
        intermediate: int,
        num_layers: int,
        causal: bool = True,
        use_rope: bool = True,
        rms_eps: float = 1e-5,
    ):
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_layers = num_layers
        self.rms_eps = float(rms_eps)

        self.layers: list[FusedLayer] = []
        for i in range(num_layers):
            layer_weights = {
                "input_layernorm.weight":          weights[f"layers.{i}.input_layernorm.weight"],
                "self_attn.qkv_proj.weight":       weights[f"layers.{i}.self_attn.qkv_proj.weight"],
                "self_attn.o_proj.weight":         weights[f"layers.{i}.self_attn.o_proj.weight"],
                "post_attention_layernorm.weight": weights[f"layers.{i}.post_attention_layernorm.weight"],
                "mlp.gate_up_proj.weight":         weights[f"layers.{i}.mlp.gate_up_proj.weight"],
                "mlp.down_proj.weight":            weights[f"layers.{i}.mlp.down_proj.weight"],
            }
            self.layers.append(FusedLayer(
                weights=layer_weights,
                rope_cos_cache=rope_cos_cache,
                rope_sin_cache=rope_sin_cache,
                hidden=hidden,
                intermediate=intermediate,
                causal=causal,
                use_rope=use_rope,
                rms_eps=rms_eps,
            ))

        self.w_final_norm = weights["norm.weight"]
        assert self.w_final_norm.is_cuda and self.w_final_norm.dtype == torch.bfloat16
        assert tuple(self.w_final_norm.shape) == (hidden,)

        # L2 persistence range (populated lazily by install_l2_persist).
        self._l2_persist_range: tuple[int, int] | None = None

    def install_l2_persist(self, max_window_bytes: int = 128 * 1024 * 1024) -> None:
        """Compute the pointer range spanning this model's layer weights and
        store it so `forward()` can mark it as hitProp=Persisting before a
        call. The CUDA ``cudaAccessPolicyWindow`` is limited to
        ``cudaDevAttrMaxAccessPolicyWindowSize`` (128 MB on sm_120a).

        If our weights are scattered across more than `max_window_bytes`,
        we clamp the window to start at the first weight's pointer and
        cover as much as we can — PyTorch's caching allocator generally
        groups sequential allocations nearby, so weights allocated
        back-to-back get partial coverage for free.
        """
        ptrs: list[tuple[int, int]] = []  # (ptr, bytes)
        for layer in self.layers:
            for t in (layer.w_in_ln, layer.w_qkv, layer.w_o, layer.w_post_ln,
                      layer.w_gu, layer.w_dn):
                ptrs.append((t.data_ptr(), t.numel() * t.element_size()))
        ptrs.append((self.w_final_norm.data_ptr(),
                     self.w_final_norm.numel() * self.w_final_norm.element_size()))
        min_ptr = min(p for p, _ in ptrs)
        max_end = max(p + sz for p, sz in ptrs)
        span = max_end - min_ptr
        win_bytes = min(span, max_window_bytes)
        self._l2_persist_range = (min_ptr, win_bytes)

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int = 1,
        kv_caches: list | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert input_embeds.is_cuda and input_embeds.dtype == torch.bfloat16
        assert input_embeds.dim() == 2 and input_embeds.size(1) == self.hidden
        orig_M = input_embeds.size(0)
        assert orig_M % batch_size == 0
        # kv_caches: list of (k_cache, v_cache) per layer, or None.
        if kv_caches is not None:
            assert len(kv_caches) == len(self.layers)

        # Pad M to 64 ONCE at the model boundary (instead of per-layer in each
        # _gemm / _gemm_residual call). Junk rows are fine for: rmsnorm (row-
        # local), GEMM (output rows we discard), RoPE (harmless extra work),
        # silu_mul (row-local), residual_add (row-local), and causal attention
        # (real rows never attend to later junk rows due to causal mask).
        #
        # NOTE: model-level padding is only safe when attention is causal AND
        # batch_size==1. Non-causal attention smears softmax across junk rows
        # (zero Q·K gives uniform attn weights across all 64 positions).
        # Batched attention (bsz>1) would also cross-pollute batches with junk.
        # In both unsafe cases, rely on per-GEMM padding inside `_gemm*`.
        causal_any = any(l.causal for l in self.layers)
        # Model-level padding once per forward (vs per-GEMM internal padding).
        # Safe when attention is causal AND batch_size==1. For the KV-cache
        # path, we extend slot_mapping with -1 for padded rows so
        # store_kvcache skips them (upstream's triton kernel treats slot=-1
        # as a no-op). Non-causal or batched paths still fall through to
        # per-GEMM padding because their attention can't tolerate junk rows.
        slot_padded = slot_mapping
        if batch_size == 1 and causal_any:
            hs, pad = _pad_M_to(input_embeds.contiguous(), 64)
        else:
            hs, pad = input_embeds.contiguous(), 0
        pos = positions
        if pad > 0:
            # Fill padded position slots with the last real position.
            pos = torch.cat([positions, positions[-1:].expand(pad).contiguous()])
            pos = pos.contiguous()
            if slot_mapping is not None:
                # Extend slot_mapping with -1 sentinels for padded rows.
                slot_padded = torch.cat([
                    slot_mapping,
                    torch.full((pad,), -1, dtype=slot_mapping.dtype,
                               device=slot_mapping.device),
                ]).contiguous()

        prefetch = os.environ.get("VOXCPM_PREFETCH") == "l2"
        side = None
        if prefetch:
            side = torch.cuda.Stream()
            evt = torch.cuda.Event()

        # L2 persistence hint: mark our model's weights as preferred-to-keep
        # in L2 so cross-iteration reuse (e.g. 9 Euler DiT calls) doesn't
        # keep paying HBM for the same weights. Only active when:
        #   (a) the model has an `_l2_persist_range` attribute (pointer + bytes)
        #   (b) VOXCPM_L2_PERSIST env is not "0" (runtime disable).
        persist = getattr(self, "_l2_persist_range", None)
        if persist is not None and os.environ.get("VOXCPM_L2_PERSIST", "1") != "0":
            base_ptr, nbytes = persist
            _ext.set_l2_persist_window(base_ptr, nbytes, 1.0)
        else:
            persist = None  # skip the matching clear below

        for i, layer in enumerate(self.layers):
            if prefetch and i + 1 < len(self.layers):
                nxt = self.layers[i + 1]
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    _ext.l2_warm(nxt.w_gu)
            if kv_caches is not None:
                kc, vc = kv_caches[i]
                hs = layer.forward(hs, pos, batch_size=batch_size,
                                   k_cache=kc, v_cache=vc,
                                   slot_mapping=slot_padded)
            else:
                hs = layer.forward(hs, pos, batch_size=batch_size)

        if prefetch:
            torch.cuda.current_stream().wait_stream(side)

        if persist is not None:
            _ext.clear_l2_persist_window()

        out = _ext.rmsnorm(hs, self.w_final_norm, self.rms_eps)
        return out[:orig_M] if pad > 0 else out
