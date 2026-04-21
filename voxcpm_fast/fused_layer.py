"""Python wrapper for the P2.2 fused non-causal transformer layer megakernel.

The layer is packaged as a persistent CUDA kernel: one host launch at startup
brings up the dispatcher + worker grid; every forward call pushes a work item
through the same pinned doorbell / HBM queue pattern as P2.1 and waits on a
pinned done flag.

Public API
----------
::

    layer = FusedNonCausalLayer(cfg, weights_dict)
    layer.start()
    out = layer.forward(hidden_states, positions)   # [N, 1024] bf16
    layer.stop()

``weights_dict`` is the state_dict of the upstream ``Cpm4DecoderLayer``:

    input_layernorm.weight          [1024]       bf16
    self_attn.qkv_proj.weight       [2560, 1024] bf16  (row-major)
    self_attn.o_proj.weight         [1024, 2048] bf16
    post_attention_layernorm.weight [1024]       bf16
    mlp.gate_up_proj.weight         [8192, 1024] bf16  (gate || up)
    mlp.down_proj.weight            [1024, 4096] bf16

All weight tensors must already live on the same CUDA device as the forward
inputs. The wrapper does a tiny ``cos_cached`` + ``sin_cached`` precompute on
construction using the encoder's MiniCPMLongRoPE settings so that the device
kernel can do a simple gather instead of recomputing frequencies.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fused_layer_noncausal_ext as _ext  # type: ignore  # noqa: E402


HIDDEN = 1024
NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
Q_DIM = NUM_HEADS * HEAD_DIM       # 2048
KV_DIM = NUM_KV_HEADS * HEAD_DIM   # 256
QKV_DIM = Q_DIM + 2 * KV_DIM       # 2560
INTERMEDIATE = 4096


class _DoneFlag:
    __slots__ = ("_addr", "_expected")

    def __init__(self, addr: int, expected: int):
        self._addr = addr
        self._expected = expected & 0xFFFFFFFF

    def wait(self, timeout: float = 5.0) -> None:
        deadline = time.perf_counter() + timeout
        while ctypes.c_uint32.from_address(self._addr).value != self._expected:
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"done flag at 0x{self._addr:x} never hit expected {self._expected}"
                )


def _build_rope_caches(head_dim: int, max_pos: int, base: float,
                       short_factor: list[float], long_factor: list[float],
                       original_max_pos: int,
                       dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Matches nanovllm's MiniCPMLongRoPE._set_cos_sin_cache exactly."""
    device = torch.device("cuda:0")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                            dtype=torch.float32) / head_dim))
    scale = max_pos / original_max_pos
    scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original_max_pos))
    t = torch.arange(max_pos, device=device, dtype=inv_freq.dtype)
    ext_factors = (
        torch.tensor(long_factor, dtype=torch.float32, device=device)
        if max_pos > original_max_pos
        else torch.tensor(short_factor, dtype=torch.float32, device=device)
    )
    freqs = torch.outer(t, 1.0 / ext_factors) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * scaling_factor).to(dtype)
    sin = (emb.sin() * scaling_factor).to(dtype)
    return cos, sin


class FusedNonCausalLayer:
    """Persistent-kernel wrapper for one non-causal Cpm4DecoderLayer."""

    def __init__(self, cfg, weights: dict[str, torch.Tensor]):
        self.cfg = cfg
        # Validate shapes.
        self._w_in_ln = weights["input_layernorm.weight"].contiguous()
        self._w_qkv = weights["self_attn.qkv_proj.weight"].contiguous()
        self._w_o = weights["self_attn.o_proj.weight"].contiguous()
        self._w_post_ln = weights["post_attention_layernorm.weight"].contiguous()
        self._w_gu = weights["mlp.gate_up_proj.weight"].contiguous()
        self._w_dn = weights["mlp.down_proj.weight"].contiguous()
        assert self._w_in_ln.shape == (HIDDEN,)
        assert self._w_qkv.shape == (QKV_DIM, HIDDEN)
        assert self._w_o.shape == (HIDDEN, Q_DIM)
        assert self._w_post_ln.shape == (HIDDEN,)
        assert self._w_gu.shape == (2 * INTERMEDIATE, HIDDEN)
        assert self._w_dn.shape == (HIDDEN, INTERMEDIATE)

        # Build RoPE caches.
        rs = cfg.rope_scaling
        cos, sin = _build_rope_caches(
            head_dim=HEAD_DIM,
            max_pos=cfg.max_position_embeddings,
            base=cfg.rope_theta,
            short_factor=list(rs.short_factor),
            long_factor=list(rs.long_factor),
            original_max_pos=rs.original_max_position_embeddings,
            dtype=torch.float32,
        )
        # We only ever look up positions 0..N-1 with N=100 in the test, but the
        # cache contains `max_position_embeddings` rows. Keep as fp32 so the
        # kernel can do the multiply in fp32 before casting.
        self._cos = cos.contiguous()
        self._sin = sin.contiguous()

        self.rms_eps = float(cfg.rms_norm_eps)
        self._started = False
        self._handle = None
        self._layer_id = 0  # multi-layer support later

    # ---------- lifecycle ----------

    def start(self, num_sms: int = 8) -> None:
        if self._started:
            return
        info = _ext.queue_info()
        (self._doorbell_slots_off,
         self._doorbell_tail_off,
         self._doorbell_head_off,
         self._slot_stride,
         self._done_flag_off,
         self._done_stride,
         self._ctrl_terminate_off,
         self._queue_capacity,
         self._ring_capacity) = info

        ptrs = _ext.launch_persistent(num_sms).tolist()
        self._doorbell = int(ptrs[0])
        self._ctrl = int(ptrs[1])
        self._done = int(ptrs[2])
        self._q_dev = int(ptrs[3])
        self._stream = int(ptrs[4])

        self._doorbell_slots_addr = self._doorbell + self._doorbell_slots_off
        self._doorbell_tail_addr = self._doorbell + self._doorbell_tail_off
        self._next_tail = 1
        self._started = True

    def stop(self, timeout: float = 0.1) -> None:
        if not self._started:
            return
        terminate_addr = self._ctrl + self._ctrl_terminate_off
        ctypes.c_uint32.from_address(terminate_addr).value = 1

        cuda = _libcudart()
        stream = ctypes.c_void_p(self._stream)
        deadline = time.perf_counter() + timeout
        while True:
            rc = cuda.cudaStreamQuery(stream)
            if rc == 0:
                break
            if time.perf_counter() > deadline:
                _ext.shutdown_persistent()
                self._started = False
                raise TimeoutError(f"persistent kernel did not exit in {timeout*1000:.1f} ms")
            time.sleep(0.0005)

        _ext.shutdown_persistent()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.stop()
        except TimeoutError:
            pass

    # ---------- forward ----------

    def current_stage(self) -> int:
        """Debug helper: returns the kernel's last-reached stage marker."""
        if not self._started:
            return -1
        # ControlBlock.stage at offset 4.
        return ctypes.c_uint32.from_address(self._ctrl + 4).value

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not self._started:
            raise RuntimeError("FusedNonCausalLayer.forward() before start()")
        assert hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
        assert hidden_states.shape[-1] == HIDDEN
        # Accept both [N, H] and [1, N, H].
        if hidden_states.dim() == 3:
            assert hidden_states.shape[0] == 1
            hs = hidden_states.squeeze(0).contiguous()
            squeeze_out = True
        else:
            hs = hidden_states.contiguous()
            squeeze_out = False
        N = hs.shape[0]
        assert N <= 256, "For P2.2 we support up to 256 tokens"
        pos = positions.to(dtype=torch.int32, device=hs.device).contiguous()
        assert pos.numel() == N

        out = torch.empty_like(hs)

        # Submit one work item and wait.
        tail = self._next_tail
        self._next_tail = (self._next_tail + 1) & 0xFFFFFFFF
        idx = tail - 1
        ring_pos = idx & (self._ring_capacity - 1)
        slot_pos = idx & (self._queue_capacity - 1)

        flag_addr = self._done + slot_pos * self._done_stride + self._done_flag_off
        ctypes.c_uint32.from_address(flag_addr).value = 0

        # Pack WorkItem: see FusedLayerWorkItem in fused_layer_noncausal.cu.
        # Fields: (u64 hs_ptr, u64 out_ptr, u64 pos_ptr,
        #          u64 w_in_ln, u64 w_qkv, u64 w_o, u64 w_post_ln, u64 w_gu, u64 w_dn,
        #          u64 cos_ptr, u64 sin_ptr,
        #          u32 seq_len, u32 layer_id, f32 rms_eps, u32 idx, u32 pad)
        slot_addr = self._doorbell_slots_addr + ring_pos * self._slot_stride
        _ext.pack_workitem(
            int(slot_addr),
            int(hs.data_ptr()),
            int(out.data_ptr()),
            int(pos.data_ptr()),
            int(self._w_in_ln.data_ptr()),
            int(self._w_qkv.data_ptr()),
            int(self._w_o.data_ptr()),
            int(self._w_post_ln.data_ptr()),
            int(self._w_gu.data_ptr()),
            int(self._w_dn.data_ptr()),
            int(self._cos.data_ptr()),
            int(self._sin.data_ptr()),
            int(N),
            int(self._layer_id),
            float(self.rms_eps),
            int(idx & 0x7FFFFFFF),
        )
        _ext.store_release_seq(self._doorbell_tail_addr, tail)

        # Wait on the done flag.
        try:
            _DoneFlag(flag_addr, tail).wait(timeout=3.0)
        except TimeoutError:
            stage = self.current_stage()
            print(f"[fused_layer] done flag never fired; kernel stuck at stage={stage}",
                  flush=True)
            raise TimeoutError(
                f"done flag never fired; kernel stuck at stage={stage}"
            )

        if squeeze_out:
            return out.unsqueeze(0)
        return out


# ---------- helpers --------------------------------------------------------

_libcudart_handle = None


def _libcudart():
    global _libcudart_handle
    if _libcudart_handle is None:
        for name in ("libcudart.so.12", "libcudart.so", "libcudart.so.12.8"):
            try:
                _libcudart_handle = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if _libcudart_handle is None:
            raise RuntimeError("libcudart.so not found")
        _libcudart_handle.cudaStreamQuery.argtypes = [ctypes.c_void_p]
        _libcudart_handle.cudaStreamQuery.restype = ctypes.c_int
    return _libcudart_handle
