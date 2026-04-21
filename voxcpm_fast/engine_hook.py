"""Monkey-patch `VoxCPM2Runner.init_model` to swap in our fast transformer
stacks AFTER upstream's weight load but BEFORE graph capture.

Scope (what we swap):
- `model.feat_encoder.encoder`   — 12-layer non-causal Cpm4Model (cacheless)
- `model.feat_decoder.estimator.decoder` — 12-layer non-causal Cpm4Model (cacheless)

Scope (what we leave upstream):
- `model.base_lm`      — uses KV cache during decode; our FusedCpm4Model has
                         no KV-write, so swapping would break the decode path
- `model.residual_lm`  — same reason

These two are ~18 % of eager decode time per topology.md; the two we swap
are ~81 % (73.5 % DiT + 7.6 % feat_encoder). At c=1 this should land us
close to the 30 ms TTFPA projection.

Usage (call once before `SyncVoxCPM2ServerPool(...)`):

    from voxcpm_fast.engine_hook import install_fast_path
    install_fast_path()

Single-GPU only. At tensor_parallel_size > 1 the extra ranks are spawned in
new processes (module-level monkey-patches don't transfer) — adjust the
entry point then.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# When imported in a spawned subprocess, the voxcpm_fast parent may not be
# on sys.path yet. Add it so `voxcpm_fast.*` is importable from anywhere.
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))


_installed = False
_original_init_model: Callable | None = None


def install_fast_path(
    enable_feat_encoder: bool = True,
    enable_dit: bool = True,
    enable_base_lm: bool = True,
    enable_residual_lm: bool = True,
    verbose: bool = True,
) -> None:
    """Patch `VoxCPM2Runner.init_model` to install fast shims post-load."""
    global _installed, _original_init_model
    if _installed:
        return

    from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Runner

    _original_init_model = VoxCPM2Runner.init_model

    def init_model_with_shims(self, model_config, model_path):
        _original_init_model(self, model_config, model_path)
        _apply_shims(
            self.model,
            model_config,
            enable_feat_encoder=enable_feat_encoder,
            enable_dit=enable_dit,
            enable_base_lm=enable_base_lm,
            enable_residual_lm=enable_residual_lm,
            verbose=verbose,
        )

    VoxCPM2Runner.init_model = init_model_with_shims
    _installed = True
    if verbose:
        print(f"[voxcpm_fast] patched VoxCPM2Runner.init_model "
              f"(feat_encoder={enable_feat_encoder}, dit={enable_dit})", flush=True)


def _apply_shims(model, model_config, *, enable_feat_encoder, enable_dit,
                 enable_base_lm, enable_residual_lm, verbose):
    """Replace transformer stacks with FusedCpm4ModelShim instances."""
    from fused_layer_chained import FusedCpm4Model
    from voxcpm_fast.benchmarks.bench_voxcpm2_forward import (  # reuse helper
        FusedCausalLMShim,
        FusedCpm4ModelShim,
        _build_fused_cpm4,
    )

    base_lm_cfg = model_config.lm_config
    rms_eps = base_lm_cfg.rms_norm_eps

    if enable_base_lm:
        bl = model.base_lm
        shim = FusedCausalLMShim(
            _build_fused_cpm4(
                bl,
                hidden=base_lm_cfg.hidden_size,
                intermediate=base_lm_cfg.intermediate_size,
                num_layers=base_lm_cfg.num_hidden_layers,
                causal=True,
                use_rope=True,
                rms_eps=rms_eps,
            ),
            hidden=base_lm_cfg.hidden_size,
            upstream_cpm4=bl,
        ).cuda().eval()
        model.base_lm = shim
        if verbose:
            print(f"[voxcpm_fast] swapped base_lm "
                  f"({base_lm_cfg.num_hidden_layers}L hidden={base_lm_cfg.hidden_size})", flush=True)

    if enable_residual_lm:
        rl = model.residual_lm
        shim = FusedCausalLMShim(
            _build_fused_cpm4(
                rl,
                hidden=base_lm_cfg.hidden_size,
                intermediate=base_lm_cfg.intermediate_size,
                num_layers=model_config.residual_lm_num_layers,
                causal=True,
                use_rope=not model_config.residual_lm_no_rope,
                rms_eps=rms_eps,
            ),
            hidden=base_lm_cfg.hidden_size,
            upstream_cpm4=rl,
        ).cuda().eval()
        model.residual_lm = shim
        if verbose:
            print(f"[voxcpm_fast] swapped residual_lm "
                  f"({model_config.residual_lm_num_layers}L hidden={base_lm_cfg.hidden_size})", flush=True)

    if enable_feat_encoder:
        enc_cfg = model_config.encoder_config
        enc = model.feat_encoder.encoder
        shim = FusedCpm4ModelShim(
            _build_fused_cpm4(
                enc,
                hidden=enc_cfg.hidden_dim,
                intermediate=enc_cfg.ffn_dim,
                num_layers=enc_cfg.num_layers,
                causal=False,
                use_rope=True,
                rms_eps=rms_eps,
            ),
            hidden=enc_cfg.hidden_dim,
        ).cuda().eval()
        model.feat_encoder.encoder = shim
        if verbose:
            print(f"[voxcpm_fast] swapped feat_encoder.encoder "
                  f"({enc_cfg.num_layers}L hidden={enc_cfg.hidden_dim})", flush=True)

    if enable_dit:
        dit_cfg = model_config.dit_config
        dit = model.feat_decoder.estimator.decoder
        shim = FusedCpm4ModelShim(
            _build_fused_cpm4(
                dit,
                hidden=dit_cfg.hidden_dim,
                intermediate=dit_cfg.ffn_dim,
                num_layers=dit_cfg.num_layers,
                causal=False,
                use_rope=True,
                rms_eps=rms_eps,
            ),
            hidden=dit_cfg.hidden_dim,
        ).cuda().eval()
        model.feat_decoder.estimator.decoder = shim
        if verbose:
            print(f"[voxcpm_fast] swapped feat_decoder.estimator.decoder "
                  f"({dit_cfg.num_layers}L hidden={dit_cfg.hidden_dim})", flush=True)


def install_prefill_graph_capture(n_buckets=(16, 32, 64, 128, 256, 512)) -> None:
    """Monkey-patch VoxCPM2Model.forward to graph-capture per prompt-length
    bucket. During real prefill, copies inputs into a bucket's stable buffers
    and replays the graph — matching our synthetic 23 ms bench measurement.
    Must be called AFTER install_fast_path (i.e. AFTER model is built).

    This runs inside the child process via fast_main_loop.
    """
    import torch
    from nanovllm_voxcpm.models.voxcpm2.model import VoxCPM2Model
    _orig_forward = VoxCPM2Model.forward

    def forward_graphed(self, positions, text_tokens, feat, feat_mask,
                        temperature, cfg_value):
        # Eager for decode and for engine's own graph-capture warmup.
        # Only engage our prefill graph capture for real prefill calls.
        from nanovllm_voxcpm.utils.context import get_context
        ctx = get_context()
        N = positions.size(0)
        can_graph = (
            ctx.is_prefill
            and N > 1
            and positions.is_cuda
            and not torch.is_grad_enabled()
            and not torch.cuda.is_current_stream_capturing()
        )
        if not can_graph:
            return _orig_forward(self, positions, text_tokens, feat,
                                 feat_mask, temperature, cfg_value)

        # Pick smallest bucket N' >= N.
        bucket = None
        for b in n_buckets:
            if N <= b:
                bucket = b
                break
        if bucket is None:
            # N too large — fall back to eager.
            return _orig_forward(self, positions, text_tokens, feat,
                                 feat_mask, temperature, cfg_value)

        cache = getattr(self, "_prefill_graphs", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_prefill_graphs", cache)

        from nanovllm_voxcpm.utils.context import set_context, reset_context, get_context

        if bucket not in cache:
            # Capture.
            import time as _t
            t0 = _t.time()
            pb = torch.zeros(bucket, dtype=positions.dtype, device="cuda")
            tb = torch.zeros(bucket, dtype=text_tokens.dtype, device="cuda")
            fb = torch.zeros(bucket, *feat.shape[1:], dtype=feat.dtype, device="cuda")
            mb = torch.zeros(bucket, dtype=feat_mask.dtype, device="cuda")
            temp_b = temperature.clone()
            cfg_b = cfg_value.clone()

            # Seed real content for warmup so compute is realistic.
            pb[:N].copy_(positions)
            tb[:N].copy_(text_tokens)
            fb[:N].copy_(feat)
            mb[:N].copy_(feat_mask)
            # Positions arange for padded slots so RoPE is well-defined.
            for j in range(N, bucket):
                pb[j] = j

            # Persistent context tensors — their addresses are baked into the
            # captured graph. We write real values into them before each replay.
            cu_q = torch.tensor([0, bucket], dtype=torch.int32, device="cuda")
            cu_k = torch.tensor([0, bucket], dtype=torch.int32, device="cuda")
            slot = torch.full((bucket,), -1, dtype=torch.int32, device="cuda")
            # block_tables and context_lens are prefill-irrelevant but the
            # graph-captured attention call passes them through. Set None.

            # Warmup (default stream).
            set_context(True, cu_q, cu_k, bucket, bucket, slot, None, None)
            try:
                for _ in range(3):
                    _ = _orig_forward(self, pb, tb, fb, mb, temp_b, cfg_b)
            finally:
                reset_context()
            torch.cuda.synchronize()

            # Side-stream warmup per PyTorch recipe.
            cs = torch.cuda.Stream()
            cs.wait_stream(torch.cuda.current_stream())
            set_context(True, cu_q, cu_k, bucket, bucket, slot, None, None)
            try:
                with torch.cuda.stream(cs):
                    for _ in range(3):
                        _ = _orig_forward(self, pb, tb, fb, mb, temp_b, cfg_b)
            finally:
                reset_context()
            torch.cuda.current_stream().wait_stream(cs)
            torch.cuda.synchronize()

            # Capture with cu_q = [0, bucket]. At replay we'll overwrite to
            # [0, N_real]. Flash-attn varlen reads cu_seqlens dynamically, and
            # `last_indices = cu_seqlens_q[1:] - 1` is a baked tensor subtraction
            # that rereads cu_seqlens_q at replay, so lm_hidden indexing picks
            # enc_outputs[N_real - 1] correctly.
            g = torch.cuda.CUDAGraph()
            set_context(True, cu_q, cu_k, bucket, bucket, slot, None, None)
            try:
                with torch.cuda.graph(g):
                    out = _orig_forward(self, pb, tb, fb, mb, temp_b, cfg_b)
            finally:
                reset_context()
            cache[bucket] = (g, pb, tb, fb, mb, temp_b, cfg_b, out, cu_q, cu_k, slot)
            print(f"[voxcpm_fast] captured VoxCPM2Model.forward @ N={bucket} "
                  f"in {(_t.time()-t0)*1000:.0f} ms", flush=True)

        g, pb, tb, fb, mb, temp_b, cfg_b, out, cu_q, cu_k, slot = cache[bucket]

        # ---- Write real request data + context into captured tensors ----
        real_ctx = get_context()

        # 1. Inputs: first N positions real, rest zeros (attn won't see them
        # because cu_seqlens tells flash_attn the real length).
        pb.zero_()
        pb[:N].copy_(positions)
        tb.zero_()
        tb[:N].copy_(text_tokens)
        fb.zero_()
        fb[:N].copy_(feat)
        mb.zero_()
        mb[:N].copy_(feat_mask)
        temp_b.copy_(temperature)
        cfg_b.copy_(cfg_value)

        # 2. Attention context: cu_seqlens = [0, N], slot_mapping filled with
        # engine's real slots in first N, -1 in rest (store_kvcache skips -1).
        cu_q[0] = 0
        cu_q[1] = N
        cu_k[0] = 0
        cu_k[1] = N
        slot.fill_(-1)
        if real_ctx.slot_mapping is not None and real_ctx.slot_mapping.numel() > 0:
            slot[:N].copy_(real_ctx.slot_mapping[:N])

        g.replay()

        # 3. Outputs: captured `out` has batch dim = number of requests.
        # At c=1 prefill it's shape (1, ...) for both latents and stop_flag.
        # No slicing needed; clone so caller owns a fresh tensor.
        if isinstance(out, dict):
            return {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in out.items()}
        return out.clone()

    VoxCPM2Model.forward = forward_graphed
    print("[voxcpm_fast] installed prefill graph capture for VoxCPM2Model.forward", flush=True)

    # --- Pre-warm all buckets at first-call time (lazy via an init hook) ---
    def _prewarm(self, feat_dim, patch_size, vocab_size):
        import time as _t
        if not hasattr(self, "_prefill_graphs"):
            object.__setattr__(self, "_prefill_graphs", {})
        for bucket in n_buckets:
            if bucket in self._prefill_graphs:
                continue
            _t0 = _t.time()
            dummy_pos = torch.arange(bucket, device="cuda", dtype=torch.int64)
            dummy_tokens = torch.zeros(bucket, device="cuda", dtype=torch.int64)
            dummy_feat = torch.zeros(bucket, patch_size, feat_dim,
                                     device="cuda", dtype=torch.bfloat16)
            dummy_mask = torch.zeros(bucket, device="cuda", dtype=torch.bool)
            dummy_temp = torch.tensor([0.7], device="cuda", dtype=torch.bfloat16)
            dummy_cfg = torch.tensor([1.5], device="cuda", dtype=torch.bfloat16)
            # Trigger capture path — must look like a real prefill call.
            from nanovllm_voxcpm.utils.context import set_context, reset_context
            cu_q = torch.tensor([0, bucket], dtype=torch.int32, device="cuda")
            cu_k = torch.tensor([0, bucket], dtype=torch.int32, device="cuda")
            slot = torch.full((bucket,), -1, dtype=torch.int32, device="cuda")
            set_context(True, cu_q, cu_k, bucket, bucket, slot, None, None)
            try:
                with torch.inference_mode():
                    _ = self(dummy_pos, dummy_tokens, dummy_feat, dummy_mask,
                            dummy_temp, dummy_cfg)
            finally:
                reset_context()
            print(f"[voxcpm_fast] pre-warmed bucket N={bucket} "
                  f"in {(_t.time()-_t0)*1000:.0f} ms  (total {len(self._prefill_graphs)} cached)", flush=True)

    VoxCPM2Model.prewarm_prefill_buckets = _prewarm

    # Patch `VoxCPM2Runner.__init__` so prewarm runs after the engine's own
    # `capture_cudagraph` and `allocate_kv_cache` — when the environment is
    # fully set up for a real forward.
    from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Runner
    _orig_runner_init = VoxCPM2Runner.__init__

    def runner_init(self, config, rank, device_idx, distributed_port, event):
        _orig_runner_init(self, config, rank, device_idx, distributed_port, event)
        # At this point: model built, weights loaded, shims installed,
        # kv_cache allocated, decode graphs captured. Prewarm our prefill
        # graphs now so the first real request never pays capture cost.
        try:
            self.model.prewarm_prefill_buckets(
                self.feat_dim, self.patch_size, config.model_config.lm_config.vocab_size
            )
        except Exception as e:
            print(f"[voxcpm_fast] prewarm skipped: {type(e).__name__}: {e}", flush=True)

    VoxCPM2Runner.__init__ = runner_init


def install_graphed_phase_probe() -> None:
    """Rewrite VoxCPM2Model.forward with inline CUDA event records (no sync)
    so they get captured into the outer prefill graph. Events are shared
    across calls. After each replay, the probe syncs + prints elapsed times
    between consecutive events for the JUST-replayed graph.

    Enable with VOXCPM_GRAPHED_PROBE=1. Must be installed BEFORE
    install_prefill_graph_capture.
    """
    import os as _os
    if _os.environ.get("VOXCPM_GRAPHED_PROBE") != "1":
        return
    import torch
    from nanovllm_voxcpm.models.voxcpm2.model import VoxCPM2Model

    labels = ["start", "feat_enc", "embed", "base_lm", "fsq+fuse",
              "res_lm", "dit", "stop"]
    # One event per boundary; reused across replays. Graph capture will
    # bake the record() calls into the captured stream.
    evs = [torch.cuda.Event(enable_timing=True) for _ in labels]
    state = {"ready": False}

    # Keep reference to the original so we can fall back during engine's
    # decode-graph warmup (cu_seqlens_q is None there).
    _orig_for_probe = VoxCPM2Model.forward

    def forward_probed(self, positions, text_tokens, feat, feat_mask,
                       temperature, cfg_value):
        from nanovllm_voxcpm.utils.context import get_context
        ctx = get_context()
        if ctx.cu_seqlens_q is None or not ctx.is_prefill:
            return _orig_for_probe(self, positions, text_tokens, feat,
                                   feat_mask, temperature, cfg_value)

        evs[0].record()
        feat_embeds = self.enc_to_lm_proj(self.feat_encoder(feat))
        feat_embeds = torch.masked_fill(feat_embeds, feat_mask.unsqueeze(-1).logical_not(), 0)
        evs[1].record()

        text_embeds = self.base_lm.embed_tokens(text_tokens)
        combined_embeds = torch.where(feat_mask.unsqueeze(-1), feat_embeds, text_embeds)
        evs[2].record()

        enc_outputs = self.base_lm(combined_embeds, positions)
        enc_outputs = torch.where(feat_mask.unsqueeze(-1), self.fsq_layer(enc_outputs), enc_outputs)
        evs[3].record()

        last_indices = ctx.cu_seqlens_q[1:] - 1
        lm_hidden = enc_outputs[last_indices].contiguous()
        residual_inputs = self.fusion_concat_proj(
            torch.cat([enc_outputs, torch.where(feat_mask.unsqueeze(-1), feat_embeds, 0)], dim=-1)
        )
        evs[4].record()

        ralm_outputs = self.residual_lm(residual_inputs, positions)
        ralm_hidden = ralm_outputs[last_indices].contiguous()
        prefix_feat_cond = feat[last_indices].contiguous()
        evs[5].record()

        dit_hidden = torch.cat([self.lm_to_dit_proj(lm_hidden), self.res_to_dit_proj(ralm_hidden)], dim=-1)
        pred_feat = self.feat_decoder(
            mu=dit_hidden,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            temperature=temperature,
            cfg_value=cfg_value,
        ).transpose(1, 2)
        evs[6].record()

        stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)
        evs[7].record()
        state["ready"] = True
        return {"latents": pred_feat, "stop_flag": stop_flag}

    VoxCPM2Model.forward = forward_probed

    # After outer prefill graph replay, sync + dump event deltas.
    from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Runner
    _orig_run = VoxCPM2Runner.run
    counter = {"n": 0}

    def run_with_dump(self, seqs, is_prefill):
        out = _orig_run(self, seqs, is_prefill)
        if is_prefill and state["ready"]:
            torch.cuda.synchronize()
            counter["n"] += 1
            try:
                ms = lambda i: evs[i].elapsed_time(evs[i + 1])
                parts = [f"{labels[i+1]}={ms(i):5.2f}" for i in range(len(labels) - 1)]
                total = evs[0].elapsed_time(evs[-1])
                print(f"[graphed probe #{counter['n']:03d}] total={total:5.2f}  " + "  ".join(parts), flush=True)
            except Exception as e:
                # Captured events + graph replay are fragile; swallow and move on.
                pass
        return out

    VoxCPM2Runner.run = run_with_dump
    print("[voxcpm_fast] graphed-phase probe installed", flush=True)


def install_model_forward_probe() -> None:
    """Split the graphed-replay model.forward into fine-grained phases so we
    can see where the 22 ms goes (base_lm vs DiT vs projections vs VAE).

    Wraps each major sub-module with a CUDA event pair. Only runs when
    `VOXCPM_FORWARD_TIMING=1`.
    """
    import os as _os
    if _os.environ.get("VOXCPM_FORWARD_TIMING") != "1":
        return
    import torch
    from nanovllm_voxcpm.models.voxcpm2.model import VoxCPM2Model
    _orig = VoxCPM2Model.forward

    state = {"count": 0}

    def forward_probed(self, positions, text_tokens, feat, feat_mask,
                       temperature, cfg_value):
        from nanovllm_voxcpm.utils.context import get_context
        ctx = get_context()
        if not ctx.is_prefill or torch.cuda.is_current_stream_capturing():
            return _orig(self, positions, text_tokens, feat, feat_mask,
                         temperature, cfg_value)

        torch.cuda.synchronize()
        labels = ["start", "feat_enc", "embed", "base_lm", "fsq+fuse",
                  "res_lm", "dit", "stop", "end"]
        evs = [torch.cuda.Event(enable_timing=True) for _ in labels]

        evs[0].record()
        feat_embeds = self.enc_to_lm_proj(self.feat_encoder(feat))
        feat_embeds = torch.masked_fill(feat_embeds, feat_mask.unsqueeze(-1).logical_not(), 0)
        evs[1].record()

        text_embeds = self.base_lm.embed_tokens(text_tokens)
        combined_embeds = torch.where(feat_mask.unsqueeze(-1), feat_embeds, text_embeds)
        evs[2].record()

        enc_outputs = self.base_lm(combined_embeds, positions)
        enc_outputs = torch.where(feat_mask.unsqueeze(-1), self.fsq_layer(enc_outputs), enc_outputs)
        evs[3].record()

        last_indices = ctx.cu_seqlens_q[1:] - 1
        lm_hidden = enc_outputs[last_indices].contiguous()
        residual_inputs = self.fusion_concat_proj(
            torch.cat([enc_outputs, torch.where(feat_mask.unsqueeze(-1), feat_embeds, 0)], dim=-1)
        )
        evs[4].record()

        ralm_outputs = self.residual_lm(residual_inputs, positions)
        ralm_hidden = ralm_outputs[last_indices].contiguous()
        prefix_feat_cond = feat[last_indices].contiguous()
        evs[5].record()

        dit_hidden = torch.cat([self.lm_to_dit_proj(lm_hidden), self.res_to_dit_proj(ralm_hidden)], dim=-1)
        pred_feat = self.feat_decoder(
            mu=dit_hidden,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            temperature=temperature,
            cfg_value=cfg_value,
        ).transpose(1, 2)
        evs[6].record()

        stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)
        evs[7].record()
        out = {"latents": pred_feat, "stop_flag": stop_flag}
        evs[8].record()

        torch.cuda.synchronize()
        state["count"] += 1
        parts = [f"{labels[i+1]}={evs[i].elapsed_time(evs[i+1]):5.2f}"
                 for i in range(len(labels) - 1)]
        print(f"[forward #{state['count']:02d}] " + "  ".join(parts), flush=True)
        return out

    VoxCPM2Model.forward = forward_probed
    print("[voxcpm_fast] forward probe installed", flush=True)


def install_timing_probe(log_every: int = 1) -> None:
    """Patch VoxCPM2Runner.run to log per-phase GPU wall time on every prefill.

    Prints a CSV line per prefill call: N,total,prep,forward,vae_decode,vae_cpu,stop_cpu,post
    Intended for one-off investigation — keep off for production.
    """
    import time as _time
    import torch
    from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Runner

    _orig_run = VoxCPM2Runner.run
    state = {"count": 0}

    def run_timed(self, seqs, is_prefill):
        if not is_prefill:
            return _orig_run(self, seqs, is_prefill)

        # Reimplement runner.run with torch.cuda.Event markers between phases.
        import numpy as np

        # -- start
        torch.cuda.synchronize()
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        evs[0].record()

        positions = self.prepare_prefill_context(seqs)
        inputs = {"positions": positions}
        text_tokens, feats, feat_masks, temperatures, cfg_values = [], [], [], [], []
        for seq in seqs:
            p = seq.custom_payload
            text_tokens.append(p.text_tokens)
            feats.append(p.feats)
            feat_masks.append(p.feat_masks)
            temperatures.append(p.temperature)
            cfg_values.append(p.cfg_value)
        inputs["text_tokens"] = torch.from_numpy(np.concatenate(text_tokens, axis=0)).cuda(non_blocking=True)
        inputs["feat"] = torch.from_numpy(np.concatenate(feats, axis=0)).cuda(non_blocking=True).to(self.dtype)
        inputs["feat_mask"] = torch.from_numpy(np.concatenate(feat_masks, axis=0)).cuda(non_blocking=True)
        inputs["temperature"] = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        inputs["cfg_value"] = torch.tensor(cfg_values, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        evs[1].record()  # after prep + H2D

        outputs = self.run_model(inputs, is_prefill)
        evs[2].record()  # after model.forward
        latents = outputs["latents"]

        pad_lengths = [
            s.custom_payload.padding_decode.shape[0] if s.custom_payload.padding_decode is not None else 0
            for s in seqs
        ]
        max_pad_decode = max(pad_lengths) + self.patch_size
        vae_decoder_inputs = torch.zeros(len(seqs), max_pad_decode, self.feat_dim, dtype=torch.float32, device="cuda")
        for i, s in enumerate(seqs):
            pad_len = pad_lengths[i]
            if pad_len > 0:
                vae_decoder_inputs[i, :pad_len] = torch.from_numpy(s.custom_payload.padding_decode).cuda(non_blocking=True)
            vae_decoder_inputs[i, pad_len : pad_len + self.patch_size] = latents[i].to(torch.float32)
        evs[3].record()  # after VAE input prep

        vae_decoder_outputs = self.vae.decode(vae_decoder_inputs.permute(0, 2, 1))[:, 0, :]
        evs[4].record()  # after VAE.decode GPU work
        vae_np = vae_decoder_outputs.cpu().numpy()
        evs[5].record()  # after VAE cpu sync

        stop_flag = outputs["stop_flag"].cpu().tolist()
        evs[6].record()  # after stop cpu sync

        ret_waveforms = []
        for i, pad_len in enumerate(pad_lengths):
            ret_waveforms.append(
                vae_np[i, pad_len * self.vae.decoder_chunk_size : (pad_len + self.patch_size) * self.vae.decoder_chunk_size]
            )
        np_latents = latents.to(torch.float32).cpu().numpy()
        evs[7].record()  # after post

        torch.cuda.synchronize()

        N = positions.size(0)
        ms = lambda a, b: evs[a].elapsed_time(evs[b])
        total = ms(0, 7)
        prep = ms(0, 1)
        forward = ms(1, 2)
        vae_prep = ms(2, 3)
        vae_fwd = ms(3, 4)
        vae_cpu = ms(4, 5)
        stop_cpu = ms(5, 6)
        post = ms(6, 7)
        state["count"] += 1
        if state["count"] % log_every == 0:
            # Collect shim replay/eager counts (if shims present)
            shim_stats = []
            for name in ("feat_encoder", "feat_decoder"):
                try:
                    if name == "feat_encoder":
                        shim = self.model.feat_encoder.encoder
                    else:
                        shim = self.model.feat_decoder.estimator.decoder
                    if hasattr(shim, "_replay_count"):
                        shim_stats.append(f"{name}:replay={shim._replay_count},eager={shim._eager_count},new={shim._new_capture_count}")
                        shim._replay_count = 0
                        shim._eager_count = 0
                        shim._new_capture_count = 0
                except AttributeError:
                    pass
            print(f"[timing prefill #{state['count']:03d}] N={N:3d}  total={total:6.2f}  "
                  f"prep={prep:5.2f}  forward={forward:6.2f}  vae_prep={vae_prep:5.2f}  "
                  f"vae_fwd={vae_fwd:5.2f}  vae_cpu={vae_cpu:5.2f}  stop_cpu={stop_cpu:5.2f}  "
                  f"post={post:5.2f}  " + "  ".join(shim_stats), flush=True)

        return [
            {"latents": np_latents[i], "stop_flag": stop_flag[i], "waveforms": ret_waveforms[i]}
            for i in range(len(seqs))
        ]

    VoxCPM2Runner.run = run_timed
    print("[voxcpm_fast] timing probe installed", flush=True)


def uninstall_fast_path() -> None:
    """Restore upstream init_model. No effect if not installed."""
    global _installed, _original_init_model
    if not _installed or _original_init_model is None:
        return
    from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Runner
    VoxCPM2Runner.init_model = _original_init_model
    _original_init_model = None
    _installed = False
