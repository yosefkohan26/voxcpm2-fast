"""P2.5.3-integration — end-to-end VoxCPM2Model.forward, upstream vs ours.

Monkey-patches `model.base_lm`, `residual_lm`, `feat_encoder.encoder`, and
`feat_decoder.estimator.decoder` with `FusedCpm4Model` shims, then drives
the *same* `VoxCPM2Model.forward(...)` code path as the upstream engine.
This gives the apples-to-apples T_first baseline: everything around the
transformer stacks (projections, masked_fill, cat, stop_head, feat_decoder
orchestration, VAE is NOT in this call) is unchanged.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"
MODEL_DIR = REPO_ROOT / "models" / "VoxCPM2"

sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))
sys.path.insert(0, str(NANOVLLM_ROOT))


def _init_dist():
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group("gloo", rank=0, world_size=1)


class FusedCausalLMShim(torch.nn.Module):
    """Drop-in for upstream causal Cpm4Model (base_lm, residual_lm) that
    routes through FusedCpm4Model AND writes KV cache via the upstream
    ``store_kvcache`` so decode can read it.

    Preserves upstream's ``embed_tokens`` attribute so ``model.base_lm.embed_tokens(...)``
    still works.
    """

    def __init__(self, fused, hidden: int, upstream_cpm4):
        super().__init__()
        self.fused = fused
        self.hidden = hidden
        # embed_tokens may be an Identity (residual_lm) or VocabParallelEmbedding
        # (base_lm). Expose it so callers of `self.base_lm.embed_tokens(...)`
        # continue to work.
        self.embed_tokens = upstream_cpm4.embed_tokens
        # Keep the upstream Cpm4Model as a fallback for decode (which needs
        # flash_attn_with_kvcache — our path runs flash_attn_func, prefill-only).
        # Also gives us reliable access to per-layer k_cache/v_cache tensors.
        self._upstream = upstream_cpm4

    def forward(self, input_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from nanovllm_voxcpm.utils.context import get_context
        ctx = get_context()
        # Decode path: our fused layer can't read KV cache. Delegate upstream.
        if not ctx.is_prefill:
            return self._upstream(input_embeds, positions)
        # Prefill path: fused forward + KV cache write.
        kv_caches = [(layer.self_attn.attn.k_cache,
                      layer.self_attn.attn.v_cache)
                     for layer in self._upstream.layers]
        return self.fused.forward(input_embeds, positions.to(torch.int32),
                                  kv_caches=kv_caches,
                                  slot_mapping=ctx.slot_mapping)


class FusedCpm4ModelShim(torch.nn.Module):
    """Drop-in for upstream ``Cpm4Model`` that accepts its 2-D or 3-D inputs
    and routes through ``FusedCpm4Model``.

    - 2-D ``input_embeds`` [N, H] → flat single-batch, matches causal stacks.
    - 3-D ``input_embeds`` [B, S, H] → flatten to [B*S, H], attention
      reshapes per-batch inside ``FusedLayer``. Output reshaped back.

    Optional internal CUDA-graph capture: shapes encountered at inference time
    get captured lazily and replayed on subsequent calls. This turns each
    layered fused forward into a single replay call, killing the 10–40 µs
    per-kernel launch overhead across ~120 layer-forwards per DiT call.
    """

    def __init__(self, fused, hidden: int, capture: bool = True):
        super().__init__()
        self.fused = fused
        self.hidden = hidden
        self._capture = capture
        # shape_key -> (graph, in_buf, pos_buf, out_buf, B)
        self._graphs: dict = {}
        self._replay_count = 0
        self._eager_count = 0
        self._new_capture_count = 0

    def _ensure_capture(self, shape_key, flat_template, pos_template, B):
        if shape_key in self._graphs:
            return
        # Warmup to populate kernel cache, force CUDA allocator memoization.
        for _ in range(3):
            _ = self.fused.forward(flat_template, pos_template, batch_size=B)
        torch.cuda.synchronize()

        in_buf = flat_template.clone()
        pos_buf = pos_template.clone()
        # Side-stream capture per PyTorch graph-capture recipe.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.fused.forward(in_buf, pos_buf, batch_size=B)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_buf = self.fused.forward(in_buf, pos_buf, batch_size=B)
        self._graphs[shape_key] = (g, in_buf, pos_buf, out_buf, B)
        import os
        if os.environ.get("VOXCPM_GRAPH_LOG") == "1":
            print(f"[voxcpm_fast] captured shim graph at {shape_key}  "
                  f"({len(self._graphs)} cached)", flush=True)

    def forward(self, input_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if input_embeds.dim() == 2:
            # 2-D path (causal stacks): runtime shape varies with prompt length,
            # so graph capture is per-N. We don't capture here unless asked
            # explicitly via env.
            out = self.fused.forward(input_embeds, positions.to(torch.int32))
            return out

        assert input_embeds.dim() == 3, input_embeds.shape
        B, S, H = input_embeds.shape
        assert H == self.hidden
        N = B * S
        flat = input_embeds.reshape(N, H).contiguous()
        if positions.shape[0] != N:
            pos_flat = positions.to(torch.int32).repeat(B)
        else:
            pos_flat = positions.to(torch.int32)

        # Internal graph capture of the DiT sub-forward. When the outer
        # stream is already capturing (i.e. the engine's prefill-graph
        # capture is recording), the internal capture is deferred: we run
        # the fused kernels eagerly on first invocation so the outer graph
        # records them inline. When the outer stream is NOT capturing, we
        # capture once per shape and replay on future calls. The "replay
        # while outer is capturing" path (nested replay) is documented to
        # work via cudaGraphAddChildGraphNode, but torch's CUDAGraph API
        # doesn't expose it cleanly, so we stick to the inline-recording
        # route for the outer graph.
        outer_capturing = torch.cuda.is_current_stream_capturing()
        if (self._capture and input_embeds.is_cuda
                and not torch.is_grad_enabled()
                and not outer_capturing):
            key = ("3d", B, S, H)
            if key not in self._graphs:
                self._new_capture_count += 1
                self._ensure_capture(key, flat, pos_flat, B)
            g, in_buf, pos_buf, out_buf, _ = self._graphs[key]
            in_buf.copy_(flat)
            pos_buf.copy_(pos_flat)
            g.replay()
            self._replay_count += 1
            return out_buf.clone().view(B, S, H)

        self._eager_count += 1
        out = self.fused.forward(flat, pos_flat, batch_size=B)
        return out.view(B, S, H)


def _collect_weights(model, dotted_prefix: str) -> dict:
    """Return model.state_dict() keys starting with `dotted_prefix`, stripped."""
    out = {}
    for k, v in model.state_dict().items():
        if k.startswith(dotted_prefix):
            out[k[len(dotted_prefix):]] = v.detach()
    return out


def _build_fused_cpm4(model_component, hidden, intermediate, num_layers,
                     causal, use_rope, rms_eps):
    """Wrap an upstream Cpm4Model into a FusedCpm4Model by extracting its weights.
    """
    from fused_layer_chained import FusedCpm4Model

    # Rebuild weight dict in upstream-key form from the nn.Module.
    weights = _collect_weights(model_component, "")
    # Merge q/k/v into qkv_proj, gate/up into gate_up_proj (same as the other
    # bench helpers do from safetensors).
    merged = dict(weights)
    for i in range(num_layers):
        q = weights.get(f"layers.{i}.self_attn.q_proj.weight")
        k_ = weights.get(f"layers.{i}.self_attn.k_proj.weight")
        v = weights.get(f"layers.{i}.self_attn.v_proj.weight")
        if q is not None and k_ is not None and v is not None:
            merged[f"layers.{i}.self_attn.qkv_proj.weight"] = torch.cat([q, k_, v], dim=0)
        # qkv_proj may already be present (if upstream uses QKVParallelLinear)
        elif weights.get(f"layers.{i}.self_attn.qkv_proj.weight") is not None:
            pass
        g = weights.get(f"layers.{i}.mlp.gate_proj.weight")
        u = weights.get(f"layers.{i}.mlp.up_proj.weight")
        if g is not None and u is not None:
            merged[f"layers.{i}.mlp.gate_up_proj.weight"] = torch.cat([g, u], dim=0)

    # RoPE cache
    if use_rope:
        rope = model_component.layers[0].self_attn.rotary_emb
        cos_cache = rope.cos_cached.to(torch.float32).contiguous()
        sin_cache = rope.sin_cached.to(torch.float32).contiguous()
    else:
        cos_cache = sin_cache = None

    need = set()
    for i in range(num_layers):
        for k in ("input_layernorm.weight", "self_attn.qkv_proj.weight",
                  "self_attn.o_proj.weight", "post_attention_layernorm.weight",
                  "mlp.gate_up_proj.weight", "mlp.down_proj.weight"):
            need.add(f"layers.{i}.{k}")
    need.add("norm.weight")
    fw = {k: v.to(torch.bfloat16).cuda().contiguous()
          for k, v in merged.items() if k in need}

    return FusedCpm4Model(
        weights=fw,
        rope_cos_cache=cos_cache,
        rope_sin_cache=sin_cache,
        hidden=hidden,
        intermediate=intermediate,
        num_layers=num_layers,
        causal=causal,
        use_rope=use_rope,
        rms_eps=rms_eps,
    )


def _load_upstream_voxcpm2():
    _init_dist()
    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.model import VoxCPM2Model
    from safetensors.torch import safe_open

    cfg = VoxCPM2Config.model_validate_json((MODEL_DIR / "config.json").read_text())
    cfg.lm_config.use_mup = False

    torch.set_default_dtype(torch.bfloat16)
    model = VoxCPM2Model(cfg, inference_timesteps=cfg.inference_timesteps).cuda()
    torch.set_default_dtype(torch.float32)

    # Load weights from safetensors directly (matches upstream's load path).
    state: dict[str, torch.Tensor] = {}
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)

    # Merge q/k/v and gate/up per layer across all sub-models the same way
    # upstream's custom loader does.
    def merge_for(prefix: str, num_layers: int):
        for i in range(num_layers):
            qk = f"{prefix}layers.{i}.self_attn.q_proj.weight"
            kk = f"{prefix}layers.{i}.self_attn.k_proj.weight"
            vk = f"{prefix}layers.{i}.self_attn.v_proj.weight"
            if qk in state and kk in state and vk in state:
                state[f"{prefix}layers.{i}.self_attn.qkv_proj.weight"] = torch.cat(
                    [state[qk], state[kk], state[vk]], dim=0)
            gk = f"{prefix}layers.{i}.mlp.gate_proj.weight"
            uk = f"{prefix}layers.{i}.mlp.up_proj.weight"
            if gk in state and uk in state:
                state[f"{prefix}layers.{i}.mlp.gate_up_proj.weight"] = torch.cat(
                    [state[gk], state[uk]], dim=0)

    merge_for("base_lm.", cfg.lm_config.num_hidden_layers)
    merge_for("residual_lm.", cfg.residual_lm_num_layers)
    merge_for("feat_encoder.encoder.", cfg.encoder_config.num_layers)
    merge_for("feat_decoder.estimator.decoder.", cfg.dit_config.num_layers)

    sd = model.state_dict()
    with torch.no_grad():
        for name, t in state.items():
            if name in sd:
                sd[name].copy_(t.to(torch.bfloat16).cuda())
    model.eval()
    return model, cfg


def _install_fast_shims(model, cfg):
    """Replace the four transformer stacks in-place with FusedCpm4Model shims.
    Returns the shims so the caller can also graph-capture them separately.
    """
    base_cfg = cfg.lm_config
    res_cfg = cfg.lm_config
    enc_cfg_overrides = cfg.encoder_config
    dit_cfg_overrides = cfg.dit_config

    base_shim = FusedCpm4ModelShim(
        _build_fused_cpm4(model.base_lm,
                          hidden=base_cfg.hidden_size,
                          intermediate=base_cfg.intermediate_size,
                          num_layers=base_cfg.num_hidden_layers,
                          causal=True, use_rope=True,
                          rms_eps=base_cfg.rms_norm_eps),
        hidden=base_cfg.hidden_size).cuda().eval()
    # Preserve embed_tokens so upstream's model.base_lm.embed_tokens(...) works.
    base_shim.embed_tokens = model.base_lm.embed_tokens

    residual_shim = FusedCpm4ModelShim(
        _build_fused_cpm4(model.residual_lm,
                          hidden=res_cfg.hidden_size,
                          intermediate=res_cfg.intermediate_size,
                          num_layers=cfg.residual_lm_num_layers,
                          causal=True, use_rope=not cfg.residual_lm_no_rope,
                          rms_eps=res_cfg.rms_norm_eps),
        hidden=res_cfg.hidden_size).cuda().eval()

    encoder_shim = FusedCpm4ModelShim(
        _build_fused_cpm4(model.feat_encoder.encoder,
                          hidden=enc_cfg_overrides.hidden_dim,
                          intermediate=enc_cfg_overrides.ffn_dim,
                          num_layers=enc_cfg_overrides.num_layers,
                          causal=False, use_rope=True,
                          rms_eps=base_cfg.rms_norm_eps),
        hidden=enc_cfg_overrides.hidden_dim).cuda().eval()

    dit_shim = FusedCpm4ModelShim(
        _build_fused_cpm4(model.feat_decoder.estimator.decoder,
                          hidden=dit_cfg_overrides.hidden_dim,
                          intermediate=dit_cfg_overrides.ffn_dim,
                          num_layers=dit_cfg_overrides.num_layers,
                          causal=False, use_rope=True,
                          rms_eps=base_cfg.rms_norm_eps),
        hidden=dit_cfg_overrides.hidden_dim).cuda().eval()

    model.base_lm = base_shim
    model.residual_lm = residual_shim
    model.feat_encoder.encoder = encoder_shim
    model.feat_decoder.estimator.decoder = dit_shim
    return base_shim, residual_shim, encoder_shim, dit_shim


def _time_iters(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record()
        fn()
        e1[i].record()
    torch.cuda.synchronize()
    return sorted(e0[i].elapsed_time(e1[i]) for i in range(iters))


def main(N=100, warmup=20, iters=50):
    torch.manual_seed(17)
    # Build TWO models: one with upstream stacks, one with our shims. Using
    # separate instances avoids state leak between replays.
    model_up, cfg = _load_upstream_voxcpm2()
    model_ours, _ = _load_upstream_voxcpm2()
    _install_fast_shims(model_ours, cfg)

    # ---------------------------------------------------------------
    # Build prefill inputs at N = prompt_len (text tokens, no feat).
    # VoxCPM2Model.forward signature:
    #   positions [N], text_tokens [N], feat [N, P, feat_dim], feat_mask [N],
    #   temperature (1,), cfg_value (1,)
    # ---------------------------------------------------------------
    P = cfg.patch_size        # 4
    feat_dim = cfg.feat_dim   # 64
    positions = torch.arange(N, device="cuda", dtype=torch.int64)
    text_tokens = torch.randint(0, cfg.lm_config.vocab_size - 100,
                                (N,), device="cuda", dtype=torch.int64)
    feat = (torch.randn(N, P, feat_dim, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    feat_mask = torch.zeros(N, device="cuda", dtype=torch.bool)
    feat_mask[N // 2:] = True  # second half is feat, first half is text
    temperature = torch.tensor([0.7], device="cuda", dtype=torch.bfloat16)
    cfg_value = torch.tensor([1.5], device="cuda", dtype=torch.bfloat16)

    from nanovllm_voxcpm.utils.context import set_context, reset_context

    cu_q = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    slot = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    def _run(model, seed=17):
        # UnifiedCFM draws z = torch.randn(...) inside its forward; seed for
        # deterministic comparison between upstream and ours.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        set_context(True, cu_q, cu_k, N, N, slot, None, None)
        try:
            with torch.inference_mode():
                out = model(positions, text_tokens, feat, feat_mask,
                            temperature, cfg_value)
        finally:
            reset_context()
        return out

    import flash_attn
    print("=" * 72)
    print(f"P2.5.3-integration — VoxCPM2Model.forward (prefill) c=1")
    print(f"date        : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"flash-attn  : {flash_attn.__version__}")
    print(f"N={N}  warmup={warmup}  iters={iters}")
    print("=" * 72)

    # Numerics (1-shot).
    up_out = _run(model_up)
    ours_out = _run(model_ours)
    for key in ("latents", "stop_flag"):
        u = up_out[key].float()
        o = ours_out[key].float()
        diff = (u - o).abs()
        max_abs = diff.max().item()
        max_val = u.abs().max().item()
        print(f"  {key:10s}: max_abs={max_abs:.3e}  max_val={max_val:.3e}  "
              f"max_rel={max_abs/max(max_val,1e-9):.3e}")

    print()
    # Eager timings.
    up_ms = _time_iters(lambda: _run(model_up), warmup, iters)
    ours_ms = _time_iters(lambda: _run(model_ours), warmup, iters)

    def pct(xs, p): return xs[int(round(p / 100.0 * (len(xs) - 1)))]

    # Graph-capture our forward. UnifiedCFM calls torch.randn inside solve_euler,
    # which is capturable when the RNG state is an explicit CUDA generator.
    # We run the forward with an RNG capture-compatible setup.
    torch.cuda.synchronize()
    # Warm capture stream per PyTorch recipe.
    cs = torch.cuda.Stream()
    cs.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cs), torch.inference_mode():
        for _ in range(5):
            _ = _run(model_ours)
    torch.cuda.current_stream().wait_stream(cs)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        set_context(True, cu_q, cu_k, N, N, slot, None, None)
        with torch.cuda.graph(g), torch.inference_mode():
            graphed_out = model_ours(positions, text_tokens, feat, feat_mask,
                                     temperature, cfg_value)
        reset_context()
        graph_ok = True
    except Exception as e:
        graph_ok = False
        print(f"(graph capture of full forward failed: {type(e).__name__}: {e})")
        print("  → continuing with eager-only measurement")

    if graph_ok:
        ours_g_ms = _time_iters(lambda: g.replay(), warmup, iters)
    else:
        ours_g_ms = None

    print(f"{'phase':28s}  {'p50':>7s}  {'p95':>7s}  {'p99':>7s}  {'mean':>7s}  (ms)")
    print(f"{'-'*28}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for name, xs in [("upstream eager", up_ms), ("ours eager", ours_ms)]:
        print(f"{name:28s}  {pct(xs,50):7.3f}  {pct(xs,95):7.3f}  {pct(xs,99):7.3f}  {statistics.mean(xs):7.3f}")
    if ours_g_ms is not None:
        print(f"{'ours graphed':28s}  {pct(ours_g_ms,50):7.3f}  {pct(ours_g_ms,95):7.3f}  "
              f"{pct(ours_g_ms,99):7.3f}  {statistics.mean(ours_g_ms):7.3f}")
    print()
    up50, ou50 = pct(up_ms, 50), pct(ours_ms, 50)
    print(f"ours eager    vs upstream : {up50/ou50:.2f}x")
    if ours_g_ms is not None:
        og50 = pct(ours_g_ms, 50)
        print(f"ours graphed  vs upstream : {up50/og50:.2f}x")
        print(f"graph savings vs ours eager: {ou50 - og50:.2f} ms  ({100.0*(1 - og50/ou50):.1f}%)")
    print()
    print(f"upstream baseline T_first p50 @ c=1 : 187.3 ms (BASELINE.md)")
    print(f"headroom to 70 ms target           : {(ou50 if ours_g_ms is None else og50) / 70:.2f}x")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    main(N=args.N, warmup=args.warmup, iters=args.iters)
