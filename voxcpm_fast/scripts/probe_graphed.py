"""Measure per-phase time of the GRAPHED prefill forward.

Events are recorded *inside* the captured graph so their latest timestamps
are updated on every replay. After N replays we synchronize and read the
elapsed times between successive events.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"
MODEL_DIR = REPO_ROOT / "models" / "VoxCPM2"

sys.path.insert(0, str(NANOVLLM_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))


def install_graphed_phase_probe():
    """Before install_prefill_graph_capture runs, install a VoxCPM2Model.forward
    wrapper that re-emits forward body with inline CUDA events. Graph capture
    will bake the event.record() calls into the captured graph."""
    import torch
    from nanovllm_voxcpm.models.voxcpm2.model import VoxCPM2Model
    _orig = VoxCPM2Model.forward

    # Pre-allocated events (global pool).
    labels = ["start", "feat_enc", "embed", "base_lm", "fsq+fuse",
              "res_lm", "dit", "stop"]
    evs = [torch.cuda.Event(enable_timing=True) for _ in labels]

    def forward_probed(self, positions, text_tokens, feat, feat_mask,
                       temperature, cfg_value):
        from nanovllm_voxcpm.utils.context import get_context
        ctx = get_context()

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
        return {"latents": pred_feat, "stop_flag": stop_flag}

    VoxCPM2Model.forward = forward_probed
    print("[probe] forward rewritten with inline CUDA events", flush=True)
    return labels, evs


def main():
    os.environ["VOXCPM_FAST_ENC"] = "1"
    os.environ["VOXCPM_FAST_DIT"] = "1"
    os.environ["VOXCPM_FAST_BASE"] = os.environ.get("VOXCPM_FAST_BASE", "0")
    os.environ["VOXCPM_FAST_RES"] = os.environ.get("VOXCPM_FAST_RES", "0")

    # Install the probe rewrite FIRST, then graph capture wraps it.
    # But spawn child imports fresh, so we need to go through fast_main_loop.
    # Trick: monkey-patch fast_main_loop to call our probe-installer before the
    # graph capture install.
    import voxcpm_fast.fast_main_loop as fml
    _orig_fast = fml.fast_main_loop

    def fast_with_probe(queue_in, queue_out, args, kwargs):
        # Install probe + hooks in child.
        fml._ensure_sys_path()
        enc = os.environ.get("VOXCPM_FAST_ENC", "1") != "0"
        dit = os.environ.get("VOXCPM_FAST_DIT", "1") != "0"
        base = os.environ.get("VOXCPM_FAST_BASE", "0") != "0"
        res = os.environ.get("VOXCPM_FAST_RES", "0") != "0"
        from voxcpm_fast.engine_hook import install_fast_path, install_prefill_graph_capture
        install_fast_path(enable_feat_encoder=enc, enable_dit=dit,
                          enable_base_lm=base, enable_residual_lm=res)
        # Rewrite forward with inline events (replaces original).
        global _probe_state
        labels, evs = install_graphed_phase_probe()
        _probe_state = (labels, evs)
        # Now install graph capture on top — it'll wrap our probed forward.
        if os.environ.get("VOXCPM_PREFILL_GRAPH", "1") != "0":
            install_prefill_graph_capture()
        from nanovllm_voxcpm.models.voxcpm2.server import main_loop as _upstream
        return _upstream(queue_in, queue_out, args, kwargs)

    fml.fast_main_loop = fast_with_probe
    fml.patch_server_module()

    from nanovllm_voxcpm.models.voxcpm2.server import SyncVoxCPM2ServerPool

    # Child runs the probe; this parent only drives the pool.
    print("loading model …", flush=True)
    t0 = time.time()
    pool = SyncVoxCPM2ServerPool(
        model_path=str(MODEL_DIR),
        inference_timesteps=10,
        max_num_seqs=8,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        devices=[0],
    )
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    text = "The quick brown fox jumps over the lazy dog."
    # Prime a few requests so per-phase cache isn't warming up.
    for _ in range(3):
        for _ in pool.generate(target_text="Hello there friend.",
                               max_generate_length=200, temperature=0.7, cfg_value=2.0):
            pass

    # Drive 10 more — the child's probe will print events-difference lines.
    for _ in range(10):
        for _ in pool.generate(target_text=text, max_generate_length=200,
                               temperature=0.7, cfg_value=2.0):
            pass
    pool.stop()


if __name__ == "__main__":
    main()
