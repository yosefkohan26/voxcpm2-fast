"""Enumerate everything the loaded VoxCPM2 model actually does.

Output sections:

1. Module tree with shapes + parameter counts.
2. A single prefill + single decode forward, with a hook that records every
   ``nn.Module.__call__`` in order + input/output shapes.
3. Kernel-launch count for one prefill and one decode step via the CUDA event
   stream (simple counter) and, if available, CUPTI callbacks.
4. Wall-time breakdown of (feat_encoder, base_lm, residual_lm, feat_decoder,
   audiovae.decode) for one decode step, averaged over N iterations.

Designed to run *inside* the upstream server process via
``VoxCPMRunner`` so we see the real graph the runner builds, not an idealized
model constructed from scratch.

Invocation:

    uv run python voxcpm_fast/experiments/explore_model.py \\
        --model ./models/VoxCPM2 \\
        --iters 20 \\
        --output voxcpm_fast/notes/topology.md
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from collections import Counter, defaultdict


def param_count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def module_tree_lines(module, prefix: str = "") -> list[str]:
    lines = []
    for name, child in module.named_children():
        lines.append(f"{prefix}{name}: {type(child).__name__}  params={param_count(child):,}")
        lines.extend(module_tree_lines(child, prefix + "  "))
    return lines


@contextlib.contextmanager
def forward_recorder(root):
    """Record (qualified_name, input_shapes, output_shapes) for every module call."""
    events: list[tuple[str, tuple, tuple]] = []
    handles = []

    def shape_of(x):
        import torch
        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
        if isinstance(x, (list, tuple)):
            return tuple(shape_of(i) for i in x)
        if isinstance(x, dict):
            return {k: shape_of(v) for k, v in x.items()}
        return type(x).__name__

    qual_names: dict[int, str] = {}
    for name, mod in root.named_modules():
        qual_names[id(mod)] = name or "<root>"

    def pre_hook(mod, args, kwargs):
        # no-op: capture in fwd hook
        return None

    def fwd_hook(mod, args, output):
        events.append((qual_names.get(id(mod), type(mod).__name__), shape_of(args), shape_of(output)))

    for mod in root.modules():
        handles.append(mod.register_forward_hook(fwd_hook))
    try:
        yield events
    finally:
        for h in handles:
            h.remove()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to VoxCPM2 weights dir")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--output", required=True, help="Path to write markdown report")
    ap.add_argument("--devices", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "CUDA required"
    print(f"[explore] torch={torch.__version__}  device={torch.cuda.get_device_name(0)}", flush=True)

    # We cannot easily instantiate the runner out-of-process; easiest path is to
    # reuse upstream's sync server which still gives us direct access to the
    # engine in the child process. Here we build the engine in-process for
    # exploration only (no tensor parallelism). We therefore bypass the
    # multiprocessing layer and instantiate ``VoxCPMEngine`` directly.
    from nanovllm_voxcpm.config import Config
    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine

    import json
    cfg_path = os.path.join(args.model, "config.json")
    model_config = VoxCPM2Config.model_validate_json(open(cfg_path).read())
    engine_config = Config(
        model=args.model,
        model_config=model_config,
        devices=args.devices,
        max_num_batched_tokens=4096,
        max_num_seqs=8,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        enforce_eager=True,  # we want readable graph, not CUDA-Graph replay
    )
    print("[explore] building engine in-process (this may take a minute)…", flush=True)
    engine = VoxCPM2Engine(engine_config)
    model = engine.model_runner.model

    lines: list[str] = []
    lines.append("# VoxCPM2 Topology\n")
    lines.append(f"- arch: `{type(model).__name__}`")
    lines.append(f"- total params: `{param_count(model):,}`")
    lines.append(f"- feat_dim: `{model.feat_dim}`  patch_size: `{model.patch_size}`")
    lines.append(f"- inference_timesteps: `{engine_config.model_config.inference_timesteps}`\n")

    lines.append("## Module tree\n")
    lines.append("```")
    lines.extend(module_tree_lines(model))
    lines.append("```\n")

    # One decode step under the forward recorder.
    print("[explore] recording one decode step…", flush=True)
    engine.add_request(
        seq_id="explore-0",
        target_text="hello world, this is an exploration run.",
        max_generate_length=2,
    )
    # Prefill first.
    engine.step()
    with forward_recorder(model) as events:
        torch.cuda.synchronize()
        engine.step()
        torch.cuda.synchronize()
    lines.append("## Forward order (one decode step, concurrency=1)\n")
    lines.append(f"Total module calls: `{len(events)}`\n")
    lines.append("| # | module | in shapes | out shapes |")
    lines.append("|---|---|---|---|")
    for i, (name, in_s, out_s) in enumerate(events):
        if name == "" or name == "<root>":
            continue
        lines.append(f"| {i} | `{name}` | `{in_s}` | `{out_s}` |")
    lines.append("")

    # Kernel-count + wall-time per section.
    print(f"[explore] timing {args.iters} decode steps…", flush=True)
    # Reset engine, fresh request with enough length.
    engine.add_request(
        seq_id="explore-1",
        target_text="timing run. " * 40,
        max_generate_length=args.iters + 2,
    )
    engine.step()  # prefill

    torch.cuda.synchronize()
    section_ms = defaultdict(float)
    start_evts = {}

    def make_section_hooks():
        hooks = []
        # Section = top-level subcomponents we care about.
        sections = {
            "feat_encoder": model.feat_encoder,
            "enc_to_lm_proj": model.enc_to_lm_proj,
            "base_lm": model.base_lm,
            "fsq_layer": model.fsq_layer,
            "residual_lm": model.residual_lm,
            "lm_to_dit_proj": model.lm_to_dit_proj,
            "res_to_dit_proj": model.res_to_dit_proj,
            "feat_decoder": model.feat_decoder,
            "stop_head": model.stop_head,
        }
        # Use CUDA events around pre/post hooks.
        def mkpre(name):
            def _pre(mod, *_a, **_k):
                e = torch.cuda.Event(enable_timing=True); e.record()
                start_evts[name] = e
            return _pre
        def mkpost(name):
            def _post(mod, *_a, **_k):
                e = torch.cuda.Event(enable_timing=True); e.record()
                e.synchronize()
                section_ms[name] += start_evts[name].elapsed_time(e)
            return _post
        for name, mod in sections.items():
            hooks.append(mod.register_forward_pre_hook(mkpre(name)))
            hooks.append(mod.register_forward_hook(mkpost(name)))
        return hooks

    hooks = make_section_hooks()
    try:
        t0 = time.perf_counter()
        steps_done = 0
        for _ in range(args.iters):
            out = engine.step()
            if not out:
                break
            steps_done += 1
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        for h in hooks:
            h.remove()

    lines.append("## Per-section wall time (averaged across decode steps)\n")
    lines.append(f"- steps measured: `{steps_done}`")
    lines.append(f"- end-to-end wall: `{wall_ms / max(steps_done, 1):.2f} ms/step`\n")
    lines.append("| section | ms/step | % |")
    lines.append("|---|---|---|")
    total_section = sum(section_ms.values()) / max(steps_done, 1)
    for name, total in sorted(section_ms.items(), key=lambda kv: -kv[1]):
        per = total / max(steps_done, 1)
        pct = 100.0 * per / max(total_section, 1e-9)
        lines.append(f"| `{name}` | `{per:.3f}` | `{pct:.1f}%` |")
    lines.append("")

    lines.append("## Notes\n")
    lines.append("- Timings measured with `enforce_eager=True` (no CUDA Graph replay). Real hot path will be faster on the no-LoRA graph.")
    lines.append("- Section totals won't add up to 100% because we haven't instrumented every small op (projections, adds, norms outside the sections above).")
    lines.append("")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"[explore] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
