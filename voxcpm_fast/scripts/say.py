"""Text-to-speech smoke test. Uses the unmodified upstream engine to prove the
model works end-to-end; our fast-path integration still has to go through the
engine, which is next session's work.

Usage:
  uv run python voxcpm_fast/scripts/say.py "Text to speak." out.wav
"""

from __future__ import annotations

import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"
MODEL_DIR = REPO_ROOT / "models" / "VoxCPM2"

sys.path.insert(0, str(NANOVLLM_ROOT))
sys.path.insert(0, str(REPO_ROOT))  # expose voxcpm_fast package
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))


def main():
    if len(sys.argv) < 3:
        print("usage: say.py <text> <out.wav> [--cfg=2.0] [--temperature=0.7] [--no-fast]")
        sys.exit(1)
    text = sys.argv[1]
    out_path = Path(sys.argv[2])

    cfg_value = 2.0
    temperature = 0.7
    fast_enc = True
    fast_dit = True
    for a in sys.argv[3:]:
        if a.startswith("--cfg="):
            cfg_value = float(a.split("=", 1)[1])
        elif a.startswith("--temperature="):
            temperature = float(a.split("=", 1)[1])
        elif a == "--no-fast":
            fast_enc = fast_dit = False
        elif a == "--no-enc":
            fast_enc = False
        elif a == "--no-dit":
            fast_dit = False

    if fast_enc or fast_dit:
        # Spawn-safe hook install: patches server.main_loop so the child
        # process installs the fast path before building VoxCPM2ServerImpl.
        os.environ["VOXCPM_FAST_ENC"] = "1" if fast_enc else "0"
        os.environ["VOXCPM_FAST_DIT"] = "1" if fast_dit else "0"
        from voxcpm_fast.fast_main_loop import patch_server_module
        patch_server_module()

    from nanovllm_voxcpm.models.voxcpm2.server import SyncVoxCPM2ServerPool

    print(f"loading model from {MODEL_DIR} ...", flush=True)
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
    info = pool.get_model_info()
    sr = int(info["output_sample_rate"])
    print(f"ready in {time.time()-t0:.1f}s  sample_rate={sr}", flush=True)

    # Warmup run — first call loads CUDA graphs lazily and takes a few s even
    # though steady-state is sub-100 ms. Do an untimed short run to settle.
    print("warming up ...", flush=True)
    wu_chunks = 0
    for _ in pool.generate(target_text="Hello.", max_generate_length=50,
                           temperature=temperature, cfg_value=cfg_value):
        wu_chunks += 1
    print(f"  warmup done ({wu_chunks} chunks)", flush=True)

    print(f"generating: {text!r}", flush=True)
    t0 = time.time()
    chunks: list[np.ndarray] = []
    first_chunk_t = None
    for chunk in pool.generate(
        target_text=text,
        max_generate_length=500,
        temperature=temperature,
        cfg_value=cfg_value,
    ):
        if first_chunk_t is None:
            first_chunk_t = time.time() - t0
            print(f"  first chunk: {first_chunk_t*1000:.1f} ms  "
                  f"({len(chunk)} samples = {len(chunk)/sr*1000:.0f} ms of audio)",
                  flush=True)
        chunks.append(chunk.astype(np.float32))
    total_wall = time.time() - t0

    if not chunks:
        print("ERROR: no chunks produced")
        pool.stop()
        sys.exit(2)

    audio = np.concatenate(chunks)
    duration = len(audio) / sr
    rtf = total_wall / duration
    print(f"done: {len(chunks)} chunks  {len(audio)} samples  "
          f"{duration:.2f}s audio in {total_wall*1000:.0f} ms  RTF={rtf:.3f}", flush=True)

    # Clip to [-1, 1] and write int16 WAV.
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    print(f"wrote {out_path}  ({out_path.stat().st_size} bytes)")

    pool.stop()


if __name__ == "__main__":
    main()
