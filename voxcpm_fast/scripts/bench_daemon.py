"""Long-lived bench daemon: loads the model once, reads JSON commands from
stdin, prints JSON results to stdout.

Wire-up (driver side):
    mkfifo /tmp/bench_in
    python voxcpm_fast/scripts/bench_daemon.py --fast < /tmp/bench_in &
    echo '{"cmd":"bench","target":"hello","trials":3}' > /tmp/bench_in

Each line of stdin is one JSON command; daemon emits one `[daemon] ...` line
per received command + one `RESULT: {...}` line per finished command so a
Monitor can grep for RESULT.

Commands (all cmd values case-sensitive):
  {"cmd":"env","set":{"KEY":"VAL"}, "unset":["KEY"]}  — mutate process env
  {"cmd":"bench","target":"...","trials":3,"voice":"bobby"|null,"warmup":1}
  {"cmd":"quit"}
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/workspace/Developments/VoxCPM2")
NANOVLLM_ROOT = REPO_ROOT / "nanovllm-voxcpm"
MODEL_DIR = REPO_ROOT / "models" / "VoxCPM2"
VOICES_DIR = REPO_ROOT / "voxcpm_fast" / "voices"

sys.path.insert(0, str(NANOVLLM_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "voxcpm_fast"))


def _stats(xs: list[float]) -> dict:
    xs_s = sorted(xs)
    return {
        "n": len(xs),
        "p50": statistics.median(xs),
        "p95": xs_s[min(len(xs) - 1, int(0.95 * len(xs)))],
        "mean": statistics.mean(xs),
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def _emit(tag: str, body: dict) -> None:
    """Emit a newline-terminated JSON line so a Monitor grep can pick it up."""
    sys.stdout.write(f"{tag}: {json.dumps(body)}\n")
    sys.stdout.flush()


def main():
    fast = "--fast" in sys.argv
    if fast:
        os.environ.setdefault("VOXCPM_FAST_ENC", "1")
        os.environ.setdefault("VOXCPM_FAST_DIT", "1")
        from voxcpm_fast.fast_main_loop import patch_server_module
        patch_server_module()

    from nanovllm_voxcpm.models.voxcpm2.server import SyncVoxCPM2ServerPool

    _emit("INFO", {"msg": f"loading mode={'FAST' if fast else 'UPSTREAM'}"})
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
    _emit("READY", {"load_s": round(time.time() - t0, 2)})

    # Lazy voice registration cache.
    voices: dict[str, str] = {}

    def _voice_id(name: str) -> str:
        if name in voices:
            return voices[name]
        d = VOICES_DIR / name
        wav = (d / "audio.wav").read_bytes()
        txt = (d / "transcript.txt").read_text().strip()
        t = time.time()
        voices[name] = pool.add_prompt(wav, "wav", txt)
        _emit("VOICE", {"name": name, "register_ms": int((time.time() - t) * 1000)})
        return voices[name]

    def _run_bench(cmd: dict) -> dict:
        target = cmd.get("target", "The quick brown fox jumps over the lazy dog.")
        trials = int(cmd.get("trials", 5))
        warmup = int(cmd.get("warmup", 1))
        voice_name = cmd.get("voice")
        voice_id = _voice_id(voice_name) if voice_name else None
        temp = float(cmd.get("temperature", 0.7))
        cfg = float(cmd.get("cfg_value", 2.0))

        # Warmup.
        for _ in range(warmup):
            for _ in pool.generate(target_text="Hi there.", prompt_id=voice_id,
                                   max_generate_length=80, temperature=temp, cfg_value=cfg):
                pass

        tfirst, total = [], []
        for _ in range(trials):
            t = time.time()
            tf = None
            for _ in pool.generate(target_text=target, prompt_id=voice_id,
                                   max_generate_length=200, temperature=temp, cfg_value=cfg):
                if tf is None:
                    tf = (time.time() - t) * 1000
            tfirst.append(tf)
            total.append((time.time() - t) * 1000)
        return {"tfirst": _stats(tfirst), "total": _stats(total),
                "raw_tfirst": [round(x, 2) for x in tfirst]}

    # Command loop.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            _emit("ERR", {"parse": str(e), "line": line[:200]})
            continue
        name = cmd.get("cmd")
        try:
            if name == "env":
                for k, v in cmd.get("set", {}).items():
                    os.environ[str(k)] = str(v)
                for k in cmd.get("unset", []):
                    os.environ.pop(str(k), None)
                _emit("ENV", {"set": cmd.get("set", {}), "unset": cmd.get("unset", []),
                              "current": {k: os.environ.get(k)
                                          for k in ("VOXCPM_PREFETCH", "VOXCPM_GEMM",
                                                    "VOXCPM_PRE_ATTN", "VOXCPM_ATTN")}})
            elif name == "bench":
                tag = cmd.get("tag", "")
                t0 = time.time()
                out = _run_bench(cmd)
                out["elapsed_s"] = round(time.time() - t0, 2)
                out["tag"] = tag
                out["env"] = {k: os.environ.get(k)
                              for k in ("VOXCPM_PREFETCH", "VOXCPM_GEMM",
                                        "VOXCPM_PRE_ATTN", "VOXCPM_ATTN")}
                _emit("RESULT", out)
            elif name == "quit":
                _emit("BYE", {})
                break
            else:
                _emit("ERR", {"unknown_cmd": name})
        except Exception as e:
            import traceback
            _emit("ERR", {"cmd": name, "exc": str(e),
                          "tb": traceback.format_exc()[-500:]})

    pool.stop()


if __name__ == "__main__":
    main()
