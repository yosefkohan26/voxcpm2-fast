#!/usr/bin/env bash
# Download VoxCPM2 weights into ../models/VoxCPM2 (relative to this script).
# Idempotent; safe to re-run.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
dest="$root/models/VoxCPM2"

repo="openbmb/VoxCPM2"
hf_home="$root/hf_cache"
mkdir -p "$hf_home" "$root/models"

echo "[download_model] repo=$repo dest=$dest hf_home=$hf_home"

cd "$root/nanovllm-voxcpm"

if [ -f "$dest/config.json" ] && [ -f "$dest/audiovae.pth" ]; then
    echo "[download_model] already populated; skipping."
    exit 0
fi

HF_HOME="$hf_home" UV_CACHE_DIR="$root/.uv_cache" uv run python - <<PY
import os, sys
from huggingface_hub import snapshot_download

repo = "$repo"
dest = "$dest"
print(f"[hf] snapshot_download {repo} -> {dest}", flush=True)
local = snapshot_download(repo_id=repo, local_dir=dest, local_dir_use_symlinks=False)
print(f"[hf] done: {local}")
PY

echo "[download_model] done. Contents:"
ls -la "$dest"
