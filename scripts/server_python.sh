#!/usr/bin/env bash
# Run a Python script that needs the server's packages (zroform, peft, safetensors) inside the server image, as
# the calling user, CPU only, no network. Used for scripts/import_adapter.py and scripts/export_hf.py.
# Usage: scripts/server_python.sh DIR[:DIR...] SCRIPT.py [args...]
#   DIRs (no ":" in them; mount a parent of store paths) are mounted read-write at the same paths; HF_HUB_CACHE read-only.
#   Set ZROFORM_DATA to the server's out dir for scripts that open its store through a profile (import_adapter.py).
# Run import_adapter.py while the server is stopped: the running server owns its store as root.
set -euo pipefail
. "$(dirname "$0")/env.sh"
REPO=$(cd "$(dirname "$0")/.." && pwd)
mounts=(); IFS=: read -ra dirs <<< "${1:?dirs}"; shift
for d in "${dirs[@]}"; do mkdir -p "$d"; d=$(readlink -f "$d"); mounts+=(-v "$d:$d"); done
exec docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=user -w "$REPO" \
  -v "$REPO:$REPO:ro" -v "$ZF_SERVER_VENV:$ZF_SERVER_VENV:ro" -v "$HF_HUB_CACHE:$HF_HUB_CACHE:ro" "${mounts[@]}" \
  -e PYTHONPATH="$REPO" -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -e ZROFORM_DATA="${ZROFORM_DATA:-/nonexistent}" \
  -e ZROFORM_API_KEYS="$ZROFORM_API_KEYS" --entrypoint "$ZF_SERVER_VENV/bin/python" "$ZF_SERVER_IMAGE" "$@"
