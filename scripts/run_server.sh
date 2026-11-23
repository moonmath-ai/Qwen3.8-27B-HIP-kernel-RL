#!/usr/bin/env bash
# Zro-form training server (trainer + SGLang sampler) on the MI300X, inside the SGLang ROCm image.
# Usage: scripts/run_server.sh PROFILE OUT_DIR [NAME]
#   PROFILE  e.g. profiles/mi300x-qwen38-27b-phased.yaml  OUT_DIR  store/metrics/logs root (ZROFORM_DATA)
# Env (scripts/env.sh): IMAGE (default $ZF_SERVER_IMAGE), VENV (default $ZF_SERVER_VENV: the server's extras over
#      the image's packages, setup_server_venv.sh), PORT (default 8100).
# Runs as root in the container (the image's AITER JIT cache is root-owned); OUT_DIR files end up root-owned,
# so the script chowns OUT_DIR back on stop (scripts/stop_server.sh).
set -euo pipefail
. "$(dirname "$0")/env.sh"
PROFILE=${1:?profile}; OUT=${2:?out dir}; NAME=${3:-zf-amd-server}
IMAGE=${IMAGE:-$ZF_SERVER_IMAGE}; VENV=${VENV:-$ZF_SERVER_VENV}; PORT=${PORT:-8100}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
amd-smi process 2>&1 | grep -q 'No running processes' || { echo "GPU has other processes; not starting" >&2; exit 1; }
docker run -d --name "$NAME" --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined \
  -v "$HF_HUB_CACHE:$HF_HUB_CACHE:ro" -e HF_HUB_CACHE="$HF_HUB_CACHE" \
  -e HF_HUB_OFFLINE=1 -v "$REPO:$REPO:ro" -v "$VENV:$VENV:ro" -v "$OUT:$OUT" -w "$REPO" \
  -e PYTHONPATH="$REPO" -e PYTHONDONTWRITEBYTECODE=1 -e PATH="$VENV/bin:/opt/venv/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin" \
  -e ZROFORM_PROFILE="$PROFILE" -e ZROFORM_DATA="$OUT" -e ZROFORM_API_KEYS="$ZROFORM_API_KEYS" \
  -e TMPDIR=/tmp --entrypoint bash "$IMAGE" -lc \
  "rm -rf /sgl-workspace/aiter/aiter/jit/build/lock_*; exec $VENV/bin/python -m zroform.launch --host 127.0.0.1 --port $PORT > $OUT/server.log 2>&1"
echo "started $NAME; log $OUT/server.log, SGLang log $OUT/sglang.log"
