#!/usr/bin/env bash
# CPU test suite inside the server image (its packages + the server venv); no GPU, no network.
# Usage: scripts/test_cpu.sh [pytest args]     expected: all pass, 1 skip (needs a GPU)
set -euo pipefail
. "$(dirname "$0")/env.sh"
REPO=$(cd "$(dirname "$0")/.." && pwd)
exec docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=user -v "$REPO:$REPO" -w "$REPO" \
  -v "$ZF_SERVER_VENV:$ZF_SERVER_VENV:ro" -v "$ZF_RUNNER_VENV:$ZF_RUNNER_VENV:ro" -e ZF_RUNNER_VENV="$ZF_RUNNER_VENV" \
  -v "$HF_HUB_CACHE:$HF_HUB_CACHE:ro" -e HF_HUB_CACHE="$HF_HUB_CACHE" -e HF_HUB_OFFLINE=1 -e PYTHONPATH="$REPO" \
  -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= --entrypoint "$ZF_SERVER_VENV/bin/python" "$ZF_SERVER_IMAGE" \
  -m pytest tests -q -p no:cacheprovider -rs "$@"
