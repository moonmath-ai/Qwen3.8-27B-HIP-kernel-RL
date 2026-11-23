#!/usr/bin/env bash
# Plain SGLang server (no trainer) on the MI300X: samples the base model's own answers for SFT (full_run.sh stage 1).
# Usage: scripts/run_sglang.sh NAME [extra sglang args...]   -> http://127.0.0.1:30100 (no API key)
# Stop: docker stop NAME && docker rm NAME. The warm image has the AITER JIT cache; stale build locks are removed.
set -euo pipefail
. "$(dirname "$0")/env.sh"
NAME=${1:?container name}; shift
IMAGE=${IMAGE:-$ZF_SERVER_IMAGE}
amd-smi process 2>&1 | grep -q 'No running processes' || { echo "GPU has other processes; not starting" >&2; exit 1; }
docker run -d --name "$NAME" --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined \
  -v "$HF_HUB_CACHE:/hf:ro" -e HF_HUB_CACHE=/hf -e HF_HUB_OFFLINE=1 --entrypoint bash "$IMAGE" -lc \
  "rm -rf /sgl-workspace/aiter/aiter/jit/build/lock_*; exec python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B \
   --host 127.0.0.1 --port 30100 --mem-fraction-static 0.85 --context-length 32768 --max-running-requests 256 \
   --disable-radix-cache --watchdog-timeout 3600 --skip-server-warmup --log-level info $*"
echo "started $NAME (docker logs $NAME)"
