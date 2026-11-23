#!/usr/bin/env bash
# Optional: save a copy of the SGLang ROCm image with its attention kernels already compiled. AITER (AMD's kernel
# library in the image) compiles them on the first requests of every new container (~3-4 min each) and keeps
# nothing afterwards, so each server, eval or check start pays that again. This starts the image once, sends a few
# requests so the kernels compile, and saves the container as a new local image. Everything works without it.
# Usage: scripts/warm_image.sh [NAME]      (default sglang-warm; ~15 min on one MI300X; the GPU must be free)
#   then: export ZF_SERVER_IMAGE=NAME      (or set it wherever you set the other env.sh variables)
set -euo pipefail
. "$(dirname "$0")/env.sh"
NAME=${1:-sglang-warm}; C=zf_warm_$$; PORT=30150
amd-smi process 2>&1 | grep -q 'No running processes' || { echo "GPU has other processes; not starting" >&2; exit 1; }
trap 'docker rm -f "$C" >/dev/null 2>&1 || true' EXIT
docker run -d --name "$C" --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined \
  -v "$HF_HUB_CACHE:/hf:ro" -e HF_HUB_CACHE=/hf -e HF_HUB_OFFLINE=1 --entrypoint bash "$ZF_SERVER_IMAGE" -lc \
  "rm -rf /sgl-workspace/aiter/aiter/jit/build/lock_*; exec python3 -m sglang.launch_server --model-path Qwen/Qwen3.8-27B \
   --host 127.0.0.1 --port $PORT --mem-fraction-static 0.85 --context-length 16384 --disable-radix-cache \
   --watchdog-timeout 3600 --skip-server-warmup" >/dev/null
echo "started $C from $ZF_SERVER_IMAGE; waiting for it to come up (first start compiles kernels)"
for _ in $(seq 360); do curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break
  docker ps --format '{{.Names}}' | grep -qx "$C" || { docker logs "$C" 2>&1 | tail -20; exit 1; }; sleep 10; done
# a short and a long prompt, a few at once: prefill and decode paths at several batch sizes
long=$(python3 -c "print('Explain GPU memory coalescing in detail. ' * 200)")
for n in 1 4 16; do
  pids=()
  for i in $(seq "$n"); do
    curl -sf -m 1800 "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
      -d "$(python3 -c "import json,sys; print(json.dumps({'text': sys.argv[1], 'sampling_params': {'max_new_tokens': 256}}))" \
            "$([ $((i % 2)) = 0 ] && echo "$long" || echo 'Write a HIP kernel that adds two vectors.')")" >/dev/null &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { echo "a warm-up request failed; not saving" >&2; exit 1; }; done
  echo "batch of $n done"
done
docker stop -t 30 "$C" >/dev/null
# keep the base image's own entrypoint and command (the scripts here always override them anyway)
ep=$(docker inspect -f '{{json .Config.Entrypoint}}' "$ZF_SERVER_IMAGE"); cmd=$(docker inspect -f '{{json .Config.Cmd}}' "$ZF_SERVER_IMAGE")
docker commit -c "ENTRYPOINT $([ "$ep" = null ] && echo '[]' || echo "$ep")" -c "CMD $([ "$cmd" = null ] && echo '[]' || echo "$cmd")" \
  "$C" "$NAME" >/dev/null
echo "saved $NAME; use it with: export ZF_SERVER_IMAGE=$NAME"
