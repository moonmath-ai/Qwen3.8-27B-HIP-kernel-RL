#!/usr/bin/env bash
# Stop a server started by run_server.sh and give OUT_DIR back to the calling user (the container ran as root).
# Usage: scripts/stop_server.sh NAME OUT_DIR
set -euo pipefail
. "$(dirname "$0")/env.sh"
NAME=${1:?container name}; OUT=${2:?out dir}
docker stop -t 30 "$NAME" >/dev/null && docker rm "$NAME" >/dev/null
docker run --rm -v "$OUT:$OUT" --entrypoint chown busybox@sha256:a46257e06c82ad85242d0d38077d0a52d9896389ca60a1a59d3463db586e5fa4 -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null \
  || docker run --rm -v "$OUT:$OUT" --entrypoint chown "${IMAGE:-$ZF_SERVER_IMAGE}" -R "$(id -u):$(id -g)" "$OUT"
echo "stopped $NAME; $OUT owned by $(id -un)"
