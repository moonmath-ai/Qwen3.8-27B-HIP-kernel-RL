#!/bin/bash
# Run a bash command in the same compile sandbox as run_compile.sh (no network, no GPU devices, read-only root,
# non-root, host /usr + ROCm + venv read-only); only WORK is writable (mounted at the same path).
# usage: sandbox.sh WORK 'bash commands'
set -euo pipefail
. "$(dirname "$0")/env.sh"
WORK=$(readlink -f "$1"); VENV=$ZF_RUNNER_VENV
exec docker run --rm --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:size=8g \
  --cpus "$(nproc)" --memory 100g --pids-limit 4096 --entrypoint /usr/bin/bash \
  -v /usr:/usr:ro -v /etc/ld.so.cache:/etc/ld.so.cache:ro -v $ZF_ROCM:$ZF_ROCM:ro -e ZF_ROCM=$ZF_ROCM \
  -v $VENV:$VENV:ro -v "$WORK":"$WORK" -e HOME=/tmp -e WORK="$WORK" -w "$WORK" "$ZF_GRADER_IMAGE" -c "$2"
