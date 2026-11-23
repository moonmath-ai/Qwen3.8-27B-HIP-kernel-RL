#!/bin/bash
# Build every torch-extension build dir under BUILD_ROOT with ninja, J at a time, inside a sandbox:
# no network, no GPU devices, read-only root, non-root, host userland (/usr) and ROCm mounted read-only,
# only BUILD_ROOT writable. Nothing built is ever loaded or executed. Prints "name rc ms" per kernel.
# usage: run_compile.sh BUILD_ROOT J [IMAGE]     (IMAGE only provides an empty merged-/usr rootfs)
set -euo pipefail
. "$(dirname "$0")/env.sh"
ROOT=$(readlink -f "$1"); J=$2; IMG=${3:-$ZF_GRADER_IMAGE}
VENV=$ZF_RUNNER_VENV
NINJA=$VENV/bin/ninja
exec docker run --rm --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:size=8g \
  --cpus "$(nproc)" --memory 100g --pids-limit 4096 --entrypoint /usr/bin/bash \
  -v /usr:/usr:ro -v /etc/ld.so.cache:/etc/ld.so.cache:ro -v $ZF_ROCM:$ZF_ROCM:ro \
  -v $VENV:$VENV:ro -v "$ROOT":"$ROOT" -e ROOT="$ROOT" -e J="$J" -e NINJA="$NINJA" -e HOME=/tmp "$IMG" -c '
one() { d=$1; s=$(date +%s%N); $NINJA -C "$d" -j1 > "$d/build.log" 2>&1; rc=$?; e=$(date +%s%N)
        echo "$(basename "$d") $rc $(( (e-s)/1000000 ))"; }
export -f one
ls -d "$ROOT"/*/ | sed "s#/\$##" | xargs -P "$J" -I{} bash -c "one {}"'
