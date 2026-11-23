#!/usr/bin/env bash
# Checks that an exported HF adapter is the trained checkpoint and works outside this repo (model card, "Use"):
#   1. reference logprobs from the training server (base + checkpoint) on 8 of the checkpoint's held-out answers
#   2. Transformers + PEFT loading the HF folder as a user would: logprobs on the same tokens
#   3. stock SGLang with the converted adapter (convert_to_sglang.py): logprobs on the same tokens
#   4. the card's build snippet on one correct held-out answer (writable dir, numeric GPU group ids)
#   5. compare.py: mean |logprob difference| per answer token, each path vs the training server
# Usage: scripts/check_release.sh RUN_DIR CHECKPOINT EVAL_DIR HF_DIR OUT_DIR
#   CHECKPOINT tinker://<session>/weights/keep0075 (in RUN_DIR/server/store); EVAL_DIR its held-out A eval
#   (RUN_DIR/rl/evals/step0075); HF_DIR the exported folder (adapter_model.safetensors, adapter_config.json).
# Pass: PEFT and SGLang within the bf16 noise floor of the training server (~0.005 nats/token; adapter vs base ~0.026).
set -uo pipefail
R=$(readlink -f "${1:?run dir}"); CK=${2:?checkpoint}; E=$(readlink -f "${3:?eval dir}"); H=$(readlink -f "${4:?hf dir}")
C=$(readlink -f "${5:?out dir}"); mkdir -p "$C/build"
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"; . scripts/env.sh
export HF_HUB_OFFLINE=1; PY=$ZF_CLIENT_VENV/bin/python; RV=$ZF_RUNNER_VENV; K=$REPO/scripts
say() { echo "[$(date -u +%FT%TZ)] checks: $*" | tee -a "$C/STATUS"; }
gpu_args=(--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined)
for g in $(stat -c %g /dev/kfd /dev/dri/renderD* | sort -u); do gpu_args+=(--group-add "$g"); done   # numeric: names
                                                       # resolve in the image's /etc/group, not the host's
server_stop() { docker ps -a --format '{{.Names}}' | grep -qx zf_eval_server || return 0
  scripts/stop_server.sh zf_eval_server "$R/server" >/dev/null; sleep 10; }
trap 'server_stop; docker rm -f zf_check_sglang >/dev/null 2>&1' EXIT

# 1. training server
scripts/run_server.sh profiles/mi300x-qwen38-27b-phased.yaml "$R/server" zf_eval_server >/dev/null
up=0; for _ in $(seq 120); do curl -sf -m 5 http://127.0.0.1:8100/api/v1/healthz >/dev/null && { up=1; break; }; sleep 10; done
[ $up = 1 ] && $PY -I $K/check_ref_logprobs.py "$C" "$E" "$CK" > "$C/ref.log" 2>&1 && say "ref logprobs: done" \
  || { say "ref logprobs: failed, see $C/ref.log"; exit 1; }
server_stop

# 2. Transformers + PEFT from the HF folder
timeout 5400 docker run --rm --name zf_check_peft --network none --ipc host "${gpu_args[@]}" \
  -v "$HF_HUB_CACHE:$HF_HUB_CACHE:ro" -e HF_HUB_CACHE="$HF_HUB_CACHE" -e HF_HUB_OFFLINE=1 -v "$H:$H:ro" -v "$C:$C" -v "$K:$K:ro" \
  -e HOME=/tmp -e USER=user --user "$(id -u):$(id -g)" --entrypoint python "$ZF_SERVER_IMAGE" $K/check_peft_logprobs.py "$C" "$H" > "$C/peft.log" 2>&1 \
  && say "peft logprobs: done" || say "peft logprobs: failed, see $C/peft.log"

# 3. stock SGLang with the converted adapter
scripts/server_python.sh "$H:$C" scripts/convert_to_sglang.py "$H" "$C/sglang_adapter" > "$C/convert.log" 2>&1
docker run -d --name zf_check_sglang --network host --ipc host --shm-size 32g "${gpu_args[@]}" \
  -v "$HF_HUB_CACHE:$HF_HUB_CACHE:ro" -e HF_HUB_CACHE="$HF_HUB_CACHE" -e HF_HUB_OFFLINE=1 -v "$C:$C:ro" --entrypoint python "$ZF_SERVER_IMAGE" \
  -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --host 127.0.0.1 --port 30200 --enable-lora \
  --lora-paths hip=$C/sglang_adapter --lora-target-modules all --max-lora-rank 64 --context-length 8192 \
  --mem-fraction-static 0.8 --disable-radix-cache > /dev/null
up=0; for _ in $(seq 120); do curl -sf -m 5 http://127.0.0.1:30200/health >/dev/null && { up=1; break; }; sleep 15; done
[ $up = 1 ] && timeout 1800 $PY -I $K/check_sglang_logprobs.py "$C" http://127.0.0.1:30200 > "$C/sglang.log" 2>&1 \
  && say "sglang logprobs: done" || say "sglang logprobs: failed (up=$up), see $C/sglang.log"
docker logs zf_check_sglang > "$C/sglang_server.log" 2>&1; docker rm -f zf_check_sglang >/dev/null

# 4. the build snippet on one correct held-out answer of the checkpoint
python3 - "$E" "$C/build" <<'EOF'
import json, sys
sys.path.insert(0, "scripts"); import hip_grade   # the grader's own extraction (answer after the reasoning)
E, B = sys.argv[1], sys.argv[2]
g = {o["id"]: o for o in map(json.loads, open(f"{E}/graded.jsonl"))}
for r in map(json.loads, open(f"{E}/rollouts.jsonl")):
    if g.get(r["id"], {}).get("correct"):
        open(f"{B}/kernel.hip", "w").write(hip_grade.extract_hip(r["response"]))
        open(f"{B}/functional.py", "w").write(r["functional"]); break
EOF
timeout 900 docker run --rm --name zf_check_build --network none "${gpu_args[@]}" --cap-drop ALL \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:size=4g,exec -e HOME=/tmp -e TORCH_EXTENSIONS_DIR=/tmp/ext -e PYTORCH_ROCM_ARCH=gfx942 \
  -e PATH="$RV/bin:$ZF_ROCM/bin:/usr/bin:/bin" -e ROCM_HOME="$ZF_ROCM" \
  -v /usr:/usr:ro -v /etc/ld.so.cache:/etc/ld.so.cache:ro -v "$ZF_ROCM:$ZF_ROCM:ro" -v "$ZF_ROCM:/opt/rocm:ro" \
  -v "$RV:$RV:ro" -v "$C:$C:ro" -v "$K:$K:ro" -w /tmp --entrypoint bash "$ZF_GRADER_IMAGE" \
  -c "cp $C/build/kernel.hip $C/build/functional.py /tmp/ && $RV/bin/python -I $K/check_build.py /tmp/kernel.hip /tmp/functional.py" \
  > "$C/build_check.log" 2>&1
grep -q BUILD_CHECK_OK "$C/build_check.log" && say "build snippet: ok" || say "build snippet: failed, see $C/build_check.log"

# 5. compare
python3 -I $K/check_compare.py "$C" > "$C/compare.log" 2>&1; say "$(tr -d '\n ' < "$C/compare.log" | cut -c1-600)"
