#!/usr/bin/env bash
# Auto-resume for full_run.sh. No root, no reboot.
# A model kernel can fault the GPU; the reset kills the training server, so rl_main exits rc 1. Then: remove the
# dead server, check the driver (no stuck ttm/kfd workers) and the GPU (gpu_health.py, process must exit fast), and
# re-run full_run.sh, which resumes from the last step's checkpoint. Gives up after MAX_RETRIES (3) restarts in
# WINDOW_S (6 h), on any exit other than rc 1 in rl_main (2 = gate stop, 3 = VRAM busy, 4 = driver wedged -> reboot),
# or when a check fails.
# Run inside tmux:  tmux new -d -s zf_super 'scripts/full_run_supervised.sh <run dir>'
# If a full_run.sh is already running in tmux zf_full, it adopts it (waits for it to end) instead of starting one.
set -uo pipefail
R=$(readlink -f "${1:?run dir}"); S=$R/STATUS
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"; . scripts/env.sh
MAX_RETRIES=${MAX_RETRIES:-3}; WINDOW_S=${WINDOW_S:-21600}
RUNNER_PY=$ZF_RUNNER_VENV/bin/python
SESSION=${SESSION:-zf_full}; SERVER=${SERVER:-zf_full_server}; FULL_RUN=${FULL_RUN:-scripts/full_run.sh}   # overridable for tests
HEALTH=${HEALTH:-"$RUNNER_PY -I scripts/gpu_health.py"}
say() { echo "[$(date -u +%FT%TZ)] supervisor: $*" | tee -a "$S"; }
# tmux starts the run with the tmux server's environment, not this shell's: pass the settings on explicitly
PASS_VARS="ZF_CLIENT_VENV ZF_SERVER_VENV ZF_RUNNER_VENV ZF_ROCM AIG_DATA ZF_SERVER_IMAGE ZF_GRADER_IMAGE HF_HOME HF_HUB_CACHE
  JOBS TINKER_API_KEY ZROFORM_API_KEYS DATA REFS REFS_WORK PILOT MAIN SWITCH TOKENS1 TOKENS2 EVAL_TOKENS EVAL_ARGS SMOKE
  REFS_N OWN_PER_LEVEL OWN_MIN OWN_TOKENS SFT_LIMIT RL_GROUPS GATE SAVE_EVERY MAX_IDLE_VRAM_GB"
launch() {
  local e=""; for v in $PASS_VARS; do [ -n "${!v+x}" ] && e+="$v=$(printf %q "${!v}") "; done
  tmux new -d -s "$SESSION" "env $e $FULL_RUN $R 2>&1 | tee -a $R.log"; }
restarts=()
tmux has-session -t "$SESSION" 2>/dev/null && say "adopting the running $SESSION" || { say "starting full_run"; launch; }
while true; do
  while tmux has-session -t "$SESSION" 2>/dev/null; do sleep "${POLL_S:-30}"; done
  last=$(grep -E '\] EXIT rc=|all stages done' "$S" | tail -1)
  case "$last" in
    *"all stages done"*|*"EXIT rc=0 "*) say "run finished"; exit 0 ;;
    *"EXIT rc=1 (stage: rl_main)"*) ;;
    *) say "not restarting after: ${last#*] }"; exit 1 ;;
  esac
  now=$(date +%s); kept=(); for t in "${restarts[@]}"; do [ $((now - t)) -lt "$WINDOW_S" ] && kept+=("$t"); done
  restarts=("${kept[@]}")
  [ "${#restarts[@]}" -lt "$MAX_RETRIES" ] || { say "gave up: ${#restarts[@]} restarts within $((WINDOW_S/3600)) h"; exit 1; }
  sleep "${SETTLE_S:-60}"   # let the dead server's processes finish tearing down
  docker ps -a --format '{{.Names}}' | grep -qx "$SERVER" && scripts/stop_server.sh "$SERVER" "$R/server" >/dev/null
  python3 -I -c "
import sys,time; sys.path.insert(0,'scripts'); import hip_grade as h
w=h.DriverWatchdog(max_d_s=60)
for _ in range(36):
    if not w.probe(): sys.exit(0)
    if w.check(): print(w.why); sys.exit(4)
    time.sleep(5)
sys.exit(4)" || { say "driver has stuck ttm/kfd workers -> reboot the VM, then re-run"; exit 4; }
  h=$(timeout 300 $HEALTH 2>&1 | tail -1) || { say "GPU health check failed: $h -> reboot the VM"; exit 4; }
  restarts+=("$(date +%s)")
  say "rl_main died (GPU fault?); health ok $h; restart ${#restarts[@]}/$MAX_RETRIES in this window"
  launch
done
