#!/usr/bin/env bash
# The whole run (RESULTS.md): dataset references graded -> base model's own answers sampled and graded ->
# SFT data -> SFT -> eval -> RL pilot (20 steps) -> base-model eval -> gate (check_pilot.py) -> RL main.
# Run inside tmux (or under full_run_supervised.sh, which restarts it after a GPU fault):
#   tmux new -s zf_full 'scripts/full_run.sh <run dir> 2>&1 | tee -a <run dir>.log'
# Re-running skips stages whose output is complete (checked by content, not markers). State: $R/STATUS.
# Answer budget: 12,288 tokens through step 57, 14,000 from step 58 (the released run's schedule). The released run
# stopped after step 125 (MAIN) by its stop rule (no gain over step 75 in 50 steps); step 75 is the released adapter.
set -euo pipefail
R=$(readlink -f "${1:?run dir}"); mkdir -p "$R"
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"; . scripts/env.sh
DATA=${DATA:-$AIG_DATA}
REFS=${REFS:-$R/refs}          # the dataset's reference answer for every train task, graded (stage 0)
PY=$ZF_CLIENT_VENV/bin/python
PROFILE=profiles/mi300x-qwen38-27b-phased.yaml
# SMOKE=1: the same stages at toy size (~40 min on the GPU; scripts/smoke.sh runs it) to check the pipeline end to end; numbers are meaningless.
if [ "${SMOKE:-0}" = 1 ]; then
  : "${REFS_N:=24}" "${OWN_PER_LEVEL:=L1:0,L2:0,L3:0}" "${OWN_MIN:=1}" "${OWN_TOKENS:=2048}" "${SFT_LIMIT:=8}"
  : "${PILOT:=2}" "${SWITCH:=3}" "${MAIN:=4}" "${RL_GROUPS:=4}" "${TOKENS1:=2048}" "${TOKENS2:=3072}" "${GATE:=warn}"
  : "${EVAL_ARGS:=--eval-limit 4 --eval-k 2}" "${EVAL_TOKENS:=2048}"
fi
PILOT=${PILOT:-20}; MAIN=${MAIN:-125}; SWITCH=${SWITCH:-57}   # steps < SWITCH at TOKENS1 (12,288), then TOKENS2 (14,000)
TOKENS1=${TOKENS1:-12288}; TOKENS2=${TOKENS2:-14000}; EVAL_TOKENS=${EVAL_TOKENS:-12288}; EVAL_ARGS=${EVAL_ARGS:-}
export HF_HUB_OFFLINE=1
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$R/STATUS"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
wait_http() {  # URL CONTAINER: up to 30 min, but give up as soon as the container has exited
  for _ in $(seq 1 180); do curl -sf -m 5 "$1" >/dev/null && return 0
    docker ps --format '{{.Names}}' | grep -qx "$2" || return 1; sleep 10; done; return 1; }
driver_ok() {  # no amdgpu ttm/kfd kernel worker stuck in D state for 60 s (hip_grade.DriverWatchdog)
  python3 -I -c "
import sys,time; sys.path.insert(0,'scripts'); import hip_grade as h
w=h.DriverWatchdog(max_d_s=60)
for _ in range(14):
    if not w.probe(): sys.exit(0)
    if w.check(): print(w.why); sys.exit(4)
    time.sleep(5)
sys.exit(4)" || { say "amdgpu driver unhealthy before ${STAGE:-?}"; return 4; }
}
vram_used_gb() { amd-smi metric -m 2>/dev/null | sed -n 's/.*USED_VRAM: \([0-9]*\) MB/\1/p' | head -1 | awk '{print int($1/1024)}'; }
server_up() {
  docker ps --format '{{.Names}}' | grep -qx zf_full_server && return 0
  # a server that died (e.g. exit 139 after a GPU hang) leaves an exited container holding the name; remove it first
  docker ps -a --format '{{.Names}}' | grep -qx zf_full_server && scripts/stop_server.sh zf_full_server "$R/server"
  # a GPU hang can leave unkillable processes holding VRAM; the colocated server needs ~all of it, so refuse to start
  u=$(vram_used_gb); [ "${u:-0}" -le "${MAX_IDLE_VRAM_GB:-8}" ] || { say "GPU has ${u} GB VRAM in use before server start: stop"; exit 3; }
  driver_ok || exit 4
  scripts/run_server.sh "$PROFILE" "$R/server" zf_full_server
  wait_http http://127.0.0.1:8100/api/v1/healthz zf_full_server || { say "server did not come up"; exit 1; }
}
server_down() { docker ps -a --format '{{.Names}}' | grep -qx zf_full_server && scripts/stop_server.sh zf_full_server "$R/server" || true; }
trap 'rc=$?; say "EXIT rc=$rc (stage: ${STAGE:-?})$([ $rc -eq 4 ] && echo ": amdgpu driver unhealthy -> reboot, check the GPU (REPRODUCE.md §4), re-run")"' EXIT

# 0. the dataset's reference answers, one per train task (seed 0), graded for correctness (SFT data, task records) -
STAGE=refs
mkdir -p "$REFS"
[ -s "$REFS/refs.jsonl" ] || python3 -I scripts/make_ref_rollouts.py --data "$DATA" --n "${REFS_N:-0}" --seed 0 --out "$REFS/refs.jsonl"
if [ "$(lines "$REFS/graded.jsonl")" -ne "$(lines "$REFS/refs.jsonl")" ]; then
  driver_ok || exit 4
  say "refs: grading $(lines "$REFS/refs.jsonl") references"
  python3 -I scripts/hip_grade.py --tasks "$REFS/refs.jsonl" --work "${REFS_WORK:-$R/work_refs}" --out "$REFS/graded.jsonl" \
    --jobs "$JOBS" --gpu --no-timing --workers 10 > "$REFS/grade.log" 2>&1
  say "refs: done, correct $(grep -c '"correct": true' "$REFS/graded.jsonl" || true) / $(lines "$REFS/graded.jsonl")"
fi

# 0b. re-profile references whose kernel profile came back empty (the profiler loses events with 10 containers at
# once; hip_grade.run_all already re-runs those alone, so this re-runs few or none) ----------------------------
STAGE=refs_fix; RG=$R/refs_graded.jsonl
if [ ! -s "$RG" ]; then
  say "refs_fix: start"; mkdir -p "$R/refs_fix"
  python3 -c "
import json,sys
g={o['id'] for o in map(json.loads,open('$REFS/graded.jsonl')) if o.get('correct') and not o.get('kernels')}
with open('$R/refs_fix/rollouts.jsonl','w') as f:
    for l in open('$REFS/refs.jsonl'):
        if json.loads(l)['id'] in g: f.write(l)"
  python3 -I scripts/hip_grade.py --tasks "$R/refs_fix/rollouts.jsonl" --work "${REFS_WORK:-$R/work_refs}" \
    --out "$R/refs_fix/graded.jsonl" --jobs "$JOBS" --gpu --no-timing --workers 1 > "$R/refs_fix/grade.log" 2>&1
  python3 -c "
import json
fix={o['id']:o for o in map(json.loads,open('$R/refs_fix/graded.jsonl'))}
with open('$RG.tmp','w') as f:
    for o in map(json.loads,open('$REFS/graded.jsonl')): f.write(json.dumps(fix.get(o['id'],o))+chr(10))"
  mv "$RG.tmp" "$RG"
  say "refs_fix: done, $(lines "$R/refs_fix/graded.jsonl") re-run, still empty $(grep -c '"kernels": \[\]' "$R/refs_fix/graded.jsonl" || true)"
fi

# 1. own answers on train tasks: all L2/L3 + 2000 L1, k=2, plain SGLang -----------------------------------------
STAGE=own_sample; O=$R/own
mkdir -p "$O"
if [ "$(lines "$O/rollouts.jsonl")" -lt "${OWN_MIN:-1000}" ]; then
  say "own_sample: start"
  docker ps --format '{{.Names}}' | grep -qx zf_own || scripts/run_sglang.sh zf_own
  wait_http http://127.0.0.1:30100/health zf_own || { say "SGLang did not come up"; exit 1; }
  python3 -I scripts/sample_sglang.py --data "$DATA" --split train --task-records "$REFS/refs.jsonl" \
    --per-level "${OWN_PER_LEVEL:-L1:2000,L2:0,L3:0}" --k 2 --concurrency 128 --seed 100000 --max-tokens "${OWN_TOKENS:-12288}" \
    --out "$O/rollouts.jsonl" > "$O/sample.log" 2>&1
  docker logs zf_own > "$O/sglang.log" 2>&1 || true
  scripts/stop_server.sh zf_own "$O"
  say "own_sample: done $(lines "$O/rollouts.jsonl") rows, errors $(grep -c '_err"' "$O/rollouts.jsonl" || true)"
fi

# 2. grade them (SGLang off), correctness only ---------------------------------------------------------------
STAGE=own_grade
driver_ok || exit 4
if [ "$(lines "$O/graded.jsonl")" -ne "$(lines "$O/rollouts.jsonl")" ]; then
  say "own_grade: start"
  python3 -I scripts/hip_grade.py --tasks "$O/rollouts.jsonl" --work "$R/work_own" --out "$O/graded.jsonl" \
    --jobs "$JOBS" --gpu --no-timing --workers 10 > "$O/grade.log" 2>&1
  say "own_grade: done, correct $(grep -c '"correct": true' "$O/graded.jsonl" || true) / $(lines "$O/graded.jsonl")"
fi

# 3. SFT data ---------------------------------------------------------------------------------------------------
STAGE=sft_data; S=$R/sft
mkdir -p "$S"
if [ ! -s "$S/sft.jsonl" ]; then
  python3 -I scripts/build_sft_data.py --data "$DATA" --own-rollouts "$O/rollouts.jsonl" --own-graded "$O/graded.jsonl" \
    --refs "$REFS/refs.jsonl" --refs-graded "$RG" --out "$S/sft.jsonl" --exclude-out "$R/exclude.txt" \
    | tee "$S/build.json"
  [ -n "${SFT_LIMIT:-}" ] && { head -n "$SFT_LIMIT" "$S/sft.jsonl" > "$S/sft.tmp"; mv "$S/sft.tmp" "$S/sft.jsonl"; }
  say "sft_data: $(cat "$S/build.json")"
fi

# 4. SFT (Zro-form server, phased restart profile) ------------------------------------------------------------
STAGE=sft
sft_ckpt() { [ -f "$S/metrics.jsonl" ] && python3 -c "
import json,sys
c=[json.loads(l).get('ckpt') for l in open('$S/metrics.jsonl')]
c=[x for x in c if x and 'sft_final' in x]; print(c[-1] if c else '')" || true; }
if [ -z "$(sft_ckpt)" ]; then
  say "sft: start"; server_up
  $PY scripts/hip_sft.py --sft "$S/sft.jsonl" --out "$S" --epochs 1 ${SFT_LIMIT:+--batch 4 --save-every 1} > "$S/train.log" 2>&1
  say "sft: done, $(sft_ckpt)"
fi
SFT=$(sft_ckpt)

# 5. eval of the SFT model on held-out A (200 x 4) ------------------------------------------------------------
STAGE=sft_eval; E=$R/sft_eval
if [ ! -s "$E/evals.jsonl" ]; then
  say "sft_eval: start"; server_up
  $PY scripts/hip_rl.py --data "$DATA" --out "$E" --init "$SFT" --eval-only --max-tokens "$EVAL_TOKENS" $EVAL_ARGS > "$E.log" 2>&1
  say "sft_eval: $(tail -1 "$E/evals.jsonl")"
fi

# 6. RL pilot ---------------------------------------------------------------------------------------------------
STAGE=rl_pilot; L=$R/rl
RLARGS=(--data "$DATA" --out "$L" --store-dir "$R" --task-records "$REFS/refs.jsonl" --exclude "$R/exclude.txt" --groups "${RL_GROUPS:-25}" --k 4 $EVAL_ARGS)
if [ "$(lines "$L/metrics.jsonl")" -lt "$PILOT" ]; then
  say "rl_pilot: start"; server_up
  if [ "$(lines "$L/metrics.jsonl")" -eq 0 ]; then
    $PY scripts/hip_rl.py "${RLARGS[@]}" --init "$SFT" --steps "$PILOT" --keep-every "$PILOT" --eval-every "$PILOT" \
      --max-tokens "$TOKENS1" >> "$L.log" 2>&1
  else
    say "rl_pilot: partial run found; resume by hand (hip_rl.py --resume CK --start-step N)"; exit 1
  fi
fi

# 6b. base-model eval on held-out A through the same server, sampler and grader as sft_eval ---------------------
STAGE=base_eval; BE=$R/base_eval
if [ ! -s "$BE/evals.jsonl" ]; then
  say "base_eval: start"; server_up
  $PY scripts/hip_rl.py --data "$DATA" --out "$BE" --eval-only --max-tokens "$EVAL_TOKENS" $EVAL_ARGS > "$BE.log" 2>&1
  say "base_eval: $(tail -1 "$BE/evals.jsonl")"
fi

# 7. gate -------------------------------------------------------------------------------------------------------
STAGE=gate
if ! python3 -I scripts/check_pilot.py --run "$L" --baseline-eval "$E/evals.jsonl" | tee -a "$R/STATUS"; then
  if [ "${GATE:-stop}" = warn ]; then say "gate: would stop (ignored, GATE=warn)"
  else say "gate: STOP, pilot looks wrong; check by hand before continuing"; server_down; exit 2; fi
fi

# 8. RL main: steps up to SWITCH at 12,288 tokens, then 14,000 (the 16k context minus the prompt) -----------------
rl_main_to() {   # $1 last step (exclusive), $2 answer budget
  STAGE=rl_main
  DONE=$(lines "$L/metrics.jsonl")
  [ "$DONE" -lt "$1" ] || return 0
  CK=$(python3 -c "
import json
c=[json.loads(l).get('ckpt') for l in open('$L/metrics.jsonl')]; c=[x for x in c if x]; print(c[-1])")
  LAST=$(python3 -c "
import json
m=[json.loads(l) for l in open('$L/metrics.jsonl')]; c=[x['step'] for x in m if x.get('ckpt')]; print(c[-1]+1)")
  # steps after the last checkpoint (the server died before the next save, e.g. a GPU hang) are re-run: move
  # their metrics rows and step dirs aside so metrics.jsonl stays one row per step
  python3 - "$L" "$LAST" <<'EOF'
import json, sys, time
from pathlib import Path
L, last = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(l) for l in open(L / "metrics.jsonl")]
late = [r for r in rows if r["step"] >= last]
if late or any(int(d.name) >= last for d in (L / "steps").iterdir()):
    aside = L / f"superseded_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
    (aside / "steps").mkdir(parents=True)
    with open(aside / "metrics.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in late)
    with open(L / "metrics.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows if r["step"] < last)
    for d in sorted((L / "steps").iterdir()):
        if int(d.name) >= last:
            d.rename(aside / "steps" / d.name)
    print(f"moved {len(late)} metrics rows and later step dirs to {aside}")
EOF
  say "rl_main: resume from step $LAST ($CK)"; server_up
  # save every step (~52 s of ~40 min): a GPU hang kills the server, and with it everything since the last save
  $PY scripts/hip_rl.py "${RLARGS[@]}" --resume "$CK" --start-step "$LAST" --steps "$1" --eval-every 25 \
    --save-every "${SAVE_EVERY:-1}" --max-tokens "$2" >> "$L.log" 2>&1
}
rl_main_to "$SWITCH" "$TOKENS1"
rl_main_to "$MAIN" "$TOKENS2"
say "all stages done"; server_down
