#!/usr/bin/env bash
# Regression benchmarks for one model, same protocol as the DeepCoder RL release: HumanEval+ (157) and MBPP+ (375)
# x2 at temperature 1 through the sandboxed grader; IFEval, MMLU-Pro, MATH-500 (300 each, greedy); 12,288 tokens.
# Usage: scripts/regress.sh RUN_DIR LABEL [sampler_path]    (no sampler_path = the base model)
#   sampler_path: tinker://<session>/weights/keep0075 (a run checkpoint) or tinker://imported/sampler_weights/<name>
# Outputs RUN_DIR/regress/<LABEL>_{humanevalplus,mbppplus}/, <LABEL>_bench/bench.json. Compare two labels with
#   scripts/eval_regress.py compare (HumanEval+/MBPP+) and scripts/compare_bench.py (the others).
# Needs the zf-code-runner:1 image and the cached datasets (REPRODUCE.md §2-3). ~35 min per model on one MI300X.
# REGRESS_LIMIT=N runs the first N problems of each benchmark only (smoke test).
set -uo pipefail
R=$(readlink -f "${1:?run dir}"); LABEL=${2:?label}; SP=${3:-}
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"; . scripts/env.sh
O=$R/regress; S=$R/STATUS; mkdir -p "$O"
say() { echo "[$(date -u +%FT%TZ)] regress $LABEL: $*" | tee -a "$S"; }
server_stop() { docker ps -a --format '{{.Names}}' | grep -qx zf_eval_server || return 0
  scripts/stop_server.sh zf_eval_server "$R/server" >/dev/null; sleep 10; }
trap server_stop EXIT
scripts/run_server.sh profiles/mi300x-qwen38-27b-phased.yaml "$R/server" zf_eval_server >/dev/null
up=0; for _ in $(seq 120); do curl -sf -m 5 http://127.0.0.1:8100/api/v1/healthz >/dev/null && { up=1; break; }; sleep 10; done
[ $up = 1 ] || { say "server did not come up"; exit 1; }
say "start"
scripts/eval_queue_regress.sh 8100 "$O" "$LABEL" $SP > "$O/$LABEL.log" 2>&1
say "$(grep -cE FAILED "$O/$LABEL.log") failed steps; $(cat "$O"/${LABEL}_*/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"
