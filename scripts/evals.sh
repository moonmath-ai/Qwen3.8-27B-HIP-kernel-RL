#!/usr/bin/env bash
# The kernel evals of RESULTS.md for one model: pass@1 over 4 samples at temperature 1, graded like training.
#   val14k  validation (100 tasks) at 14,000 tokens      ha14k  held-out A (200 tasks) at 14,000 tokens
#   ha20k   held-out A at 20,000 tokens (server restarted with the 24k-context profile)
# Usage: scripts/evals.sh RUN_DIR LABEL [hip_rl weight args]
#   RUN_DIR/server is the server's store (the run's own checkpoints, or an imported adapter); results go to
#   RUN_DIR/final/<LABEL>_<set>/evals/init/{rollouts,graded}.jsonl and .../evals.jsonl.
#   no weight args = the base model;  --resume tinker://<session>/weights/keep0075 = a run checkpoint;
#   --sampler-path tinker://imported/sampler_weights/<name> = an imported adapter (REPRODUCE.md §7)
# Env: SETS (default "val14k ha14k ha20k"), EVAL_LIMIT (first N tasks only, for a quick check).
# Starts the server, runs the sets, stops the server (frees the GPU). ~2-3 h per set on one MI300X.
set -uo pipefail
R=$(readlink -f "${1:?run dir}"); LABEL=${2:?label}; shift 2; W=("$@")
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"; . scripts/env.sh
export HF_HUB_OFFLINE=1; PY=$ZF_CLIENT_VENV/bin/python; DATA=${DATA:-$AIG_DATA}
F=$R/final; S=$R/STATUS; SETS=${SETS:-"val14k ha14k ha20k"}; mkdir -p "$F"
P16=profiles/mi300x-qwen38-27b-phased.yaml; P24=profiles/mi300x-qwen38-27b-phased-24k.yaml
say() { echo "[$(date -u +%FT%TZ)] evals $LABEL: $*" | tee -a "$S"; }
server_stop() { docker ps -a --format '{{.Names}}' | grep -qx zf_eval_server || return 0
  scripts/stop_server.sh zf_eval_server "$R/server" >/dev/null; sleep 10; }
server_start() {   # $1 profile
  scripts/run_server.sh "$1" "$R/server" zf_eval_server >/dev/null
  for _ in $(seq 120); do curl -sf -m 5 http://127.0.0.1:8100/api/v1/healthz >/dev/null && return 0; sleep 10; done
  say "server did not come up with $1"; server_stop; exit 1; }
ev() {   # $1 set name, rest: hip_rl args
  local n=${LABEL}_$1; shift; say "$n: start"
  if $PY scripts/hip_rl.py --data "$DATA" --out "$F/$n" --eval-only ${EVAL_LIMIT:+--eval-limit $EVAL_LIMIT} "$@" "${W[@]}" \
       > "$F/$n.log" 2>&1; then say "$n: $(tail -1 "$F/$n/evals.jsonl" | cut -c1-400)"
  else say "$n: failed (rc $?), see $F/$n.log"; fi; }
trap server_stop EXIT
case " $SETS " in *" val14k "*|*" ha14k "*) server_start "$P16" ;; esac
case " $SETS " in *" val14k "*) ev val14k --eval-split validation --max-tokens 14000 ;; esac
case " $SETS " in *" ha14k "*) ev ha14k --eval-split heldout_a --max-tokens 14000 ;; esac
case " $SETS " in *" ha20k "*) server_stop; server_start "$P24"
  ev ha20k --eval-split heldout_a --max-seq 24576 --max-tokens 20000 ;; esac
say "done; server stopped, GPU free"
