#!/usr/bin/env bash
# Pipeline smoke test (~1.5-2 h on one MI300X): every stage of the run at toy size, then the eval, export, import,
# release-check and regression paths. Checks that the scripts work end to end; the numbers mean nothing.
#   1. full_run.sh with SMOKE=1 under the supervisor: 24 references, own answers on them, SFT on 8 rows,
#      2 pilot RL steps, base eval, gate (warn only), RL steps 3-4 (resume), 2,048-3,072-token answers
#   2. evals.sh on the last checkpoint (2 validation tasks at 14k, 2 held-out A tasks at 20k with the 24k profile)
#   3. export_hf.py -> import_adapter.py -> evals.sh --sampler-path (the released-adapter path, REPRODUCE.md §7)
#   4. check_release.sh on the export (training server, Transformers + PEFT, stock SGLang, build snippet)
#   5. regress.sh with 5 problems per benchmark
# Usage: scripts/smoke.sh RUN_DIR [CHECK_EVAL_DIR]
#   CHECK_EVAL_DIR: a held-out A eval with finished, correct answers for check_release.sh (default: this smoke's 20k eval,
#   which may have none at this size; pass e.g. a real run's rl/evals/step0075).
set -uo pipefail
R=$(readlink -f "${1:?run dir}"); mkdir -p "$R"
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"; . scripts/env.sh
S=$R/STATUS; fails=0
say() { echo "[$(date -u +%FT%TZ)] smoke: $*" | tee -a "$S"; }
ok() { say "PASS $1"; }; bad() { say "FAIL $1"; fails=$((fails + 1)); }

say "1. full_run.sh SMOKE=1 under the supervisor"
SESSION=zf_smoke_full FULL_RUN="env SMOKE=1 scripts/full_run.sh" scripts/full_run_supervised.sh "$R" \
  && grep -q "all stages done" "$S" && ok "training pipeline" || { bad "training pipeline (see $S, $R.log)"; exit 1; }
CK=$(python3 -c "
import json
c=[json.loads(l).get('ckpt') for l in open('$R/rl/metrics.jsonl')]; print([x for x in c if x][-1])")
STORE=$R/server/store/${CK#tinker://}
say "last checkpoint $CK; $(wc -l < "$R/rl/metrics.jsonl") RL steps; resumed: $(grep -c 'rl_main: resume' "$S")"

say "2. evals.sh on the checkpoint"
EVAL_LIMIT=2 SETS="val14k ha20k" scripts/evals.sh "$R" smoke --resume "$CK"
for x in val14k ha20k; do [ -s "$R/final/smoke_$x/evals.jsonl" ] && ok "eval $x" || bad "eval $x"; done

say "3. export -> import -> eval of the imported adapter"
scripts/server_python.sh "$R/server:$R/hf" scripts/export_hf.py --src "$STORE" --out "$R/hf" --note "smoke" \
  > "$R/export.log" 2>&1 && [ -s "$R/hf/adapter_model.safetensors" ] && ok export || bad "export (see $R/export.log)"
SP=$(ZROFORM_DATA=$R/server scripts/server_python.sh "$R/server:$R/hf" scripts/import_adapter.py \
  --name smoke_hf --src "$R/hf" 2> "$R/import.log" | tail -1)
[ "$SP" = tinker://imported/sampler_weights/smoke_hf ] && ok "import ($SP)" || bad "import (see $R/import.log)"
EVAL_LIMIT=2 SETS=val14k scripts/evals.sh "$R" imported --sampler-path "$SP"
[ -s "$R/final/imported_val14k/evals.jsonl" ] && ok "eval of the imported adapter" || bad "eval of the imported adapter"

say "4. release checks"
scripts/check_release.sh "$R" "$CK" "${2:-$R/final/smoke_ha20k/evals/init}" "$R/hf" "$R/checks"
grep -q "ref logprobs: done" "$R/checks/STATUS" && grep -q "peft logprobs: done" "$R/checks/STATUS" \
  && grep -q "sglang logprobs: done" "$R/checks/STATUS" && ok "logprob checks" || bad "logprob checks (see $R/checks/STATUS)"
grep -q "build snippet: ok" "$R/checks/STATUS" && ok "build snippet" || bad "build snippet (see $R/checks/build_check.log)"

say "5. regressions, 5 problems each"
REGRESS_LIMIT=5 scripts/regress.sh "$R" smoke "$CK"
n=$(ls "$R"/regress/smoke_{humanevalplus,mbppplus}/summary.json "$R"/regress/smoke_bench/bench.json 2>/dev/null | wc -l)
[ "$n" = 3 ] && ! grep -q FAILED "$R/regress/smoke.log" && ok regressions || bad "regressions (see $R/regress/smoke.log)"

say "done: $fails failed"; [ $fails = 0 ]
