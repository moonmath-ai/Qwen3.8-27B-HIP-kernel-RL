#!/usr/bin/env bash
# Regression eval queue for one model (the same protocol as the DeepCoder RL release): HumanEval+ x2 and MBPP+ x2 through the safe
# grader (eval_regress.py), then IFEval, MMLU-Pro, MATH-500 (300 each, greedy) via eval_bench.py.
#   scripts/eval_queue_regress.sh <port> <out_dir> <label> [sampler_path]   (no sampler_path = the base model)
#   REGRESS_LIMIT=N: first N problems of each benchmark only (smoke test)
set -uo pipefail
port=$1 out=$2 label=$3 sp=${4:-}
. "$(dirname "$0")/env.sh"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
# Everything is cached; offline avoids the Hub rate limit (429) when several evals start at once.
# eval_bench is the exception: the cookbook's load_dataset(<repo>) cannot resolve hub-cache-only datasets offline.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$(dirname "$0")/.."
export ZF_GRADER_CONCURRENCY=48
for b in humanevalplus mbppplus; do
  echo "[$(date -Is)] $label: $b x2"
  sg docker -c "$ZF_CLIENT_VENV/bin/python scripts/eval_regress.py run --bench $b --n 2 ${sp:+--sampler-path $sp} ${REGRESS_LIMIT:+--limit $REGRESS_LIMIT} \
      --base-url http://127.0.0.1:$port --out $out/${label}_$b" || echo "[$(date -Is)] $label: $b FAILED"
done
echo "[$(date -Is)] $label: ifeval, mmlu_pro, math500 (300 each, greedy)"
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "$ZF_CLIENT_VENV/bin/python" scripts/eval_bench.py --base-url "http://127.0.0.1:$port" --model Qwen/Qwen3.8-27B \
    --renderer qwen3_8_low_reasoning --max-tokens 12288 --max-examples "${REGRESS_LIMIT:-300}" --bench ifeval mmlu_pro math500 \
    ${sp:+--adapter $label=$sp --skip-base} --out "$out/${label}_bench/bench.json" || echo "[$(date -Is)] $label: eval_bench FAILED"
echo "[$(date -Is)] $label: regression queue done"
touch "$out/$label.regress.done"
