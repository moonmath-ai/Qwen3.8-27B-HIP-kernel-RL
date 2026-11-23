"""Paired comparison of two eval_bench.py runs (IFEval, MMLU-Pro, MATH-500; greedy) on the same examples: score,
paired bootstrap 95% CI over examples (5,000 resamples, seed 0), examples better / worse.
Each argument is a bench output directory holding <bench>/trajectories.jsonl (e.g. RUN/regress/base_bench/bench_base).
  python3 -I scripts/compare_bench.py BASE_DIR OTHER_DIR
"""
import json
import random
import sys


def load(path: str) -> dict:
    return {r["example_id"]: r["reward"] for r in map(json.loads, open(path))}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    out = {}
    for b in ("ifeval", "mmlu_pro", "math500"):
        try:
            x, y = (load(f"{d}/{b}/trajectories.jsonl") for d in sys.argv[1:])
        except FileNotFoundError:
            continue
        keys = sorted(set(x) & set(y)); d = [y[k] - x[k] for k in keys]
        rng = random.Random(0)
        boot = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
        out[b] = {"examples": len(keys), "base": round(sum(x[k] for k in keys) / len(keys), 3),
                  "other": round(sum(y[k] for k in keys) / len(keys), 3), "gain_points": round(100 * sum(d) / len(d), 1),
                  "ci95_points": [round(100 * boot[124], 1), round(100 * boot[4874], 1)],
                  "better": sum(v > 0 for v in d), "worse": sum(v < 0 for v in d)}
    print(json.dumps(out, indent=1))
