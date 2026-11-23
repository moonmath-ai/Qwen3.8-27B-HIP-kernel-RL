"""Paired comparison of two kernel evals on the same tasks (RESULTS.md): correct rate, paired bootstrap 95% CI
over tasks (5,000 resamples, seed 0), tasks up / down, by level, and the other per-answer numbers of the model card.
Each argument is an eval directory with rollouts.jsonl + graded.jsonl (e.g. RUN/final/base_ha14k/evals/init).
  python3 -I scripts/compare.py BASE_DIR OTHER_DIR
"""
import json
import random
import statistics
import sys


def load(d: str) -> dict:
    rows = {}
    for line in open(f"{d}/graded.jsonl"):
        r = json.loads(line)
        rows[r["id"].split("_", 1)[1]] = r          # id = e<label>_<task>_<sample>
    for line in open(f"{d}/rollouts.jsonl"):
        r = json.loads(line)
        k = r["id"].split("_", 1)[1]
        rows[k].update(finish=r["finish"], level=r["difficulty"], task=r["task"])
    return rows


def by_task(rows: dict) -> dict:
    t: dict = {}
    for r in rows.values():
        t.setdefault(r["task"], []).append(float(bool(r.get("correct"))))
    return {k: sum(v) / len(v) for k, v in t.items()}


def summary(rows: dict) -> dict:
    v = list(rows.values()); n = len(v)
    sp = [r["speedup"] for r in v if r.get("correct") and r.get("speedup")]
    lv: dict = {}
    for r in v:
        lv.setdefault(r["level"], []).append(bool(r.get("correct")))
    return {"answers": n, "correct": round(sum(bool(r.get("correct")) for r in v) / n, 3),
            "compiled": round(sum(bool(r.get("compiled")) for r in v) / n, 3),
            "truncated": round(sum(r["finish"] == "length" for r in v) / n, 3),
            "correct_and_ge_1.1x": sum(s >= 1.1 for s in sp),
            "median_speedup_correct": round(statistics.median(sp), 2) if sp else None,
            "correct_by_level": {k: round(sum(x) / len(x), 3) for k, x in sorted(lv.items())}}


def paired(a: dict, b: dict, n_boot: int = 5000, seed: int = 0) -> dict:
    ta, tb = by_task(a), by_task(b)
    keys = sorted(set(ta) & set(tb))
    d = [tb[k] - ta[k] for k in keys]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {"tasks": len(keys), "gain_points": round(100 * sum(d) / len(d), 1),
            "ci95_points": [round(100 * boot[int(0.025 * n_boot) - 1], 1), round(100 * boot[int(0.975 * n_boot) - 1], 1)],
            "tasks_up": sum(x > 0 for x in d), "tasks_down": sum(x < 0 for x in d)}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    print(json.dumps({"base": summary(a), "other": summary(b), "other_vs_base": paired(a, b)}, indent=1))
