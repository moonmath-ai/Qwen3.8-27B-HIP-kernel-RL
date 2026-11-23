"""Gate between the RL pilot and the main run (continue automatically unless something looks
wrong). Reads <run>/metrics.jsonl and evals.jsonl; exit 0 = continue, 1 = stop (reasons printed).

Stops if, over the pilot: the reward trend is clearly down (mean of the last third < first third - 0.1); format_ok
fell by > 0.15 (answers stop arriving: truncation or broken format); the library-time share among correct answers
rose above 0.3 (reward hacking via library calls); a step had no datums for 3 steps in a row (no learning signal);
or the eval after the pilot is more than 5 points below the eval before it (--baseline-eval).
"""
import argparse
import os
import json
import sys


def check(metrics: list[dict], eval_before: float | None, eval_after: float | None) -> list[str]:
    bad = []
    n = len(metrics)
    if n < 6:
        return [f"only {n} steps logged"]
    third = max(n // 3, 1)
    first = sum(m["reward_mean"] for m in metrics[:third]) / third
    last = sum(m["reward_mean"] for m in metrics[-third:]) / third
    if last < first - 0.1:
        bad.append(f"reward down: first third {first:.3f}, last third {last:.3f}")
    f0 = sum(m["format_ok"] for m in metrics[:third]) / third
    f1 = sum(m["format_ok"] for m in metrics[-third:]) / third
    if f1 < f0 - 0.15:
        bad.append(f"format_ok down: {f0:.2f} -> {f1:.2f}")
    shares = [m["other_time_share"] for m in metrics[-third:] if m.get("other_time_share") is not None]
    if shares and max(shares) > 0.3:
        bad.append(f"library time share up to {max(shares):.2f}")
    run = 0
    for m in metrics:
        run = run + 1 if not m.get("datums") else 0
        if run >= 3:
            bad.append("3 steps in a row without training data")
            break
    if eval_before is not None and eval_after is not None and eval_after < eval_before - 0.05:
        bad.append(f"held-out correct fell: {eval_before:.3f} -> {eval_after:.3f}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline-eval", help="evals.jsonl of the starting point (SFT eval)")
    a = ap.parse_args()
    metrics = [json.loads(line) for line in open(f"{a.run}/metrics.jsonl")]
    ef = f"{a.run}/evals.jsonl"
    evals = [json.loads(line) for line in open(ef)] if os.path.exists(ef) else []
    before = None
    if a.baseline_eval:
        before = [json.loads(line) for line in open(a.baseline_eval)][-1]["correct"]
    after = evals[-1]["correct"] if evals else None
    bad = check(metrics, before, after)
    print(json.dumps({"steps": len(metrics), "eval_before": before, "eval_after": after, "problems": bad}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
