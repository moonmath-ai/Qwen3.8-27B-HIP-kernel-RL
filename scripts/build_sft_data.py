"""SFT data for PyTorch -> HIP (RESULTS.md §4): the base model's own correct answers + dataset references.

Inputs (all graded by hip_grade.py on the MI300X):
  --own-rollouts/--own-graded   the base model's samples on train tasks (sample_sglang.py --split train)
  --refs/--refs-graded          the dataset's reference answers, one per train task (make_ref_rollouts.py)
Selection:
  - own: per task up to --own-per-task correct, untruncated samples (their full text: reasoning + answer);
  - refs: only for tasks that were sampled but never solved (the hard ones), with an empty reasoning block, capped
    at --ref-ratio x the number of own examples so the model is not taught to skip thinking;
  - tasks whose correct reference spends > --lib-share of its GPU time in non-candidate kernels are written to
    --exclude-out (nothing to win: RL skips them, SFT doesn't use them).
Output rows: {"task", "source": "own"|"ref", "record_id", "user": <prompt text>, "completion": <text after the
generation prompt's "<think>\\n">}.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sample_sglang", HERE / "sample_sglang.py")
smp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smp)

EMPTY_THINK = "\n</think>\n\n"


def jl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def lib_share(o: dict) -> float | None:
    own, other = o.get("own_us") or 0.0, o.get("other_us") or 0.0
    return other / (own + other) if own + other > 0 else None


def select(own_rows, own_graded, ref_rows, ref_graded, own_per_task=2, ref_ratio=1.0, lib_max=0.9, seed=0):
    """Pure selection logic: returns (examples, excluded task ids). Rows and graded outcomes align by index."""
    rng = random.Random(seed)
    exclude = set()
    ref_ok = {}
    for r, o in zip(ref_rows, ref_graded):
        if o.get("correct") and o.get("own_kernels"):
            ls = lib_share(o)
            if ls is not None and ls > lib_max:
                exclude.add(r["task"])
            else:
                ref_ok[r["task"]] = r
    by_task = defaultdict(list)
    sampled = set()
    for r, o in zip(own_rows, own_graded):
        sampled.add(r["task"])
        if o.get("correct") and o.get("own_kernels") and r.get("finish") != "length" and r["task"] not in exclude:
            by_task[r["task"]].append(r)
    own = []
    for task, rs in sorted(by_task.items()):
        for r in rs[:own_per_task]:
            own.append({"task": task, "source": "own", "record_id": r.get("record_id"), "completion": r["response"]})
    unsolved = sorted(t for t in sampled if t not in by_task and t in ref_ok)
    rng.shuffle(unsolved)
    refs = []
    for task in unsolved[:int(ref_ratio * len(own))]:
        r = ref_ok[task]
        code = ref_code(r["response"])
        refs.append({"task": task, "source": "ref", "record_id": r["id"],
                     "completion": EMPTY_THINK + "```cpp\n" + code + "\n```"})
    return own + refs, exclude


def ref_code(resp: str) -> str:
    s = resp.strip()
    if s.startswith("```cpp\n") and s.endswith("```"):
        return s[len("```cpp\n"):-3].rstrip("\n")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--own-rollouts", required=True)
    ap.add_argument("--own-graded", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--refs-graded", required=True)
    ap.add_argument("--own-per-task", type=int, default=2)
    ap.add_argument("--ref-ratio", type=float, default=1.0)
    ap.add_argument("--lib-share", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-out", required=True)
    a = ap.parse_args()
    own_rows, own_g = jl(a.own_rollouts), {o["id"]: o for o in jl(a.own_graded)}
    ref_rows, ref_g = jl(a.refs), {o["id"]: o for o in jl(a.refs_graded)}
    own_rows = [r for r in own_rows if r["id"] in own_g]
    ex, exclude = select(own_rows, [own_g[r["id"]] for r in own_rows], ref_rows, [ref_g[r["id"]] for r in ref_rows],
                         a.own_per_task, a.ref_ratio, a.lib_share, a.seed)
    recs = {}
    for p in (smp.CUDAAGENT, smp.AICUDA):
        for r in json.load(open(os.path.join(a.data, p), encoding="utf-8")):
            recs[r["id"]] = r
    with open(a.out, "w") as f:
        for e in ex:
            e["user"] = smp.prompt(recs[e["record_id"]])
            f.write(json.dumps(e) + "\n")
    Path(a.exclude_out).write_text("".join(t + "\n" for t in sorted(exclude)))
    n_own = sum(e["source"] == "own" for e in ex)
    print(json.dumps({"examples": len(ex), "own": n_own, "ref": len(ex) - n_own, "own_tasks":
                      len({e["task"] for e in ex if e["source"] == "own"}), "excluded_tasks": len(exclude)}))


if __name__ == "__main__":
    main()
