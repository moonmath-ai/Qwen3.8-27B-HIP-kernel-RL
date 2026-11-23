"""Grader input from the dataset's own reference answers (a sanity check of hip_grade.py, and later SFT-data filtering).

Picks N train-split tasks (data/split_manifest.json) with a fixed seed, one reference record per task, and writes
hip_grade.py rows: {"id", "task", "subset", "difficulty", "module", "functional", "response"}, the response being the
record's last answer file in a ```cpp block.

  python3 -I scripts/make_ref_rollouts.py --data <AIG-Datasets>/v0.2 --n 24 --seed 0 --out refs.jsonl
  (--n 0: every train task)
"""
import argparse
import importlib.util
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("split", HERE / "split.py")
split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split)


def code_of(rec: dict, version: str) -> str | None:
    if version == "module":
        return split.module_code(rec)
    return next((f["content"] for f in rec.get("benchmark_code", []) if f.get("version") == "functional"), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", default=str(HERE.parent / "data/split_manifest.json"))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    train = {t["task"] for t in json.load(open(a.manifest))["train"]}
    recs, _ = split.load(a.data)
    groups = [g for g in split.group(recs) if split.describe(g)["task"] in train]
    rng = random.Random(a.seed)
    picked = groups if a.n == 0 else rng.sample(groups, a.n)
    by_id = {}
    for subset, path in (("aicuda", split.AICUDA), ("cudaagent", split.CUDAAGENT)):
        for r in json.load(open(os.path.join(a.data, path), encoding="utf-8")):
            by_id[r["id"]] = r
    with open(a.out, "w") as f:
        for g in picked:
            meta = rng.choice(sorted(g, key=lambda r: r["id"]))
            r = by_id[meta["id"]]
            f.write(json.dumps({"id": r["id"], "task": split.describe(g)["task"], "subset": meta["subset"],
                                "difficulty": meta["difficulty"], "module": code_of(r, "module"),
                                "functional": code_of(r, "functional"),
                                "response": "```cpp\n" + r["answer_code"][-1]["content"] + "\n```"}) + "\n")
    print(len(picked), "rows ->", a.out)


if __name__ == "__main__":
    main()
