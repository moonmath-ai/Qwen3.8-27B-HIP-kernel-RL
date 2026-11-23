"""PyTorch -> HIP RL task split (held-out A, validation, train), deterministic.

Writes data/split_manifest.json (hashes, record ids and op names; no code text).

Source: amd/AIG-Datasets v0.2, two subsets with a PyTorch reference (needed for the reward):
  - HIP-AICUDA: 14,282 records but only ~187 distinct KernelBench tasks (many agent kernels per task, each
    as pytorch_to_hip and cuda_to_hip). KernelBench is public, so these tasks are "likely seen".
  - HIP-CudaAgent: 5,388 synthetic multi-op tasks; only verified_status == "pass" is used.
The unit of splitting is the underlying PyTorch task, never the record. Records are grouped (union-find) when
they share any of: the normalised module code (comments/docstrings/whitespace removed), the same code with
numeric literals masked (same op at other sizes), or the same CudaAgent op name (same op combination).

Splits:
  - held-out A (headline): 200 CudaAgent groups from the upstream "val" split, stratified L1/L2/L3;
  - held-out KB (secondary, comparable to AMD's KernelBench eval, contamination caveat): 30 AICUDA groups;
  - validation: 100 CudaAgent groups from upstream "val";
  - training: every other group (both subsets).
Drawing CudaAgent held-out/validation from the upstream val split keeps them clear of anything trained on the
upstream train split (e.g. AMD's SFT data).

  python3 -I scripts/split.py --data <AIG-Datasets>/v0.2 [--out data/split_manifest.json]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import random
import re
import tokenize

AICUDA = "pytorch_hip_kernel_aicuda_engineer/pytorch_hip_kernel.json"
CUDAAGENT = "pytorch_hip_kernel_cuda_agent_ops_6k/pytorch_hip_kernel_cuda_agent_ops_6k.json"
SEED = 20261110
HELDOUT_A = {"L1": 120, "L2": 50, "L3": 30}
VALIDATION = {"L1": 60, "L2": 25, "L3": 15}
HELDOUT_KB = 30  # AICUDA, spread over difficulties in proportion
_SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
         tokenize.ENDMARKER}


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def norm_code(src: str) -> str:
    """Python tokens joined by spaces, without comments, blank lines or statement-level strings (docstrings)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return re.sub(r"\s+", " ", src).strip()
    out, at_stmt_start = [], True
    for t in toks:
        if t.type in _SKIP:
            at_stmt_start = at_stmt_start or t.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
            continue
        if t.type == tokenize.STRING and at_stmt_start:
            continue
        out.append(t.string)
        at_stmt_start = False
    return " ".join(out)


def mask_numbers(s: str) -> str:
    return re.sub(r"\b\d+(\.\d+)?([eE][-+]?\d+)?\b", "N", s)


def module_code(rec: dict) -> str | None:
    for f in rec.get("question_code", []) + rec.get("benchmark_code", []):
        if f.get("version") == "module" or (f.get("language") == "python" and f.get("version") != "functional"):
            return f.get("content")
    return None


def has_functional(rec: dict) -> bool:
    return any(f.get("version") == "functional" and f.get("content") for f in rec.get("benchmark_code", []))


def load(data_dir: str) -> tuple[list[dict], dict]:
    """One entry per usable record: id, subset, difficulty, op, upstream split, grouping keys."""
    out, dropped = [], collections.Counter()
    for subset, path in (("aicuda", AICUDA), ("cudaagent", CUDAAGENT)):
        for r in json.load(open(os.path.join(data_dir, path), encoding="utf-8")):
            meta, extra = r.get("meta") or {}, (r.get("meta") or {}).get("extra") or {}
            if subset == "cudaagent" and extra.get("verified_status") != "pass":
                dropped[f"cudaagent_{extra.get('verified_status')}"] += 1
                continue
            code = module_code(r)
            if not code or not has_functional(r) or not r.get("answer_code"):
                dropped[f"{subset}_missing_reference_or_answer"] += 1
                continue
            n = norm_code(code)
            keys = [f"code:{h(n)}", f"masked:{h(mask_numbers(n))}"]
            if subset == "cudaagent":
                keys.append(f"op:{meta.get('operation')}")
            out.append({"id": r["id"], "subset": subset, "task_type": meta.get("task_type"),
                        "difficulty": meta.get("difficulty_level"), "op": meta.get("operation"),
                        "upstream_split": extra.get("split"), "keys": keys})
    return out, dict(dropped)


def group(records: list[dict]) -> list[list[dict]]:
    """Union-find over shared keys; groups sorted by their smallest key for determinism."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[str, int] = {}
    for i, r in enumerate(records):
        for k in r["keys"]:
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    groups = collections.defaultdict(list)
    for i, r in enumerate(records):
        groups[find(i)].append(r)
    return sorted(groups.values(), key=lambda g: min(k for r in g for k in r["keys"]))


def describe(g: list[dict]) -> dict:
    subsets = sorted({r["subset"] for r in g})
    diffs = collections.Counter(r["difficulty"] for r in g)
    ops = collections.Counter(r["op"] for r in g)
    kb = [o for o in ops if re.match(r"^\d+_\d+_", o or "")]  # AICUDA: "<kernel>_<KernelBench no.>_<name>"
    return {"task": h(min(k for r in g for k in r["keys"])), "subset": "+".join(subsets),
            "difficulty": diffs.most_common(1)[0][0], "op": (sorted(kb) or sorted(o or "" for o in ops))[0],
            "n_records": len(g), "n_reference_answers": sum(r["task_type"] == "pytorch_to_hip" for r in g),
            "has_functional_reference": True, "upstream_split": sorted({str(r["upstream_split"]) for r in g}),
            "record_ids": sorted(r["id"] for r in g)}


def pick(rng: random.Random, pool: list[dict], quota: dict[str, int]) -> list[dict]:
    out = []
    for d in sorted(quota):
        cand = [t for t in pool if t["difficulty"] == d]
        if len(cand) < quota[d]:
            raise SystemExit(f"only {len(cand)} {d} tasks for a quota of {quota[d]}")
        out += rng.sample(cand, quota[d])
    return out


def split(data_dir: str) -> dict:
    records, dropped = load(data_dir)
    tasks = [describe(g) for g in group(records)]
    rng = random.Random(SEED)
    ca_val = [t for t in tasks if t["subset"] == "cudaagent" and t["upstream_split"] == ["val"]]
    heldout_a = pick(rng, ca_val, HELDOUT_A)
    taken = {t["task"] for t in heldout_a}
    validation = pick(rng, [t for t in ca_val if t["task"] not in taken], VALIDATION)
    taken |= {t["task"] for t in validation}
    kb = [t for t in tasks if t["subset"] == "aicuda"]
    by_d = collections.Counter(t["difficulty"] for t in kb)
    kb_quota = {d: max(1, round(HELDOUT_KB * n / len(kb))) for d, n in by_d.items()}
    heldout_kb = pick(rng, kb, kb_quota)
    taken |= {t["task"] for t in heldout_kb}
    train = [t for t in tasks if t["task"] not in taken]

    def stats(ts):
        return {"tasks": len(ts), "records": sum(t["n_records"] for t in ts),
                "by_subset_difficulty": dict(sorted(collections.Counter(f"{t['subset']}/{t['difficulty']}"
                                                                        for t in ts).items()))}

    strip = lambda ts, ids: [{k: v for k, v in t.items() if ids or k != "record_ids"} for t in ts]
    return {
        "version": 1, "dataset": "amd/AIG-Datasets v0.2", "seed": SEED,
        "files": {p: hashlib.sha256(open(os.path.join(data_dir, p), "rb").read()).hexdigest()
                  for p in (AICUDA, CUDAAGENT)},
        "rules": {"unit": "PyTorch task: union of records sharing normalised module code, numeric-masked code, "
                          "or CudaAgent op name",
                  "cudaagent_filter": "verified_status == pass",
                  "heldout_a": HELDOUT_A, "validation": VALIDATION, "heldout_kb": kb_quota,
                  "heldout_and_validation_source": "CudaAgent upstream split == val"},
        "counts": {"records_used": len(records), "records_dropped": dropped, "tasks": len(tasks),
                   "tasks_by_subset": dict(collections.Counter(t["subset"] for t in tasks)),
                   "tasks_spanning_both_subsets": sum("+" in t["subset"] for t in tasks),
                   "heldout_a": stats(heldout_a), "heldout_kb": stats(heldout_kb),
                   "validation": stats(validation), "train": stats(train)},
        "heldout_a": strip(heldout_a, True), "heldout_kb": strip(heldout_kb, True),
        "validation": strip(validation, True), "train": strip(train, False),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="AIG-Datasets v0.2 directory")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../data/split_manifest.json"))
    a = ap.parse_args()
    m = split(a.data)
    with open(a.out, "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps(m["counts"], indent=1))
    print("wrote", os.path.normpath(a.out))


if __name__ == "__main__":
    main()
