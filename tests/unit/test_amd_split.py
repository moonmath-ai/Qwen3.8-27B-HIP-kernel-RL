"""scripts/split.py on a tiny synthetic AIG-Datasets tree: grouping by task, determinism, disjoint splits."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("amd_split", Path(__file__).parents[2] / "scripts/split.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

MODULE = '''import torch
class Model(torch.nn.Module):
    """{doc}"""
    def forward(self, x):
        return x.{k}() * 2  # {comment}
batch = {n}
'''


def rec(i, subset, diff, k, n=16, op=None, split="val", status="pass", doc="d", comment="c"):
    code = MODULE.format(doc=doc, k=k, comment=comment, n=n)
    meta = {"difficulty_level": diff, "operation": op or f"{i}_{k}_op", "task_type": "pytorch_to_hip",
            "extra": {"split": split, "verified_status": status} if subset == "cudaagent" else {}}
    return {"id": f"{subset}-{i}", "question_code": [{"path": "m.py", "version": "module", "content": code}],
            "benchmark_code": [{"path": "f.py", "version": "functional", "content": "def module_fn(x): ..."}],
            "answer_code": [{"path": "k.hip", "content": "// hip"}], "meta": meta}


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "HELDOUT_A", {"L1": 3, "L2": 1, "L3": 1})
    monkeypatch.setattr(sp, "VALIDATION", {"L1": 2, "L2": 1, "L3": 1})
    monkeypatch.setattr(sp, "HELDOUT_KB", 2)
    ca, i = [], 0
    for diff, n_tasks in (("L1", 12), ("L2", 4), ("L3", 4)):
        for t in range(n_tasks):
            ca.append(rec(i, "cudaagent", diff, k=f"op_{diff}_{t}", op=f"fused_{diff}_{t}")); i += 1
    # Same task as cudaagent-0 at another size and with other comments: must join its group.
    ca.append(rec(i, "cudaagent", "L1", k="op_L1_0", n=64, op="other_name", doc="x", comment="y")); i += 1
    ca.append(rec(i, "cudaagent", "L1", k="op_bad", status="mismatch")); i += 1  # dropped
    ai = [rec(j, "aicuda", "L1" if j % 2 else "L2", k=f"kb_{j % 4}", op=f"{j}_{j % 4}_kb") for j in range(16)]
    (tmp_path / Path(sp.AICUDA).parent).mkdir(parents=True)
    (tmp_path / Path(sp.CUDAAGENT).parent).mkdir(parents=True)
    (tmp_path / sp.AICUDA).write_text(json.dumps(ai))
    (tmp_path / sp.CUDAAGENT).write_text(json.dumps(ca))
    return str(tmp_path)


def test_groups_by_task_not_record(data):
    m = sp.split(data)
    assert m["counts"]["records_dropped"] == {"cudaagent_mismatch": 1}
    assert m["counts"]["tasks_by_subset"] == {"aicuda": 4, "cudaagent": 20}  # 16 AICUDA records = 4 tasks
    every = m["heldout_a"] + m["heldout_kb"] + m["validation"] + m["train"]
    multi = [t for t in every if t["subset"] == "cudaagent" and t["n_records"] == 2]
    assert len(multi) == 1  # the resized copy joined its original


def test_splits_disjoint_and_sized(data):
    m = sp.split(data)
    names = ("heldout_a", "heldout_kb", "validation", "train")
    keys = [{t["task"] for t in m[n]} for n in names]
    assert sum(map(len, keys)) == len(set().union(*keys)) == m["counts"]["tasks"]
    assert m["counts"]["heldout_a"]["tasks"] == 5 and m["counts"]["validation"]["tasks"] == 4
    assert all(t["upstream_split"] == ["val"] for t in m["heldout_a"] + m["validation"])
    held_ids = {i for n in ("heldout_a", "heldout_kb", "validation") for t in m[n] for i in t["record_ids"]}
    assert len(held_ids) == sum(t["n_records"] for n in ("heldout_a", "heldout_kb", "validation") for t in m[n])


def test_deterministic(data):
    assert json.dumps(sp.split(data)) == json.dumps(sp.split(data))


def test_norm_code_ignores_comments_and_docstrings():
    a = MODULE.format(doc="one", k="relu", comment="first", n=8)
    b = MODULE.format(doc="two", k="relu", comment="second", n=8)
    assert sp.norm_code(a) == sp.norm_code(b)
    assert sp.mask_numbers(sp.norm_code(a)) == sp.mask_numbers(sp.norm_code(MODULE.format(doc="", k="relu", comment="", n=9)))
