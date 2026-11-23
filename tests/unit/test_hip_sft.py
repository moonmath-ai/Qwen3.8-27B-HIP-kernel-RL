"""CPU tests for scripts/build_sft_data.py selection and hip_sft.py datum building."""
import importlib.util
from pathlib import Path

A = Path(__file__).parents[2] / "scripts"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, A / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bs, sft = _mod("build_sft_data"), _mod("hip_sft")
OK = {"correct": True, "own_kernels": ["k"], "own_us": 10.0, "other_us": 1.0}


def test_select_own_first_refs_for_unsolved_capped_and_library_tasks_excluded():
    own_rows = ([{"task": "t1", "record_id": "r1", "response": f"think{i}</think>ans", "finish": "stop"}
                 for i in range(3)] +
                [{"task": "t2", "record_id": "r2", "response": "x", "finish": "length"},   # truncated: not used
                 {"task": "t3", "record_id": "r3", "response": "y", "finish": "stop"},
                 {"task": "t4", "record_id": "r4", "response": "z", "finish": "stop"}])
    own_g = [OK, OK, OK, OK, {"correct": False}, OK]
    ref_rows = [{"task": t, "id": f"ref{t}", "response": "```cpp\nCODE\n```"} for t in ("t1", "t2", "t3", "t4")]
    ref_g = [OK, OK, OK, {**OK, "own_us": 1.0, "other_us": 99.0}]   # t4's reference is a library call
    ex, exclude = bs.select(own_rows, own_g, ref_rows, ref_g, own_per_task=2, ref_ratio=1.0)
    own = [e for e in ex if e["source"] == "own"]
    refs = [e for e in ex if e["source"] == "ref"]
    assert [e["task"] for e in own] == ["t1", "t1"]            # capped per task; t4 excluded
    assert exclude == {"t4"}
    assert {e["task"] for e in refs} <= {"t2", "t3"} and len(refs) == 2   # unsolved, up to ratio x own
    assert refs[0]["completion"] == "\n</think>\n\n```cpp\nCODE\n```"


def test_sft_fields_mask_prompt():
    d = sft.sft_fields([1, 2, 3], [7, 8, 9])
    assert d["model_input"] == [1, 2, 3, 7, 8]
    assert d["target_tokens"] == [2, 3, 7, 8, 9]
    assert d["weights"] == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_batches_cover_each_epoch_once():
    b = sft.batches(10, 4, 2, 0)
    assert [len(x) for x in b] == [4, 4, 2, 4, 4, 2]
    assert sorted(i for x in b[:3] for i in x) == list(range(10))
