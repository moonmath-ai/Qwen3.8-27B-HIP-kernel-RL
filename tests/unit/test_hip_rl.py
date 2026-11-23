"""CPU tests for scripts/hip_rl.py's pure parts (no server, no GPU)."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("hip_rl", Path(__file__).parents[2] / "scripts/hip_rl.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)

TASKS = ([{"task": f"a{i}", "difficulty": "L1"} for i in range(50)] +
         [{"task": f"b{i}", "difficulty": "L2"} for i in range(5)] +
         [{"task": f"c{i}", "difficulty": "L3"} for i in range(5)])


def test_sampler_balances_levels_and_skips_excluded():
    s = rl.TaskSampler(TASKS, {"L1": 1, "L2": 1, "L3": 1}, exclude={"b0"}, seed=1)
    drawn = [t for _ in range(200) for t in s.draw(3)]
    lv = [t["difficulty"] for t in drawn]
    assert all(abs(lv.count(x) / len(lv) - 1 / 3) < 0.08 for x in ("L1", "L2", "L3"))
    assert "b0" not in {t["task"] for t in drawn}
    assert all(len({t["task"] for t in s.draw(6)}) == 6 for _ in range(20))   # no repeats within a step


def test_sampler_drops_tasks_without_variance():
    s = rl.TaskSampler(TASKS[:2], {"L1": 1}, dead_after=2, seed=0)
    s.record("a0", [1.0, 1.0, 1.0, 1.0])
    assert not s.dead("a0")
    s.record("a0", [0.0, 0.0, 0.0, 0.0])
    assert s.dead("a0")
    s.record("a1", [0.0, 0.0])
    s.record("a1", [0.0, 1.0])            # variance in the latest group keeps it alive
    assert not s.dead("a1")
    assert {t["task"] for t in s.draw(2)} == {"a1"}


def test_level_without_tasks_is_ignored():
    s = rl.TaskSampler(TASKS[:3], rl.parse_levels("L1:0.4,L2:0.3,L3:0.3"))
    assert {t["difficulty"] for t in s.draw(3)} == {"L1"}


def test_advantages_group_centred():
    assert rl.advantages([0.0, 1.0, 1.0, 2.0]) == pytest.approx([-1.0, 0.0, 0.0, 1.0])
    assert rl.advantages([0.1, 0.1]) == pytest.approx([0.0, 0.0])


def test_datum_alignment():
    d = rl.rl_datum_fields([1, 2, 3], [7, 8], [-0.5, -0.25], 0.75)
    assert d["model_input"] == [1, 2, 3, 7]
    assert d["target_tokens"] == [2, 3, 7, 8]
    assert d["logprobs"] == [0.0, 0.0, -0.5, -0.25]
    assert d["advantages"] == [0.0, 0.0, 0.75, 0.75]
    assert len({len(v) for v in d.values()}) == 1


def test_step_metrics():
    outs = [{"format_ok": False, "reward": 0.0},
            {"format_ok": True, "compiled": True, "correct": False, "reward": 0.1},
            {"format_ok": True, "compiled": True, "correct": True, "speedup": 2.0, "own_us": 30.0, "other_us": 10.0,
             "reward": 1.5},
            {"format_ok": True, "compiled": True, "correct": True, "speedup": 1.0, "own_us": 10.0, "other_us": 0.0,
             "reward": 1.0, "pass1_error": "crash"}]
    m = rl.step_metrics(outs, {"t1": [0.0, 0.1], "t2": [1.5, 1.5]}, ["length", "stop", "stop", "stop"])
    assert m["correct"] == 0.5 and m["format_ok"] == 0.75 and m["truncated"] == 0.25
    assert m["groups_mixed"] == 1 and m["speedup_ge_1_1"] == 1 and m["pass1_retries"] == 1
    assert m["other_time_share"] == pytest.approx(0.2)
    assert m["reward_mean"] == pytest.approx(0.65)


def test_training_client_waits_for_reaped_slot(monkeypatch):
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)
    tries = []

    class Svc:
        def create_lora_training_client(self, base_model, rank):
            tries.append(rank)
            if len(tries) < 3:
                raise RuntimeError("Request failed: no free training slot on this server")
            return "tc"

    assert rl.training_client(Svc(), "m", 64) == "tc" and len(tries) == 3

    class Bad:
        def create_lora_training_client(self, base_model, rank):
            raise RuntimeError("base_model not served here")

    with pytest.raises(RuntimeError, match="not served"):
        rl.training_client(Bad(), "m", 64)


def test_train_split_without_task_records_fails_clearly(tmp_path):
    m = tmp_path / "m.json"
    m.write_text('{"train": [{"task": "a", "difficulty": "L1"}]}')
    with pytest.raises(ValueError, match="--task-records"):
        rl.load_tasks(str(tmp_path), str(m), "train")


def test_sampler_accepts_no_tasks():          # --eval-only builds it with no training tasks
    rl.TaskSampler([], rl.parse_levels("L1:1"))
