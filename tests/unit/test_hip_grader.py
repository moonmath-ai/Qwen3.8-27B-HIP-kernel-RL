"""scripts/hip_grade.py + hip_grade_worker.py on the CPU: extraction, static scan, reward policies, and the
worker's correctness checks against honest and cheating stand-ins (a Python `forward` instead of a built .so)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


hg = _load("hip_grade", "scripts/hip_grade.py")

HIP = '''#include <torch/extension.h>
__global__ void __launch_bounds__(256) scale_kernel(const float* x, float* y, int n) { }
__global__ void other_kernel(float* y) { }
torch::Tensor forward(torch::Tensor x, double s) { return x; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, "fwd"); }
'''

MODULE = '''import torch
class Model(torch.nn.Module):
    def __init__(self, s):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(8))
        self.s = s
    def forward(self, x):
        return torch.relu(x * self.w) * self.s
x = torch.randn(4, 8)
def get_inputs():
    return [x]
def get_init_inputs():
    return [2.0]
'''
FUNCTIONAL = '''import torch
def module_fn(x, w, s):
    return torch.relu(x * w) * s
class Model(torch.nn.Module):
    def __init__(self, s):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(8))
        self.s = s
    def forward(self, x, fn=module_fn):
        return fn(x, self.w, self.s)
x = torch.randn(4, 8)
def get_inputs():
    return [x]
def get_init_inputs():
    return [2.0]
'''
FNS = {
    "honest": "import torch\ndef forward(x, w, s):\n    return torch.relu(x * w) * s\n",
    "wrong": "import torch\ndef forward(x, w, s):\n    return torch.relu(x * w) * s + 1e-2\n",
    # memorises the first call's output: passes AIG's single fixed input, fails on a fresh seed
    "caching": "import torch\n_c = {}\ndef forward(x, w, s):\n    _c.setdefault('y', torch.relu(x * w) * s)\n    return _c['y']\n",
    "mutates_input": "import torch\ndef forward(x, w, s):\n    x.mul_(w).relu_().mul_(s)\n    return x.clone()\n",
    "aliases_input": "import torch\ndef forward(x, w, s):\n    x.mul_(0)\n    return x\n",
    "nan": "import torch\ndef forward(x, w, s):\n    y = torch.relu(x * w) * s\n    y[0, 0] = float('nan')\n    return y\n",
    "wrong_dtype": "import torch\ndef forward(x, w, s):\n    return (torch.relu(x * w) * s).double()\n",
    "raises": "def forward(x, w, s):\n    raise RuntimeError('hipErrorIllegalAddress')\n",
}


def test_extract_takes_last_block_with_pybind_forward():
    text = f"thinking...\n```cpp\nint draft;\n```\nfinal:\n```cpp\n{HIP}```\n```python\nprint(1)\n```"
    assert hg.extract_hip(text) == HIP
    assert hg.extract_hip("```cpp\n__global__ void k() {}\n```") is None  # no binding: format failure
    assert hg.extract_hip("no code") is None
    # only the final answer counts: a stray fence in the reasoning must not misalign pairing, drafts don't count
    final = f"```cpp\n{HIP}```"
    assert hg.extract_hip("think ``` odd fence\n```cpp\nint d;\n```\n</think>\n" + final) == HIP
    assert hg.extract_hip(f"draft:\n{final}\n</think>\nno code after all") is None
    assert hg.extract_hip(f"still thinking {final}", truncated=True) is None


def test_scan_flags_kernels_forbidden_and_library_calls():
    s = hg.scan(HIP)
    assert s == {"src_kernels": ["other_kernel", "scale_kernel"], "forbidden": [], "library": []}
    bad = HIP + '\nvoid h(){ system("x"); FILE* f = fopen("/etc/passwd","r"); getenv("HOME"); }\n' \
                'torch::Tensor g(torch::Tensor a){ return torch::matmul(a, a); }\n'
    s = hg.scan(bad)
    assert s["forbidden"] == ["env", "file_io", "process"]
    assert s["library"] == ["aten_compute"]
    assert hg.scan(HIP + "// system(\"x\") torch::matmul(a,b)\n")["forbidden"] == []  # comments don't count


@pytest.mark.parametrize("o,scheme,want", [
    ({"format_ok": False}, "correct_speedup", 0.0),
    ({"format_ok": True, "forbidden": ["env"], "compiled": True, "correct": True, "speedup": 3.0}, "hierarchical", 0.0),
    ({"format_ok": True, "compiled": True, "correct": False}, "correct_speedup", 0.0),
    ({"format_ok": True, "compiled": True, "correct": True, "speedup": 2.0}, "correct_speedup", 2.6),
    ({"format_ok": True, "compiled": True, "correct": True, "speedup": 50.0}, "correct_speedup", 10.6),
    ({"format_ok": True, "compiled": True, "correct": True, "speedup": 2.0, "other_kernels": ["Cijk_x"]},
     "correct_speedup", 0.3),
    ({"format_ok": True, "compiled": True, "correct": False}, "hierarchical", 0.4),
    ({"format_ok": True, "compiled": True, "correct": True, "speedup": 4.0}, "hierarchical", 2.9),
    ({"format_ok": True, "compiled": True, "correct": True, "speedup": 0.5}, "binary", 1.0),
])
def test_reward_policies(o, scheme, want):
    assert hg.reward(o, scheme) == pytest.approx(want)


K = {"format_ok": True, "src_kernels": ["k"], "compiled": True}


@pytest.mark.parametrize("o,want", [
    ({"format_ok": False}, 0.0),
    ({**K, "forbidden": ["env"], "correct": True, "speedup": 3.0}, 0.0),
    ({**K, "src_kernels": [], "correct": True, "speedup": 3.0}, 0.0),           # no own __global__ in the source
    ({**K, "compiled": False}, 0.0),
    ({**K, "correct": False}, 0.1),
    ({**K, "correct": True, "own_kernels": [], "other_kernels": ["Cijk"], "speedup": 3.0}, 0.0),  # wraps a library
    ({**K, "correct": True, "speedup": 1.05}, 1.0),                         # dead zone
    ({**K, "correct": True, "speedup": 0.5}, 1.0),
    ({**K, "correct": True, "speedup": 2.0}, 1.5),
    ({**K, "correct": True, "speedup": 4.0, "own_kernels": ["k"], "other_kernels": ["Cijk"]}, 2.0),  # library ok
    ({**K, "correct": True, "speedup": 100.0}, 2.5),                        # clipped at 8x
    ({**K, "correct": True}, 1.0),                                          # no timing run
])
def test_reward_zro(o, want):
    assert hg.reward(o) == pytest.approx(want)


def test_docker_cmd_is_isolated(tmp_path):
    cmd = hg.docker_cmd(tmp_path, "k_x", ["scale_kernel"])
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--pids-limit", "--memory", "no-new-privileges"):
        assert flag in joined
    assert cmd[:3] == ["docker", "run", "--rm"] and cmd[cmd.index("--name") + 1].startswith("zfg_k_x_")  # killable by name
    assert "--privileged" not in cmd


def _worker(tmp_path, fn_src, seeds="0,1,2"):
    (tmp_path / "m.py").write_text(MODULE)
    (tmp_path / "f.py").write_text(FUNCTIONAL)
    (tmp_path / "fn.py").write_text(fn_src)
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/hip_grade_worker.py"), "--module", str(tmp_path / "m.py"),
                    "--functional", str(tmp_path / "f.py"), "--fn-py", str(tmp_path / "fn.py"), "--device", "cpu",
                    "--seeds", seeds, "--out", str(out)], check=True, cwd=tmp_path)
    return json.loads(out.read_text())


def test_worker_passes_an_honest_kernel(tmp_path):
    r = _worker(tmp_path, FNS["honest"])
    assert r["loaded"] and r["correct"], r
    assert [s["seed"] for s in r["seeds"]] == [0, 1, 2]


@pytest.mark.parametrize("kind,why", [
    ("wrong", "values differ"),
    ("caching", "values differ"),          # fresh inputs per seed defeat memorisation
    ("mutates_input", "inputs left in a different state"),
    ("aliases_input", "output aliases an input"),
    ("nan", "NaN pattern differs"),
    ("wrong_dtype", "dtype"),
    ("raises", "runtime RuntimeError"),
])
def test_worker_rejects_cheats_and_bugs(tmp_path, kind, why):
    r = _worker(tmp_path, FNS[kind])
    assert r["loaded"] and not r["correct"], r
    assert why in r["why"], r


def test_worker_reports_load_failure(tmp_path):
    r = _worker(tmp_path, "this is not python")
    assert r["loaded"] is False and r["error_type"] == "load"


def test_parallel_run_retries_crashes_alone_and_times_only_correct(tmp_path, monkeypatch):
    calls = []
    first = {"a": {"correct": True}, "b": {"correct": False, "error_type": "timeout"},
             "c": {"correct": False, "error_type": "mismatch", "why": "values differ"}}

    def fake(work, o, timeout_s, timing):
        calls.append((o["name"], timing))
        if not timing and sum(1 for n, t in calls if n == o["name"] and not t) == 1:
            return dict(first[o["name"]])
        return {"correct": True, "speedup": 2.0} if timing or o["name"] == "b" else {"correct": True}

    monkeypatch.setattr(hg, "_grade_one", fake)
    outs = [{"name": n, "format_ok": True} for n in "abc"]
    hg.run_all(tmp_path, outs, {"a": True, "b": True, "c": True}, workers=3)
    assert sorted(c for c in calls if not c[1]) == [("a", False), ("b", False), ("b", False), ("c", False)]
    assert sorted(c for c in calls if c[1]) == [("a", True), ("b", True)]   # c failed correctness: never timed
    assert outs[1]["pass1_error"] == "timeout" and outs[1]["correct"]       # a neighbour's fault is not its verdict
    assert not outs[2]["correct"] and "pass1_error" not in outs[2]


def test_every_pass_gets_source_kernel_names(tmp_path, monkeypatch):
    """Both passes give the worker the source's own __global__ names, not the *launched* names that pass 1 writes
    into o["kernels"]; otherwise correct answers come back from pass 2 with own_kernels == []."""
    seen = []

    def fake_cmd(work, name, kernels, timeout_s=120, seeds="0,1,2", timing=True):
        seen.append((kernels, timing))
        return ["true", "--name", name]

    def fake_run(cmd, name, timeout_s):
        (tmp_path / "tasks" / name).mkdir(parents=True, exist_ok=True)
        (tmp_path / "tasks" / name / "result.json").write_text(
            '{"correct": true, "kernels": ["scale_kernel(float*, int) [clone .kd]"], "speedup": 1.0}')
        return 0

    monkeypatch.setattr(hg, "docker_cmd", fake_cmd)
    monkeypatch.setattr(hg, "run_container", fake_run)
    outs = [{"name": "a", "format_ok": True, "src_kernels": ["scale_kernel"]}]
    hg.run_all(tmp_path, outs, {"a": True}, workers=2)
    assert seen == [(["scale_kernel"], False), (["scale_kernel"], True)]


def test_no_timing_reruns_empty_profiles_alone(tmp_path, monkeypatch):
    """With several containers at once the profiler sometimes records no kernels; without timing, correct
    answers with an empty kernel list (= no own kernel) are re-run alone."""
    calls = []

    def fake(work, o, timeout_s, timing):
        calls.append(o["name"])
        if calls.count(o["name"]) == 1:
            return {"correct": True, "kernels": [] if o["name"] == "a" else ["k_kernel"]}
        return {"correct": True, "kernels": ["k_kernel"], "own_kernels": ["k_kernel"]}

    monkeypatch.setattr(hg, "_grade_one", fake)
    outs = [{"name": n, "format_ok": True} for n in "ab"]
    hg.run_all(tmp_path, outs, {"a": True, "b": True}, workers=2, timing=False)
    assert sorted(calls) == ["a", "a", "b"]
    assert outs[0]["own_kernels"] == ["k_kernel"] and outs[0]["profile_retry"]
    assert "profile_retry" not in outs[1]


def test_unkillable_container_is_counted_not_fatal(tmp_path, monkeypatch):
    """A GPU hang can leave containers in D state, so `docker run` does not exit after docker kill. The batch
    must not crash on the TimeoutExpired: the answer gets error_type "stuck" and is not re-run alone."""
    import subprocess

    class P:
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("docker", timeout)

    monkeypatch.setattr(hg.subprocess, "Popen", lambda *a, **k: P())
    monkeypatch.setattr(hg.subprocess, "run", lambda *a, **k: None)
    hg.STUCK.clear()
    assert hg.run_container(["docker"], "zfg_x", 1) is None and "zfg_x" in hg.STUCK
    monkeypatch.setattr(hg, "docker_cmd", lambda work, name, *a, **k: ["docker", "--name", "zfg_" + name])
    monkeypatch.setattr(hg, "run_container", lambda cmd, name, t: hg.STUCK.add(name))
    (tmp_path / "tasks" / "y").mkdir(parents=True)
    outs = [{"name": "y", "format_ok": True}, {"name": "z", "format_ok": True}]
    (tmp_path / "tasks" / "z").mkdir(parents=True)
    hg.run_all(tmp_path, outs, {"y": True, "z": True}, workers=2, timing=False)
    assert [o["error_type"] for o in outs] == ["stuck", "stuck"] and "pass1_error" not in outs[0]


def test_pass1_resumes_from_cache(tmp_path, monkeypatch):
    calls = []

    def fake(work, o, timeout_s, timing):
        calls.append(o["name"])
        return {"correct": False, "error_type": "mismatch"}

    monkeypatch.setattr(hg, "_grade_one", fake)
    (tmp_path / "pass1_cache.jsonl").write_text('{"name": "a", "result": {"correct": false, "error_type": "mismatch"}}\n')
    outs = [{"name": n, "format_ok": True} for n in "ab"]
    hg.run_all(tmp_path, outs, {"a": True, "b": True}, workers=2, timing=False)
    assert calls == ["b"] and all(o["error_type"] == "mismatch" for o in outs)
    assert len((tmp_path / "pass1_cache.jsonl").read_text().splitlines()) == 2


def test_watchdog_trips_on_a_long_stuck_driver_worker():
    seen = {1: "kworker/u27:1+ttm"}
    wd = hg.DriverWatchdog(max_d_s=60, probe=lambda: dict(seen))
    hg.STUCK.clear()
    assert not wd.check(now=100)
    assert not wd.check(now=150)                      # 50 s: a big free can take a few seconds, not a fault
    seen.clear()
    assert not wd.check(now=155)                      # it went away: forgotten
    seen[2] = "kworker/7:1+kfd_process_wq"
    assert not wd.check(now=200)
    assert wd.check(now=260) and wd.onset == 200 and "kfd_process_wq" in wd.why
    assert wd.check(now=1000) and wd.onset == 200    # sticky


def test_stuck_driver_threads_ignores_other_d_state_tasks():
    assert all("ttm" in c or "kfd" in c for c in hg.stuck_driver_threads().values())


def test_unhealthy_driver_stops_grading_and_drops_late_results(tmp_path, monkeypatch):
    """After a hang every grade times out; those must not be cached as if they were the answer's fault. Grading
    stops (DriverUnhealthy) and cached rows from after the onset are moved aside."""
    hg.STUCK.clear()
    clock = {"t": 1000.0}
    monkeypatch.setattr(hg.time, "time", lambda: clock["t"])
    wd = hg.DriverWatchdog(max_d_s=60, probe=lambda: {})
    calls = []

    def fake(work, o, timeout_s, timing):
        calls.append((o["name"], wd.bad))
        if o["name"] == "b":                         # the driver wedges while b runs (after a is cached)
            for _ in range(200):
                if '"a"' in (tmp_path / "pass1_cache.jsonl").read_text():
                    break
                threading.Event().wait(0.01)
            clock["t"] = 1020.0
            wd.onset, wd.why = 1015.0, "test"
        return {"correct": False, "error_type": "timeout"}

    monkeypatch.setattr(hg, "_grade_one", fake)
    (tmp_path / "pass1_cache.jsonl").write_text('{"name": "old", "result": {"correct": true}}\n')
    outs = [{"name": n, "format_ok": True} for n in "abcd"]
    import pytest
    import threading
    with pytest.raises(hg.DriverUnhealthy):
        hg.run_all(tmp_path, outs, {n: True for n in "abcd"}, workers=1 + 1, timeout_s=1, watchdog=wd)
    kept = [json.loads(l)["name"] for l in open(tmp_path / "pass1_cache.jsonl")]
    dropped = [json.loads(l)["name"] for l in open(tmp_path / "pass1_cache.dropped_1015.jsonl")]
    assert "old" in kept and "a" in kept and "b" in dropped and "b" not in kept
    assert not any(bad for _, bad in calls)          # nothing new starts once it tripped


@pytest.mark.skipif(not Path(hg.RUNNER_VENV, "bin/python").exists(), reason="needs the ROCm torch runner venv")
def test_gen_ninja_skips_a_source_that_hangs_hipify(tmp_path):
    # `<<<...>>(` (missing a '>') sends torch's hipify launch parser into an endless, memory-growing loop
    # (until the OOM killer stops it); gen_ninja must give up on that file and still write the others' build files
    kdir, bdir = tmp_path / "k", tmp_path / "b"
    kdir.mkdir()
    (kdir / "k_bad.hip").write_text("__global__ void kk(float* x) {}\nvoid f(float* x) {\n"
                                    "  kk<float><<<\n    dim3(1), 1\n  >>(x);\n}\n")
    (kdir / "k_good.hip").write_text(HIP)
    p = subprocess.run([f"{hg.RUNNER_VENV}/bin/python", "-I", str(ROOT / "scripts/gen_ninja.py"),
                        "--hipify-timeout", "3", str(kdir), str(bdir)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "hipify failed for k_bad.hip: TimeoutError" in p.stderr
    assert not (bdir / "k_bad" / "build.ninja").exists()
    assert (bdir / "k_good" / "build.ninja").exists()
