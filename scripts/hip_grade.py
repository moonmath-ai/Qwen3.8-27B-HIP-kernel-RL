"""HIP kernel grader for PyTorch->HIP RL tasks on an MI300X (docs/REWARD.md).

Pipeline per rollout (each stage recorded; the reward is a separate, swappable policy at the end):
  extract  the last ```cpp/hip block of the response that defines PYBIND11_MODULE (else: format failure)
  scan     static flags: own __global__ kernels, forbidden calls (files, processes, env, sockets, dlopen),
           library compute calls (ATen/torch ops, rocBLAS/hipBLAS/MIOpen) — recorded, the policy decides
  compile  torch.utils.cpp_extension build files (gen_ninja.py) + ninja / precompiled-header build in the
           no-network, no-GPU sandbox (run_compile.sh / compile_pch_inner.sh); nothing is imported here
  run      hip_grade_worker.py in a fresh container per kernel: GPU devices, no network, read-only root,
           non-root, time / memory / pid limits; multi-seed correctness, launched kernels, timing
  reward   reward(outcome, scheme)

This module is importable for tests (no GPU, no Docker needed for extract / scan / reward / docker_cmd).
GPU stages run only with --gpu.

  python3 -I scripts/hip_grade.py --tasks rollouts.jsonl --work /path/work [--gpu] --out results.jsonl
  rollouts.jsonl lines: {"id", "module" (code), "functional" (code), "response" (model text)}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
# scripts/env.sh documents these; the defaults are the layout the result was produced with
RUNNER_VENV = os.environ.get("ZF_RUNNER_VENV", os.path.expanduser("~/venvs/zf-runner"))  # torch 2.14.1+rocm7.2: build AND run ABI must match
IMAGE = os.environ.get("ZF_GRADER_IMAGE", "vllm/vllm-openai-rocm@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905")  # only a rootfs; host /usr, ROCm and venv are mounted
ROCM = os.environ.get("ZF_ROCM", "/opt/rocm-7.2.4")

_BLOCK = re.compile(r"```(?:cpp|c\+\+|hip|cuda|c)?[ \t]*\n(.*?)```", re.S)
_LB = r"(?:__launch_bounds__\s*\([^)]*\)\s*)?"
_GLOBAL = re.compile(r"__global__\s+" + _LB + r"(?:static\s+)?void\s+" + _LB + r"(\w+)\s*\(")
FORBIDDEN = {
    "file_io": r"\b(fopen|freopen|std::ofstream|std::ifstream|std::fstream|open\s*\(|unlink|remove\s*\()",
    "process": r"\b(system\s*\(|popen|fork\s*\(|exec[lv]p?e?\s*\(|posix_spawn)",
    "env": r"\b(getenv|setenv|putenv)\b",
    "network": r"\b(socket\s*\(|connect\s*\(|gethostbyname|curl_)",
    "dynamic_load": r"\b(dlopen|dlsym)\b",
}
LIBRARY = {
    "aten_compute": r"\b(?:at|torch)::(?:matmul|mm|bmm|addmm|baddbmm|linear|conv\w*|softmax|log_softmax|"
                    r"layer_norm|batch_norm|group_norm|instance_norm|einsum|sum|mean|max|min|amax|amin|"
                    r"relu|gelu|sigmoid|tanh|exp|log|sqrt|pow|div|mul|add|sub|cumsum|sort|topk|"
                    r"scaled_dot_product_attention|embedding|pad|interpolate|upsample\w*|pool\w*|"
                    r"\w+_pool\w*d|clamp|where|cat|stack|transpose|permute)\s*\(",
    "tensor_method_compute": r"\.(?:matmul|mm|bmm|softmax|sum|mean|exp|log|pow|sqrt)\s*\(",
    "blas": r"\b(rocblas_|hipblas|hipblasLt|miopen|MIOpen)",
}


def extract_hip(text: str, truncated: bool = False) -> str | None:
    """The last fenced block of the final answer that defines PYBIND11_MODULE with a `forward`; None if none.
    Only text after the last </think> counts: code drafted in the reasoning is not an answer (in base-rate sampling, 94 length-truncated
    responses had a draft there), and an odd number of fences in the reasoning misaligns the block pairing
    (in base-rate sampling, that would miss 61 complete answers)."""
    text = text or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif truncated or "<think>" in text:
        return None  # the reasoning never ended (the chat template opens <think> in the prompt)
    for block in reversed(_BLOCK.findall(text)):
        if "PYBIND11_MODULE" in block and re.search(r'm\.def\(\s*"forward"', block):
            return block
    return None


def scan(src: str) -> dict:
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", src, flags=re.S)  # comments can't trigger flags
    return {
        "src_kernels": sorted(set(_GLOBAL.findall(code))),
        "forbidden": sorted(k for k, p in FORBIDDEN.items() if re.search(p, code)),
        "library": sorted(k for k, p in LIBRARY.items() if re.search(p, code)),
    }
# ---- reward policies ("zro" is the Zro-form reward used for the result; docs/REWARD.md) -------------------

# ---- reward policies ("zro" was used for the result; docs/REWARD.md) --------------------------------
def reward(o: dict, scheme: str = "zro") -> float:
    """o: outcome with format_ok, forbidden, kernels, compiled, correct, speedup, own_kernels, other_kernels.
    - zro:             0 on format failure / forbidden call / no own __global__ kernel (in the source, or launched
                       when correct); 0.1 if compiled but not correct; if correct 1 + 0.5*clip(log2 s, 0, 3) with
                       s < 1.1 counted as 1 (timing-noise dead zone). Library calls allowed: s is end to end vs eager.
                       Same formula for every task; difficulty is handled by group advantages and task sampling.
    - binary:          1 if correct else 0
    - correct_speedup: AMD rl4kernel_hip default: 0 on any failure, else 0.3 + clip(speedup, 0, 10)
                       (+0.3 if speedup >= 1.1); speedup counts only if no library kernels ran
    - hierarchical:    AMD Triton paper: 0.4*compiled + 1.0*correct + min(0.375*log2(s)^2, 1.5) if correct and s>=1
    Forbidden calls and format failures always score 0.
    """
    if not o.get("format_ok") or o.get("forbidden"):
        return 0.0
    compiled, correct = bool(o.get("compiled")), bool(o.get("correct"))
    s = o.get("speedup")
    s = s if isinstance(s, (int, float)) and math.isfinite(s) and s > 0 else None
    if o.get("other_kernels"):
        s = None  # time spent in library kernels is not the candidate's speedup
    if scheme == "zro":
        if not o.get("src_kernels"):
            return 0.0
        if not correct:
            return 0.1 if compiled else 0.0
        if "own_kernels" in o and not o["own_kernels"]:
            return 0.0
        sp = o.get("speedup")
        sp = sp if isinstance(sp, (int, float)) and math.isfinite(sp) and sp > 0 else 1.0
        return 1.0 + (0.5 * min(math.log2(sp), 3.0) if sp >= 1.1 else 0.0)
    if scheme == "binary":
        return 1.0 if correct else 0.0
    if scheme == "correct_speedup":
        if not correct:
            return 0.0
        sp = min(max(s or 0.0, 0.0), 10.0)
        return 0.3 + sp + (0.3 if (s or 0) >= 1.1 else 0.0)
    if scheme == "hierarchical":
        bonus = min(0.375 * math.log2(s) ** 2, 1.5) if (correct and s and s >= 1) else 0.0
        return 0.4 * compiled + 1.0 * correct + bonus
    raise ValueError(f"unknown reward scheme {scheme!r}")


# ---- stages that touch Docker / the GPU ---------------------------------------------------------------------
def prepare(rows: list[dict], work: Path) -> list[dict]:
    """Write per-rollout dirs: kernels/<name>.hip (for gen_ninja), tasks/<name>/{module,functional}.py."""
    (work / "kernels").mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        name = "k_" + hashlib.sha256((r["id"] + (r.get("response") or "")).encode()).hexdigest()[:16]
        src = extract_hip(r.get("response", ""), truncated=r.get("finish") == "length")
        o = {"id": r["id"], "name": name, "format_ok": src is not None}
        if src is not None:
            o.update(scan(src))
            (work / "kernels" / f"{name}.hip").write_text(src)
            t = work / "tasks" / name
            t.mkdir(parents=True, exist_ok=True)
            (t / "module.py").write_text(r["module"])
            (t / "functional.py").write_text(r["functional"])
        out.append(o)
    return out


def compile_all(work: Path, jobs: int, pch: bool = True) -> dict[str, bool]:
    """Build every prepared kernel in the CPU sandbox; returns name -> built."""
    build = work / "build"
    env = {**os.environ, "PYTORCH_ROCM_ARCH": "gfx942"}
    subprocess.run([f"{RUNNER_VENV}/bin/python", "-I", str(HERE / "gen_ninja.py"), str(work / "kernels"),
                    str(build)], check=True, env=env, capture_output=True)
    if pch:
        inner = build / "_inner.sh"
        inner.write_text((HERE / "compile_pch_inner.sh").read_text())
        subprocess.run([str(HERE / "sandbox.sh"), str(build), f"bash {inner} {build} {jobs}"], check=True,
                       capture_output=True)
    else:
        subprocess.run([str(HERE / "run_compile.sh"), str(build), str(jobs)], check=True, capture_output=True)
    return {d.name: (d / f"{d.name}.so").exists() for d in build.iterdir() if d.is_dir() and not d.name.startswith("_")}


def gpu_groups() -> list[str]:
    """Numeric GIDs of the device nodes: group *names* inside the image map to other GIDs (non-root sees no GPU)."""
    nodes = [Path("/dev/kfd"), *sorted(Path("/dev/dri").glob("renderD*"))]
    gids = sorted({p.stat().st_gid for p in nodes if p.exists()})
    return [x for g in gids for x in ("--group-add", str(g))]


def docker_cmd(work: Path, name: str, kernels: list[str], timeout_s: int = 120, seeds: str = "0,1,2",
               timing: bool = True) -> list[str]:
    """One throwaway container per kernel: GPU, no network, read-only root, non-root, limits."""
    t, so = work / "tasks" / name, work / "build" / name / f"{name}.so"
    return ["docker", "run", "--rm", "--name", f"zfg_{name}_{os.getpid()}", "--stop-timeout", "1", "--network", "none", "--read-only", "--user", f"{os.getuid()}:{os.getgid()}",
            "--device", "/dev/kfd", "--device", "/dev/dri", *gpu_groups(),
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--memory", "32g", "--pids-limit", "256",
            "--tmpfs", "/tmp:size=1g", "-e", "HOME=/tmp", "--entrypoint", f"{RUNNER_VENV}/bin/python",
            "-v", "/usr:/usr:ro", "-v", "/etc/ld.so.cache:/etc/ld.so.cache:ro",
            "-v", f"{ROCM}:{ROCM}:ro", "-v", f"{RUNNER_VENV}:{RUNNER_VENV}:ro",
            "-v", f"{HERE / 'hip_grade_worker.py'}:/w/worker.py:ro", "-v", f"{t}:/task", "-v", f"{so}:/w/{name}.so:ro",
            IMAGE, "-I", "/w/worker.py", "--module", "/task/module.py", "--functional", "/task/functional.py",
            "--ext", f"/w/{name}.so", "--ext-name", name, "--kernel-names", ",".join(kernels),
            "--seeds", seeds, "--out", "/task/result.json", *([] if timing else ["--no-timing"])]


def run_container(cmd: list[str], name: str, timeout_s: int) -> int | None:
    """Run one grading container; on timeout kill the *container* (killing the `docker run` client alone leaves
    it running, and a hung kernel keeps the GPU busy). None means timed out."""
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:  # a GPU hang can leave containers in D state (unkillable);
            STUCK.add(name)                  # don't crash the whole batch, count it and move on
            print(f"[hip_grade] container {name} did not die after kill (GPU hang?); abandoned", flush=True)
        return None


STUCK: set[str] = set()   # container names that survived docker kill


class DriverUnhealthy(RuntimeError):
    """The amdgpu driver looks wedged: stop grading, the GPU needs a reboot/reset before results mean anything."""


def stuck_driver_threads() -> dict[int, str]:
    """pid -> comm of amdgpu kernel workers (ttm / kfd work queues) in uninterruptible sleep (D)."""
    out = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            st = (d / "stat").read_text()
        except OSError:
            continue
        comm, state = st[st.index("(") + 1:st.rindex(")")], st[st.rindex(")") + 2]
        if state == "D" and comm.startswith("kworker") and ("ttm" in comm or "kfd" in comm):
            out[int(d.name)] = comm
    return out


class DriverWatchdog:
    """A graded kernel can hang the GPU so that amdgpu `ttm`/`kfd` kernel workers sit in D state for hours; every
    later grade then times out, and only a VM reboot fixes it.
    A worker in D state for a few seconds is normal (freeing a large process); one stuck >= `max_d_s`, or a
    container that survived `docker kill`, means the driver is wedged. `onset` = when the first such thread was
    seen: results produced after it are suspect."""

    def __init__(self, max_d_s: float = 60, poll_s: float = 5, probe=stuck_driver_threads):
        self.max_d_s, self.poll_s, self.probe = max_d_s, poll_s, probe
        self.first_seen: dict[int, float] = {}
        self.onset: float | None = None
        self.why = ""
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def check(self, now: float | None = None) -> bool:
        """One poll; True if unhealthy (sticky)."""
        if self.onset is not None:
            return True
        now = time.time() if now is None else now
        cur = self.probe()
        self.first_seen = {p: self.first_seen.get(p, now) for p in cur}
        old = {p: t for p, t in self.first_seen.items() if now - t >= self.max_d_s}
        if old:
            self.onset = min(old.values())
            self.why = f"{len(old)} amdgpu kernel worker(s) in D state >= {self.max_d_s:.0f} s: " \
                       + ", ".join(sorted({cur[p] for p in old}))[:300]
        elif STUCK:
            self.onset = now
            self.why = f"container(s) survived docker kill: {sorted(STUCK)[:3]}"
        return self.onset is not None

    def start(self) -> "DriverWatchdog":
        def loop():
            while not self._stop.wait(self.poll_s):
                if self.check():
                    print(f"[hip_grade] DRIVER UNHEALTHY: {self.why}", flush=True)
                    return
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    @property
    def bad(self) -> bool:
        return self.onset is not None


def _grade_one(work: Path, o: dict, timeout_s: int, timing: bool) -> dict:
    rf = work / "tasks" / o["name"] / "result.json"
    rf.unlink(missing_ok=True)
    cmd = docker_cmd(work, o["name"], o.get("src_kernels", []), timeout_s, timing=timing)
    cname = cmd[cmd.index("--name") + 1]
    rc = run_container(cmd, cname, timeout_s + 30)
    if rf.exists():
        return json.loads(rf.read_text())
    return {"correct": False, "error_type": "stuck" if cname in STUCK else "timeout" if rc is None else "crash",
            "exit_code": rc}


def run_all(work: Path, outcomes: list[dict], built: dict[str, bool], timeout_s: int = 120, workers: int = 1,
            timing: bool = True, watchdog: DriverWatchdog | None = None) -> None:
    """Pass 1: correctness, `workers` containers at once, no timing. A GPU memory fault can take down other GPU
    processes running at that moment, so every crash/timeout from pass 1 is re-run ALONE before it
    counts. Pass 2: timing, one container at a time, only for kernels that passed (skipped if timing=False)."""
    wd = watchdog   # None: no driver checks (tests, CPU-only callers)
    todo = []
    for o in outcomes:
        if o["format_ok"]:
            o["compiled"] = built.get(o["name"], False)
            if o["compiled"] and not o.get("forbidden"):
                todo.append(o)
    if workers > 1:
        # pass-1 results are appended to a cache so a restarted batch resumes (a GPU hang can kill an hours-long batch)
        cache_f = work / "pass1_cache.jsonl"
        cache = {}
        if cache_f.exists():
            for line in open(cache_f):
                try:
                    c = json.loads(line)
                    cache[c["name"]] = c["result"]
                except (json.JSONDecodeError, KeyError):
                    pass

        def one(o):
            if o["name"] in cache:
                return cache[o["name"]]
            if wd is not None and wd.bad:   # wedged driver: start nothing new
                return None
            r = _grade_one(work, o, timeout_s, False)
            with open(cache_f, "a") as f:
                f.write(json.dumps({"name": o["name"], "result": r, "t": time.time()}) + "\n")
            return r
        with ThreadPoolExecutor(workers) as ex:
            for o, r in zip(todo, ex.map(one, todo)):
                if r is not None:
                    o.update(r)
        _abort_if_unhealthy(wd, cache_f)
        for o in todo:
            if o.get("error_type") in ("crash", "timeout", "runtime") or o.get("harness_error"):
                o["pass1_error"] = o.get("error_type") or "harness"
                for k in ("error_type", "exit_code", "why", "error", "harness_error"):
                    o.pop(k, None)
                _abort_if_unhealthy(wd, cache_f)
                o.update(_grade_one(work, o, timeout_s, False))
        # The torch profiler loses all events now and then with several GPU processes at once (on the train
        # references: 997 of 4443 correct ones had an empty kernel list in pass 1, 0 of 88 when re-run alone), which
        # would read as "no own kernel" (reward 0). Timing re-runs every correct one alone anyway; without timing,
        # re-run alone only the correct ones whose profile came back empty.
        timed = [o for o in todo if o.get("correct") and (timing or not o.get("kernels"))]
    else:
        timed = todo
    for o in timed:
        _abort_if_unhealthy(wd, work / "pass1_cache.jsonl")
        if workers > 1 and not timing:
            o["profile_retry"] = True
        r = _grade_one(work, o, timeout_s, timing)
        if workers > 1 and not r.get("correct"):
            r["flaky_after_pass1"] = True
        o.update(r)


def _abort_if_unhealthy(wd: DriverWatchdog | None, cache_f: Path) -> None:
    """Raise DriverUnhealthy; first drop pass-1 cache rows written after the onset (they may be hang victims)."""
    if wd is None or not wd.check():
        return
    if cache_f.exists():
        rows = [ln for ln in open(cache_f) if ln.strip()]
        late = [json.loads(ln).get("t", 0) >= wd.onset for ln in rows]
        if any(late):
            cache_f.with_name(f"pass1_cache.dropped_{int(wd.onset)}.jsonl").write_text(
                "".join(ln for ln, x in zip(rows, late) if x))
            cache_f.write_text("".join(ln for ln, x in zip(rows, late) if not x))
    raise DriverUnhealthy(wd.why)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gpu", action="store_true", help="run correctness/timing on the GPU (runs model-written code; see REPRODUCE.md §0)")
    ap.add_argument("--scheme", default="zro")
    ap.add_argument("--workers", type=int, default=1, help="parallel correctness containers (timing stays serial)")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per grading container")
    ap.add_argument("--no-timing", action="store_true", help="correctness only (e.g. SFT-data filtering)")
    a = ap.parse_args()
    work = Path(a.work).resolve()
    rows = [json.loads(line) for line in open(a.tasks)]
    outcomes = prepare(rows, work)
    if any(o["format_ok"] for o in outcomes):
        built = compile_all(work, a.jobs)
        for o in outcomes:
            if o["format_ok"]:
                o["compiled"] = built.get(o["name"], False)
        if a.gpu:
            wd = DriverWatchdog().start()
            try:
                run_all(work, outcomes, built, a.timeout, a.workers, timing=not a.no_timing, watchdog=wd)
            except DriverUnhealthy as e:
                print(f"[hip_grade] stopped: amdgpu driver unhealthy ({e}); reboot the VM, then re-run (pass 1 "
                      f"resumes from {work / 'pass1_cache.jsonl'})", file=sys.stderr, flush=True)
                sys.exit(4)
            finally:
                wd.stop()
    with open(a.out, "w") as f:
        for o in outcomes:
            o["reward"] = reward(o, a.scheme)
            f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
