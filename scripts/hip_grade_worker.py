"""Grade ONE built HIP extension against an AIG-Datasets PyTorch reference. Runs inside a throwaway container
(scripts/hip_grade.py starts it: GPU devices, no network, read-only root, time/memory/pid limits).

Checks, each recorded separately (the reward policy lives in hip_grade.py, not here):
  1. load      the .so imports and exposes `forward`
  2. correct   for each seed, the module file is re-executed after torch.manual_seed(seed) so the inputs AND the
               init weights are fresh (AIG's get_inputs() returns tensors made once at import, which a caching
               kernel could memorise); the reference `Model` (nn version) runs, then the functional `Model` with
               fn=forward; outputs compared with allclose (NaN only where the reference has NaN), shapes and
               dtypes must match, and inputs must be left as the reference leaves them
  3. kernels   GPU kernels launched during one candidate call (torch.profiler): candidate's own __global__ names
               vs anything else (ATen / rocBLAS / hipBLASLt / Tensile "Cijk_" ...); recorded, policy decides
  4. aliasing  candidate output sharing storage with an input when the reference's doesn't
  5. timing    only if correct: median of CUDA-event timings with a 1 GiB L2/infinity-cache flush before each
               iteration, candidate vs reference (PyTorch eager) on the same inputs, same process
Writes one JSON to --out. `--device cpu --fn-py` is for CPU tests (a Python stand-in for the extension).

  python hip_grade_worker.py --module m.py --functional f.py --ext name.so --kernel-names k1,k2 --out r.json
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import time
import traceback

import torch


def exec_file(path: str, seed: int, name: str):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    spec = importlib.util.spec_from_file_location(f"{name}_{seed}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(mod, seed: int, device: str):
    """Model(*get_init_inputs()) with the same RNG state for module and functional versions (as AIG does)."""
    init = mod.get_init_inputs()
    torch.manual_seed(seed)
    if isinstance(init, (list, tuple)) and len(init) == 2 and isinstance(init[0], list) and isinstance(init[1], dict):
        m = mod.Model(*init[0], **init[1])
    else:
        m = mod.Model(*init)
    return m.to(device).eval()


def to_dev(xs, device):
    return [x.to(device).clone() if torch.is_tensor(x) else x for x in xs]


def flatten(o):
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, (list, tuple)):
        return [t for x in o for t in flatten(x)]
    if isinstance(o, dict):
        return [t for k in sorted(o) for t in flatten(o[k])]
    return [o]


def compare(ref, cand, rtol: float, atol: float) -> dict:
    r, c = flatten(ref), flatten(cand)
    if len(r) != len(c):
        return {"ok": False, "why": f"output count {len(c)} != {len(r)}"}
    worst = 0.0
    for a, b in zip(r, c):
        if torch.is_tensor(a) != torch.is_tensor(b):
            return {"ok": False, "why": "tensor vs non-tensor output"}
        if not torch.is_tensor(a):
            if a != b:
                return {"ok": False, "why": "non-tensor output differs"}
            continue
        if a.shape != b.shape:
            return {"ok": False, "why": f"shape {tuple(b.shape)} != {tuple(a.shape)}"}
        if a.dtype != b.dtype:
            return {"ok": False, "why": f"dtype {b.dtype} != {a.dtype}"}
        if a.numel() == 0:
            continue
        af, bf = a.detach().double(), b.detach().double()
        if not torch.equal(torch.isnan(af), torch.isnan(bf)):
            return {"ok": False, "why": "NaN pattern differs"}
        if not torch.allclose(af, bf, rtol=rtol, atol=atol, equal_nan=True):
            d = (af - bf).abs().nan_to_num(0)
            return {"ok": False, "why": "values differ", "max_abs_err": float(d.max())}
        d = (af - bf).abs().nan_to_num(0)
        worst = max(worst, float(d.max()))
    return {"ok": True, "max_abs_err": worst}


def shares_storage(outs, inputs) -> bool:
    ptrs = {x.untyped_storage().data_ptr() for x in inputs if torch.is_tensor(x) and x.numel()}
    return any(torch.is_tensor(o) and o.numel() and o.untyped_storage().data_ptr() in ptrs for o in flatten(outs))


def launched_kernels(call, device: str) -> dict[str, float]:
    """GPU kernel name -> total device time (us) during one call."""
    if device == "cpu":
        return {}
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    us: dict[str, float] = {}
    for e in prof.events():
        if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type):
            t = getattr(e, "device_time", None)
            t = getattr(e, "cuda_time", 0.0) if t is None else t
            us[e.name] = us.get(e.name, 0.0) + float(t or 0.0)
    return us


def timed(call, device: str, flush: torch.Tensor | None, budget_s: float = 0.5, max_iters: int = 200) -> float:
    """Median ms over iterations, each after an L2 flush; GPU events around the call only."""
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    times, t_end = [], time.time() + budget_s
    while len(times) < max_iters and (len(times) < 10 or time.time() < t_end):
        if flush is not None:
            flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        call()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def load_fn(a):
    if a.fn_py:  # CPU tests: a Python file defining forward(...)
        return exec_file(a.fn_py, 0, "fn").forward
    loader = importlib.machinery.ExtensionFileLoader(a.ext_name, a.ext)
    spec = importlib.util.spec_from_file_location(a.ext_name, a.ext, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod.forward


def grade(a) -> dict:
    res = {"loaded": False, "correct": False, "seeds": [], "error_type": None}
    dev = a.device
    try:
        fn = load_fn(a)
        res["loaded"] = True
    except Exception as e:  # noqa: BLE001
        res.update(error_type="load", error=f"{type(e).__name__}: {str(e)[:300]}")
        return res
    own = [k for k in (a.kernel_names or "").split(",") if k]
    for seed in [int(s) for s in a.seeds.split(",")]:
        rec = {"seed": seed}
        res["seeds"].append(rec)
        try:
            modu = exec_file(a.module, seed, "modu")
            func = exec_file(a.functional, seed, "func")
            ref_m, cand_m = build(modu, seed, dev), build(func, seed, dev)
            inputs = modu.get_inputs()
            x_ref, x_cand = to_dev(inputs, dev), to_dev(inputs, dev)
            with torch.no_grad():
                ref = ref_m(*x_ref)
                cand = cand_m(*x_cand, fn=fn)
            if dev != "cpu":
                torch.cuda.synchronize()
            cmp = compare(ref, cand, a.rtol, a.atol)
            in_cmp = compare(x_ref, x_cand, 0.0, 0.0)
            rec.update(cmp)
            if cmp["ok"] and not in_cmp["ok"]:
                rec.update(ok=False, why="inputs left in a different state than the reference leaves them")
            if shares_storage(cand, x_cand) and not shares_storage(ref, x_ref):
                rec.update(ok=False, why="output aliases an input")
        except torch.cuda.OutOfMemoryError:
            rec.update(ok=False, why="out of memory")
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, why=f"runtime {type(e).__name__}: {str(e)[:200]}")
        if not rec["ok"]:
            res["error_type"] = "runtime" if rec["why"].startswith(("runtime", "out of memory")) else "mismatch"
            res["why"] = rec["why"]
            return res
    res["correct"] = True
    # kernels and timing on the last seed's models/inputs
    with torch.no_grad():
        us = launched_kernels(lambda: cand_m(*x_cand, fn=fn), dev)
    names = sorted(us)
    res["kernels"] = names
    res["own_kernels"] = [n for n in names if any(k in n for k in own)]
    res["other_kernels"] = [n for n in names if not any(k in n for k in own)]
    # library-share monitor (docs/REWARD.md): GPU time in own vs other kernels
    res["own_us"] = round(sum(us[n] for n in res["own_kernels"]), 2)
    res["other_us"] = round(sum(us[n] for n in res["other_kernels"]), 2)
    if dev != "cpu" and not a.no_timing:
        flush = torch.empty(1 << 28, dtype=torch.float32, device=dev)  # 1 GiB > MI300X L2 + Infinity Cache
        with torch.no_grad():
            t_ref = timed(lambda: ref_m(*x_ref), dev, flush)
            t_cand = timed(lambda: cand_m(*x_cand, fn=fn), dev, flush)
        res.update(ref_ms=t_ref, cand_ms=t_cand, speedup=(t_ref / t_cand) if t_cand > 0 else None)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--functional", required=True)
    ap.add_argument("--ext", help="built .so")
    ap.add_argument("--ext-name", default="candidate", help="module name the .so was built with")
    ap.add_argument("--fn-py", help="CPU tests: Python file defining forward(...) instead of --ext")
    ap.add_argument("--kernel-names", default="", help="comma-separated __global__ names from the source")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    try:
        res = grade(a)
    except Exception:  # noqa: BLE001 - the harness itself failed: never a candidate verdict
        res = {"harness_error": traceback.format_exc()[-2000:]}
    res["worker_s"] = round(time.time() - t0, 3)
    with open(a.out, "w") as f:
        json.dump(res, f, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else str(x))


if __name__ == "__main__":
    main()
