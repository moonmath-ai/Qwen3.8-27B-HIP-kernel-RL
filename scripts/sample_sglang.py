"""Sample the base model with plain SGLang (no trainer): tasks of a split (data/split_manifest.json) -> K samples each from a running SGLang
server -> hip_grade.py rows. Grade afterwards with SGLang STOPPED (a candidate's GPU memory fault can kill a decoding
SGLang).

Prompt = the dataset's instruction + context + the nn.Module code (what AIG shows) + the functional version and an
explicit interface: the grader calls `Model(..).forward(*inputs, fn=<extension>.forward)`, i.e. the extension's
`forward` gets `module_fn`'s arguments in order. Without the functional file the model cannot know that signature.

  SGLANG_API_KEY=... python3 -I scripts/sample_sglang.py --data <AIG v0.2> --split heldout_a --k 4 --out rollouts.jsonl

Train split (full_run.sh stage own_sample, for SFT): train entries have no record_ids, so pass --task-records <refs.jsonl from
make_ref_rollouts.py> (task -> the record whose reference is graded), --per-level L1:2000,L2:0,L3:0 (0 = all) and
--concurrency to keep queued requests under the HTTP timeout.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import time
import urllib.request

SYSTEM = ("You are an expert GPU kernel engineer targeting AMD Instinct MI300X (gfx942, ROCm/HIP). "
          "Answer with one complete .hip file in a single ```cpp code block.")
CUDAAGENT = "pytorch_hip_kernel_cuda_agent_ops_6k/pytorch_hip_kernel_cuda_agent_ops_6k.json"
AICUDA = "pytorch_hip_kernel_aicuda_engineer/pytorch_hip_kernel.json"
INTERFACE = """Interface (this is how your file is built and checked):
- It is compiled with torch.utils.cpp_extension for ROCm (hipcc, gfx942) and must define
  `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, ...); }`.
- `forward` replaces `module_fn` in the functional version below: it receives the same arguments in the same order
  (torch.Tensor -> torch::Tensor, float -> double, int -> int64_t, bool -> bool, str -> std::string,
  tuple/list of ints -> std::vector<int64_t>) and must return the same tensor(s), shape and dtype.
- Inputs live on the GPU. Do the computation in your own __global__ kernels.
- For the current stream, `#include <ATen/cuda/CUDAContext.h>` and use `at::cuda::getCurrentCUDAStream()`
  (the build hipifies both)."""


def code(rec: dict, version: str) -> str | None:
    for f in rec.get("question_code", []) + rec.get("benchmark_code", []):
        if version == "module" and (f.get("version") == "module" or
                                    (f.get("language") == "python" and f.get("version") != "functional")):
            return f.get("content")
        if version == "functional" and f.get("version") == "functional":
            return f.get("content")
    return None


def prompt(rec: dict) -> str:
    return (f"{rec['question_instruction']}\n\n{rec.get('question_context', '')}\n\n"
            f"PyTorch module:\n```python\n{code(rec, 'module')}\n```\n\n"
            f"Functional version (your `forward` replaces `module_fn`):\n```python\n{code(rec, 'functional')}\n```\n\n"
            f"{INTERFACE}")


def call(url: str, text: str, n: int, max_tokens: int, effort: str, seed: int) -> dict:
    body = {"model": "default", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            "n": n, "temperature": 1.0, "max_tokens": max_tokens, "seed": seed,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    hdr = {"Content-Type": "application/json"}
    if os.environ.get("SGLANG_API_KEY"):
        hdr["Authorization"] = "Bearer " + os.environ["SGLANG_API_KEY"]
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.load(r)
    return {"latency_s": time.time() - t0, "prompt_tokens": out["usage"]["prompt_tokens"],
            "choices": [{"text": c["message"].get("content") or "", "finish": c["finish_reason"]}
                        for c in out["choices"]]}


def pick_tasks(tasks: list[dict], per_level: str, seed: int) -> list[dict]:
    """per_level "L1:2000,L2:0" -> up to 2000 random L1 tasks, all L2 (0 = all), levels not named are dropped."""
    want = {k: int(v) for k, v in (kv.split(":") for kv in per_level.split(","))}
    rng, out = random.Random(seed), []
    for lv, n in want.items():
        ts = [t for t in tasks if t["difficulty"] == lv]
        out += ts if n == 0 or n >= len(ts) else rng.sample(ts, n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "../data/split_manifest.json"))
    ap.add_argument("--split", default="heldout_a")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-records", help="jsonl with {id: record id, task}: record per task (train split)")
    ap.add_argument("--per-level", help="e.g. L1:2000,L2:0,L3:0 (0 = all of that level)")
    ap.add_argument("--concurrency", type=int, default=0, help="max requests in flight (0 = all at once)")
    a = ap.parse_args()
    tasks = json.load(open(a.manifest))[a.split]
    if a.task_records:
        rid = {json.loads(line)["task"]: json.loads(line)["id"] for line in open(a.task_records)}
        tasks = [{**t, "record_ids": [rid[t["task"]]]} for t in tasks if t["task"] in rid]
    if a.per_level:
        tasks = pick_tasks(tasks, a.per_level, a.seed)
    if a.limit:
        tasks = tasks[:a.limit]
    recs = {}
    for p in (CUDAAGENT, AICUDA):
        for r in json.load(open(os.path.join(a.data, p), encoding="utf-8")):
            recs[r["id"]] = r
    jobs = []
    for t in tasks:
        rec = recs[sorted(t["record_ids"])[0]]  # one fixed record per task
        if code(rec, "functional") and code(rec, "module"):
            jobs.append((t, rec))
    t0 = time.time()
    n_tok = 0
    with open(a.out, "w") as f, cf.ThreadPoolExecutor(a.concurrency or len(jobs)) as ex:
        futs = {ex.submit(call, a.url, prompt(rec), a.k, a.max_tokens, a.effort, a.seed + i): (t, rec)
                for i, (t, rec) in enumerate(jobs)}
        for fut in cf.as_completed(futs):
            t, rec = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                f.write(json.dumps({"id": f"{t['task']}_err", "task": t["task"], "error": repr(e)[:300]}) + "\n")
                continue
            for i, c in enumerate(res["choices"]):
                f.write(json.dumps({"id": f"{t['task']}_{i}", "task": t["task"], "record_id": rec["id"],
                                    "difficulty": t["difficulty"], "finish": c["finish"],
                                    "module": code(rec, "module"), "functional": code(rec, "functional"),
                                    "response": c["text"]}) + "\n")
    print(json.dumps({"tasks": len(jobs), "k": a.k, "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
