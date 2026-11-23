"""PyTorch -> HIP RL on one MI300X: a minimal synchronous loop against the Zro-form server (Tinker SDK 0.33.1).

One step = sample -> grade -> train, never overlapping (RESULTS.md §2):
  1. sample   B train tasks x K samples through the server (SGLang decodes).
  2. grade    hip_grade.py stages on the finished batch. First a zero-weight forward_backward moves the server's
              PhaseGate to "train", which in `restart` mode STOPS SGLang: a candidate's GPU memory fault can kill a
              decoding SGLang, and stopping it also gives the grading containers its memory.
  3. train    group-centred advantages (same reward formula for every task: hip_grade.reward "zro"),
              importance_sampling forward_backward + optim_step, then save_weights_and_get_sampling_client
              (the next sample restarts SGLang with the new adapter, ~52 s).

Task sampling does the weighting: levels drawn with --level-weights, uniform within a level; a task whose
last --dead-after groups all had zero reward variance (all pass or all fail) is skipped; --exclude drops tasks (e.g.
references that are a single library op). Groups with zero variance give zero advantage and are not sent.

Per step: <out>/metrics.jsonl (one line), <out>/steps/<n>/rollouts.jsonl (+ graded outcomes) for inspection.

  (. scripts/env.sh) $ZF_CLIENT_VENV/bin/python scripts/hip_rl.py \\
      --data ~/data/AIG-Datasets/v0.2 --out <run_dir>/rl --steps 2 --groups 4 --k 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


hg = _load("hip_grade")
wd = None   # hip_grade.DriverWatchdog, started under __main__
smp = _load("sample_sglang")   # prompt(), SYSTEM, code(): the prompt of every stage


# ---- pure parts (unit-tested) -----------------------------------------------------------------------------
class TaskSampler:
    """Level-weighted task draw that skips excluded and 'dead' tasks (no reward variance in the last N groups)."""

    def __init__(self, tasks: list[dict], level_weights: dict[str, float], dead_after: int = 2,
                 exclude: set[str] = frozenset(), seed: int = 0):
        self.by_level: dict[str, list[dict]] = defaultdict(list)
        for t in tasks:
            if t["task"] not in exclude:
                self.by_level[t["difficulty"]].append(t)
        self.w = {lv: w for lv, w in level_weights.items() if self.by_level.get(lv)}
        self.dead_after, self.rng = dead_after, random.Random(seed)
        self.history: dict[str, list[bool]] = defaultdict(list)   # task -> [group had variance?]

    def dead(self, task: str) -> bool:
        h = self.history[task]
        return self.dead_after > 0 and len(h) >= self.dead_after and not any(h[-self.dead_after:])

    def draw(self, n: int) -> list[dict]:
        out, seen = [], set()
        for _ in range(n * 50):
            if len(out) == n:
                break
            lv = self.rng.choices(list(self.w), weights=list(self.w.values()))[0]
            t = self.rng.choice(self.by_level[lv])
            if t["task"] in seen or self.dead(t["task"]):
                continue
            seen.add(t["task"])
            out.append(t)
        return out

    def record(self, task: str, rewards: list[float]) -> None:
        self.history[task].append(max(rewards) - min(rewards) > 1e-9)

    def state(self) -> dict:
        return {"history": dict(self.history), "rng": self.rng.getstate()}


def advantages(rewards: list[float]) -> list[float]:
    """Group-centred (the cookbook's compute_advantages): reward minus the group mean, no std scaling."""
    m = sum(rewards) / len(rewards)
    return [r - m for r in rewards]


def rl_datum_fields(prompt: list[int], tokens: list[int], logprobs: list[float], adv: float) -> dict:
    """Inputs/targets for one sampled sequence, shifted by one, prompt positions masked (advantage 0)."""
    n = len(prompt)
    return {"model_input": prompt + tokens[:-1],
            "target_tokens": prompt[1:] + tokens,
            "logprobs": [0.0] * (n - 1) + list(logprobs),
            "advantages": [0.0] * (n - 1) + [adv] * len(tokens)}


def step_metrics(outcomes: list[dict], groups: dict[str, list[float]], finishes: list[str]) -> dict:
    n = max(len(outcomes), 1)
    mixed = sum(1 for r in groups.values() if max(r) - min(r) > 1e-9)
    correct = [o for o in outcomes if o.get("correct")]
    sp = sorted(o["speedup"] for o in correct if isinstance(o.get("speedup"), (int, float)))
    own = sum(o.get("own_us") or 0.0 for o in correct)
    other = sum(o.get("other_us") or 0.0 for o in correct)
    return {
        "n": len(outcomes),
        "reward_mean": sum(o["reward"] for o in outcomes) / n,
        "format_ok": sum(bool(o.get("format_ok")) for o in outcomes) / n,
        "truncated": sum(f == "length" for f in finishes) / max(len(finishes), 1),
        "compiled": sum(bool(o.get("compiled")) for o in outcomes) / n,
        "correct": len(correct) / n,
        "forbidden": sum(bool(o.get("forbidden")) for o in outcomes) / n,
        "groups": len(groups), "groups_mixed": mixed,
        "speedup_median": sp[len(sp) // 2] if sp else None,
        "speedup_ge_1_1": sum(s >= 1.1 for s in sp),
        # library-share monitor (docs/REWARD.md): GPU time in non-candidate kernels among correct rollouts
        "other_time_share": (other / (own + other)) if own + other > 0 else None,
        "pass1_retries": sum(1 for o in outcomes if o.get("pass1_error")),
    }


def parse_levels(s: str) -> dict[str, float]:
    return {k: float(v) for k, v in (kv.split(":") for kv in s.split(","))}


# ---- the loop (needs the server) --------------------------------------------------------------------------
def load_tasks(data: str, manifest: str, split_name: str, task_records: str | None = None) -> list[dict]:
    """Held-out/validation entries carry record_ids; train entries don't: task_records (make_ref_rollouts.py output,
    {id: record id, task}) names the record per task, the same one whose reference answer was graded."""
    tasks = json.load(open(manifest))[split_name]
    if not task_records and any("record_ids" not in t for t in tasks):
        raise ValueError(f"split {split_name!r} has no record ids in the manifest: pass --task-records")
    if task_records:
        rid = {}
        for line in open(task_records):
            r = json.loads(line)
            rid[r["task"]] = r["id"]
        tasks = [{**t, "record_ids": [rid[t["task"]]]} for t in tasks if t["task"] in rid]
    recs = {}
    for p in (smp.CUDAAGENT, smp.AICUDA):
        for r in json.load(open(os.path.join(data, p), encoding="utf-8")):
            recs[r["id"]] = r
    out = []
    for t in tasks:
        rec = recs[sorted(t["record_ids"])[0]]  # one fixed record per task
        if smp.code(rec, "functional") and smp.code(rec, "module"):
            out.append({"task": t["task"], "difficulty": t["difficulty"], "record_id": rec["id"],
                        "text": smp.prompt(rec), "module": smp.code(rec, "module"),
                        "functional": smp.code(rec, "functional")})
    return out



def training_client(svc, model: str, rank: int, wait_s: float = 300.0):
    """The server has one training slot; a previous client's slot frees only when the server reaps its session
    (no heartbeat for 60 s, zroform/api.py SESSION_IDLE_SEC). Retry until then instead of failing."""
    t_end = time.time() + wait_s
    while True:
        try:
            return svc.create_lora_training_client(base_model=model, rank=rank)
        except Exception as e:  # tinker.RequestFailedError
            if "no free training slot" not in str(e) or time.time() > t_end:
                raise
            print("training slot busy (previous run's session not reaped yet); retrying in 15 s", flush=True)
            time.sleep(15)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--renderer", default="qwen3_8_low_reasoning")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", default=str(HERE.parent / "data/split_manifest.json"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--groups", type=int, default=16, help="tasks per step")
    ap.add_argument("--k", type=int, default=4, help="samples per task")
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--max-seq", type=int, default=16384, help="server max_model_len (prompt + response)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--loss", default="importance_sampling")
    ap.add_argument("--level-weights", default="L1:0.4,L2:0.3,L3:0.3")
    ap.add_argument("--dead-after", type=int, default=2)
    ap.add_argument("--exclude", help="file with one task id per line to skip")
    ap.add_argument("--task-records", help="train split: jsonl {id, task} (make_ref_rollouts.py refs.jsonl)")
    ap.add_argument("--init", help="tinker:// state to start from, weights only (e.g. the SFT result)")
    ap.add_argument("--eval-split", default="heldout_a")
    ap.add_argument("--eval-every", type=int, default=0, help="eval on --eval-split every N steps (0 = never)")
    ap.add_argument("--eval-k", type=int, default=4)
    ap.add_argument("--eval-limit", type=int, default=0, help="first N eval tasks only (smoke tests)")
    ap.add_argument("--eval-only", action="store_true", help="evaluate the --init/--resume weights, no training")
    ap.add_argument("--sampler-path", help="with --eval-only: sample from these tinker:// sampler weights instead, e.g. "
                    "an imported HF adapter (scripts/import_adapter.py)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", os.cpu_count() or 8)))
    # checkpoint policy: rolling resume states expire, milestones are kept; sampler weights per
    # step live in RAM only (save_weights_and_get_sampling_client persists nothing)
    ap.add_argument("--save-every", type=int, default=5, help="rolling resume state (weights + Adam)")
    ap.add_argument("--rolling-ttl", type=int, default=86400, help="seconds a rolling state is kept")
    ap.add_argument("--keep-every", type=int, default=25, help="milestone states kept without TTL (+ last step)")
    ap.add_argument("--min-free-gb", type=float, default=50.0, help="save and stop when the store's disk has less free")
    ap.add_argument("--store-dir", default=".", help="a path on the checkpoint store's filesystem (for --min-free-gb)")
    ap.add_argument("--resume", help="tinker:// state path from a previous save (optimizer included)")
    ap.add_argument("--start-step", type=int, default=0, help="with --resume: the step that state was saved after")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import tinker
    from tinker import types as t
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    out = Path(a.out).expanduser()
    (out / "steps").mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)
    tok = get_tokenizer(a.model)
    renderer = renderers.get_renderer(a.renderer, tok)
    exclude = set(Path(a.exclude).read_text().split()) if a.exclude else set()
    # --eval-only trains nothing: don't load (or require --task-records for) the training split
    train_tasks = [] if a.eval_only else load_tasks(a.data, a.manifest, a.split, a.task_records)
    sampler = TaskSampler(train_tasks, parse_levels(a.level_weights), a.dead_after, exclude, a.seed)
    eval_tasks = load_tasks(a.data, a.manifest, a.eval_split) if (a.eval_every or a.eval_only) else []
    if a.eval_limit:
        eval_tasks = eval_tasks[:a.eval_limit]

    svc = tinker.ServiceClient(base_url=a.base_url)
    tc = training_client(svc, a.model, a.rank)
    if a.resume:
        tc.load_state_with_optimizer(a.resume).result()
    elif a.init:
        tc.load_state(a.init).result()
    if a.sampler_path and not a.eval_only:
        raise SystemExit("--sampler-path needs --eval-only")
    # the training client stays: its forward_backward switches the phased server to the train phase before grading
    sc = svc.create_sampling_client(model_path=a.sampler_path) if a.sampler_path else tc.save_weights_and_get_sampling_client()
    flip = t.Datum(model_input=t.ModelInput.from_ints([0]), loss_fn_inputs={"target_tokens": [0], "weights": [0.0]})

    def sample_and_grade(tasks, k, tag, sdir, seed_base):
        """Sample k per task with the current sampling client, then stop SGLang and grade. Returns rows, seqs,
        outcomes, timings."""
        tm, t0 = {}, time.time()
        prompts, futs = [], []
        for i, task in enumerate(tasks):
            msgs = [{"role": "system", "content": smp.SYSTEM}, {"role": "user", "content": task["text"]}]
            p = renderer.build_generation_prompt(msgs).to_ints()
            mt = min(a.max_tokens, a.max_seq - len(p) - 1)
            prompts.append(p)
            futs.append(sc.sample(t.ModelInput.from_ints(p), k, t.SamplingParams(
                max_tokens=mt, temperature=a.temperature, seed=seed_base + i, stop=renderer.get_stop_sequences())))
        rows, seqs = [], []
        for task, p, f in zip(tasks, prompts, futs):
            for j, s in enumerate(f.result().sequences):
                toks = list(s.tokens)
                rows.append({"id": f"{tag}_{task['task']}_{j}", "task": task["task"],
                             "difficulty": task["difficulty"], "finish": str(s.stop_reason).split(".")[-1],
                             "module": task["module"], "functional": task["functional"],
                             "response": tok.decode(toks, skip_special_tokens=False)})
                seqs.append((p, toks, list(s.logprobs)))
        tm["sample_s"] = time.time() - t0
        with open(sdir / "rollouts.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        t1 = time.time()
        tc.forward_backward([flip], "cross_entropy").result()   # PhaseGate -> train: SGLang stops (restart mode)
        tm["phase_to_train_s"] = time.time() - t1
        t1 = time.time()
        work = sdir / "work"
        outcomes = hg.prepare(rows, work)
        built = hg.compile_all(work, a.jobs) if any(o["format_ok"] for o in outcomes) else {}
        hg.run_all(work, outcomes, built, a.timeout, a.workers, watchdog=wd)   # DriverUnhealthy -> exit 4
        for o in outcomes:
            o["reward"] = hg.reward(o, "zro")
        tm["grade_s"] = time.time() - t1
        with open(sdir / "graded.jsonl", "w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")
        return rows, seqs, outcomes, tm

    def run_eval(label: str) -> dict:
        edir = out / "evals" / label
        edir.mkdir(parents=True, exist_ok=True)
        rows, _, outcomes, tm = sample_and_grade(eval_tasks, a.eval_k, f"e{label}", edir, 10_000_000)
        groups: dict[str, list[float]] = defaultdict(list)
        for r, o in zip(rows, outcomes):
            groups[r["task"]].append(o["reward"])
        m = {"eval": label, "split": a.eval_split, **step_metrics(outcomes, groups, [r["finish"] for r in rows]),
             **{k: round(v, 1) for k, v in tm.items()}}
        by_lv: dict[str, list[bool]] = defaultdict(list)
        for r, o in zip(rows, outcomes):
            by_lv[r["difficulty"]].append(bool(o.get("correct")))
        m["correct_by_level"] = {lv: sum(v) / len(v) for lv, v in sorted(by_lv.items())}
        with open(out / "evals.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        print(json.dumps({k: m[k] for k in ("eval", "correct", "reward_mean", "correct_by_level")}), flush=True)
        return m

    if a.eval_only:
        run_eval("init")
        return

    hist = out / "task_history.json"
    if a.start_step and hist.exists():
        for task, h in json.load(open(hist)).items():
            sampler.history[task] = h
    for step in range(a.start_step, a.steps):
        sdir = out / "steps" / f"{step:04d}"
        sdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        tasks = sampler.draw(a.groups)
        rows, seqs, outcomes, tm = sample_and_grade(tasks, a.k, f"s{step}", sdir, a.seed * 1_000_003 + step * 1000)
        # 3. train ------------------------------------------------------------------------------------------
        t1 = time.time()
        groups: dict[str, list[float]] = defaultdict(list)
        for r, o in zip(rows, outcomes):
            groups[r["task"]].append(o["reward"])
        adv = {task: advantages(rs) for task, rs in groups.items()}
        seen: dict[str, int] = defaultdict(int)
        data = []
        for r, (p, toks, lps) in zip(rows, seqs):
            a_i = adv[r["task"]][seen[r["task"]]]
            seen[r["task"]] += 1
            if abs(a_i) < 1e-9 or not toks:
                continue
            d = rl_datum_fields(p, toks, lps, a_i)
            data.append(t.Datum(model_input=t.ModelInput.from_ints(d["model_input"]),
                                loss_fn_inputs={k: d[k] for k in ("target_tokens", "logprobs", "advantages")}))
        fb = tc.forward_backward(data, a.loss).result() if data else None
        if data:
            tc.optim_step(t.AdamParams(learning_rate=a.lr)).result()
        tm["train_s"] = time.time() - t1
        for task, rs in groups.items():
            sampler.record(task, rs)
        t1 = time.time()
        sc = tc.save_weights_and_get_sampling_client()
        tm["save_s"] = time.time() - t1
        ckpt, n = None, step + 1
        free_gb = shutil.disk_usage(a.store_dir).free / 1e9
        low_disk = free_gb < a.min_free_gb
        if (a.keep_every and n % a.keep_every == 0) or n == a.steps or low_disk:
            ckpt = tc.save_state(f"keep{n:04d}").result().path                     # milestone: no TTL
        elif a.save_every and n % a.save_every == 0:
            ckpt = tc.save_state(f"roll{n:04d}", ttl_seconds=a.rolling_ttl).result().path
        m = {"step": step, **step_metrics(outcomes, groups, [r["finish"] for r in rows]),
             "datums": len(data), "tokens": sum(len(s[1]) for s in seqs),
             "loss_metrics": dict(fb.metrics) if fb else None, "ckpt": ckpt, "free_gb": round(free_gb),
             **{k: round(v, 1) for k, v in tm.items()}, "step_s": round(time.time() - t0, 1)}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        json.dump(sampler.state()["history"], open(out / "task_history.json", "w"))
        print(json.dumps({k: m[k] for k in ("step", "reward_mean", "correct", "groups_mixed", "datums", "step_s")}),
              flush=True)
        if a.eval_every and (n % a.eval_every == 0 or n == a.steps):
            run_eval(f"step{n:04d}")
        if low_disk:
            print(f"STOP: {free_gb:.0f} GB free < --min-free-gb {a.min_free_gb}; state saved at {ckpt}", flush=True)
            break


if __name__ == "__main__":
    wd = hg.DriverWatchdog().start()   # a wedged amdgpu driver stops the run: reboot, then --resume
    try:
        main()
    except hg.DriverUnhealthy as e:
        print(f"[hip_rl] stopped: amdgpu driver unhealthy ({e}); reboot the VM, then resume from the last "
              f"checkpoint (--resume CK --start-step N)", flush=True)
        raise SystemExit(4)
