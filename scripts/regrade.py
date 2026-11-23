"""Grade saved eval rollouts without resampling, e.g. after the amdgpu driver wedged during an eval's grading
(rollouts.jsonl is written before grading starts). Pass 1 resumes from work/pass1_cache.jsonl; drop the
cache rows written after the fault first. Same grader, reward and metrics as hip_rl.run_eval; appends to
<out>/evals.jsonl. Run with the client venv, the server stopped (the GPU must be free for grading):
  $ZF_CLIENT_VENV/bin/python -I scripts/regrade.py <out_dir> <label> [split]   (label: init for evals.sh)"""
import importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

A = Path(__file__).resolve().parent
def load(name):
    s = importlib.util.spec_from_file_location(name, A / f"{name}.py"); m = importlib.util.module_from_spec(s)
    sys.path.insert(0, str(A)); s.loader.exec_module(m); return m

hr = load("hip_rl"); hg = hr.hg
out, label = Path(sys.argv[1]), sys.argv[2]; split = sys.argv[3] if len(sys.argv) > 3 else "heldout_a"
edir = out / "evals" / label
rows = [json.loads(l) for l in open(edir / "rollouts.jsonl")]
wd = hg.DriverWatchdog().start()
try:
    work = edir / "work"   # pass 1 resumes from work/pass1_cache.jsonl (rows after the onset were dropped)
    outcomes = hg.prepare(rows, work)
    built = hg.compile_all(work, 13) if any(o["format_ok"] for o in outcomes) else {}
    hg.run_all(work, outcomes, built, 120, 6, watchdog=wd)
except hg.DriverUnhealthy as e:
    print(f"[regrade] driver unhealthy: {e}", flush=True); raise SystemExit(4)
for o in outcomes:
    o["reward"] = hg.reward(o, "zro")
with open(edir / "graded.jsonl", "w") as f:
    for o in outcomes:
        f.write(json.dumps(o) + "\n")
groups = defaultdict(list)
for r, o in zip(rows, outcomes):
    groups[r["task"]].append(o["reward"])
m = {"eval": label, "split": split, "regraded": True, **hr.step_metrics(outcomes, groups, [r["finish"] for r in rows])}
by_lv = defaultdict(list)
for r, o in zip(rows, outcomes):
    by_lv[r["difficulty"]].append(bool(o.get("correct")))
m["correct_by_level"] = {lv: sum(v) / len(v) for lv, v in sorted(by_lv.items())}
with open(out / "evals.jsonl", "a") as f:
    f.write(json.dumps(m) + "\n")
print(json.dumps(m), flush=True)
