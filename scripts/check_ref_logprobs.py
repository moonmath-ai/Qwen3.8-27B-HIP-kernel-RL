"""Training-server reference: pick 8 of the checkpoint's own held-out A answers (from its eval), tokenize prompt +
answer with the HF chat template, and get per-token logprobs from the running Zro-form server for the base model
(fresh LoRA) and the checkpoint. Writes CHECK_DIR/seqs.json and ref.json. Run with the client venv, server on :8100:
  python -I scripts/check_ref_logprobs.py CHECK_DIR EVAL_DIR CHECKPOINT      (EVAL_DIR has rollouts.jsonl + graded.jsonl)"""
import json, os, random, sys
from pathlib import Path
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import sample_sglang as smp, hip_rl
import tinker
from tinker import types as t
from transformers import AutoTokenizer
C, E, CK = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
tasks = {x["task"]: x for x in hip_rl.load_tasks(os.environ.get("AIG_DATA", os.path.expanduser("~/data/AIG-Datasets/v0.2")),
         str(A.parent / "data/split_manifest.json"), "heldout_a")}
graded = {json.loads(l)["id"]: json.loads(l) for l in open(f"{E}/graded.jsonl")}
rows = [json.loads(l) for l in open(f"{E}/rollouts.jsonl")]
random.seed(0); random.shuffle(rows)
seqs, levels = [], {}
for r in rows:
    if r["finish"] != "stop" or r["task"] not in tasks: continue
    msgs = [{"role": "system", "content": smp.SYSTEM}, {"role": "user", "content": tasks[r["task"]]["text"]}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, reasoning_effort="low", tokenize=True, return_dict=True)["input_ids"]
    a = tok.encode(r["response"], add_special_tokens=False)
    if len(p) + len(a) > 6000 or levels.get(r["difficulty"], 0) >= 3: continue
    levels[r["difficulty"]] = levels.get(r["difficulty"], 0) + 1
    seqs.append({"id": r["id"], "difficulty": r["difficulty"], "correct": bool(graded.get(r["id"], {}).get("correct")),
                 "n_prompt": len(p), "ids": list(p) + a})
    if len(seqs) == 8: break
json.dump(seqs, open(f"{C}/seqs.json", "w"))
print("seqs", [(s["difficulty"], s["correct"], len(s["ids"])) for s in seqs], flush=True)
if os.environ.get("DRY"): sys.exit(0)
svc = tinker.ServiceClient(base_url="http://127.0.0.1:8100")
tc = hip_rl.training_client(svc, "Qwen/Qwen3.8-27B", 64)
out = {}
sc = tc.save_weights_and_get_sampling_client()
out["base"] = [sc.compute_logprobs(t.ModelInput.from_ints(s["ids"])).result() for s in seqs]
tc.load_state(CK).result()
sc = tc.save_weights_and_get_sampling_client()
out["adapter"] = [sc.compute_logprobs(t.ModelInput.from_ints(s["ids"])).result() for s in seqs]
json.dump(out, open(f"{C}/ref.json", "w"))
print("ref done", flush=True)
