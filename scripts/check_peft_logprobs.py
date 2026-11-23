"""Transformers + PEFT on the GPU, loading the HF repo folder exactly as a user would. Writes peft.json."""
import json, sys, time, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
import transformers, peft
C, REPO = sys.argv[1], sys.argv[2]
seqs = json.load(open(f"{C}/seqs.json"))
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.8-27B", dtype=torch.bfloat16, device_map="cuda")
m = PeftModel.from_pretrained(m, REPO); m.eval()
print("loaded", type(m.base_model.model).__name__, f"{time.time()-t0:.0f}s", transformers.__version__, peft.__version__, flush=True)
def lp(ids):
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        logits = m(input_ids=x).logits[0, :-1].float()
    return [None] + torch.log_softmax(logits, -1).gather(-1, x[0, 1:, None])[:, 0].tolist()
out = {"adapter": [lp(s["ids"]) for s in seqs]}
with m.disable_adapter():
    out["base"] = [lp(s["ids"]) for s in seqs]
json.dump(out, open(f"{C}/peft.json", "w")); print("peft done", flush=True)
