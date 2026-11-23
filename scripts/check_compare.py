"""Mean |logprob difference| per answer token: each path vs the training server, and adapter vs base."""
import json, sys
C = sys.argv[1]; seqs = json.load(open(f"{C}/seqs.json")); ref = json.load(open(f"{C}/ref.json"))
def mad(a, b):
    d = [abs(x - y) for s, xa, ya in zip(seqs, a, b) for x, y in zip(xa[s["n_prompt"]:], ya[s["n_prompt"]:])]
    return sum(d) / len(d), len(d)
def mean_lp(a):
    v = [x for s, xa in zip(seqs, a) for x in xa[s["n_prompt"]:]]; return sum(v) / len(v)
res = {"tokens": mad(ref["adapter"], ref["base"])[1],
       "train_server: adapter vs base": mad(ref["adapter"], ref["base"])[0],
       "mean answer logprob: base / adapter (train server)": (mean_lp(ref["base"]), mean_lp(ref["adapter"]))}
for name in ("peft", "sglang"):
    try: o = json.load(open(f"{C}/{name}.json"))
    except FileNotFoundError: continue
    res[f"{name} adapter vs train_server adapter"] = mad(o["adapter"], ref["adapter"])[0]
    res[f"{name} base vs train_server base"] = mad(o["base"], ref["base"])[0]
    res[f"{name} adapter vs train_server base"] = mad(o["adapter"], ref["base"])[0]
print(json.dumps(res, indent=1)); json.dump(res, open(f"{C}/compare.json", "w"), indent=1)
