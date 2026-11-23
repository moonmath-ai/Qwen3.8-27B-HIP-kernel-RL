"""Stock SGLang server (launched separately with the converted adapter as 'hip'): prompt logprobs. Writes sglang.json."""
import json, sys, urllib.request
C, URL = sys.argv[1], sys.argv[2]
seqs = json.load(open(f"{C}/seqs.json"))
def lp(ids, lora):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True, "logprob_start_len": 0, **({"lora_path": lora} if lora else {})}
    req = urllib.request.Request(f"{URL}/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return [x[0] for x in r["meta_info"]["input_token_logprobs"]]
out = {"adapter": [lp(s["ids"], "hip") for s in seqs], "base": [lp(s["ids"], None) for s in seqs]}
json.dump(out, open(f"{C}/sglang.json", "w")); print("sglang done", flush=True)
