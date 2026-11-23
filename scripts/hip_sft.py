"""SFT stage before RL (RESULTS.md §4): LoRA cross-entropy on build_sft_data.py rows through the
Zro-form server, with exactly the RL prompt (renderer qwen3_8_low_reasoning, sample_sglang.SYSTEM + the row's user text).

Loss only on the completion (the text after the generation prompt's "<think>\\n", plus the end-of-turn token).
Sequences longer than --max-seq are dropped. Linear LR decay to 0 over all steps. Checkpoints follow the RL policy
(hip_rl.py): rolling states with a TTL, the final state kept.

  (. scripts/env.sh) $ZF_CLIENT_VENV/bin/python scripts/hip_sft.py \\
      --sft sft.jsonl --out <run_dir>/sft --epochs 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sample_sglang", HERE / "sample_sglang.py")
smp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smp)


def sft_fields(prompt: list[int], completion: list[int]) -> dict:
    """Shifted inputs/targets; weight 1 on completion targets only."""
    toks = prompt + completion
    return {"model_input": toks[:-1], "target_tokens": toks[1:],
            "weights": [0.0] * (len(prompt) - 1) + [1.0] * len(completion)}


def batches(n: int, size: int, epochs: int, seed: int) -> list[list[int]]:
    rng, out = random.Random(seed), []
    for _ in range(epochs):
        idx = list(range(n))
        rng.shuffle(idx)
        out += [idx[i:i + size] for i in range(0, n, size)]
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
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq", type=int, default=16384)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--rolling-ttl", type=int, default=86400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import tinker
    from tinker import types as t
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)
    tok = get_tokenizer(a.model)
    renderer = renderers.get_renderer(a.renderer, tok)
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    rows = [json.loads(line) for line in open(a.sft)]
    ex, dropped = [], 0
    for r in rows:
        msgs = [{"role": "system", "content": smp.SYSTEM}, {"role": "user", "content": r["user"]}]
        p = renderer.build_generation_prompt(msgs).to_ints()
        c = tok.encode(r["completion"], add_special_tokens=False) + [eot]
        if len(p) + len(c) > a.max_seq:
            dropped += 1
            continue
        ex.append(sft_fields(p, c))
    plan = batches(len(ex), a.batch, a.epochs, a.seed)
    print(json.dumps({"examples": len(ex), "dropped_too_long": dropped, "steps": len(plan),
                      "completion_tokens": sum(sum(e["weights"]) for e in ex)}), flush=True)

    svc = tinker.ServiceClient(base_url=a.base_url)
    tc = training_client(svc, a.model, a.rank)
    for step, idx in enumerate(plan):
        t0 = time.time()
        data = [t.Datum(model_input=t.ModelInput.from_ints(ex[i]["model_input"]),
                        loss_fn_inputs={"target_tokens": ex[i]["target_tokens"], "weights": ex[i]["weights"]})
                for i in idx]
        fb = tc.forward_backward(data, "cross_entropy").result()
        lr = a.lr * (1 - step / len(plan))
        tc.optim_step(t.AdamParams(learning_rate=lr)).result()
        n_tok = sum(sum(ex[i]["weights"]) for i in idx)
        ckpt = None
        if step + 1 == len(plan):
            ckpt = tc.save_state("sft_final").result().path
        elif a.save_every and (step + 1) % a.save_every == 0:
            ckpt = tc.save_state(f"sft{step + 1:04d}", ttl_seconds=a.rolling_ttl).result().path
        m = {"step": step, "lr": lr, "n": len(idx), "completion_tokens": n_tok,
             "loss_per_token": (fb.metrics.get("loss:sum") or 0.0) / max(n_tok, 1), "ckpt": ckpt,
             "step_s": round(time.time() - t0, 1)}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        print(json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
