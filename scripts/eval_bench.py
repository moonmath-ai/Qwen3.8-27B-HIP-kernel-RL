"""Regression/quality benchmarks through the training server, via the unmodified Tinker SDK.

Runs tinker-cookbook's benchmark suite on the base model and on each adapter, and reports each
benchmark's score with a 95% CI and the adapter-minus-base difference. Benchmarks needing a sandbox
(mbpp, livecodebench) or a judge model are not in the default set.

  (. scripts/env.sh) $ZF_CLIENT_VENV/bin/python scripts/eval_bench.py \
      --adapter step75=tinker://<session>/weights/keep0075 \
      --bench math500 gsm8k ifeval mmlu_pro --out <dir>/bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.eval.benchmarks import run_benchmarks
from tinker_cookbook.eval.benchmarks._types import BenchmarkConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


async def main(a) -> None:
    tok = get_tokenizer(a.model)
    renderer = renderers.get_renderer(a.renderer or model_info.get_recommended_renderer_name(a.model), tok)
    svc = tinker.ServiceClient(base_url=a.base_url)
    targets = {**({} if a.skip_base else {"base": None}), **dict(x.split("=", 1) for x in a.adapter)}
    results: dict[str, dict] = {}
    try:
        for name, path in targets.items():
            sc = (svc.create_sampling_client(base_model=a.model) if path is None
                  else svc.create_sampling_client(model_path=path))
            cfg = BenchmarkConfig(max_examples=a.max_examples or None, max_tokens=a.max_tokens,
                                  temperature=a.temperature, num_samples=a.num_samples, timeout_seconds=a.timeout,
                                  save_dir=str(Path(a.out).parent / f"bench_{name}"))
            res = await run_benchmarks(a.bench, sc, renderer, cfg)
            results[name] = {b: {"score": r.score, "n": r.num_examples, "errors": r.num_errors,
                                 "truncated": r.num_truncated, "seconds": round(r.time_seconds, 1)}
                             for b, r in res.items()}
    finally:
        svc.close("success").result()

    print(f"{'benchmark':<12}" + "".join(f"{t:>22}" for t in targets))
    for b in a.bench:
        row = f"{b:<12}"
        for t in targets:
            r = results[t][b]
            cell = f"{100 * r['score']:.1f}±{100 * ci95(r['score'], r['n']):.1f}"
            if t != "base" and "base" in results:
                cell += f" ({100 * (r['score'] - results['base'][b]['score']):+.1f})"
            row += f"{cell:>22}"
        print(row)
    print("(± = 95% CI per score; (+x) = adapter minus base, in points)")
    Path(a.out).write_text(json.dumps({"model": a.model, "config": vars(a), "results": results}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", action="append", default=[], help="NAME=tinker://... (repeatable)")
    ap.add_argument("--bench", nargs="+", default=["math500", "gsm8k", "ifeval", "mmlu_pro"])
    ap.add_argument("--max-examples", type=int, default=0, help="per benchmark; 0 = all")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--renderer", help="default: the cookbook's recommended renderer for the model")
    ap.add_argument("--skip-base", action="store_true", help="evaluate only the --adapter models")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-samples", type=int, default=1)
    # per-example; the cookbook default (300 s) counts queueing on a busy server as an error
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
