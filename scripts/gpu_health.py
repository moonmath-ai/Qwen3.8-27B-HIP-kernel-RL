"""Quick check that the MI300X is healthy after a GPU fault (REPRODUCE.md §4): bf16 matmul throughput,
memory bandwidth, a 150 GB allocation, and a fast exit. A degraded amdgpu driver shows ~1 TB/s adds and
minutes-long teardown; healthy is ~640 TFLOPS and 3.6-4.1 TB/s. Run with the ROCm torch venv on an idle GPU.

usage: gpu_health.py [--min-tflops 550] [--min-gbs 3000]     exit 0 = healthy, 1 = not
"""
import argparse, json, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--min-tflops", type=float, default=550)
ap.add_argument("--min-gbs", type=float, default=3000)
a = ap.parse_args()


def timed(fn, n):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return time.time() - t


x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
tflops = 2 * 8192**3 * 20 / timed(lambda: x @ x, 20) / 1e12
b = torch.empty(2**28, device="cuda", dtype=torch.float32)
gbs = 2 * 4 * 2**28 * 20 / timed(lambda: b.add_(1), 20) / 1e9
big = torch.empty(150 * 2**30, device="cuda", dtype=torch.uint8)
torch.cuda.synchronize()
ok = tflops >= a.min_tflops and gbs >= a.min_gbs
print(json.dumps({"tflops": round(tflops), "add_gbs": round(gbs), "alloc_150gb": True, "healthy": ok}), flush=True)
sys.exit(0 if ok else 1)
