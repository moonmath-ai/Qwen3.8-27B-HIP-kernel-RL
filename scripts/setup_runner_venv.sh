#!/usr/bin/env bash
# Create ZF_RUNNER_VENV: the ROCm PyTorch that builds the graded kernels (gen_ninja.py, hipcc via the host ROCm) and
# runs them in the grading containers. Build and run must use the same torch. Needs uv and network (~5 GB).
# Usage: scripts/setup_runner_venv.sh
set -euo pipefail
. "$(dirname "$0")/env.sh"
[ -e "$ZF_RUNNER_VENV" ] && { echo "$ZF_RUNNER_VENV exists; remove it first to rebuild" >&2; exit 1; }
uv venv "$ZF_RUNNER_VENV" -p 3.12
uv pip install -p "$ZF_RUNNER_VENV" --index-url https://download.pytorch.org/whl/rocm7.2 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  "torch==2.14.1+rocm7.2" ninja==1.13.2 numpy==2.5.4 pybind11==3.1.0
"$ZF_RUNNER_VENV/bin/python" -c 'import torch; print("ok: torch", torch.__version__, "hip", torch.version.hip)'
