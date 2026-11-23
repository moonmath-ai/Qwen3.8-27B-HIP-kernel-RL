#!/usr/bin/env bash
# Create ZF_SERVER_VENV: the server's extra packages on top of the SGLang ROCm image's own (torch, SGLang,
# transformers stay the image's). Built inside the image so the interpreter matches; the venv lives on the host and
# run_server.sh mounts it. Exact versions of the run, installed without dependencies so nothing replaces the image's
# ROCm torch. (torch_memory_saver is not needed: the profile uses the restart phase mode.)
# Usage: scripts/setup_server_venv.sh       (needs network; ~1 min)
set -euo pipefail
. "$(dirname "$0")/env.sh"
V=$ZF_SERVER_VENV; mkdir -p "$(dirname "$V")"
[ -e "$V" ] && { echo "$V exists; remove it first to rebuild" >&2; exit 1; }
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=user -v "$(dirname "$V"):$(dirname "$V")" --entrypoint bash \
  "${IMAGE:-lmsysorg/sglang:v0.5.21-rocm724-mi30x}" -c "
set -e
/opt/venv/bin/python -m venv --system-site-packages $V
# a plain path line would miss the image venv's nested .pth files; addsitedir processes them
echo \"import site; site.addsitedir('/opt/venv/lib/python3.12/site-packages')\" > $V/lib/python3.12/site-packages/zz_image_site.pth
$V/bin/pip install -q --no-deps tinker==0.33.1 peft==0.21.2 flash-linear-attention==0.5.2 fla-core==0.5.2 \
  anyio==4.16.0 certifi==2026.10.23 click==8.5.1 h11==0.16.0 h2==4.4.1 hpack==4.2.0 httpcore==1.0.9 httpx==0.28.1 \
  hyperframe==6.1.0 idna==3.20 markdown-it-py==4.2.0 mdurl==0.1.2 opentelemetry-api==1.45.2 orjson==3.13.0 \
  pygments==2.21.0 pyqwest==0.14.0 pytest-asyncio==1.4.1.post1 pyyaml==6.0.3 rich==15.0.0 typing-extensions==4.16.0 \
  zstandard==0.25.0
$V/bin/python -c 'import torch, sglang, transformers, peft, tinker, fla; print(\"ok: torch\", torch.__version__, \"transformers\", transformers.__version__, \"peft\", peft.__version__, \"tinker\", tinker.__version__)'
"
