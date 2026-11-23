# Machine paths and images, sourced by every shell script here; Python scripts read the same variables.
# Override any of them in the environment (REPRODUCE.md §1). Defaults are the layout the result was produced with.
: "${ZF_CLIENT_VENV:=$HOME/venvs/zf-client}"     # tinker SDK + tinker-cookbook (requirements-client.txt)
: "${ZF_SERVER_VENV:=$HOME/venvs/zf-amd}"        # server extras on top of the SGLang ROCm image (setup_server_venv.sh)
: "${ZF_RUNNER_VENV:=$HOME/venvs/zf-runner}"     # ROCm PyTorch that builds and runs graded kernels (same ABI for both)
: "${ZF_ROCM:=/opt/rocm-7.2.4}"                  # host ROCm, mounted read-only into the grading containers
: "${AIG_DATA:=$HOME/data/AIG-Datasets/v0.2}"    # amd/AIG-Datasets snapshot (REPRODUCE.md §2)
: "${ZF_SERVER_IMAGE:=lmsysorg/sglang@sha256:d139bdfced47eb5f9345f17a69d227460ba51ba4c1d0eec67a5efe9d560acff8}"  # lmsysorg/sglang:v0.5.21-rocm724-mi30x (REPRODUCE.md §1: optional warm copy)
: "${ZF_GRADER_IMAGE:=vllm/vllm-openai-rocm@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905}"  # only a root filesystem for the grading containers
: "${HF_HOME:=$HOME/.cache/huggingface}"         # model + datasets cache (HF_HOME/hub), mounted read-only into the server
: "${HF_HUB_CACHE:=$HF_HOME/hub}"
: "${JOBS:=$(nproc)}"                            # parallel kernel builds
: "${TINKER_API_KEY:=tml-zroform-dev}"           # any "tml-" key; the server accepts the keys in ZROFORM_API_KEYS
: "${ZROFORM_API_KEYS:=$TINKER_API_KEY}"
export ZF_CLIENT_VENV ZF_SERVER_VENV ZF_RUNNER_VENV ZF_ROCM AIG_DATA ZF_SERVER_IMAGE ZF_GRADER_IMAGE HF_HOME HF_HUB_CACHE JOBS \
       TINKER_API_KEY ZROFORM_API_KEYS
export TINKER_TELEMETRY=0
