# Reproducing the Qwen3.8-27B HIP kernel RL result

End to end: environment, data, grader checks, the training run (SFT, then 125 RL steps), the evals and regressions,
the adapter export and its checks. §7 evaluates the released adapter without training. What happened in the run,
and why each choice was made: [RESULTS.md](RESULTS.md).

## 0. What you need, and the hazard

| | Used for the result | Notes |
|---|---|---|
| GPU | 1× AMD Instinct MI300X (192 GB; a virtual function in a VM), gfx942 | trainer and sampler take turns on the one GPU (docs/SERVER.md) |
| CPU, RAM | 13 vCPUs, 220 GB RAM | kernel builds run `JOBS` (default: the vCPU count) at a time |
| Host | Ubuntu 24.04, kernel 6.8.0-124, ROCm 7.2.4, amdgpu-dkms 6.16.13 | see the kernel note in §1 |
| Docker | the user in the `docker`, `video` and `render` groups | the server, every build and every graded kernel run in containers |
| Disk | ~300 GB | model 55 GB, checkpoint store ~160 GB (5.1 GB per saved state), images, grading work dirs (~1 MB per answer) |
| Time | ~5 days | data + SFT ~1 day; RL ~41 min per step × 125; each 200 × 4 eval ~3 h |

Versions of the result: SGLang 0.5.21 ROCm image (torch 2.11.0+rocm7.2, transformers 5.12.1) + peft 0.21.2,
tinker 0.33.1, flash-linear-attention 0.5.2 (server); torch 2.14.1+rocm7.2 (kernel build and grading);
tinker 0.33.1, tinker-cookbook 0.5.7 (client); model revision `37561816e66b`; dataset revision `d43e2530`.

**Hazard: the grader runs model-written GPU code.** Each answer is built in a container with no network and no GPU,
and run in its own throwaway container (no network, read-only root, non-root, memory/pid/time limits), so it cannot
reach your files or the network. It *can* fault the GPU: a GPU reset is recoverable (it kills the training server;
the supervisor checks the GPU and resumes from the last checkpoint), a wedged amdgpu driver needs a reboot. Both
happened during our run; use a machine where that is acceptable (§4, exit codes).

## 1. Environment

```bash
git clone https://github.com/moonmath-ai/Qwen3.8-27B-HIP-kernel-RL && cd Qwen3.8-27B-HIP-kernel-RL
```

**Host.** ROCm 7.2.4 with the amdgpu-dkms driver. Unattended kernel upgrades can boot a kernel the dkms driver was
not built for; the stock driver then passes simple checks and fails under load. After every reboot:
`modinfo -F filename amdgpu` must point to `updates/dkms/`.

**Paths and images** are in [`scripts/env.sh`](scripts/env.sh); every script sources it. Override any of
them in your environment (`ZF_CLIENT_VENV`, `ZF_SERVER_VENV`, `ZF_RUNNER_VENV`, `ZF_ROCM`, `AIG_DATA`,
`ZF_SERVER_IMAGE`, `ZF_GRADER_IMAGE`, `HF_HUB_CACHE`).

**Images.**

```bash
. scripts/env.sh                                     # the image references below are its defaults
docker pull "$ZF_SERVER_IMAGE"     # lmsysorg/sglang:v0.5.21-rocm724-mi30x, pinned by digest: server and sampler
docker pull "$ZF_GRADER_IMAGE"     # vllm/vllm-openai-rocm, pinned by digest: only a root filesystem for grading
docker pull busybox@sha256:a46257e06c82ad85242d0d38077d0a52d9896389ca60a1a59d3463db586e5fa4   # stop_server.sh: gives the store back to you
docker build -t zf-code-runner:1 -f scripts/runner.Dockerfile .   # HumanEval+/MBPP+ sandbox (regressions)
```

The server image compiles its AITER attention kernels on the first requests of each new container (~3–4 min each)
and keeps nothing afterwards; the profiles' `--watchdog-timeout 3600` covers the wait. Training pays it rarely (the
server restarts SGLang inside one container), but every eval, check and smoke-test start pays it again. Optional,
once (~15 min):

```bash
scripts/warm_image.sh sglang-warm        # starts the image, compiles the kernels, saves it as sglang-warm
export ZF_SERVER_IMAGE=sglang-warm
```

**Three Python environments**, kept apart on purpose:

```bash
. scripts/env.sh                  # sets the ZF_* defaults in this shell
scripts/setup_server_venv.sh      # ZF_SERVER_VENV: server extras over the image's packages (built inside the image)
scripts/setup_runner_venv.sh      # ZF_RUNNER_VENV: torch 2.14.1+rocm7.2 that builds and runs graded kernels
uv venv "$ZF_CLIENT_VENV" -p 3.12     # ZF_CLIENT_VENV: tinker SDK + cookbook, evals (CPU torch)
uv pip install -p "$ZF_CLIENT_VENV" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-client.txt
```

**CPU tests** (in the server image, no GPU):

```bash
scripts/test_cpu.sh        # 141 pass, 1 skip (needs a GPU)
```

## 2. Model and data

| What | Source | Used by |
|---|---|---|
| Model | `Qwen/Qwen3.8-27B` (revision `37561816e66b`), in `HF_HUB_CACHE` | server, regressions |
| Tasks | [amd/AIG-Datasets](https://huggingface.co/datasets/amd/AIG-Datasets) revision `d43e25303c518a26eb6847dab448be39cd82b2d5`, folders `v0.2/pytorch_hip_kernel_cuda_agent_ops_6k` and `v0.2/pytorch_hip_kernel_aicuda_engineer` | everything (`AIG_DATA` = the `v0.2` folder) |
| Coding regressions | `evalplus/humanevalplus`, `evalplus/mbppplus` | `scripts/eval_regress.py` |
| General regressions | `google/IFEval`, `TIGER-Lab/MMLU-Pro`, `HuggingFaceH4/MATH-500` | `scripts/eval_bench.py` |

```bash
hf download Qwen/Qwen3.8-27B --revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0
hf download amd/AIG-Datasets --repo-type dataset --revision d43e25303c518a26eb6847dab448be39cd82b2d5 \
    --include "v0.2/pytorch_hip_kernel_cuda_agent_ops_6k/*" "v0.2/pytorch_hip_kernel_aicuda_engineer/*" \
    --local-dir "$(dirname "$AIG_DATA")"
for d in evalplus/humanevalplus evalplus/mbppplus google/IFEval TIGER-Lab/MMLU-Pro HuggingFaceH4/MATH-500; do
  hf download "$d" --repo-type dataset; done
```

sha256 of the two task files: `5f3fd9d6…09b86` (`pytorch_hip_kernel_cuda_agent_ops_6k.json`) and `c68e62f2…f819e704`
(`pytorch_hip_kernel.json`).

**The split** is fixed in [`data/split_manifest.json`](data/split_manifest.json): held-out A (200 tasks: L1 120,
L2 50, L3 30), validation (100), a small held-out KB set (31, unused here) and 4,572 train tasks. Records are grouped
into tasks by normalised code, number-masked code and op name, so near-duplicates stay on one side.
`scripts/split.py --data "$AIG_DATA"` regenerates it (`tests/unit/test_amd_split.py`).

## 3. Grader checks (before training)

```bash
scripts/grader_selftest.sh /tmp/zf_selftest 24
#   references correct 24/24 ...; output+1 correct 0/24 (expect 0); no own kernel rewarded 0/24 (expect 0)
sg docker -c "$ZF_CLIENT_VENV/bin/python scripts/regress_selftest.py"
#   humanevalplus: canonical pass 157/157; wrong pass 0/20; os._exit pass 0/20   (mbppplus: 375/375, 0/20, 0/20)
```

The first grades 24 dataset reference answers on the GPU, the same answers with `+ 1` on the result, and the same
answers without `__global__` (the reward needs an own kernel). The grader's design and the reward:
[docs/REWARD.md](docs/REWARD.md) §1; the CPU tests cover extraction, the safety scan, the reward and the container
command lines (`tests/unit/test_hip_grader.py`).

## 3b. Pipeline smoke test (~1.5–2 h)

Before committing five days of GPU, run every stage at toy size: the whole `full_run.sh` (24 tasks, 8 SFT rows,
4 RL steps with a resume, short answers) under the supervisor, then the eval, export → import → eval, release-check
and regression paths. The numbers mean nothing; it checks that everything runs on your machine.

```bash
scripts/smoke.sh $HOME/results/hip_rl_smoke      # prints PASS/FAIL per part into the run dir's STATUS
```

## 4. Training

One driver runs every stage and skips stages whose output is complete, so re-running it resumes. Run it under the
supervisor, which restarts it after a GPU fault (health check first) and stops on anything else:

```bash
R=$HOME/results/hip_rl_run          # run dir: STATUS log, rollouts, metrics, server store
tmux new -d -s zf_super "scripts/full_run_supervised.sh $R"
tail -f $R/STATUS
```

| Stage | What | Result in our run |
|---|---|---|
| `refs` | the dataset's reference answer for each of the 4,572 train tasks, graded for correctness | 4,443 correct |
| `own_sample`, `own_grade` | plain SGLang samples the base model on train tasks (all L2/L3 + 2,000 L1, 2 each), graded | 1,931 / 5,064 correct (~6 h sampling, several hours grading) |
| `sft_data`, `sft` | 2,343 verified answers (1,722 own + 621 references), 1 epoch, LR 1e-4 | ~2.5 h |
| `sft_eval` | SFT model on held-out A (200 × 4, 12,288 tokens) | 0.254 |
| `rl_pilot` | RL steps 1–20 from the SFT model | held-out A 0.396 at step 20 |
| `base_eval` | base model on held-out A, same server and grader | 0.274 |
| `gate` | `check_pilot.py`: stops if the pilot looks broken | passed |
| `rl_main` | RL to step 125: 12,288 tokens through step 57, 14,000 after; checkpoint every step, milestone every 25, held-out A eval every 25 | 0.415 (25), 0.429 (50, 12k), **0.515 (75)**, 0.4925 (100), 0.509 (125) |

The run stopped at step 125 by its stop rule (no gain over step 75), and step 75 is the released adapter.
Checkpoints are in `$R/server/store/`; `$R/rl/metrics.jsonl` lists each step's `ckpt`
(`tinker://<session>/weights/keep0075` for step 75).

Expect your numbers to differ by a few points: sampling at temperature 1 on the GPU is not bit-reproducible, and
two runs of one checkpoint on held-out A differ by about that much (RESULTS.md §6).

**Exit codes** of `full_run.sh`: 1 = the server died (the supervisor checks the GPU and resumes), 2 = gate stop,
3 = GPU memory in use before start, 4 = driver wedged → reboot, check `modinfo`, re-run the supervisor. A healthy
GPU passes `$ZF_RUNNER_VENV/bin/python -I scripts/gpu_health.py` *and* that process exits within seconds with
the GPU memory back near 0; a slow exit or held memory means the driver is still wedged.

## 5. Evals and regressions

```bash
CK=tinker://<session>/weights/keep0075                         # from $R/rl/metrics.jsonl
scripts/evals.sh $R base                              # validation 14k, held-out A 14k, held-out A 20k
SETS="val14k ha20k" scripts/evals.sh $R keep0075 --resume $CK   # held-out A 14k is the RL eval at step 75
python3 -I scripts/compare.py $R/final/base_val14k/evals/init $R/final/keep0075_val14k/evals/init
python3 -I scripts/compare.py $R/final/base_ha14k/evals/init $R/rl/evals/step0075
```

`compare.py` prints both models' numbers (correct, compiled, truncated, correct and ≥ 1.1×, median speedup, by
level) and the paired gain with its bootstrap CI over tasks. Ours: validation +20.5 [14.3, 27.0], held-out A
+22.6 [18.1, 27.1] (the CI endpoints can move by 0.1 with the resampling order).

Regressions (~35 min per model):

```bash
scripts/regress.sh $R base
scripts/regress.sh $R keep0075 $CK
$ZF_CLIENT_VENV/bin/python scripts/eval_regress.py compare $R/regress/base_humanevalplus $R/regress/keep0075_humanevalplus
$ZF_CLIENT_VENV/bin/python scripts/eval_regress.py compare $R/regress/base_mbppplus $R/regress/keep0075_mbppplus
python3 -I scripts/compare_bench.py $R/regress/base_bench/bench_base $R/regress/keep0075_bench/bench_keep0075
```

If the driver wedges during an eval's grading (the samples are already saved), reboot, drop the cache rows written
after the fault from `work/pass1_cache.jsonl`, and run `scripts/regrade.py <eval out dir> init`.

## 6. Export and release checks

```bash
STORE=$R/server/store/<session>/weights/keep0075               # the directory behind $CK
scripts/server_python.sh "$R/server:$R/hf" scripts/export_hf.py --src "$STORE" --out "$R/hf" --note "RL step 75"
scripts/check_release.sh $R $CK $R/rl/evals/step0075 $R/hf $R/hf_checks
```

`check_release.sh` compares per-token logprobs on 8 held-out answers from the training server, Transformers + PEFT loading
the exported folder, and stock SGLang loading it after `convert_to_sglang.py`, and runs the model card's build
snippet on a correct answer. Ours: PEFT 0.0054 nats/token from the training server (base vs base 0.0049, the bf16
noise floor), stock SGLang identical (it is the same SGLang), adapter vs base 0.026; build ok. The
[HF repo](https://huggingface.co/ik6626/Qwen3.8-27B-HIP-kernel-RL) is that folder plus the model card and its
standalone tools (prompt builder, answer checker, SGLang converter, an example).

## 7. Evaluate the released adapter (no training)

```bash
D=$(hf download ik6626/Qwen3.8-27B-HIP-kernel-RL --include "adapter_*" "training_info.json")   # step 75; 1.8 GB
R=$HOME/results/hip_rl_eval
ZROFORM_DATA=$R/server scripts/server_python.sh "$R/server" scripts/import_adapter.py --name hf75 --src "$D"
#   -> tinker://imported/sampler_weights/hf75
scripts/evals.sh $R hf75 --sampler-path tinker://imported/sampler_weights/hf75
scripts/evals.sh $R base
python3 -I scripts/compare.py $R/final/base_val14k/evals/init $R/final/hf75_val14k/evals/init
scripts/regress.sh $R hf75 tinker://imported/sampler_weights/hf75
```

`EVAL_LIMIT=20 SETS=val14k scripts/evals.sh ...` is a quick check (20 tasks × 4, ~30 min). For `step125/`,
download that folder (`--include "step125/*"`) and pass `--src "$D/step125"`.
