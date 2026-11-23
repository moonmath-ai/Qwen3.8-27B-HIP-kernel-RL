# Qwen3.8-27B HIP kernel RL

Reinforcement learning of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (LoRA) to write HIP GPU
kernels for AMD Instinct MI300X, trained and evaluated on **one MI300X**. Given a PyTorch module, the model answers
with one `.hip` file whose `forward` computes the same result in its own `__global__` kernels. A sandboxed grader
builds every answer, checks it against PyTorch on three random inputs, and times it.

This repo holds everything that produced the adapter
[ik6626/Qwen3.8-27B-HIP-kernel-RL](https://huggingface.co/ik6626/Qwen3.8-27B-HIP-kernel-RL): the grader, the RL and
SFT clients, the training server (Zro-form, a Tinker-compatible LoRA server) with its one-GPU phased schedule, the
data split, the evals and the release checks. To *use* the adapter you need none of it; see the model card.

## Result

Correct = compiles and matches PyTorch (`rtol = atol = 1e-4`) on 3 random inputs. Pass@1 over 4 samples per task at
temperature 1; gain in points with a paired bootstrap 95% CI over tasks.

| Set | Base | Adapter (RL step 75) | Gain |
|---|---|---|---|
| Validation: 100 tasks, never used for training or checkpoint choice (14,000 tokens) | 0.335 | **0.540** | +20.5 [14.3, 27.0] |
| Held-out A: 200 tasks, used to pick the checkpoint (14,000 tokens) | 0.289 | **0.515** | +22.6 [18.1, 27.1] |

- The gain is correctness, not speed: more than half of the adapter's correct answers are slower than PyTorch eager.
- All of it comes from RL; SFT before RL did not raise the held-out score. RL plateaued at step 75 (of 125).
- Regressions vs base: HumanEval+, MBPP+, IFEval, MATH-500 within ±1.2 points; MMLU-Pro −3.3 [−6.7, 0.0].
- One seed; all splits come from the same public datasets (in-distribution).

Short write-up: [docs/PAPER.md](docs/PAPER.md). Decisions, full results and lessons: [RESULTS.md](RESULTS.md).
The reward: [docs/REWARD.md](docs/REWARD.md).

## Layout

| Path | What |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | environment, data, grader checks, a pipeline smoke test, the training run, evals, export checks, and evaluating the released adapter without training |
| [RESULTS.md](RESULTS.md) | decisions and full results; [docs/PAPER.md](docs/PAPER.md) is the short write-up |
| `scripts/` | everything you run. Training: `full_run.sh` (all stages) and `full_run_supervised.sh` (auto-resume), `hip_rl.py`, `hip_sft.py`, `build_sft_data.py`, `split.py`. Grader: `hip_grade.py`, `hip_grade_worker.py` and the sandboxed build. Evals: `evals.sh`, `regress.sh`, `compare*.py`. Export: `export_hf.py`, `import_adapter.py`, `convert_to_sglang.py`, `check_release.sh`. Setup and checks: `setup_*.sh`, `warm_image.sh` (optional), `test_cpu.sh`, `grader_selftest.sh`, `smoke.sh`. Machine paths: `env.sh` |
| `zroform/` | the training server: Tinker wire protocol, engine (incl. the phased schedule), PEFT trainer, SGLang sampler, checkpoint store |
| `profiles/` | the server profiles of the run (MI300X, 16k and 24k context) |
| `data/split_manifest.json` | the held-out / validation / train split (task and record ids only) |
| `docs/` | the write-up, the reward, server notes |
| `tests/` | CPU tests (grader, RL client, SFT data, split, server) |

## Requirements in short

One AMD Instinct MI300X (192 GB) with ROCm 7.2.4 and the amdgpu-dkms driver, Docker, ~300 GB of disk. The run took
about 5 days: ~1 day of data preparation and SFT, ~3.6 days for 125 RL steps (~41 min each), plus evals (~3 h per
200 × 4 set). Model-written kernels run on the GPU: read the hazards in [REPRODUCE.md](REPRODUCE.md) §0 first.

## Data and licence

Code: Apache 2.0 ([LICENSE](LICENSE)). Tasks come from the `pytorch_hip_kernel_cuda_agent_ops_6k` and
`pytorch_hip_kernel_aicuda_engineer` subsets of [amd/AIG-Datasets](https://huggingface.co/datasets/amd/AIG-Datasets),
built from [CUDA-Agent-Ops-6K](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K)
(BytedTsinghua-SIA, CC-BY-4.0) and [AI-CUDA-Engineer-Archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive)
(SakanaAI, CC-BY-4.0); see that dataset's `LICENSES_AND_PROVENANCE.md`. This repo contains no dataset content, only
task and record ids in `data/split_manifest.json`. The regression benchmarks are downloaded from their own sources.
