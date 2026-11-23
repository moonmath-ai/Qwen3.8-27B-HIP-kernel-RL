# RL for HIP kernels teaches correctness, not speed: a case study with Qwen3.8-27B on one MI300X

*Technical report. Code and results: [this repository](../README.md). Adapter:
[ik6626/Qwen3.8-27B-HIP-kernel-RL](https://huggingface.co/ik6626/Qwen3.8-27B-HIP-kernel-RL).*

## Abstract

We trained a rank-64 LoRA on Qwen3.8-27B to translate PyTorch modules into HIP kernels for AMD Instinct MI300X,
with SFT followed by 125 steps of GRPO-style RL, on a single MI300X. A sandboxed grader builds every answer, checks
it against PyTorch on three random inputs, and times it; the reward pays for correctness and, on a log scale, for
speedup over PyTorch eager (the Zro-form reward). On 100 validation tasks never used for training or checkpoint choice, the share of
correct kernels rose from 0.335 to 0.540 (+20.5 points, paired 95% CI [14.3, 27.0]); on 200 held-out tasks, from
0.289 to 0.515 (+22.6 [18.1, 27.1]). Nearly all of the gain is better solutions, not fewer answers running out of
tokens. But the kernels did not get faster: the median speedup of correct answers fell from 1.10× to 0.90× PyTorch
eager, because the reward pays the same for a slow correct kernel as for one 1.05× faster. General benchmarks
(HumanEval+, MBPP+, IFEval, MATH-500) moved by at most 1.2 points; MMLU-Pro fell 3.3 points [−6.7, 0.0].

## 1. What is new here, and what isn't

**Not new:** RL for GPU kernel generation with a compile-check-time reward has been done before, for CUDA and
Triton and on AMD hardware (§11), and our reward is a variant of the speedup rewards used there. The training loop
is a standard group-relative policy gradient on a self-hosted Tinker-compatible server.

**What we think is useful to others:**

1. **What a standard speedup reward actually teaches** (§7): almost all of the reward earned was for correctness,
   and the policy drifted to correct-but-slow kernels. If speed matters, the reward has to say so.
2. **Containment for model-written GPU code** (§2): building in a no-GPU sandbox, running each answer in its own
   container, comparing outputs outside it, and a driver watchdog plus per-step checkpoints, because a bad kernel
   can reset the GPU or wedge the driver, and both happened during training.
3. **A full 27B RL run on one GPU** (§4): trainer and sampler take turns on one 192 GB MI300X, with SGLang
   restarted at each phase switch.
4. **An evaluation protocol with controls** (§5–§6): a split by task with near-duplicates grouped, a validation set
   used once, paired bootstrap CIs, a matched-budget baseline, a gain decomposition, and regressions.
5. **Two lessons that cost us time** (§8): SFT on references with empty reasoning taught the model to skip
   reasoning, and several grader bugs appeared only at full scale on real model output.
6. **A released, verified adapter** with standalone tools to build prompts and check answers.

## 2. Task and grader

Given a PyTorch module and the same computation as a function `module_fn`, the model answers with one `.hip` file
whose `forward` replaces `module_fn` and does the computation in its own `__global__` kernels. The prompt adds the
build interface (how the file is compiled and called); it is single turn, with no tools or compiler feedback, and
uses the chat template's `reasoning_effort="low"`.

Each answer goes through these checks; the first failure sets the reward:

| Check | How | Reward if it fails |
|---|---|---|
| Format | one ```` ```cpp ```` block after the reasoning, defining `PYBIND11_MODULE` with `forward` | 0 |
| Safety scan | no file, process, environment, network or dynamic-loading calls | 0 |
| Own kernel | at least one `__global__` kernel in the source | 0 |
| Compile | `torch.utils.cpp_extension` build in a container with no network and no GPU | 0 |
| Correctness | fresh container per answer (no network, read-only root, non-root, memory/pid/time limits); 3 input seeds, inputs and weights re-made per seed; `rtol = atol = 1e-4`, NaNs only where the reference has them, inputs left untouched, no output aliasing an input; compared outside the container | 0.1 |
| Own kernel launched | the profiler must see one of the answer's own kernels run | 0 |

A correct answer scores `1 + 0.5 · clip(log2 s, 0, 3)`, where `s` is the end-to-end speedup against PyTorch eager
(median of CUDA-event timings, cache flushed, one container at a time), with `s < 1.1` counted as 1. Library calls
(rocBLAS, MIOpen) are allowed: eager already uses them, so they cannot beat the baseline on their own, and the
library share of GPU time is logged (median 0.04% per step, above 1% in 7 of 125 steps).

On the GPU the grader passes 24/24 dataset reference answers, 0/24 of the same answers with `+ 1` on the output,
and rewards 0/24 with the `__global__` qualifier removed.

## 3. Data and split

Tasks come from the `pytorch_hip_kernel_cuda_agent_ops_6k` and `pytorch_hip_kernel_aicuda_engineer` subsets of
amd/AIG-Datasets, which come from CUDA-Agent-Ops-6K and the AI CUDA Engineer archive. Records are grouped into
tasks by normalised code, number-masked code and op name, so near-duplicates (the same module with other shapes or
constants) land on one side of the split:

| Set | Size | Use |
|---|---|---|
| Held-out A | 200 tasks (L1 single op 120, L2 fused ops 50, L3 larger modules 30) × 4 samples | evaluation every 25 steps; checkpoint choice |
| Validation | 100 tasks × 4 | evaluated once, for the headline |
| Train | 4,572 tasks | SFT and RL |

All held-out and validation tasks are from CUDA-Agent-Ops. The sources are public and agent-generated, so the
base model may have seen similar kernels; this affects base and adapter alike, since every comparison is paired.

## 4. Training

**Model and adapter.** Qwen3.8-27B (Gated DeltaNet linear attention plus full attention), bf16. LoRA rank 64,
alpha 32, on attention, linear attention, MLP and the LM head.

**SFT.** 2,343 verified answers (1,722 of the base model's own correct answers on train tasks, plus 621 dataset
reference kernels), one epoch, LR 1e-4. It did not raise the held-out score (0.274 → 0.254 at 12,288 tokens) but
halved truncation (§8).

**RL.** 25 tasks × 4 samples per step, group-centred advantages (reward minus the group mean), importance-sampling
policy gradient, no KL or entropy term, LR 1e-5, temperature 1; groups with equal rewards are skipped, and tasks
are drawn L1 : L2 : L3 = 0.4 : 0.3 : 0.3, skipping tasks whose last two groups had no variance. Answer budget
12,288 tokens for steps 1–57 and 14,000 from step 58, when 22% of held-out answers were being cut off. Stop rule:
end the run if step 125 is not better than step 75 on held-out A; it was not (−0.6 [−4.4, +3.3]).

**One GPU.** The trainer (bf16 weights, LoRA, Adam, activations for 16k-token sequences) and an SGLang sampler with
KV cache for ~100 long rollouts do not fit on a 192 GB GPU together. The server stops SGLang for training and
restarts it to sample, reloading the adapter from memory (SGLang's memory-saver mode crashed on ROCm). Each step:
sample (median 13 min, ~1.8k tokens/s with the LoRA at up to 128 concurrent requests), grade (18 min, SGLang off,
so timing has the GPU alone), train (8 min); median 41 min, 125 steps in ~3.6 days. Every step is checkpointed
(52 s) and a supervisor resumes after a GPU fault, once a health check passes.

## 5. Evaluation protocol

- Evaluation uses the training environment: same prompt, grader and server.
- Correct = compiles and matches PyTorch on 3 seeds. Pass@1 over 4 samples per task at temperature 1.
- Gains are paired bootstrap 95% CIs over tasks (5,000 resamples). Two checkpoints on held-out A are resolvable to
  about ±4 points; re-runs of one checkpoint differ by a few points.
- Base and adapter at the same answer budget (14,000 tokens), on the same server.
- Validation was held out from the start and evaluated once, after the checkpoint was chosen on held-out A.
- Regressions use the same protocol as our earlier coding-RL release: HumanEval+ and MBPP+ (2 samples at
  temperature 1, sandboxed grader), IFEval, MMLU-Pro and MATH-500 (300 each, greedy), 12,288 tokens.

## 6. Results

| Set (14,000 tokens) | Base | Step 75 | Gain [95% CI] | Tasks up / down |
|---|---|---|---|---|
| Validation (100) | 0.335 | **0.540** | +20.5 [14.3, 27.0] | 57 / 10 |
| Held-out A (200) | 0.289 | **0.515** | +22.6 [18.1, 27.1] | 107 / 14 |

By difficulty on held-out A: single ops 0.34 → 0.53, fused ops 0.21 → 0.50, larger modules 0.23 → 0.51.

Held-out A over training (12,288 tokens through step 50, 14,000 after): 0.274 (base), 0.254 (SFT), 0.396 (step
20), 0.415 (25), 0.429 / 0.455 at 14k (50), **0.515 (75)**, 0.4925 (100), 0.509 (125). Most of the gain comes in
the first 20 steps, with a second rise between steps 50 and 75.

**Controls.**
- *Better solutions or shorter answers?* Every truncated answer scores 0. On held-out A, the share of answers that
  finish barely changes (0.810 → 0.836), and the pass rate of finished answers rises from 0.356 to 0.616. Changing
  one factor at a time from base, completion alone gives +0.9 points and the finished-answer pass rate +21.0; on
  validation, +3.5 and +15.5. The gain is better kernels, not shorter answers.
- *Is the budget the limit?* At 20,000 tokens truncation falls from 16% to 3%, but correctness rises only
  +2.1 [−1.5, +5.8]: most answers that finish with more room are wrong.
- *Matched budget.* The base model at 14,000 vs 12,288 tokens: +1.5 [−2.4, +5.5], so the gain is not a budget
  artefact.
- *Reward hacking.* No forbidden calls in any of the 12,500 training answers. The library-time share stayed small
  (above) and an own kernel must run, so calling a library and wrapping it in a trivial kernel does not pay.

**Regressions** (step 75 vs base, paired CIs over problems):

| Benchmark | Base | Step 75 | Gain |
|---|---|---|---|
| HumanEval+ (157 × 2) | 0.933 | 0.943 | +1.0 [−1.0, +3.2] |
| MBPP+ (375 × 2) | 0.809 | 0.821 | +1.2 [−0.1, +2.7] |
| IFEval (300) | 0.887 | 0.883 | −0.3 [−3.0, +2.3] |
| MMLU-Pro (300) | 0.790 | 0.757 | −3.3 [−6.7, 0.0] |
| MATH-500 (300) | 0.943 | 0.950 | +0.7 [−1.3, +2.7] |

Only MMLU-Pro drops, at the edge of significance (19 questions worse, 9 better). Our coding-RL adapter on the same
base model raised it (0.740 → 0.773), so narrow kernel-only training may cost a little general knowledge.

## 7. What the reward taught

| Held-out A, 14,000 tokens | Base | Step 75 |
|---|---|---|
| Answers that compile | 0.64 | 0.78 |
| Correct and ≥ 1.1× faster than eager (of 800) | 116 | 169 |
| Median speedup of correct answers | 1.10× | 0.90× |

In steps 58–66, 80% of the reward earned came from correctness, 17% from the speed bonus and 4% from "compiles but
wrong". Within a group, turning a wrong answer into a slow correct one gains +0.9; turning a slow correct one into
a 2× one gains +0.5. The policy follows that gradient: it solves more tasks, including more with a real speedup
(116 → 169), but more than half of its correct kernels are slower than eager, and 58% of correct training answers
were, scoring the same 1.0 as a 1.05× kernel.

To train for speed, the reward has to penalise slow kernels (for example `clip(log2 s, −1, 3)`, so 0.5× scores
0.5) or pay more for speed, and runs should be compared on held-out correctness *and* speedup, not on reward. Both
change what "better" means, so we left them to a separate run. [REWARD.md](REWARD.md) §4 lists the options and
their risks.

## 8. Two lessons

1. **SFT on references with empty reasoning taught the model to skip reasoning.** The 621 reference kernels have no
   reasoning text, so after SFT 268 of 737 evaluated answers skipped it, and only 1.1% of those were correct. RL
   undid this within 20 steps, but SFT added nothing measurable. Train on the model's own correct answers only, or
   skip SFT.
2. **Grader bugs appear at scale, on real model output.** Smoke tests on reference kernels passed, yet the full
   pipeline found: the torch profiler losing all events with 10 grading containers at once (997 correct
   references wrongly marked "no own kernel"; re-run alone, 0 of 88), hipify crashing or looping forever on
   malformed kernel launches, and model kernels that hang or fault the GPU. Each now has a test.

## 9. Limitations

- In-distribution: held-out and validation tasks come from the same sources as training, all from CUDA-Agent-Ops.
- One seed, one model, one hyperparameter setting; no ablation of LoRA rank, group size or learning rate.
- Correctness is checked on 3 random inputs at the task's shapes, not on other shapes or edge cases.
- Mostly general ML operators (elementwise, reductions, losses, small fused modules), not tuned serving kernels.
- Single turn, no compiler feedback; multi-turn refinement is the obvious next step.
- RL plateaued at ~0.51–0.54 after 75 steps with this data, prompt and reward.

## 10. Reproducibility

[REPRODUCE.md](../REPRODUCE.md) covers the environment, data, grader checks, a pipeline smoke test, training,
evaluation, and evaluating the released adapter without training. [RESULTS.md](../RESULTS.md) records the decisions
and full results. The released adapter, served by stock SGLang after the exact conversion in the repo, gives the
same per-token logprobs as the training server; loaded with Transformers + PEFT it agrees to 0.0054 nats/token
(base vs base: 0.0049, the bf16 noise floor), against 0.026 between adapter and base.

## 11. Related work

- KernelBench: Ouyang et al., 2025, https://github.com/ScalingIntelligence/KernelBench
- The AI CUDA Engineer: Sakana AI, 2025; archive at https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive
- CUDA-Agent-Ops-6K: BytedTsinghua-SIA, https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K
- AIG-Datasets: AMD, https://huggingface.co/datasets/amd/AIG-Datasets
- Kevin: multi-turn RL for CUDA kernels (Baronio et al.)
- AutoTriton: RL for Triton kernels (Li et al., 2025)
- GRPO: Shao et al., *DeepSeekMath*, 2024
- Tinker and tinker-cookbook: Thinking Machines Lab, https://github.com/thinking-machines-lab/tinker-cookbook
- SGLang: https://github.com/sgl-project/sglang · PEFT: https://github.com/huggingface/peft
- Our earlier coding-RL release on the same base model: https://huggingface.co/ik6626/Qwen3.8-27B-deepcoder-RL
