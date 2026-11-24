# Training record: RL for HIP kernels, Qwen3.8-27B on one MI300X

What we did, why, what came out of it and what we would do differently (one run).
Reward analysis: [`REWARD.md`](docs/REWARD.md). How to rerun it: [`REPRODUCE.md`](REPRODUCE.md). The adapter:
[ik6626/Qwen3.8-27B-HIP-kernel-RL](https://huggingface.co/ik6626/Qwen3.8-27B-HIP-kernel-RL).

## 1. Result

LoRA trained with SFT then RL raises the share of correct HIP kernels by about 20 points over the base model, on
tasks never used for training or checkpoint choice. Pass@1 over 4 samples per task at temperature 1; gains are
paired bootstrap 95% CIs over tasks.

| Set | Budget | Base | Final adapter (step 75) | Gain |
|---|---|---|---|---|
| **Validation**, 100 tasks (never used for training or selection) | 14k | 0.335 | **0.540** | **+20.5 [+14.3, +27.0]** (57 up, 10 down) |
| Held-out A, 200 tasks (used to pick step 75) | 14k | 0.289 | 0.515 | +22.6 [+18.1, +27.1] (107 up, 14 down) |
| Held-out A at 20k | 20k | - | 0.536 | +2.1 [−1.5, +5.8] vs the adapter at 14k; +24.8 [+20.5, +29.0] vs base at 14k |

- The gain is correctness, not speed: on tasks both models solve, step 75's kernels are as fast as base's (paired
  1.05× [0.92, 1.23] on held-out A); newly solved tasks come in slow (median 0.65×), so the pooled median falls.
- SFT alone did not help (0.254 vs base 0.274); the gain is from RL.
- Learning plateaued at step 75; steps 75–125 are within noise of each other.
- Regressions: within ±1.2 points on HumanEval+, MBPP+, IFEval, MATH-500; MMLU-Pro −3.3 [−6.7, 0.0] (§5).
- What it does not show: run-to-run variance (one seed), out-of-distribution tasks (all splits come from the same
  sources), speed.

## 2. Setup

| | |
|---|---|
| Model | Qwen/Qwen3.8-27B, bf16, revision `37561816e66b` |
| Adapter | LoRA rank 64, alpha 32, on attention (`q/k/v/o_proj`), Gated DeltaNet (`in_proj_qkv`, `in_proj_z`, `out_proj`), MLP and `lm_head`; float32; 1.8 GB (+3.6 GB Adam in a resume state) |
| Hardware | one AMD Instinct MI300X VF (192 GB), 13 vCPUs, ROCm 7.2.4, amdgpu-dkms 6.16.13, kernel 6.8.0-124 |
| Server | Zro-form (Tinker-compatible), phased: trainer and SGLang 0.5.21 share the GPU, SGLang restarts at each phase switch (`profiles/mi300x-qwen38-27b-phased.yaml`, `max_model_len` 16384, 128 running requests) |
| Client | `scripts/hip_rl.py` (RL and evals), `hip_sft.py`; driver `scripts/full_run.sh` + `full_run_supervised.sh` |
| Data | 4,572 train tasks from the CUDA-Agent-Ops-6K and AI-CUDA-Engineer subsets of `amd/AIG-Datasets` v0.2. Records grouped into tasks by normalised code, number-masked code and op name; held-out A (200: L1 120, L2 50, L3 30), validation (100), held-out KB (31) split off by task (`split_manifest.json`) |
| Prompt | system prompt + task (instruction, module, functional code, build interface), `reasoning_effort="low"`, single turn, no tools or compiler feedback |
| Grader | `scripts/hip_grade.py`: `torch.utils.cpp_extension` build (hipcc, gfx942), each answer runs in its own network-less container, outputs compared outside it on 3 seeds (`rtol = atol = 1e-4`), correct answers re-run alone for timing and profiling |
| Reward (Zro-form reward, `zro` in code) | 0 for format failure, forbidden call or no own `__global__` kernel; 0.1 compiles but wrong; correct: 1 + 0.5 · clip(log2 s, 0, 3), s < 1.1 counted as 1 |
| SFT | 2,343 verified answers (1,722 of the base model's own correct answers + 621 reference kernels), 1 epoch, 74 steps, LR 1e-4 linear decay |
| RL | GRPO-style: 25 tasks × 4 samples per step, group-centred advantages, importance-sampling loss, no KL, LR 1e-5, temperature 1; 125 steps; budget 12,288 tokens for steps 1–57, 14,000 from step 58 |
| Step time | median 41 min: sample 13, grade 18, train 8 (125 steps); an eval of 200 × 4 ~2.2–3.5 h (validation 100 × 4: ~50 min sampling, 55–77 min grading) |

## 3. Run sequence

| Stage | Outcome |
|---|---|
| Bring-up: SGLang alone, colocated server, grader | trainer vs SGLang logprobs at the bf16 noise floor (0.0071 nats/token); memory-saver phasing fails on ROCm → restart phasing |
| Base rate with the final prompt | 31.9% correct on 200 × 4 (plain SGLang) |
| Dataset references graded; base model's own answers sampled and graded | 4,443 / 4,572 references correct; 1,931 / 5,064 own answers correct |
| SFT | held-out A 0.254 (base 0.274, both at 12,288 tokens) |
| RL pilot (20 steps), gate | step 20: 0.396 → gate passed |
| RL main, budget 12,288 → 14,000 tokens at step 58 | step 75: 0.515 (+6.0 [+2.3, +9.9] vs step 50 at 14k) |
| Steps 100, 125 | 0.4925, 0.509: plateau → stop at 125 |
| Final evals | validation +20.5; held-out A at 20k 0.536 |
| Base on held-out A at 14k, release checks, regressions | 0.289; PEFT, SGLang and build checks pass; regressions §5 |

## 4. Decisions and why

1. **Task: PyTorch → HIP**. Verifiable by building and running; the AIG datasets give thousands of tasks.
2. **The Zro-form reward** (`zro` in code). Log speedup, not AMD's linear `clip(s, 0, 10)`: speedups are heavy-tailed, so linear
   reward lets a few easy big wins and their timing noise dominate. Small 0.1 for compiling keeps a gradient when a
   whole group fails without paying for "compiles but wrong". One formula for all tasks; difficulty comes from the
   group-centred advantage and the task sampler, not per-task weights.
3. **Library calls allowed, with guards**. Speedup is end to end against eager, which already uses
   rocBLAS/MIOpen, so calling a library cannot by itself win. At least one own kernel must launch; the library time
   share is logged per step (median 0.04%, above 1% in 7 of 125 steps).
4. **SFT on own answers + references**. In hindsight a mistake: the 621 references have empty thinking, so
   SFT taught the model to skip thinking (268/737 answers skipped it, 1.1% of those correct). RL undid it within 20
   steps. Next time: own answers only, or no SFT.
5. **Phased schedule with SGLang restarts**. `torch_memory_saver` breaks SGLang's Triton kernels on ROCm; a
   restart costs ~1–2 min per switch, small next to a 41-min step.
6. **Watchdog + auto-resume supervisor**. Model kernels can hang or wedge the GPU; a run must stop
   cleanly (exit 4 = reboot needed) or restart from the last checkpoint after a health check (exit 1). Save every
   step (52 s) so a fault costs at most one step.
7. **Budget 12k → 14k at step 58**. 22% of held-out answers were cut off at 12k, and ~45% of finished answers
   in the 10–12k band were correct. Train at 14k to the end, no 20k trial: training truncation was ~6%, and
   switching mid-run breaks comparability.
8. **Stop rule**: stop if step 125 is not above step 75. It was not (−0.6 [−4.4, +3.3]), so RL stopped there.
9. **Final checkpoint = step 75**: best held-out A; step 125 kept as `step125/` in the release.
10. **Validation used once, for the headline**. Held-out A had been used to choose; validation gives the
    unbiased gain. Held-out A at 20k answers "is the budget limiting?" (no).

## 5. Results in detail

**Held-out A over training** (200 × 4; 14k unless noted):

| | Base (12k / 14k) | SFT (12k) | Step 20 (12k) | 25 (12k) | 50 (12k / 14k) | **75** | 100 | 125 | 75 at 20k |
|---|---|---|---|---|---|---|---|---|---|
| Correct | 0.274 / 0.289 | 0.254 | 0.396 | 0.415 | 0.429 / 0.455 | **0.515** | 0.4925 | 0.509 | 0.536 |
| Truncated | 0.29 / 0.19 | 0.14 | 0.21 | 0.21 | 0.22 / 0.15 | 0.16 | 0.16 | 0.16 | 0.03 |
| Compiled | 0.55 / 0.64 | 0.52 | 0.69 | 0.70 | 0.70 / 0.76 | 0.78 | 0.78 | 0.78 | 0.85 |
| Correct ≥ 1.1× (of 800) | 108 / 116 | 99 | 154 | 150 | 153 / 164 | 169 | 179 | 173 | 180 |
| Median speedup, correct | 1.08 / 1.10 | 1.04 | 1.04 | 1.01 | 0.96 / 0.99 | 0.90 | 1.01 | 0.96 | 0.94 |

Paired comparisons (95% CI): step 75 vs base, both at 14k, +22.6 [+18.1, +27.1]; base at 14k vs 12k +1.5 [−2.4, +5.5]; step 75 vs 50 at 14k +6.0 [+2.3, +9.9]; step 50 at 14k vs 12k +2.6 [−1.1, +6.3];
step 100 vs 75 −2.3 [−6.1, +1.6]; step 125 vs 75 −0.6 [−4.4, +3.3]; step 75 at 20k vs 14k +2.1 [−1.5, +5.8].

**By level** (held-out A, base → step 75, both at 14k): L1 0.34 → 0.53, L2 0.21 → 0.50, L3 0.23 → 0.51. The
largest gain is on fused ops (L2).

**Where the reward came from** (steps 58–66): 80% correctness, 17% speed bonus, 4% compiles-but-wrong. 58% of
correct answers were slower than eager and scored the same 1.0 as a 1.05× kernel (`REWARD.md` §3).

**Speed on the same tasks** (step 75 vs base at 14k; per task, the median speedup of its correct answers; geometric
mean of the ratio, bootstrap 95% CI over tasks): held-out A 1.05× [0.92, 1.23] on the 108 tasks both solve (step 75
faster on 54), median 0.65× on the 40 tasks only step 75 solves; validation 1.11× [0.94, 1.31] on 64 (faster on 32),
0.81× on 17. The pooled median falls (1.10 → 0.90 on held-out A) because of the new tasks, not slower kernels.

**Regressions** (step 75 vs base, same server, 12,288 tokens, low reasoning; HumanEval+/MBPP+ 2 samples at
temperature 1 through the safe grader, the others greedy; paired bootstrap over problems;
`scripts/regress.sh`):

| Benchmark | Base | Step 75 | Gain |
|---|---|---|---|
| HumanEval+ (157 × 2) | 0.933 | 0.943 | +1.0 [−1.0, +3.2] |
| MBPP+ (375 × 2) | 0.809 | 0.821 | +1.2 [−0.1, +2.7] |
| IFEval (300) | 0.887 | 0.883 | −0.3 [−3.0, +2.3] |
| MMLU-Pro (300) | 0.790 | 0.757 | −3.3 [−6.7, 0.0] (19 worse, 9 better) |
| MATH-500 (300) | 0.943 | 0.950 | +0.7 [−1.3, +2.7] |

Only MMLU-Pro drops, at the edge of significance; our coding-RL adapter on the same base model
([Qwen3.8-27B-deepcoder-RL](https://huggingface.co/ik6626/Qwen3.8-27B-deepcoder-RL)) raised it (0.740 → 0.773),
so narrow kernel-only RL may cost a little general knowledge. Base numbers match that card's base within noise
except MMLU-Pro (0.790 here vs 0.740 on NVIDIA; ~2 SE).

## 6. What we learned

1. **RL on a verifiable grader works here; SFT on mixed data did not.** The whole gain is from RL, most of it in the
   first 20 steps (+12 points) and a second rise around steps 50–75.
2. **It plateaus at ~0.51–0.54 after ~75 steps** with this data, prompt and reward. Training batches flat at ~0.6
   correct from step ~75. More steps of the same will not help; change data, reward or context.
3. **The reward teaches correctness, not speed.** With s < 1.1 counted as 1 and no penalty for slow kernels, the
   policy solves new tasks with slow kernels (median 0.65×) and leaves speed on already-solved tasks unchanged
   (paired 1.05×), so the pooled median falls (1.08 → ~0.95). To get speed, change the reward (`REWARD.md` §4:
   penalise s < 1, or a bigger bonus).
4. **Budget is not the limit at 14k.** At 20k truncation falls from 16% to 3%, but correctness rises only +2 points
   (not significant): most answers that finish with more room are wrong.
5. **Evaluation noise.** Two adapters on held-out A are resolvable to about ±4 points; temperature-1 re-runs of one
   checkpoint differ by a few points. Smaller differences need more tasks (held-out A + validation) or 8 samples.
6. **Model-written kernels are a GPU hazard.** A faulting kernel can reset the GPU (recoverable) or wedge the
   driver (only a reboot clears it), and a reset kills every GPU context, including the idle server.
   Containment has to live in the harness: fresh process per grade, watchdog, per-step checkpoints, resumable
   grading caches.
7. **Pre-existing grader bugs show up only at scale.** Profiler losing events with 10 parallel containers (997
   false "no own kernel"), hipify crashing or looping on malformed launches, the SGLang API key starting with "-"
   1 time in 64. Each was found by running the full pipeline on real model output; smoke runs on references missed
   them.

## 7. Hardware limits

- Sampling: decode ~1.8k tok/s with the LoRA at 96–128 running requests (plain SGLang 2.5k); the linear-attention
  state pool, not KV, limits concurrency (128 requests → 501k KV tokens).
- Trainer: ~1.5k tok/s warm; peak ~62 GB allocated / 80 GB reserved; checkpoint 5.1 GB, 52 s to save.
- 20k answers fit (`max_model_len` 24576, `profiles/mi300x-qwen38-27b-phased-24k.yaml`); more retraction, no OOM.
- Grading: 6 parallel correctness containers + serial timing; compile ~13 jobs on 13 vCPUs; ~1–2 h per 800 answers, longer when more are correct (serial timing pass).
- Disk: checkpoint store ~160 GB with per-step rolling states (24 h TTL) and milestones; grading work dirs ~0.9 MB per answer.

## 8. Comparing a new run against this one

What we would hold fixed to compare a change (for example new training data) against this run, with step 75 as the
baseline:
- Same splits (`split_manifest.json`), prompt, Zro-form reward, grader, 25 × 4 per step, LR, 14k budget for training
  and evaluation, eval every 25 steps, the same stop rule.
- Deduplicate new data against held-out A and validation with the same task grouping (normalised code,
  number-masked code, op name) before training.
- Pick checkpoints on held-out A; use validation only for the final comparison (it has been used once, for this
  run's headline).
- Report paired CIs against both base and step 75; differences under ~4 points need held-out A + validation or 8
  samples per task.
- A second seed, to know whether a few-point difference is real.
- SFT data without empty-thinking references (or no SFT); a speed term (`REWARD.md` §4) only in a separate run,
  since it changes what "better" means.
