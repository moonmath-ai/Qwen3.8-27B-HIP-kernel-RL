# The Zro-form reward (PyTorch → HIP RL)

How an answer is scored and turned into a learning signal, what feeds into it, and what a next run could change.
The whole run used this reward unchanged. Code: `scripts/hip_grade.py` (`reward`, scheme name `zro`),
`scripts/hip_grade_worker.py`, `scripts/hip_rl.py`. Why this design: [`RESULTS.md`](../RESULTS.md) §4.

## 1. Scoring one answer

Each answer goes through these checks in order. The first failure sets the reward.

| # | Check | How | Fails → reward |
|---|---|---|---|
| 1 | **Format** | The last ```` ```cpp/hip ```` block after `</think>` that defines `PYBIND11_MODULE` with `forward`. Code in the reasoning does not count. | 0 |
| 2 | **Safety scan** | No file, process, env, network or `dlopen` calls (regex on the source, comments stripped). | 0 |
| 3 | **Own kernel** | The source defines at least one `__global__` kernel. | 0 |
| 4 | **Compile** | torch `cpp_extension` build in a sandbox with no network and no GPU; hipify capped at 30 s per file. | 0 |
| 5 | **Correctness** | Fresh container (no network, read-only, 32 GB, 120 s limit). 3 seeds; inputs *and* weights are re-made per seed. Checks: `allclose(rtol=1e-4, atol=1e-4)`, NaN only where the reference has NaN, same shapes and dtypes, inputs left untouched, and no output aliasing an input. | 0.1 (compiled but wrong) |
| 6 | **Own kernel launched** | The profiler must see one of the answer's own kernels run. | 0 |
| 7 | **Speed** | Only if correct. Median of CUDA-event timings with a 1 GiB cache flush before each run, against PyTorch eager on the same inputs. Timing runs one container at a time. | (sets the bonus) |

A truncated answer (it hit the token budget) usually has no closed code block, so it fails check 1 and scores 0.

**Reward for a correct answer:** `1.0 + 0.5 · min(log2(s), 3)` if `s ≥ 1.1`, else `1.0`, where `s` is the
end-to-end speedup.

| Outcome | Reward |
|---|---|
| Format / safety / no-own-kernel failure, or doesn't compile | 0 |
| Compiles, wrong | 0.1 |
| Correct, `s` < 1.1× (**including slower than eager**) | 1.0 |
| Correct, 1.1× / 1.5× / 2× / 4× | 1.07 / 1.29 / 1.5 / 2.0 |
| Correct, ≥ 8× (cap) | 2.5 |

Library calls (ATen, rocBLAS, MIOpen) are allowed. Speed is measured end to end against eager, which uses the
same libraries, so calling them can't beat the baseline by itself. The guard is that the answer's own kernel must
launch (check 6). Library GPU time is logged per step as `other_time_share`: median 0.04%, above 1% in 7 of 125 steps.

## 2. From reward to learning signal

- **Groups:** each step samples 25 tasks × 4 answers. Advantage = reward − the group mean (no std scaling), so an
  answer is judged only against the other answers to the same task. This is why one formula serves all tasks: a
  correct answer on a hard task, where the others failed, already gets a large advantage.
- **Zero-variance groups are dropped:** if all 4 answers score the same, there is no signal, and the group isn't
  sent.
- **Task sampling does the weighting:**
  - levels are drawn L1 : L2 : L3 = 0.4 : 0.3 : 0.3, uniform within a level;
  - a task with no reward variance in its last 2 groups (always all pass or all fail) is skipped;
  - tasks whose reference is a single library op are excluded.
- **Update:** importance-sampling policy gradient on the answer tokens only (prompt masked). LoRA rank 64,
  Adam, lr 1e-5, sampling temperature 1.0. No KL or entropy term, and no length penalty.

## 3. What contributes, in practice

Steps 58–66 (900 answers):
- 80% of the reward earned came from correctness;
- 17% from the speed bonus;
- 4% from "compiles but wrong".

Of the correct answers, 58% were slower than eager; they scored the same 1.0 as a 1.05× kernel.

Within a group:
- turning a wrong answer into a slow correct one gains +0.9;
- turning a slow correct one into a 2× one gains +0.5.

The policy follows this gradient. Held-out correctness rose from 27% (base) to 43% (step 50; 52% at step 75), while the median
speedup of correct answers drifted from about 1.1× to below 1×.

Factors, from biggest to smallest effect:
1. Correctness: the base 1.0.
2. Speedup above 1.1×: up to +1.5.
3. Truncation, through the token budget: a cut-off answer scores 0. The budget was 12k, and 14k from step 58.
4. Compiling: 0.1.
5. Hard zeros: format, safety and own-kernel failures.

Task mix (level weights, dead-task skipping) and group size change *which* differences get learned. They don't
change the score.

## 4. What a next run could take into account

Each item lists what it would fix and what it risks.

1. **Penalise slower-than-eager kernels.** For example `1 + 0.5 · clip(log2 s, −1, 3)`, so 0.5× scores 0.5.
   - Fixes: no reward for regressions.
   - Risks: the policy may avoid hard tasks, where any correct answer is likely slow. A wrong answer would still
     score below a slow correct one (0.1 vs 0.5).
2. **A larger speed bonus** (for example `1.0 · log2 s`).
   - Pushes for speed without punishing correct-but-slow answers.
   - Makes timing noise matter more; it needs the noise floor measured per task first.
3. **Per-task normalised speedup.** Score `s` against a known-good HIP reference or the best seen, not against
   eager.
   - Stops a few tasks with easy big wins from dominating, and shows how much headroom each task really has.
4. **Length-aware reward.** A small penalty per token over some budget, or a reward for finishing early.
   - Cuts truncation (22% of held-out answers at 12k) and step time.
   - Risks shortening the reasoning that makes answers correct; tune it against held-out correctness.
5. **Partial correctness credit.** Credit per seed passed, or per output within a looser tolerance.
   - More signal on hard tasks where whole groups fail.
   - Risks rewarding near-misses that are numerically wrong.
6. **Tolerances by dtype and op.** A single `1e-4` is strict for fp16/bf16 and loose for some fp32 reductions.
   Per-dtype tolerances would make "correct" mean the same thing across tasks.
7. **Robustness to input shapes.** Test correctness and speed on 2–3 shapes, not only the dataset's own.
   - Rewards kernels that generalise.
   - Penalises shape-specialised hacks.
8. **Advantage normalisation and regularisation.** Group std scaling, a KL or entropy term.
   - Helps stability over long runs.
   - Today's run is stable without them.
9. **Library-share guard.** Score speed on the answer's own kernel time when the library share is high, if
   `other_time_share` starts to grow. In this run it stayed small (median 0.04%), so no action was needed.
10. **Resource use.** Penalise very high register or LDS use, or long compiles, if deployment cares. Not measured
    today.

Suggested first changes for the next run: items 1 or 2 (speed), then 4 (length). Item 3 makes speed rewards
comparable across tasks once item 1 or 2 is in. Compare runs on held-out correctness *and* held-out speedup
(median, and share ≥ 1.1×), not on reward, since reward changes with the formula.
