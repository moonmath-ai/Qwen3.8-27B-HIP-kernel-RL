# The Zro-form server

A Tinker-compatible LoRA training server: the unmodified Tinker SDK (pinned 0.33.1) and tinker-cookbook
recipes run against it. Setup and the run itself: [REPRODUCE.md](../REPRODUCE.md).

## Modules

| Module | Role |
|---|---|
| `zroform/api.py` | FastAPI server for the Tinker wire protocol: version gate, auth, long-poll futures, telemetry sink. |
| `zroform/engine.py` | Sessions, training runs mapped to adapter slots, strict per-model `seq_id` ordering and dedupe, owner isolation, trainer→sampler hand-off, session reaper. |
| `zroform/trainer.py` | HF + PEFT LoRA trainer: one adapter per slot, alpha 32, gradients accumulate until `optim_step`, per-step `logprob_gap:mean`, chunked LM-head logprobs (long rollouts). |
| `zroform/sampler.py` | A stock SGLang 0.5.21 server on localhost with a per-boot random key. Adapters are loaded from in-memory tensors (CUDA IPC); Qwen3.5/3.8 linear-attention LoRA is converted to SGLang's fused layout (exact). |
| `zroform/kernels.py` | Linear-attention trainer kernels (`fla`, `causal-conv1d`), checked against PyTorch at start-up, with a fallback. |
| `zroform/losses.py` | `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro`, following Tinker's docs. All are sums. |
| `zroform/wire.py` | Protobuf and JSON codec, the inverse of the pinned SDK's own converters. |
| `zroform/futures.py`, `store.py`, `config.py`, `launch.py` | In-memory futures; local checkpoint store; profiles; launcher that hides the GPUs a profile doesn't use before CUDA initialises. |

## Configuration

- **Profiles** (`profiles/*.yaml`) set the model, GPU placement (`trainer_device`, and `sampler_gpus`,
  whose count is SGLang's TP; empty = colocated on the trainer's GPU), memory and concurrency limits, and slots
  per model. This run: `profiles/mi300x-qwen38-27b-phased.yaml` (16k context) and `-24k.yaml` (20k-token evals).
- **Machine paths and images** come from `scripts/env.sh` (override in the environment).
- **Start:** `scripts/run_server.sh <profile> <out_dir> [name]` runs the server in the SGLang ROCm image;
  `<out_dir>` holds the checkpoint store (`ZROFORM_DATA`), metrics and logs. Clients use
  `TINKER_API_KEY=tml-zroform-dev` (a local dev key; the accepted list is `ZROFORM_API_KEYS`).

## One-GPU phased schedule

A 27B model's trainer (bf16 weights, LoRA rank 64, Adam, activations for 16k-token sequences) and an SGLang
sampler with enough KV cache for ~100 long rollouts do not fit on one 192 GB GPU at the same time. With
`sampler_phased_memory: restart` (requires `slots_per_base: 1`, one synchronous client) they take turns:

- `zroform.engine.PhaseGate` puts the server in a *sample* or *train* phase. A sampling request while training
  (or a `forward_backward` while sampling) waits for the other phase's in-flight work, then switches.
- Switching to train stops SGLang and frees all its memory; switching to sample restarts it (a warm start,
  ~35-60 s) and reloads the live adapters from in-memory copies. One RL step therefore pays two switches.
- The RL client (`scripts/hip_rl.py`) samples, sends one zero-weight `forward_backward` to switch to train
  (so grading runs with SGLang off and kernel timing has the GPU alone), grades, then trains.
- The alternative, SGLang's `--enable-memory-saver` (`release`: free only the KV cache), crashed on ROCm
  (`Triton Error [HIP] 709` on saver-allocated memory), so `restart` is the mode used.

## Verification

CPU tests: see REPRODUCE.md §1.

| Property | Evidence |
|---|---|
| Wire compatibility with the pinned SDK | `tests/unit/test_wire.py`: round trips through the SDK's own encoders and decoders |
| Losses match Tinker's definitions | `tests/unit/test_losses.py`: formulas, input-key rules and gradient identities |
| RL, SFT and DPO verbs work through the unmodified SDK | `tests/e2e/test_sdk_cpu.py` (CPU, tiny random Qwen3) |
| `seq_id` ordering, retry dedupe, tenant isolation | `tests/unit/test_engine.py`, `test_tenants_cannot_see_each_other_and_do_not_interfere` |
| Phased schedule: phase switches, waiting, memory hand-back | `tests/unit/test_phased_memory.py` |
| Linear-attention LoRA: trainer and SGLang agree | `tests/unit/test_linear_attn_lora.py`; on the GPU, trainer vs SGLang 0.0071 nats/token on the MI300X, the bf16 noise floor (RESULTS.md) |
| Trainer↔sampler logprob gap | `logprob_gap:mean` in every RL `forward_backward` |

## Not supported

Top-k logprobs, prompt alt tokens, target prompt logprobs, image/audio chunks, 2-D (top-K) cross-entropy
targets, `save_weights_external`, checkpoint archive download. Each is rejected with a clear user error.
Checkpoints are stored in plaintext, and auth is a dev key list.
