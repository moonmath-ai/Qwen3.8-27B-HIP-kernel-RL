"""Sampler backend: a stock SGLang HTTP server on localhost, launched and owned by the engine.

- Bound to 127.0.0.1 with a random per-boot API key held only in memory, so its admin endpoints
  (adapter load/unload, flush) are reachable only by this engine.
- Adapters arrive as in-memory tensors (CUDA IPC handles via SGLang's own serializer): nothing is
  written to disk.
- SGLang's request logging is left off (its default); no prompt or token content is logged here.
"""

from __future__ import annotations

import asyncio
import base64
import os
import secrets
import signal
import subprocess
import sys
import time

import httpx

from zroform.wire import Sequence


class SamplerError(RuntimeError):
    pass


# PEFT (HF) module names -> SGLang's LoRA module names, where they differ. SGLang packs Qwen3.5's
# linear-attention in_proj_qkv + in_proj_z into one in_proj_qkvz.
_SGLANG_NAME = {"in_proj_qkv": "in_proj_qkvz", "in_proj_z": "in_proj_qkvz", "in_proj_q": "in_proj_qkvz",
                "in_proj_k": "in_proj_qkvz", "in_proj_v": "in_proj_qkvz"}


def sglang_target_modules(names: list[str]) -> list[str]:
    return list(dict.fromkeys(_SGLANG_NAME.get(n, n) for n in names))


# LoRA modules SGLang 0.5.21 supports but its --lora-target-modules flag doesn't list as choices.
_CLI_UNLISTED = {"in_proj_qkvz", "out_proj"}


def sglang_cli_targets(names: list[str]) -> list[str]:
    """--lora-target-modules values. `all` (SGLang auto-detects every LoRA-capable module of the loaded
    model, plus embed_tokens) only when a needed module can't be named on the command line."""
    targets = sglang_target_modules(names)
    return ["all"] if _CLI_UNLISTED & set(targets) else targets


def to_sglang_adapter(tensors: dict, config: dict, qkv_split: tuple[int, int, int] | None):
    """Rewrite a PEFT adapter into the layout SGLang loads. SGLang accepts linear-attention LoRA either
    split (in_proj_q/k/v/z) or merged with one shared A (in_proj_qkvz); HF/PEFT has in_proj_qkv + in_proj_z.
    LoRA on a fused qkv equals per-slice LoRA with the same A and B split by rows, so this is exact.
    The config's target_modules are derived from the tensors actually sent."""
    out = {}
    for k, v in tensors.items():
        if ".in_proj_qkv." not in k:
            out[k] = v
        elif "lora_A" in k:
            for s in "qkv":
                out[k.replace(".in_proj_qkv.", f".in_proj_{s}.")] = v
        else:
            if qkv_split is None:
                raise SamplerError("in_proj_qkv adapter weights but no q/k/v split for this model")
            for s, part in zip("qkv", v.split(list(qkv_split), dim=0)):
                out[k.replace(".in_proj_qkv.", f".in_proj_{s}.")] = part.contiguous()
    leaves = {k.split(".lora_")[0].rsplit(".", 1)[-1] for k in out}
    return out, {**config, "target_modules": sorted(sglang_target_modules(sorted(leaves)))}



def group_alive(pgid: int) -> bool:
    """Any live (non-zombie) process left in this process group? Zombies hold no GPU memory, and orphaned ones may
    never be reaped here (the server can be PID 1 in its container), so they don't count."""
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                rest = f.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(rest[2]) == pgid and rest[0] not in ("Z", "X"):
            return True
    return False

class SGLangSampler:
    def __init__(self, base_model: str, port: int, max_lora_rank: int, target_modules: list[str],
                 max_loaded_loras: int, mem_fraction: float, context_length: int, log_path: str,
                 max_running_requests: int = 256, extra_args: list[str] | None = None,
                 gpus: list[int] | None = None, qkv_split: tuple[int, int, int] | None = None,
                 phase_mode: str = ""):
        self.base_model = base_model
        # Phased schedule (Profile.sampler_phased_memory): "restart" stops/starts this process, "release" uses
        # SGLang's memory saver. With "restart", loaded adapters are kept as CPU copies (memory only, never disk) so they
        # can be loaded again after a restart.
        self.phase_mode = phase_mode
        self._held: dict[str, tuple[dict, dict]] = {}
        self.port = port
        self.qkv_split = qkv_split  # linear-attention in_proj_qkv row sizes (Qwen3.5), for to_sglang_adapter
        # GPU indices as this process sees them; SGLang runs TP=len(gpus), rank i on gpus[i].
        self.gpus = list(gpus or [0])
        self._key = secrets.token_hex(32)  # not token_urlsafe: a leading "-" makes argparse read it as a flag
        self._cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", base_model, "--host", "127.0.0.1", "--port", str(port),
            "--api-key", self._key, "--tp-size", str(len(self.gpus)),
            "--enable-lora", "--max-lora-rank", str(max_lora_rank),
            "--lora-target-modules", *sglang_cli_targets(target_modules),
            "--max-loaded-loras", str(max_loaded_loras), "--max-loras-per-batch", str(max_loaded_loras),
            "--mem-fraction-static", str(mem_fraction), "--context-length", str(context_length),
            "--max-running-requests", str(max_running_requests),
            "--log-level", "warning",
            *(["--enable-memory-saver"] if phase_mode == "release" else []),
            *(extra_args or []),
        ]
        self._log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._http: httpx.AsyncClient | None = None

    async def start(self, timeout: float = 900) -> None:
        log = open(self._log_path, "ab")  # SGLang's own log: startup/errors only, no request content
        # Request logging stays off: we never pass --log-requests (SGLang's default is off).
        # Our indices are relative to our own CUDA_VISIBLE_DEVICES, if set: map them to its entries.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devs = [visible.split(",")[g] for g in self.gpus] if visible else [str(g) for g in self.gpus]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devs)}
        # The trainer's allocator setting (Profile.trainer_cuda_alloc_conf) is not for SGLang: with expandable
        # segments its TP custom all-reduce fails to share CUDA-graph buffers ("invalid argument").
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        # Own process group: SGLang's scheduler/detokenizer children hold the GPU memory, and stop() must wait for
        # all of them, not just the launcher (otherwise a restart soon after a stop can find the memory still
        # taken, and SGLang refuses to start).
        self._proc = subprocess.Popen(self._cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self._http = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}",
                                       headers={"Authorization": f"Bearer {self._key}"}, timeout=None)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self._proc.poll() is not None:
                raise SamplerError(f"SGLang exited with {self._proc.returncode}; see {self._log_path}")
            try:
                if (await self._http.get("/health", timeout=5)).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(2)
        raise SamplerError("SGLang did not become healthy in time")

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()
        if self._proc is None:
            return
        pgid = self._proc.pid  # start_new_session: the launcher leads its own group
        for sig, wait_s in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
            if not group_alive(pgid):
                break
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + wait_s
            while group_alive(pgid) and time.monotonic() < deadline:
                self._proc.poll()  # reap the launcher
                await asyncio.sleep(0.5)
        self._proc.poll()
        if group_alive(pgid):
            raise SamplerError(f"SGLang process group {pgid} survived SIGKILL")

    async def load_adapter(self, name: str, tensors: dict, config: dict) -> None:
        if self.phase_mode == "restart":
            self._held[name] = ({k: v.detach().to("cpu", copy=True) for k, v in tensors.items()}, dict(config))
        await self._send_adapter(name, tensors, config)

    async def _send_adapter(self, name: str, tensors: dict, config: dict) -> None:
        # SGLang's own serializer: CUDA tensors travel as IPC handles (same host, no disk). One payload
        # per TP rank, each on that rank's GPU: SGLang sees only its own GPUs, so it can't open a handle
        # to the trainer's GPU. The full adapter goes to every rank; SGLang slices it for TP itself.
        # The torch patch names devices by UUID, not index, since SGLang's CUDA_VISIBLE_DEVICES differs
        # from ours; SGLang's TP workers apply the same patch when deserializing.
        import torch
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
        monkey_patch_torch_reductions()
        tensors, config = to_sglang_adapter(tensors, config, self.qkv_split)
        per_rank = [{k: v.to(f"cuda:{g}") for k, v in tensors.items()} for g in self.gpus]
        for g in self.gpus:
            torch.cuda.synchronize(g)
        payloads = [base64.b64encode(MultiprocessingSerializer.serialize(t)).decode() for t in per_rank]
        body = {"lora_name": name, "config_dict": config, "serialized_named_tensors": payloads}
        r = await self._http.post("/load_lora_adapter_from_tensors", json=body)
        if r.status_code != 200:
            raise SamplerError(f"adapter load failed ({r.status_code})")
        # `per_rank` must stay alive until SGLang has copied them; the await above guarantees it.

    async def release_kv(self) -> None:
        """Give the GPU memory to the trainer: stop SGLang ("restart") or free its KV cache ("release")."""
        if self.phase_mode == "restart":
            await self.stop()
            return
        r = await self._http.post("/release_memory_occupation", json={"tags": ["kv_cache"]})
        if r.status_code != 200:
            raise SamplerError(f"release_memory_occupation failed ({r.status_code})")

    async def resume_kv(self) -> None:
        if self.phase_mode == "restart":
            # The driver hands a stopped SGLang's ~100 GB back over some time after its processes are gone; starting
            # too soon makes SGLang size its pools from a short free-memory reading and refuse to start (e.g. a
            # restart ~1 s after a stop, when a step had nothing to train). Wait for free memory to settle, and
            # retry a failed start after a pause.
            for attempt in range(3):
                free = await self._wait_free_memory_settled()
                print(f"[sampler] starting SGLang (attempt {attempt + 1}); free GPU memory GB (first, settled, "
                      f"waited s): {free}", flush=True)
                try:
                    await self.start()
                    break
                except SamplerError:
                    await self.stop()
                    if attempt == 2:
                        raise
                    await asyncio.sleep(30)
            for name, (tensors, config) in list(self._held.items()):
                await self._send_adapter(name, tensors, config)
            return
        r = await self._http.post("/resume_memory_occupation", json={"tags": ["kv_cache"]})
        if r.status_code != 200:
            raise SamplerError(f"resume_memory_occupation failed ({r.status_code})")

    async def _wait_free_memory_settled(self, min_s: float = 3.0, max_s: float = 180.0, tol_gb: float = 0.5):
        """Poll free memory on our GPUs until two readings 2 s apart agree within tol_gb (at least min_s, at most
        max_s). Returns (first reading, settled reading, seconds waited), GB per GPU; None without CUDA (CPU tests)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
        except ImportError:
            return None

        def read():
            return [round(torch.cuda.mem_get_info(g)[0] / 2**30, 1) for g in self.gpus]
        loop = asyncio.get_running_loop()
        t0, prev = loop.time(), read()
        first = prev
        await asyncio.sleep(min_s)
        while True:
            cur = read()
            if all(abs(a - b) <= tol_gb for a, b in zip(cur, prev)) or loop.time() - t0 > max_s:
                return first, cur, round(loop.time() - t0, 1)
            prev = cur
            await asyncio.sleep(2)

    async def unload_adapter(self, name: str) -> None:
        self._held.pop(name, None)
        if self.phase_mode == "restart" and (self._proc is None or self._proc.poll() is not None):
            return  # stopped for training: dropping the held copy is enough, the restart won't load it
        await self._http.post("/unload_lora_adapter", json={"lora_name": name})

    async def sample(self, tokens: list[int], n: int, params: dict, adapter: str | None,
                     prompt_logprobs: bool) -> tuple[list[Sequence], list | None]:
        sp = {"temperature": params.get("temperature", 1.0), "top_p": params.get("top_p", 1.0),
              "top_k": params.get("top_k", -1)}
        if params.get("max_tokens") is not None:
            sp["max_new_tokens"] = params["max_tokens"]
        if params.get("seed") is not None:
            sp["sampling_seed"] = params["seed"]
        stop = params.get("stop")
        if stop == []:                    # Tinker: [] disables EOS
            sp["ignore_eos"] = True
        elif isinstance(stop, str):
            sp["stop"] = [stop]
        elif stop and isinstance(stop[0], int):
            sp["stop_token_ids"] = stop
        elif stop:
            sp["stop"] = stop
        body = {"input_ids": [tokens] * n, "sampling_params": sp, "return_logprob": True,
                "logprob_start_len": 0 if prompt_logprobs else -1}
        if adapter:
            body["lora_path"] = [adapter] * n
        r = await self._http.post("/generate", json=body)
        if r.status_code != 200:
            raise SamplerError(f"generate failed ({r.status_code})")
        outs = r.json()
        seqs = []
        for o in outs:
            mi = o["meta_info"]
            pairs = mi["output_token_logprobs"]  # [[logprob, token_id, text], ...], incl. the stop token
            reason = (mi.get("finish_reason") or {}).get("type", "stop")
            seqs.append(Sequence(tokens=[int(p[1]) for p in pairs], logprobs=[float(p[0]) for p in pairs],
                                 stop_reason="length" if reason == "length" else "stop"))
        plp = None
        if prompt_logprobs:
            plp = [None if p[0] is None else float(p[0]) for p in outs[0]["meta_info"]["input_token_logprobs"]]
        return seqs, plp
