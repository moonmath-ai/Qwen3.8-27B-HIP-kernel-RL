"""One-GPU phased schedule (Profile.sampler_phased_memory): PhaseGate ordering and the memory hand-over calls."""

import asyncio

import pytest

from zroform.config import load_profile
from zroform.engine import PhaseGate
from zroform.telemetry import MetricsLog


class FakeSampler:
    def __init__(self, log):
        self.log = log

    async def release_kv(self):
        self.log.append("release_kv")

    async def resume_kv(self):
        self.log.append("resume_kv")


class FakeTrainer:
    def __init__(self, log):
        self.log = log

    def free_cache(self):
        self.log.append("free_cache")


def gate(enabled=True):
    log = []
    return PhaseGate(enabled, FakeSampler(log), FakeTrainer(log), MetricsLog(None)), log


async def test_switching_phases_moves_memory_once_per_switch():
    g, log = gate()
    async with g.hold("sample"):          # starts in the sample phase: nothing to do
        pass
    async with g.hold("train"):
        pass
    async with g.hold("train"):           # still training: no second release
        pass
    async with g.hold("sample"):
        pass
    assert log == ["release_kv", "free_cache", "resume_kv"]


async def test_training_waits_for_in_flight_samples_and_sampling_waits_for_training():
    g, log = gate()
    events = []
    sample_may_finish = asyncio.Event()

    async def sample():
        async with g.hold("sample"):
            events.append("sample start")
            await sample_may_finish.wait()
            events.append("sample end")

    async def train():
        async with g.hold("train"):
            events.append("train")
            await asyncio.sleep(0.05)
            events.append("train end")

    s = asyncio.create_task(sample())
    await asyncio.sleep(0.01)
    t = asyncio.create_task(train())
    await asyncio.sleep(0.02)
    assert events == ["sample start"]     # training blocked while the sample is in flight
    late = asyncio.create_task(sample())  # arrives during training below; must wait for it
    sample_may_finish.set()
    await t
    await asyncio.gather(s, late)
    assert events[:4] == ["sample start", "sample end", "train", "train end"]
    assert log == ["release_kv", "free_cache", "resume_kv"]


async def test_disabled_gate_never_touches_memory():
    g, log = gate(enabled=False)
    async with g.hold("train"):
        async with g.hold("sample"):      # would deadlock if the gate were active
            pass
    assert log == []


def test_profile_rejects_phased_memory_with_shared_slots(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("name: x\nslots_per_base: 2\nsampler_phased_memory: restart\nmodels: {m: {}}\n")
    with pytest.raises(ValueError, match="slots_per_base: 1"):
        load_profile(p)


async def test_restart_mode_stops_sglang_and_reloads_live_adapters_from_memory(monkeypatch):
    import torch

    from zroform.sampler import SGLangSampler
    s = SGLangSampler("m", 30100, 8, ["q_proj"], 4, 0.5, 1024, "/dev/null", phase_mode="restart")
    calls = []

    async def start():
        calls.append("start")

    async def stop():
        calls.append("stop")
        s._proc = None

    async def send(name, tensors, config):
        calls.append(("send", name, float(next(iter(tensors.values())).sum())))

    monkeypatch.setattr(s, "start", start)
    monkeypatch.setattr(s, "stop", stop)
    monkeypatch.setattr(s, "_send_adapter", send)
    assert "--enable-memory-saver" not in s._cmd

    live = {"a.lora_A.weight": torch.ones(2, 2)}
    await s.load_adapter("v1", live, {"r": 8})
    await s.load_adapter("v0", {"a.lora_A.weight": torch.zeros(2, 2)}, {"r": 8})
    live["a.lora_A.weight"].add_(1)              # trainer keeps updating its tensors: the held copy must not move
    await s.release_kv()
    await s.unload_adapter("v0")                 # while stopped: forget it, no HTTP call to a dead server
    await s.resume_kv()
    assert calls == [("send", "v1", 4.0), ("send", "v0", 0.0), "stop", "start", ("send", "v1", 4.0)]


def test_release_mode_adds_memory_saver_flag():
    from zroform.sampler import SGLangSampler
    s = SGLangSampler("m", 30100, 8, ["q_proj"], 4, 0.5, 1024, "/dev/null", phase_mode="release")
    assert "--enable-memory-saver" in s._cmd


def test_profile_rejects_unknown_phase_mode(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("name: x\nsampler_phased_memory: true\nmodels: {m: {}}\n")
    with pytest.raises(ValueError, match="'restart' or 'release'"):
        load_profile(p)


def test_stop_waits_for_the_whole_process_group(tmp_path):
    """SGLang's children hold the GPU memory, so waiting only for the launcher lets the next phase restart it
    while the memory is still taken. The launcher leads its own group and stop() waits for all of it."""
    import asyncio
    import subprocess
    from zroform.sampler import SGLangSampler, group_alive

    s = SGLangSampler("m", 1, 8, ["q_proj"], 1, 0.5, 128, str(tmp_path / "log"), phase_mode="restart")
    # a launcher that ignores SIGTERM in a child, like a slow-to-exit scheduler process
    s._proc = subprocess.Popen(["sh", "-c", "(trap '' TERM; sleep 300) & sleep 300"], start_new_session=True)
    pgid = s._proc.pid
    assert group_alive(pgid)
    asyncio.run(asyncio.wait_for(s.stop(), 120))
    assert not group_alive(pgid)


def test_restart_retries_a_failed_start(monkeypatch):
    """SGLang can refuse to start right after a stop (memory not yet back); retry, then give up."""
    import asyncio
    from zroform import sampler as sm

    s = sm.SGLangSampler("m", 30100, 8, ["q_proj"], 4, 0.5, 1024, "/dev/null", phase_mode="restart")
    calls = []

    async def start():
        calls.append("start")
        if calls.count("start") < 3:
            raise sm.SamplerError("SGLang exited with 1")

    async def stop():
        calls.append("stop")

    async def no_sleep(_):
        pass

    monkeypatch.setattr(s, "start", start)
    monkeypatch.setattr(s, "stop", stop)
    monkeypatch.setattr(sm.asyncio, "sleep", no_sleep)
    asyncio.run(s.resume_kv())
    assert calls == ["start", "stop", "start", "stop", "start"]
    calls.clear()

    async def always_fail():
        calls.append("start")
        raise sm.SamplerError("SGLang exited with 1")

    monkeypatch.setattr(s, "start", always_fail)
    import pytest
    with pytest.raises(sm.SamplerError):
        asyncio.run(s.resume_kv())
    assert calls.count("start") == 3
