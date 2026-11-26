# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.metadata
import importlib.util
import logging
import sys
import threading
import types
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, ClassVar

import faiss
import numpy as np
import pytest
from haystack import Document

from haystack_integrations.components.preprocessors.emeans import EmeansFitter, _backend, fitter
from haystack_integrations.components.preprocessors.emeans._backend import (
    BackendChoice,
    EnvInfo,
    ProbeResult,
    _backend_hint,
    default_rungs,
    hidden_gpu_reason,
    mismatch_reason,
    resolve_precision,
    select_backend,
)
from haystack_integrations.components.preprocessors.emeans._model import load_model

AMD_MISMATCH = (
    "AMD GPU found (/dev/kfd) but PyTorch 2.15.0 is not a ROCm build; "
    "if this GPU is supported by ROCm, install the ROCm build of PyTorch and do not use the [gpu] extra"
)
NVIDIA_MISMATCH = (
    "NVIDIA GPU found but PyTorch 2.15.0 is a CPU-only build; "
    "install a CUDA build of PyTorch, for example with the [gpu] extra"
)
ROCM_ON_NVIDIA = (
    "NVIDIA GPU found but PyTorch 2.15.0 is a ROCm build; "
    "install a CUDA build of PyTorch, for example with the [gpu] extra"
)
AMD_DRIVER_CLAUSE = (
    "check that the amdgpu driver is loaded and that this user can open /dev/kfd and /dev/dri (render and video groups)"
)
AMD_BASE = "AMD GPU found (/dev/kfd) and PyTorch 2.15.0 is a ROCm build, but PyTorch sees no GPU; "
AMD_HIDDEN = AMD_BASE + AMD_DRIVER_CLAUSE
NVIDIA_BASE = "NVIDIA GPU found and PyTorch 2.15.0 is a CUDA 12.8 build, but PyTorch sees no GPU; "
NVIDIA_HIDDEN = (
    NVIDIA_BASE
    + "check that the NVIDIA driver is loaded and supports CUDA 12.8, and that this user can open /dev/nvidia*"
)


def visible_clause(name: str, value: str) -> str:
    return f"{name}={value!r} is set and may hide it; unset it, or use device='cpu' to run on CPU"


HINT = "(pip install emeans-rocm for Triton kernels and BF16/FP16 on RX 9000 GPUs)"


def warnings_in(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelname == "WARNING"]


def gpu_env(**overrides: Any) -> EnvInfo:
    params: dict[str, Any] = {
        "torch_importable": True,
        "gpu_available": True,
        "is_rocm": False,
        "emeans_rocm": False,
        "device_name": "Fake GPU",
        "torch_version": "2.15.0",
        "hip_version": None,
    }
    params.update(overrides)
    return EnvInfo(**params)


OK = ProbeResult(True, "auto", None)


def fail(reason: str) -> ProbeResult:
    return ProbeResult(False, None, reason)


class FakeProbe:
    """Probe stand-in that records its calls; rungs not listed pass."""

    def __init__(self, results: dict[str, ProbeResult] | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, rung_name: str, requested: str, precision: str) -> ProbeResult:
        self.calls.append((rung_name, requested, precision))
        return self.results.get(rung_name, OK)


def choose(env: EnvInfo, probe: FakeProbe, device: str = "auto") -> BackendChoice:
    return select_backend(device=device, n=100, k=10, gpu_threshold=1, env=env, gpu_rungs=default_rungs(probe_fn=probe))


# mismatch_reason table


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"amd_node": True, "nvidia_node": False, "hip_version": None, "cuda_version": None, "gpu_available": False},
            AMD_MISMATCH,
        ),
        (
            {
                "amd_node": True,
                "nvidia_node": False,
                "hip_version": "7.16",
                "cuda_version": None,
                "gpu_available": False,
            },
            None,
        ),
        (
            {
                "amd_node": True,
                "nvidia_node": False,
                "hip_version": None,
                "cuda_version": "12.8",
                "gpu_available": True,
            },
            None,
        ),
        (
            {"amd_node": False, "nvidia_node": True, "hip_version": None, "cuda_version": None, "gpu_available": False},
            NVIDIA_MISMATCH,
        ),
        (
            {
                "amd_node": False,
                "nvidia_node": True,
                "hip_version": None,
                "cuda_version": "12.8",
                "gpu_available": False,
            },
            None,
        ),
        (
            {
                "amd_node": False,
                "nvidia_node": False,
                "hip_version": None,
                "cuda_version": None,
                "gpu_available": False,
            },
            None,
        ),
    ],
)
def test_mismatch_reason_table(kwargs: dict[str, Any], expected: str | None) -> None:
    assert mismatch_reason(torch_version="2.15.0", **kwargs) == expected


def _nodes(*, amd: bool, nvidia: bool, hip: str | None, cuda: str | None) -> dict[str, Any]:
    return {"amd_node": amd, "nvidia_node": nvidia, "hip_version": hip, "cuda_version": cuda, "gpu_available": False}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (_nodes(amd=False, nvidia=True, hip="7.16", cuda=None), ROCM_ON_NVIDIA),  # ROCm torch on an NVIDIA machine
        (_nodes(amd=True, nvidia=True, hip=None, cuda="12.8"), None),  # APU + NVIDIA, CUDA build: not a build issue
        (_nodes(amd=True, nvidia=True, hip=None, cuda=None), NVIDIA_MISMATCH),  # APU + NVIDIA, CPU-only build
        (_nodes(amd=True, nvidia=True, hip="7.16", cuda=None), None),  # both, ROCm build: possibly hidden on purpose
        (_nodes(amd=True, nvidia=False, hip=None, cuda="12.8"), AMD_MISMATCH),  # CUDA build on an AMD-only machine
    ],
)
def test_mismatch_reason_mixed_layouts(kwargs: dict[str, Any], expected: str | None) -> None:
    assert mismatch_reason(torch_version="2.15.0", **kwargs) == expected


def test_mismatch_reason_text_names_fix() -> None:
    text = mismatch_reason(
        amd_node=True,
        nvidia_node=False,
        torch_version="2.15.0",
        hip_version=None,
        cuda_version=None,
        gpu_available=False,
    )
    assert text is not None
    for fragment in ("AMD GPU found", "2.15.0", "not a ROCm build", "[gpu]"):
        assert fragment in text


# hidden_gpu_reason table (SA-02)

_AMD = {"amd_node": True, "nvidia_node": False, "hip_version": "7.16", "cuda_version": None, "gpu_available": False}
_NVIDIA = {"amd_node": False, "nvidia_node": True, "hip_version": None, "cuda_version": "12.8", "gpu_available": False}


@pytest.mark.parametrize(
    ("kwargs", "environ", "expected"),
    [
        (_AMD, {}, AMD_HIDDEN),
        *[
            (_AMD, {name: ""}, AMD_BASE + visible_clause(name, ""))
            for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        ],
        (_NVIDIA, {}, NVIDIA_HIDDEN),
        (_NVIDIA, {"CUDA_VISIBLE_DEVICES": "-1"}, NVIDIA_BASE + visible_clause("CUDA_VISIBLE_DEVICES", "-1")),
        (_NVIDIA, {"HIP_VISIBLE_DEVICES": ""}, NVIDIA_HIDDEN),
        ({**_AMD, "gpu_available": True}, {}, None),
        ({**_NVIDIA, "gpu_available": True}, {}, None),
        ({**_AMD, "amd_node": False}, {}, None),
        ({**_AMD, "hip_version": None}, {}, None),
        ({**_NVIDIA, "cuda_version": None}, {}, None),
    ],
)
def test_hidden_gpu_reason_table(kwargs: dict[str, Any], environ: dict[str, str], expected: str | None) -> None:
    assert hidden_gpu_reason(torch_version="2.15.0", environ=environ, **kwargs) == expected


def test_hidden_gpu_reason_escapes_control_characters() -> None:
    text = hidden_gpu_reason(torch_version="2.15.0", environ={"ROCR_VISIBLE_DEVICES": "0\nFAKE"}, **_AMD)
    assert text is not None
    assert "\n" not in text


# ladder


def test_nvidia_all_ok_picks_auto() -> None:
    probe = FakeProbe()
    choice = choose(gpu_env(), probe)
    assert choice == BackendChoice("nvidia-auto", "Fake GPU", None, requested="auto")


def test_nvidia_auto_fails_picks_triton(caplog: pytest.LogCaptureFixture) -> None:
    probe = FakeProbe({"nvidia-auto": fail("ImportError: CuTe assignment requires nvidia-cutlass-dsl")})
    with caplog.at_level(logging.DEBUG):
        choice = choose(gpu_env(), probe)
    assert (choice.name, choice.requested) == ("nvidia-triton", "triton")
    assert warnings_in(caplog) == []


def test_nvidia_auto_and_triton_fail_picks_native() -> None:
    probe = FakeProbe({"nvidia-auto": fail("a"), "nvidia-triton": fail("b")})
    choice = choose(gpu_env(), probe)
    assert (choice.name, choice.requested) == ("native", "native")


def test_amd_with_emeans_rocm_picks_rocm_triton() -> None:
    probe = FakeProbe()
    choice = choose(gpu_env(is_rocm=True, hip_version="7.16", emeans_rocm=True), probe)
    assert (choice.name, choice.requested) == ("rocm-triton", "auto")
    assert [call[0] for call in probe.calls] == ["rocm-triton"]


def test_amd_without_emeans_rocm_picks_native_silently(caplog: pytest.LogCaptureFixture) -> None:
    probe = FakeProbe()
    with caplog.at_level(logging.DEBUG):
        choice = choose(gpu_env(is_rocm=True, hip_version="7.16"), probe)
    assert (choice.name, choice.requested) == ("native", "native")
    assert [call[0] for call in probe.calls] == ["native"]
    assert warnings_in(caplog) == []


class CompilationError(Exception):
    """Stands in for a Triton compilation error that is not a RuntimeError."""


@pytest.mark.parametrize("error", [ImportError, RuntimeError, CompilationError])
def test_emeans_rocm_probe_failure_falls_to_native(error: type[Exception]) -> None:
    probe = FakeProbe({"rocm-triton": fail(f"{error.__name__}: boom")})
    choice = choose(gpu_env(is_rocm=True, hip_version="7.16", emeans_rocm=True), probe)
    assert (choice.name, choice.requested) == ("native", "native")


def test_all_rungs_fail_auto_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    probe = FakeProbe({"nvidia-auto": fail("r1"), "nvidia-triton": fail("r2"), "native": fail("r3")})
    with caplog.at_level(logging.DEBUG):
        choice = choose(gpu_env(), probe)
    assert choice.name == "cpu"
    reason = choice.fallback_reason
    assert reason is not None
    assert reason.startswith("no GPU backend usable: ")
    for fragment in ("nvidia-auto: r1", "nvidia-triton: r2", "native: r3"):
        assert fragment in reason
    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert warnings[0].getMessage() == reason


def test_all_rungs_fail_cuda_raises() -> None:
    probe = FakeProbe({"nvidia-auto": fail("r1"), "nvidia-triton": fail("r2"), "native": fail("r3")})
    with pytest.raises(RuntimeError) as info:
        choose(gpu_env(), probe, device="cuda")
    for fragment in ("nvidia-auto: r1", "nvidia-triton: r2", "native: r3"):
        assert fragment in str(info.value)


# silent and warning fallbacks


def test_torch_missing_auto_silent(caplog: pytest.LogCaptureFixture) -> None:
    env = gpu_env(torch_importable=False, gpu_available=False, torch_version=None)
    with caplog.at_level(logging.DEBUG):
        choice = choose(env, FakeProbe())
    assert choice.fallback_reason == "no GPU backend usable: PyTorch is not installed"
    assert warnings_in(caplog) == []


def test_torch_missing_cuda_raises() -> None:
    env = gpu_env(torch_importable=False, gpu_available=False, torch_version=None)
    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        choose(env, FakeProbe(), device="cuda")


@pytest.mark.parametrize("text", [AMD_MISMATCH, NVIDIA_MISMATCH])
def test_mismatch_auto_warns_once_with_fallback_reason(text: str, caplog: pytest.LogCaptureFixture) -> None:
    probe = FakeProbe()
    with caplog.at_level(logging.DEBUG):
        choice = choose(gpu_env(gpu_available=False, mismatch_reason=text), probe)
    assert choice.name == "cpu"
    assert choice.fallback_reason is not None
    assert text in choice.fallback_reason
    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert warnings[0].getMessage() == choice.fallback_reason
    assert probe.calls == []


@pytest.mark.parametrize("text", [AMD_MISMATCH, NVIDIA_MISMATCH])
def test_mismatch_cuda_raises_with_text(text: str) -> None:
    with pytest.raises(RuntimeError) as info:
        choose(gpu_env(gpu_available=False, mismatch_reason=text), FakeProbe(), device="cuda")
    assert text in str(info.value)


def test_hidden_gpu_auto_warns_once_with_reason(caplog: pytest.LogCaptureFixture) -> None:
    env = gpu_env(gpu_available=False, is_rocm=True, hip_version="7.16", mismatch_reason=AMD_HIDDEN)
    with caplog.at_level(logging.DEBUG):
        choice = choose(env, FakeProbe())
    assert choice.name == "cpu"
    assert choice.fallback_reason == f"no GPU backend usable: {AMD_HIDDEN}"
    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert warnings[0].getMessage() == choice.fallback_reason


def test_hidden_gpu_cuda_raises_with_reason() -> None:
    env = gpu_env(gpu_available=False, is_rocm=True, hip_version="7.16", mismatch_reason=AMD_HIDDEN)
    with pytest.raises(RuntimeError) as info:
        choose(env, FakeProbe(), device="cuda")
    assert AMD_HIDDEN in str(info.value)


def test_rocm_torch_without_reason_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    env = gpu_env(gpu_available=False, is_rocm=True, hip_version="7.16")
    with caplog.at_level(logging.DEBUG):
        choice = choose(env, FakeProbe())
    assert choice.fallback_reason == "no GPU backend usable: no GPU visible to PyTorch"
    assert warnings_in(caplog) == []


def test_cpu_torch_without_device_node_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    env = gpu_env(gpu_available=False)
    with caplog.at_level(logging.DEBUG):
        choice = choose(env, FakeProbe())
    assert choice.name == "cpu"
    assert warnings_in(caplog) == []


# memoisation and warning repetition


def test_probe_runs_once_per_rung_but_warning_repeats(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[tuple[str, str]] = []

    def counting(requested: str, precision: str) -> ProbeResult:
        calls.append((requested, precision))
        return fail("nope")

    monkeypatch.setattr(_backend, "_run_probe", counting)
    env = gpu_env()
    with caplog.at_level(logging.DEBUG):
        for _ in range(2):
            choice = select_backend(device="auto", n=100, k=10, gpu_threshold=1, env=env, gpu_rungs=default_rungs())
            assert choice.name == "cpu"
    assert len(calls) == 3  # nvidia-auto, nvidia-triton, native; the second call reuses the memo
    assert len(warnings_in(caplog)) == 2


def test_probe_propagates_exception_and_caches_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(requested: str, precision: str) -> ProbeResult:  # noqa: ARG001
        msg = "out of memory"
        raise MemoryError(msg)

    monkeypatch.setattr(_backend, "_run_probe", boom)
    with pytest.raises(MemoryError):
        _backend.probe("native", "native", "fp32")
    assert _backend._PROBE_CACHE == {}


def test_probe_memo_keeps_rungs_with_the_same_name_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake(requested: str, precision: str) -> ProbeResult:
        calls.append((requested, precision))
        return ProbeResult(requested == "triton", requested, None)

    monkeypatch.setattr(_backend, "_run_probe", fake)
    assert _backend.probe("custom", "triton", "fp32").ok
    assert not _backend.probe("custom", "native", "fp32").ok
    assert _backend.probe("custom", "triton", "fp32").ok
    assert calls == [("triton", "fp32"), ("native", "fp32")]


@pytest.mark.parametrize(
    ("reason", "remembered"),
    [
        ("RuntimeError: HIP error: out of memory", False),
        ("RuntimeError: CUDA error: device or resource busy", False),
        ("OutOfResources: out of resource: shared memory", True),
        ("CompilationError: bad kernel", True),
    ],
)
def test_probe_does_not_remember_transient_failures(
    monkeypatch: pytest.MonkeyPatch, reason: str, remembered: bool
) -> None:
    calls: list[str] = []

    def fake(requested: str, precision: str) -> ProbeResult:  # noqa: ARG001
        calls.append(requested)
        return ProbeResult(False, None, reason)

    monkeypatch.setattr(_backend, "_run_probe", fake)
    for _ in range(2):
        assert not _backend.probe("nvidia-triton", "triton", "fp32").ok
    assert len(calls) == (1 if remembered else 2)


def test_probe_of_one_key_does_not_block_another(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def fake(requested: str, precision: str) -> ProbeResult:  # noqa: ARG001
        if requested == "triton":
            started.set()
            assert release.wait(5)
        return ProbeResult(True, requested, None)

    monkeypatch.setattr(_backend, "_run_probe", fake)
    slow = threading.Thread(target=_backend.probe, args=("nvidia-triton", "triton", "fp32"))
    slow.start()
    try:
        assert started.wait(5)
        assert _backend.probe("native", "native", "fp32").ok  # would deadlock on a global probe lock
    finally:
        release.set()
        slow.join(5)
    assert _backend._PROBE_CACHE[("nvidia-triton", "triton", "fp32")].ok


# probe out of memory (SA-01)


class FakeOutOfMemoryError(RuntimeError):
    pass


class OomProbe(FakeProbe):
    """FakeProbe that raises a fake out-of-memory error for the rungs in `oom`."""

    def __init__(self, oom: set[str], results: dict[str, ProbeResult] | None = None) -> None:
        super().__init__(results)
        self.oom = oom

    def __call__(self, rung_name: str, requested: str, precision: str) -> ProbeResult:
        if rung_name in self.oom:
            self.calls.append((rung_name, requested, precision))
            msg = "CUDA out of memory"
            raise FakeOutOfMemoryError(msg)
        return super().__call__(rung_name, requested, precision)


@pytest.fixture
def fake_torch_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(OutOfMemoryError=FakeOutOfMemoryError))


@pytest.mark.usefixtures("fake_torch_oom")
def test_auto_every_rung_oom_falls_back_to_cpu_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    probe = OomProbe({"nvidia-auto", "nvidia-triton", "native"})
    with caplog.at_level(logging.DEBUG):
        choice = choose(gpu_env(), probe)
    assert choice.name == "cpu"
    reason = choice.fallback_reason
    assert reason is not None
    assert reason.startswith("no GPU backend usable:")
    assert reason.count("out of GPU memory") == 3
    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert warnings[0].getMessage() == reason


@pytest.mark.usefixtures("fake_torch_oom")
def test_auto_oom_on_one_rung_tries_the_next(caplog: pytest.LogCaptureFixture) -> None:
    probe = OomProbe({"nvidia-auto"})
    with caplog.at_level(logging.DEBUG):
        choice = choose(gpu_env(), probe)
    assert choice.name == "nvidia-triton"
    assert choice.fallback_reason is None
    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert "nvidia-auto" in warnings[0].getMessage()
    assert "out of memory" in warnings[0].getMessage()


@pytest.mark.usefixtures("fake_torch_oom")
def test_cuda_probe_oom_propagates_unwrapped() -> None:
    probe = OomProbe({"nvidia-auto"})
    with pytest.raises(FakeOutOfMemoryError):
        choose(gpu_env(), probe, device="cuda")
    assert [call[0] for call in probe.calls] == ["nvidia-auto"]


@pytest.mark.usefixtures("fake_torch_oom")
def test_auto_probe_oom_is_not_memoised(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(requested: str, precision: str) -> ProbeResult:  # noqa: ARG001
        msg = "CUDA out of memory"
        raise FakeOutOfMemoryError(msg)

    monkeypatch.setattr(_backend, "_run_probe", boom)
    _backend.clear_probe_cache()
    choice = select_backend(device="auto", n=100, k=10, gpu_threshold=1, env=gpu_env(), gpu_rungs=default_rungs())
    assert choice.name == "cpu"
    assert _backend._PROBE_CACHE == {}


class RaisingProbe(FakeProbe):
    """FakeProbe that raises `error` for every rung."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def __call__(self, rung_name: str, requested: str, precision: str) -> ProbeResult:
        self.calls.append((rung_name, requested, precision))
        raise self.error


@pytest.mark.usefixtures("fake_torch_oom")
def test_auto_non_oom_error_propagates() -> None:
    with pytest.raises(ValueError, match="not memory"):
        choose(gpu_env(), RaisingProbe(ValueError("not memory")))


def test_auto_runtime_error_propagates_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with pytest.raises(RuntimeError, match="plain failure"):
        choose(gpu_env(), RaisingProbe(RuntimeError("plain failure")))


def test_cpu_and_below_threshold_never_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> EnvInfo:
        msg = "detect_environment must not run"
        raise AssertionError(msg)

    probe = FakeProbe()
    monkeypatch.setattr(_backend, "detect_environment", forbidden)
    assert (
        select_backend(device="cpu", n=100, k=10, gpu_threshold=1, gpu_rungs=default_rungs(probe_fn=probe)).name
        == "cpu"
    )
    assert (
        select_backend(device="auto", n=10, k=2, gpu_threshold=10**9, gpu_rungs=default_rungs(probe_fn=probe)).name
        == "cpu"
    )
    assert probe.calls == []


# fork hint


def test_cpu_fit_info_has_no_hint() -> None:
    x = np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    result = _backend.fit_model(
        x,
        k=3,
        spherical=True,
        nepochs=1.0,
        seed=0,
        max_points_per_centroid=None,
        backend=BackendChoice("cpu", "cpu", None),
    )
    assert result.info["backend_hint"] is None


@pytest.mark.parametrize(
    ("is_rocm", "assignment_backend", "emeans_rocm", "expected"),
    [
        (True, "native", False, HINT),
        (False, "native", False, None),
        (True, "triton", False, None),
        (True, "native", True, None),
        (False, "triton", True, None),
    ],
)
def test_backend_hint_table(is_rocm: bool, assignment_backend: str, emeans_rocm: bool, expected: str | None) -> None:
    assert _backend_hint(is_rocm=is_rocm, assignment_backend=assignment_backend, emeans_rocm=emeans_rocm) == expected


def test_fitter_info_line_carries_hint(
    tmp_path: Path,
    make_docs: Callable[..., list[Document]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real = fitter.fit_model

    def with_hint(*args: Any, **kwargs: Any) -> _backend.FitResult:
        result = real(*args, **kwargs)
        result.info["backend_hint"] = HINT
        return result

    monkeypatch.setattr(fitter, "fit_model", with_hint)
    with caplog.at_level(logging.INFO):
        stats = EmeansFitter(model_path=str(tmp_path / "m.npz"), k=3).run(documents=make_docs(40))["stats"]
    infos = [r for r in caplog.records if r.levelname == "INFO" and r.name == fitter.__name__]
    assert len(infos) == 1
    assert infos[0].getMessage().endswith(HINT)
    assert "backend_hint" not in stats


# resolve_precision

TRITON = BackendChoice("nvidia-triton", "Fake GPU", None, requested="triton")
FAIL = ProbeResult(False, None, "RuntimeError: nope")


def test_resolve_precision_fp32_skips_probe() -> None:
    fake = FakeProbe()
    assert resolve_precision(TRITON, "fp32", fake) == ("fp32", None)
    assert fake.calls == []


@pytest.mark.parametrize(
    ("precision", "ok", "expected"),
    [("auto", True, "bf16"), ("auto", False, "fp32"), ("bf16", True, "bf16"), ("fp16", True, "fp16")],
)
def test_resolve_precision_silent_rows(precision: str, ok: bool, expected: str) -> None:
    fake = FakeProbe({} if ok else {"nvidia-triton": FAIL})
    assert resolve_precision(TRITON, precision, fake) == (expected, None)
    wanted = "bf16" if precision == "auto" else precision
    assert fake.calls == [("nvidia-triton", "triton", wanted)]


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_resolve_precision_explicit_failure_warns(precision: str) -> None:
    used, warning = resolve_precision(TRITON, precision, FakeProbe({"nvidia-triton": FAIL}))
    assert used == "fp32"
    assert warning is not None
    assert precision in warning
    assert "nvidia-triton" in warning


def test_resolve_precision_rejects_cpu_choice() -> None:
    with pytest.raises(ValueError, match="GPU backends only"):
        resolve_precision(BackendChoice("cpu", "cpu", None), "bf16", FakeProbe())


def test_resolve_precision_probe_memoised(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def counting(requested: str, precision: str) -> ProbeResult:
        calls.append((requested, precision))
        return OK

    monkeypatch.setattr(_backend, "_run_probe", counting)
    _backend.clear_probe_cache()
    try:
        assert resolve_precision(TRITON, "bf16") == ("bf16", None)
        assert resolve_precision(TRITON, "auto") == ("bf16", None)
    finally:
        _backend.clear_probe_cache()
    assert calls == [("triton", "bf16")]


@pytest.mark.parametrize(("precision", "warns"), [("auto", False), ("bf16", True), ("fp16", True)])
def test_resolve_precision_native_rung_is_fp32(precision: str, warns: bool) -> None:
    fake = FakeProbe()
    used, warning = resolve_precision(BackendChoice("native", "Fake GPU", None, requested="native"), precision, fake)
    assert used == "fp32"
    assert (warning is not None) is warns
    assert fake.calls == []


# GPU memory pre-flight

GIB = 2**30
GPU_CHOICE = BackendChoice("nvidia-triton", "Fake GPU", None, requested="triton")
PREFLIGHT_STATS_KEYS = {
    "k", "d", "n_documents", "n_skipped", "n_train", "spherical", "backend", "assignment_backend", "device_name",
    "precision_used", "fallback_reason", "memory_estimate_bytes", "emeans_source", "faiss_version", "nepochs",
    "batch_size", "alpha0", "n_splits", "imbalance", "train_seconds",
}  # fmt: skip


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def to(self, _device: str) -> "FakeTensor":
        return self

    def half(self) -> "FakeTensor":
        return self

    def detach(self) -> "FakeTensor":
        return self

    def float(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape


class ProbeRows(np.ndarray):
    """Stands in for the probe's random torch tensor."""

    def norm(self, dim: int, keepdim: bool) -> "ProbeRows":
        return np.linalg.norm(self, axis=dim, keepdims=keepdim).view(ProbeRows)

    def half(self) -> "ProbeRows":
        return self


class FakeEngine:
    instances: ClassVar[list["FakeEngine"]] = []
    assignment_backend: ClassVar[str] = "triton"

    def __init__(self, d: int, k: int, **kwargs: Any) -> None:
        self.d, self.k, self.kwargs = d, k, kwargs
        self.trained = False
        self.n_splits = 0
        self.iteration_stats: list[dict[str, Any]] = []
        self.recipe = {
            "assignment_backend": self.assignment_backend,
            "representation": "torch.float32",
            "nepochs": 1.0,
            "batch_size": 64,
            "alpha0": 0.5,
        }
        FakeEngine.instances.append(self)

    def train(self, _x: FakeTensor, init_centroids: object = None) -> tuple[FakeTensor, None]:  # noqa: ARG002
        self.trained = True
        return FakeTensor(np.random.default_rng(0).normal(size=(self.k, self.d)).astype(np.float32)), None


def _find_spec_without_emeans_rocm() -> Callable[..., ModuleSpec | None]:
    real = importlib.util.find_spec

    def fake(name: str, *args: Any) -> ModuleSpec | None:
        if name == "emeans_rocm":
            return None
        return real(name, *args)

    return fake


class StockEngine(FakeEngine):
    """Recorded by the stock faiss.contrib.e_means shadow."""


class RocmEngine(FakeEngine):
    """Recorded by the fake emeans_rocm module."""


class FakeGpu:
    """A fake torch plus a stock faiss.contrib.e_means shadow and, optionally, a fake emeans_rocm, in sys.modules.

    `emeans_rocm` is the installed version string, `None` for an installed package without metadata, or `False`
    when the package is not installed.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        free: int = GIB,
        estimate: Any = GIB,
        emeans_rocm: str | bool | None = "0.2.5",
    ) -> None:
        # The CPU fallback imports e-means submodules lazily; load them before the fake package shadows the real one.
        importlib.import_module("faiss.contrib.e_means.e_means_spherical")
        FakeEngine.instances = []
        self.free = free
        self.estimate_calls: list[tuple[Any, ...]] = []
        self.mem_get_info_calls = 0
        self.from_numpy_calls = 0
        self.float32, self.float16, self.bfloat16 = "float32", "float16", "bfloat16"
        gpu = self

        def mem_get_info() -> tuple[int, int]:
            gpu.mem_get_info_calls += 1
            return gpu.free, 16 * GIB

        def from_numpy(array: np.ndarray) -> FakeTensor:
            gpu.from_numpy_calls += 1
            return FakeTensor(array)

        self.torch = types.SimpleNamespace(
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
            from_numpy=from_numpy,
            Generator=lambda **_kwargs: types.SimpleNamespace(manual_seed=lambda _seed: None),
            randn=lambda rows, cols, **_kwargs: np.ones((rows, cols), dtype=np.float32).view(ProbeRows),
            isfinite=lambda tensor: types.SimpleNamespace(all=lambda: np.isfinite(tensor.array).all()),
            cuda=types.SimpleNamespace(mem_get_info=mem_get_info, synchronize=lambda: None, empty_cache=lambda: None),
            version=types.SimpleNamespace(hip=None, cuda="12.8"),
            OutOfMemoryError=FakeOutOfMemoryError,
            are_deterministic_algorithms_enabled=lambda: False,
        )
        self.e_means = types.SimpleNamespace(QuantizedEmeans=StockEngine, Emeans=StockEngine)
        self.rocm = types.ModuleType("emeans_rocm")
        self.rocm.__spec__ = ModuleSpec("emeans_rocm", None)
        self.rocm.QuantizedEmeans = RocmEngine  # type: ignore[attr-defined]
        self.rocm.Emeans = RocmEngine  # type: ignore[attr-defined]
        if estimate is not None:

            def memory_estimate(*args: Any, **kwargs: Any) -> int:
                gpu.estimate_calls.append((*args, kwargs))
                if isinstance(estimate, Exception):
                    raise estimate
                return int(estimate)

            self.e_means.memory_estimate = memory_estimate
            self.rocm.memory_estimate = memory_estimate  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", self.torch)
        monkeypatch.setitem(sys.modules, "faiss.contrib.e_means", self.e_means)
        if emeans_rocm is not False:
            monkeypatch.setitem(sys.modules, "emeans_rocm", self.rocm)
            monkeypatch.setattr(_backend, "_dist_version", lambda _dist: emeans_rocm)  # type: ignore[arg-type,return-value]
            # The fake module stands for the installed wheel's copy; provenance has its own tests.
            monkeypatch.setattr(_backend, "_emeans_rocm_is_packaged", lambda: emeans_rocm is not None)
        else:
            monkeypatch.delitem(sys.modules, "emeans_rocm", raising=False)
            monkeypatch.setattr(importlib.util, "find_spec", _find_spec_without_emeans_rocm())


def run_fit(
    n: int = 40, d: int = 8, *, allow_cpu_fallback: bool, precision: str = "fp32", mpc: int | None = None
) -> _backend.FitResult:
    x = np.random.default_rng(0).normal(size=(n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return _backend.fit_model(
        x,
        k=3,
        spherical=True,
        nepochs=1.0,
        seed=0,
        max_points_per_centroid=mpc,
        backend=GPU_CHOICE,
        precision=precision,
        allow_cpu_fallback=allow_cpu_fallback,
    )


def test_preflight_auto_over_budget_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gpu = FakeGpu(monkeypatch, free=1 * GIB, estimate=3 * GIB)
    with caplog.at_level(logging.DEBUG):
        result = run_fit(allow_cpu_fallback=True)
    info = result.info
    assert info["backend"] == "cpu"
    assert info["device_name"] == "cpu"
    assert info["precision_used"] == "fp32"
    assert info["memory_estimate_bytes"] == 3 * GIB
    assert "3.00 GiB" in info["fallback_reason"]
    assert "1.00 GiB" in info["fallback_reason"]
    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert warnings[0].getMessage() == info["fallback_reason"]
    assert not any(engine.trained for engine in FakeEngine.instances)
    assert gpu.from_numpy_calls == 0
    assert result.centroids.shape == (3, 8)
    assert result.centroids.dtype == np.float32


def test_preflight_auto_within_budget_runs_on_gpu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gpu = FakeGpu(monkeypatch, free=2 * GIB, estimate=GIB)
    with caplog.at_level(logging.DEBUG):
        info = run_fit(allow_cpu_fallback=True).info
    assert info["backend"] == "gpu-triton"
    assert info["fallback_reason"] is None
    assert info["memory_estimate_bytes"] == GIB
    assert gpu.from_numpy_calls == 1
    assert warnings_in(caplog) == []


def test_preflight_cuda_only_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    gpu = FakeGpu(monkeypatch, free=1 * GIB, estimate=3 * GIB)
    with caplog.at_level(logging.DEBUG):
        info = run_fit(allow_cpu_fallback=False).info
    assert info["backend"] == "gpu-triton"
    assert info["memory_estimate_bytes"] == 3 * GIB
    assert FakeEngine.instances[0].trained
    assert gpu.from_numpy_calls == 1
    assert warnings_in(caplog) == []
    infos = [r for r in caplog.records if r.levelname == "INFO"]
    assert len(infos) == 1
    assert "3.00 GiB" in infos[0].getMessage()


def test_preflight_upstream_without_memory_estimate(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gpu = FakeGpu(monkeypatch, free=1, estimate=None)
    with caplog.at_level(logging.DEBUG):
        info = run_fit(allow_cpu_fallback=True).info
    assert info["backend"] == "gpu-triton"
    assert info["memory_estimate_bytes"] is None
    assert gpu.mem_get_info_calls == 0
    assert warnings_in(caplog) == []


def test_preflight_skipped_when_training_on_a_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = FakeGpu(monkeypatch, free=1, estimate=3 * GIB)
    info = run_fit(n=40, allow_cpu_fallback=True, mpc=2).info  # 2 points per centroid * k=3 < 40
    assert gpu.estimate_calls == []
    assert info["memory_estimate_bytes"] is None
    assert info["backend"] == "gpu-triton"


def test_preflight_no_subsampling_still_estimates(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = FakeGpu(monkeypatch, free=2 * GIB, estimate=GIB)
    info = run_fit(n=40, allow_cpu_fallback=True, mpc=1000).info
    assert len(gpu.estimate_calls) == 1
    assert info["memory_estimate_bytes"] == GIB


def test_preflight_estimate_error_is_ignored(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    FakeGpu(monkeypatch, free=1, estimate=RuntimeError("unsupported representation"))
    with caplog.at_level(logging.DEBUG):
        info = run_fit(allow_cpu_fallback=True).info
    assert info["backend"] == "gpu-triton"
    assert info["memory_estimate_bytes"] is None
    assert warnings_in(caplog) == []


def test_preflight_estimate_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = FakeGpu(monkeypatch, free=2 * GIB, estimate=GIB)
    run_fit(n=40, allow_cpu_fallback=True)
    model, n, dtype, device, kwargs = gpu.estimate_calls[0]
    assert model is FakeEngine.instances[0]
    assert (n, dtype, device, kwargs) == (40, "float32", "cuda", {})
    monkeypatch.setattr(_backend, "probe", lambda *_a, **_k: ProbeResult(True, "triton", None))
    run_fit(n=40, allow_cpu_fallback=True, precision="fp16")
    assert gpu.estimate_calls[1][2] == "float16"


def test_preflight_fitter_reports_cpu_fallback(
    tmp_path: Path, make_docs: Callable[..., list[Document]], monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeGpu(monkeypatch, free=1 * GIB, estimate=3 * GIB)
    monkeypatch.setattr(fitter, "select_backend", lambda **_kwargs: GPU_CHOICE)
    path = tmp_path / "m.npz"
    stats = EmeansFitter(device="auto", k=3, model_path=str(path)).run(documents=make_docs(40))["stats"]
    assert stats["backend"] == "cpu"
    assert "GiB" in stats["fallback_reason"]
    assert stats["memory_estimate_bytes"] == 3 * GIB
    assert set(stats) == PREFLIGHT_STATS_KEYS
    assert path.exists()
    assert load_model(str(path)).centroids.shape == (3, 16)


def test_cpu_fit_info_memory_and_fallback_keys() -> None:
    x = np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    info = _backend.fit_model(
        x,
        k=3,
        spherical=True,
        nepochs=1.0,
        seed=0,
        max_points_per_centroid=None,
        backend=BackendChoice("cpu", "cpu", "some reason"),
    ).info
    assert info["memory_estimate_bytes"] is None
    assert info["fallback_reason"] == "some reason"


# engine selection: emeans_rocm on any GPU when installed, stock faiss.contrib.e_means otherwise
def engines_built() -> set[type]:
    return {type(engine) for engine in FakeEngine.instances}


@pytest.mark.parametrize("is_rocm", [True, False], ids=["amd", "nvidia"])
@pytest.mark.parametrize(("installed", "engine"), [("0.2.5", RocmEngine), (False, StockEngine)])
def test_all_gpu_sites_use_the_selected_engine(
    monkeypatch: pytest.MonkeyPatch, is_rocm: bool, installed: str | bool, engine: type[FakeEngine]
) -> None:
    gpu = FakeGpu(monkeypatch, emeans_rocm=installed)
    gpu.torch.version = types.SimpleNamespace(hip="7.16" if is_rocm else None, cuda=None if is_rocm else "12.8")

    assert _backend._run_probe("auto", "fp32").ok
    assert engines_built() == {engine}

    FakeEngine.instances = []
    run_fit(allow_cpu_fallback=False)
    assert engines_built() == {engine}

    FakeEngine.instances = []
    _backend.assigner_for(GPU_CHOICE, np.eye(3, 8, dtype=np.float32), spherical=True)
    assert engines_built() == {engine}


@pytest.mark.parametrize("is_rocm", [True, False], ids=["amd", "nvidia"])
def test_probe_reduced_precision_uses_emeans_class_from_emeans_rocm(
    monkeypatch: pytest.MonkeyPatch, is_rocm: bool
) -> None:
    gpu = FakeGpu(monkeypatch)
    gpu.torch.version = types.SimpleNamespace(hip="7.16" if is_rocm else None, cuda=None)
    assert _backend._run_probe("auto", "bf16").ok
    assert engines_built() == {RocmEngine}


class BrokenEmeansRocmFinder:
    """Makes `import emeans_rocm` raise ImportError, as a partially uninstalled package or a mismatched faiss does."""

    def find_spec(self, name: str, *_args: Any) -> ModuleSpec | None:
        if name == "emeans_rocm":
            msg = "cannot import name 'NumpyEmeans' from partially initialized module"
            raise ImportError(msg)
        return None


def _break_emeans_rocm_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "emeans_rocm")
    monkeypatch.setattr(sys, "meta_path", [BrokenEmeansRocmFinder(), *sys.meta_path])
    monkeypatch.setattr(_backend, "emeans_rocm_importable", lambda: True)  # found on the path, fails to import


AMD_EMEANS_ROCM = {"is_rocm": True, "emeans_rocm": True, "hip_version": "7.16"}


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_broken_emeans_rocm_import_falls_back_to_the_stock_gpu_engine(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, device: str
) -> None:
    FakeGpu(monkeypatch)
    _break_emeans_rocm_import(monkeypatch)
    env = gpu_env(**AMD_EMEANS_ROCM)
    with caplog.at_level(logging.DEBUG):
        choice = select_backend(device=device, n=100, k=10, gpu_threshold=1, env=env)
    assert (choice.name, choice.requested, choice.engine) == ("native-stock", "native", _backend.STOCK_ENGINE)
    reason = choice.fallback_reason
    assert reason is not None
    for fragment in ("rocm-triton: ImportError", "native: ImportError", "stock faiss e-means engine"):
        assert fragment in reason
    assert engines_built() == {StockEngine}  # the stock probe ran; nothing was built from emeans_rocm
    assert [w.getMessage() for w in warnings_in(caplog)] == [reason]
    # A broken install is not transient: it is remembered, so the next selection does not probe again.
    assert not _backend._is_transient_failure(reason)

    FakeEngine.instances = []
    monkeypatch.setattr(FakeEngine, "assignment_backend", "native")
    x = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    info = _backend.fit_model(
        x, k=3, spherical=True, nepochs=1.0, seed=0, max_points_per_centroid=None, backend=choice, precision="fp32",
    ).info  # fmt: skip
    assert engines_built() == {StockEngine}
    assert info["backend"] == "gpu-pytorch"
    assert info["emeans_source"] == f"faiss-cpu {faiss.__version__}"
    assert info["fallback_reason"] == reason

    FakeEngine.instances = []
    _backend.assigner_for(choice, np.eye(3, 8, dtype=np.float32), spherical=True)
    assert engines_built() == {StockEngine}


class BrokenEngine(FakeEngine):
    def train(self, _x: FakeTensor, init_centroids: object = None) -> tuple[FakeTensor, None]:  # noqa: ARG002
        msg = "no kernel image is available"
        raise RuntimeError(msg)


def test_emeans_rocm_and_stock_engine_both_failing_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gpu = FakeGpu(monkeypatch)
    gpu.rocm.QuantizedEmeans = BrokenEngine  # type: ignore[attr-defined]
    gpu.e_means.QuantizedEmeans = BrokenEngine
    env = gpu_env(**AMD_EMEANS_ROCM)
    with caplog.at_level(logging.DEBUG):
        choice = select_backend(device="auto", n=100, k=10, gpu_threshold=1, env=env)
    assert choice.name == "cpu"
    reason = choice.fallback_reason
    assert reason is not None
    for rung in ("rocm-triton", "native", "native-stock"):
        assert f"{rung}: RuntimeError: no kernel image is available" in reason
    assert [w.getMessage() for w in warnings_in(caplog)] == [reason]
    with pytest.raises(RuntimeError, match="native-stock: RuntimeError"):
        select_backend(device="cuda", n=100, k=10, gpu_threshold=1, env=env)


def test_failed_engine_memory_is_released_before_the_stock_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = FakeGpu(monkeypatch)
    events: list[str] = []

    class RecordingBroken(BrokenEngine):
        def __init__(self, d: int, k: int, **kwargs: Any) -> None:
            events.append("emeans-rocm built")
            super().__init__(d, k, **kwargs)

    class RecordingStock(StockEngine):
        def __init__(self, d: int, k: int, **kwargs: Any) -> None:
            events.append("stock built")
            super().__init__(d, k, **kwargs)

    gpu.rocm.QuantizedEmeans = RecordingBroken  # type: ignore[attr-defined]
    gpu.e_means.QuantizedEmeans = RecordingStock
    gpu.torch.cuda.empty_cache = lambda: events.append("empty_cache")
    monkeypatch.setattr(_backend.gc, "collect", lambda: events.append("gc") or 0)
    choice = select_backend(device="auto", n=100, k=10, gpu_threshold=1, env=gpu_env(**AMD_EMEANS_ROCM))
    assert choice.name == "native-stock"
    stock = events.index("stock built")
    assert events[stock - 2 : stock] == ["gc", "empty_cache"]  # released after the last failed emeans-rocm probe
    assert events.count("emeans-rocm built") == 2  # rocm-triton and native


def test_stock_rung_is_tried_after_every_emeans_rocm_rung_fails() -> None:
    probe = FakeProbe({"rocm-triton": fail("t"), "native": fail("n")})
    choice = choose(gpu_env(**AMD_EMEANS_ROCM), probe)
    assert (choice.name, choice.engine) == ("native-stock", _backend.STOCK_ENGINE)
    assert [call[0] for call in probe.calls] == ["rocm-triton", "native", "native-stock"]
    assert choice.fallback_reason is not None
    assert "rocm-triton: t" in choice.fallback_reason
    assert "native: n" in choice.fallback_reason


def test_stock_rung_is_skipped_without_emeans_rocm() -> None:
    probe = FakeProbe({"nvidia-auto": fail("a"), "nvidia-triton": fail("b"), "native": fail("c")})
    choice = choose(gpu_env(), probe)
    assert choice.name == "cpu"
    assert [call[0] for call in probe.calls] == ["nvidia-auto", "nvidia-triton", "native"]
    assert choice.fallback_reason is not None
    assert "native-stock" not in choice.fallback_reason


def test_emeans_rocm_rung_success_carries_no_reason() -> None:
    choice = choose(gpu_env(**AMD_EMEANS_ROCM), FakeProbe())
    assert (choice.name, choice.engine, choice.fallback_reason) == ("rocm-triton", None, None)


def test_nvidia_with_emeans_rocm_picks_nvidia_auto_rung() -> None:
    probe = FakeProbe()
    choice = choose(gpu_env(is_rocm=False, emeans_rocm=True), probe)
    assert (choice.name, choice.requested) == ("nvidia-auto", "auto")


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ("0.2.5", "emeans-rocm 0.2.5"),
        (None, "emeans-rocm (unversioned source checkout)"),
        (False, f"faiss-cpu {faiss.__version__}"),
    ],
)
def test_gpu_fit_reports_engine_label(
    monkeypatch: pytest.MonkeyPatch, installed: str | bool | None, expected: str
) -> None:
    FakeGpu(monkeypatch, emeans_rocm=installed)
    assert run_fit(allow_cpu_fallback=False).info["emeans_source"] == expected


def test_model_metadata_carries_emeans_rocm_label(
    tmp_path: Path, make_docs: Callable[..., list[Document]], monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeGpu(monkeypatch)
    monkeypatch.setattr(fitter, "select_backend", lambda **_kwargs: GPU_CHOICE)
    path = tmp_path / "m.npz"
    stats = EmeansFitter(device="auto", k=3, spherical=False, model_path=str(path)).run(documents=make_docs(40))[
        "stats"
    ]
    assert stats["emeans_source"] == "emeans-rocm 0.2.5"
    assert load_model(str(path)).metadata["emeans_source"] == "emeans-rocm 0.2.5"


@pytest.mark.parametrize(("installed", "hint"), [("0.2.5", None), ("0.3.0", HINT), (False, HINT)])
def test_native_rung_hint_depends_on_emeans_rocm(
    monkeypatch: pytest.MonkeyPatch, installed: str | bool, hint: str | None
) -> None:
    gpu = FakeGpu(monkeypatch, emeans_rocm=installed)
    gpu.torch.version = types.SimpleNamespace(hip="7.16", cuda=None)
    monkeypatch.setattr(FakeEngine, "assignment_backend", "native")
    native = BackendChoice("native", "Fake GPU", None, requested="native")
    x = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    info = _backend.fit_model(
        x, k=3, spherical=True, nepochs=1.0, seed=0, max_points_per_centroid=None, backend=native, precision="fp32",
        allow_cpu_fallback=False,
    ).info  # fmt: skip
    assert info["backend_hint"] == hint


# supported emeans-rocm range: outside it the stock engine runs and a warning names the version


@pytest.mark.parametrize("version", ["0.2.5", "0.2.9", "0.2.6.dev1"])
def test_emeans_rocm_inside_supported_range_is_used(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, version: str
) -> None:
    FakeGpu(monkeypatch, emeans_rocm=version)
    with caplog.at_level(logging.DEBUG):
        assert _backend.emeans_rocm_unsupported_reason() is None
        assert _backend.emeans_rocm_enabled()
        info = run_fit(allow_cpu_fallback=False).info
    assert engines_built() == {RocmEngine}
    assert info["emeans_source"] == f"emeans-rocm {version}"
    assert warnings_in(caplog) == []


@pytest.mark.parametrize("version", ["0.2.4", "0.3.0", "0.3.0rc1", "1.0", "not-a-version"])
def test_emeans_rocm_outside_supported_range_uses_stock_engine(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, version: str
) -> None:
    FakeGpu(monkeypatch, emeans_rocm=version)
    with caplog.at_level(logging.DEBUG):
        reason = _backend.emeans_rocm_unsupported_reason()
        assert reason is not None
        assert f"emeans-rocm {version}" in reason
        assert ">=0.2.5,<0.3" in reason
        assert not _backend.emeans_rocm_enabled()
        info = run_fit(allow_cpu_fallback=False).info
        assert _backend._run_probe("auto", "fp32").ok
    assert engines_built() == {StockEngine}
    assert info["emeans_source"] == f"faiss-cpu {faiss.__version__}"
    warnings = warnings_in(caplog)
    assert [w.getMessage() for w in warnings] == [reason]  # warned once per process, not on every engine lookup


def test_rocm_triton_rung_names_the_unsupported_version(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeGpu(monkeypatch, emeans_rocm="0.3.0")
    env = gpu_env(is_rocm=True, emeans_rocm=_backend.emeans_rocm_enabled(), hip_version="7.16")
    choice = select_backend(device="auto", n=100, k=10, gpu_threshold=1, env=env, gpu_rungs=default_rungs(FakeProbe()))
    assert choice.name == "native"
    reason = _backend._needs_emeans_rocm(env)
    assert reason is not None
    assert "emeans-rocm 0.3.0 is outside the supported range" in reason


@pytest.mark.parametrize("missing", ["assignment_backend", "representation", "nepochs", "batch_size", "alpha0"])
def test_missing_recipe_key_does_not_discard_the_fit(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    FakeGpu(monkeypatch)
    real_train = FakeEngine.train

    def train(self: FakeEngine, x: FakeTensor, init_centroids: object = None) -> tuple[FakeTensor, None]:
        del self.recipe[missing]
        return real_train(self, x, init_centroids)

    monkeypatch.setattr(FakeEngine, "train", train)
    result = run_fit(allow_cpu_fallback=False)
    assert result.centroids.shape == (3, 8)
    info = result.info
    assert info["emeans_source"] == "emeans-rocm 0.2.5"
    assert info["precision_used"] == "fp32"
    assert info["nepochs"] == 1.0
    assert missing not in info["recipe"]


def test_missing_engine_attributes_do_not_discard_the_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeGpu(monkeypatch)
    real_train = FakeEngine.train

    def train(self: FakeEngine, x: FakeTensor, init_centroids: object = None) -> tuple[FakeTensor, None]:
        out = real_train(self, x, init_centroids)
        del self.recipe, self.n_splits, self.iteration_stats
        return out

    monkeypatch.setattr(FakeEngine, "train", train)
    info = run_fit(allow_cpu_fallback=False).info
    assert info["assignment_backend"] == GPU_CHOICE.requested  # the requested backend when unreported
    assert info["batch_size"] is None
    assert info["alpha0"] is None
    assert info["n_splits"] is None
    assert info["recipe"] == {}


# engine provenance: the label names the code that runs, not only the installed metadata (IN-02)


class FakeDist:
    """Stands in for the installed emeans-rocm distribution, rooted at `site`."""

    def __init__(self, site: Path, files: list[str]) -> None:
        self.site = site
        self.files = [Path(f) for f in files]

    def locate_file(self, entry: Path) -> Path:
        return self.site / entry


def _provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, module_at: str, dist: FakeDist | None, version: str | None
) -> None:
    """Make `emeans_rocm` resolve to `tmp_path/module_at/emeans_rocm/__init__.py`, next to the given distribution."""
    origin = tmp_path / module_at / "emeans_rocm" / "__init__.py"
    origin.parent.mkdir(parents=True, exist_ok=True)
    origin.touch()
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: Any) -> ModuleSpec | None:
        if name == "emeans_rocm":
            return ModuleSpec("emeans_rocm", None, origin=str(origin))
        return real_find_spec(name, *args)

    def distribution(name: str) -> FakeDist:
        if dist is None or name != "emeans-rocm":
            raise importlib.metadata.PackageNotFoundError(name)
        return dist

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(_backend, "_dist_version", lambda _dist: version)


def test_installed_wheel_copy_is_labelled_with_its_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dist = FakeDist(tmp_path / "site", ["emeans_rocm/__init__.py", "emeans_rocm/e_means.py"])
    _provenance(monkeypatch, tmp_path, module_at="site", dist=dist, version="0.2.5")
    assert _backend._emeans_rocm_is_packaged()
    assert _backend.detect_emeans_source() == "emeans-rocm 0.2.5"


def test_checkout_ahead_of_the_wheel_is_labelled_as_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dist = FakeDist(tmp_path / "site", ["emeans_rocm/__init__.py"])
    _provenance(monkeypatch, tmp_path, module_at="checkout", dist=dist, version="0.2.5")
    assert not _backend._emeans_rocm_is_packaged()
    label = _backend.detect_emeans_source()
    assert label == "emeans-rocm 0.2.5 (source checkout)"
    assert str(tmp_path) not in label


def test_editable_install_without_module_files_is_labelled_as_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dist = FakeDist(tmp_path / "site", ["emeans_rocm-0.2.5.dist-info/METADATA", "_emeans_rocm.pth"])
    _provenance(monkeypatch, tmp_path, module_at="checkout", dist=dist, version="0.2.5")
    assert _backend.detect_emeans_source() == "emeans-rocm 0.2.5 (source checkout)"


def test_checkout_without_any_distribution_is_unversioned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _provenance(monkeypatch, tmp_path, module_at="checkout", dist=None, version=None)
    assert not _backend._emeans_rocm_is_packaged()
    assert _backend.detect_emeans_source() == "emeans-rocm (unversioned source checkout)"


def test_checkout_is_not_range_checked_against_the_wheel_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The installed metadata does not describe a checkout that runs instead, so its version is not range checked.
    dist = FakeDist(tmp_path / "site", ["emeans_rocm/__init__.py"])
    _provenance(monkeypatch, tmp_path, module_at="checkout", dist=dist, version="0.3.0")
    assert _backend.emeans_rocm_unsupported_reason() is None
    assert _backend.detect_emeans_source() == "emeans-rocm 0.3.0 (source checkout)"


def test_wheel_copy_outside_the_range_is_still_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dist = FakeDist(tmp_path / "site", ["emeans_rocm/__init__.py"])
    _provenance(monkeypatch, tmp_path, module_at="site", dist=dist, version="0.3.0")
    assert _backend.emeans_rocm_unsupported_reason() is not None
    assert _backend.detect_emeans_source() == f"faiss-cpu {faiss.__version__}"
