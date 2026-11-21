# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging
import sys
from importlib.machinery import ModuleSpec

import numpy as np
import pytest

from haystack_integrations.components.preprocessors.emeans import _backend
from haystack_integrations.components.preprocessors.emeans._backend import (
    BackendChoice,
    EnvInfo,
    detect_emeans_source,
    detect_environment,
    fit_model,
    select_backend,
)

CPU = BackendChoice("cpu", "cpu", None)


def unit_rows(n: int = 100, d: int = 8, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=(n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def warnings_in(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelname == "WARNING"]


def fit(**kwargs: object) -> _backend.FitResult:
    params: dict[str, object] = {
        "k": 4,
        "spherical": True,
        "nepochs": 2.0,
        "seed": 1,
        "max_points_per_centroid": None,
        "backend": CPU,
    }
    params.update(kwargs)
    return fit_model(unit_rows(), **params)  # type: ignore[arg-type]


def test_select_cpu() -> None:
    assert select_backend(device="cpu", n=10, k=2, gpu_threshold=1) == BackendChoice("cpu", "cpu", None)


def test_select_auto_below_threshold(caplog: pytest.LogCaptureFixture) -> None:
    choice = select_backend(device="auto", n=10, k=2, gpu_threshold=100)
    assert choice.name == "cpu"
    assert choice.fallback_reason is None
    assert not caplog.records


def test_select_auto_at_threshold_falls_back_silently(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        choice = select_backend(device="auto", n=10, k=2, gpu_threshold=20)
    assert choice.fallback_reason == "no GPU backend usable: PyTorch is not installed"
    assert warnings_in(caplog) == []


def test_select_cuda_raises() -> None:
    with pytest.raises(RuntimeError, match="no GPU backend usable"):
        select_backend(device="cuda", n=10, k=2, gpu_threshold=1)


def test_select_unknown_device() -> None:
    with pytest.raises(ValueError, match="tpu"):
        select_backend(device="tpu", n=10, k=2, gpu_threshold=1)


GPU_ENV = EnvInfo(
    torch_importable=True,
    gpu_available=True,
    is_rocm=False,
    fork_detected=False,
    device_name="Fake GPU",
    torch_version="0",
    hip_version=None,
)


def test_select_empty_gpu_rungs_never_picks_gpu() -> None:
    choice = select_backend(device="auto", n=10, k=2, gpu_threshold=1, env=GPU_ENV, gpu_rungs=())
    assert choice.name == "cpu"
    assert choice.fallback_reason is not None
    assert choice.fallback_reason.endswith("no GPU backend registered")
    with pytest.raises(RuntimeError, match="no GPU backend usable"):
        select_backend(device="cuda", n=10, k=2, gpu_threshold=1, env=GPU_ENV, gpu_rungs=())


def test_select_uses_usable_rung_and_reports_reasons() -> None:
    class Rung:
        def __init__(self, name: str, reason: str | None) -> None:
            self.name = name
            self.requested = "native"
            self._reason = reason

        def usable(self, env: EnvInfo) -> str | None:  # noqa: ARG002
            return self._reason

    ok = select_backend(
        device="cuda", n=10, k=2, gpu_threshold=1, env=GPU_ENV, gpu_rungs=[Rung("a", "bad"), Rung("b", None)]
    )
    assert ok == BackendChoice("b", "Fake GPU", None, requested="native")
    with pytest.raises(RuntimeError, match="a: bad"):
        select_backend(device="cuda", n=10, k=2, gpu_threshold=1, env=GPU_ENV, gpu_rungs=[Rung("a", "bad")])
    with pytest.raises(RuntimeError, match="a: bad; c: worse"):
        select_backend(
            device="cuda", n=10, k=2, gpu_threshold=1, env=GPU_ENV, gpu_rungs=[Rung("a", "bad"), Rung("c", "worse")]
        )


def test_emeans_source_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec

    def fake(name: str, *args: object) -> ModuleSpec | None:
        if name == "faiss.contrib.e_means.device_profile":
            return None
        return real(name, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    assert detect_emeans_source() == "faiss-cpu"


def test_emeans_source_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec

    def fake(name: str, *args: object) -> ModuleSpec | None:
        if name == "faiss.contrib.e_means.device_profile":
            return ModuleSpec(name, None)
        return real(name, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    assert detect_emeans_source() == "e-means-rocm"
    assert detect_environment().fork_detected is True


@pytest.mark.parametrize("error", [ValueError, ImportError])
def test_emeans_source_find_spec_errors(monkeypatch: pytest.MonkeyPatch, error: type[Exception]) -> None:
    def boom(*_args: object) -> None:
        raise error

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert detect_emeans_source() == "faiss-cpu"


@pytest.mark.parametrize("outcome", [None, ImportError, ValueError])
def test_detect_environment_without_torch(monkeypatch: pytest.MonkeyPatch, outcome: type[Exception] | None) -> None:
    real = importlib.util.find_spec
    imported_before = "torch" in sys.modules

    def fake(name: str, *args: object) -> ModuleSpec | None:
        if name == "torch":
            if outcome is not None:
                raise outcome
            return None
        return real(name, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    env = detect_environment()
    assert env.torch_importable is False
    assert env.gpu_available is False
    assert ("torch" in sys.modules) == imported_before


def test_fit_model_info_plain_types() -> None:
    info = fit().info
    assert info["backend"] == "cpu"
    assert info["device_name"] == "cpu"
    assert info["precision_used"] == "fp32"
    assert info["assignment_backend"] == "faiss"
    assert info["emeans_source"] == detect_emeans_source()
    assert isinstance(info["recipe"], dict)
    assert isinstance(info["faiss_version"], str)
    for key, expected in {
        "nepochs": float,
        "batch_size": int,
        "alpha0": float,
        "n_splits": int,
        "imbalance": float,
        "train_seconds": float,
        "n_train": int,
    }.items():
        assert type(info[key]) is expected, key


@pytest.mark.parametrize("stats", [[], [{}]])
def test_fit_model_imbalance_nan_without_iteration_stats(monkeypatch: pytest.MonkeyPatch, stats: list) -> None:
    class NoStats(_backend.NumpyEmeans):  # type: ignore[misc,name-defined]
        def train(self, x: np.ndarray) -> object:
            result = super().train(x)
            self.iteration_stats = list(stats)
            return result

    monkeypatch.setattr(_backend, "NumpyEmeans", NoStats)
    assert np.isnan(fit().info["imbalance"])


def test_fit_model_n_train_subsample() -> None:
    assert fit(max_points_per_centroid=5).info["n_train"] == 20
    assert fit().info["n_train"] == 100


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_fit_unsupported_precision_warns_once(precision: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        info = fit(precision=precision).info
    assert info["precision_used"] == "fp32"
    assert len(warnings_in(caplog)) == 1


@pytest.mark.parametrize("precision", ["auto", "fp32"])
def test_fit_supported_precision_no_warning(precision: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        fit(precision=precision)
    assert warnings_in(caplog) == []


def test_fit_deterministic_cpu(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        a = fit(deterministic=True).centroids
        b = fit(deterministic=True).centroids
    assert warnings_in(caplog) == []
    np.testing.assert_array_equal(a, b)
