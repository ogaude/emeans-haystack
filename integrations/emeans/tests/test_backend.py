# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging
import os
import sys
import types
from importlib.machinery import ModuleSpec

import faiss
import numpy as np
import pytest

from haystack_integrations.components.preprocessors.emeans import _backend
from haystack_integrations.components.preprocessors.emeans._backend import (
    BackendChoice,
    EnvInfo,
    detect_emeans_source,
    detect_environment,
    emeans_rocm_importable,
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
    emeans_rocm=False,
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


def _patch_find_spec(monkeypatch: pytest.MonkeyPatch, *, emeans_rocm: ModuleSpec | None) -> None:
    real = importlib.util.find_spec

    def fake(name: str, *args: object) -> ModuleSpec | None:
        if name == "emeans_rocm":
            return emeans_rocm
        return real(name, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_emeans_rocm_importable_when_spec_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_find_spec(monkeypatch, emeans_rocm=ModuleSpec("emeans_rocm", None))
    assert emeans_rocm_importable() is True


def test_emeans_rocm_not_importable_when_spec_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_find_spec(monkeypatch, emeans_rocm=None)
    assert emeans_rocm_importable() is False


@pytest.mark.parametrize("error", [ValueError, ImportError])
def test_emeans_rocm_importable_find_spec_errors(monkeypatch: pytest.MonkeyPatch, error: type[Exception]) -> None:
    def boom(*_args: object) -> None:
        raise error

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert emeans_rocm_importable() is False
    assert detect_emeans_source() == f"faiss-cpu {faiss.__version__}"


def test_emeans_source_names_package_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_find_spec(monkeypatch, emeans_rocm=ModuleSpec("emeans_rocm", None))
    monkeypatch.setattr(_backend, "_dist_version", lambda _dist: "0.2.5")
    monkeypatch.setattr(_backend, "_emeans_rocm_is_packaged", lambda: True)  # the installed wheel's copy
    assert detect_emeans_source() == "emeans-rocm 0.2.5"
    assert detect_environment().emeans_rocm is True


def test_emeans_source_without_distribution_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_find_spec(monkeypatch, emeans_rocm=ModuleSpec("emeans_rocm", None))
    monkeypatch.setattr(_backend, "_dist_version", lambda _dist: None)
    assert detect_emeans_source() == "emeans-rocm (unversioned source checkout)"


def test_emeans_source_falls_back_to_faiss_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_find_spec(monkeypatch, emeans_rocm=None)
    assert detect_emeans_source() == f"faiss-cpu {faiss.__version__}"
    assert detect_environment().emeans_rocm is False


def test_dist_version_missing_distribution_is_none() -> None:
    assert _backend._dist_version("emeans-haystack-no-such-distribution") is None


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


def _pretend_torch_installed(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    real_find = importlib.util.find_spec
    real_import = importlib.import_module

    def find(name: str, *args: object) -> ModuleSpec | None:
        if name == "torch":
            return ModuleSpec(name, None)
        return real_find(name, *args)  # type: ignore[arg-type]

    def load(name: str, *args: object) -> object:
        if name == "torch":
            if isinstance(module, BaseException):
                raise module
            return module
        return real_import(name, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", find)
    monkeypatch.setattr(importlib, "import_module", load)


@pytest.mark.parametrize(
    "error",
    [
        OSError("libamdhip64.so: cannot open"),
        RuntimeError("bad build"),
        ImportError("libamdhip64.so: cannot open shared object file"),
        ModuleNotFoundError("No module named 'sympy'"),
    ],
)
def test_detect_environment_broken_torch_import(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    _pretend_torch_installed(monkeypatch, error)
    env = detect_environment()
    assert env.torch_importable is False
    assert env.gpu_available is False
    assert env.mismatch_reason is not None
    assert type(error).__name__ in env.mismatch_reason
    with pytest.raises(RuntimeError, match="could not be imported"):
        select_backend(device="cuda", n=10, k=2, gpu_threshold=1, env=env)


def test_detect_environment_gpu_init_error_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_index: int) -> str:
        msg = "HIP error: invalid device ordinal"
        raise RuntimeError(msg)

    fake_torch = types.SimpleNamespace(
        __version__="2.14.0+rocm10.1.0",
        version=types.SimpleNamespace(hip="10.1", cuda=None),
        cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=boom),
    )
    _pretend_torch_installed(monkeypatch, fake_torch)
    env = detect_environment()
    assert env.torch_importable is True
    assert env.gpu_available is False
    assert env.mismatch_reason is not None
    assert "could not initialise the GPU" in env.mismatch_reason
    with caplog.at_level(logging.DEBUG):
        choice = select_backend(device="auto", n=10, k=2, gpu_threshold=1, env=env)
    assert choice.name == "cpu"
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def _hidden_rocm_torch() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        __version__="2.14.0+rocm10.1.0",
        version=types.SimpleNamespace(hip="10.1", cuda=None),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )


def test_detect_environment_present_gpu_hidden_from_matching_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    _pretend_torch_installed(monkeypatch, _hidden_rocm_torch())
    monkeypatch.setattr(_backend, "_device_nodes", lambda: (True, False))
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    env = detect_environment()
    assert env.mismatch_reason is not None
    assert "ROCm build" in env.mismatch_reason
    assert "sees no GPU" in env.mismatch_reason
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    env = detect_environment()
    assert env.mismatch_reason is not None
    assert "ROCR_VISIBLE_DEVICES=''" in env.mismatch_reason


def test_detect_environment_no_device_node_stays_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    _pretend_torch_installed(monkeypatch, _hidden_rocm_torch())
    monkeypatch.setattr(_backend, "_device_nodes", lambda: (False, False))
    assert detect_environment().mismatch_reason is None


def test_cpu_fit_label_ignores_emeans_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_backend, "emeans_rocm_importable", lambda: True)
    assert fit().info["emeans_source"] == f"faiss-cpu {faiss.__version__}"


def test_fit_model_info_plain_types() -> None:
    info = fit().info
    assert info["backend"] == "cpu"
    assert info["device_name"] == "cpu"
    assert info["precision_used"] == "fp32"
    assert info["assignment_backend"] == "faiss"
    assert info["emeans_source"] == f"faiss-cpu {faiss.__version__}"
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
        "unused_threshold": float,
    }.items():
        assert type(info[key]) is expected, key


@pytest.mark.parametrize(("d", "expected"), [(1, 0.0), (2, 0.0), (3, 0.1), (768, 0.1)])
def test_unused_threshold_helper(d: int, expected: float) -> None:
    assert _backend._unused_threshold(d) == expected
    assert type(_backend._unused_threshold(d)) is float


@pytest.mark.parametrize("spherical", [True, False])
@pytest.mark.parametrize(("d", "expected"), [(2, 0.0), (3, 0.1)])
def test_cpu_constructors_get_the_unused_threshold(
    monkeypatch: pytest.MonkeyPatch, d: int, expected: float, spherical: bool
) -> None:
    seen: list[dict[str, object]] = []

    class Spy(_backend.NumpyEmeans):  # type: ignore[misc,name-defined]
        def __init__(self, *args: object, **kwargs: object) -> None:
            seen.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(_backend, "NumpyEmeans", Spy)
    x = np.random.default_rng(0).normal(size=(40, d)).astype(np.float32)
    if spherical:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    result = fit_model(x, k=3, spherical=spherical, nepochs=1.0, seed=0, max_points_per_centroid=None, backend=CPU)
    _backend.cpu_assigner(np.asarray(result.centroids, dtype=np.float32), spherical=spherical)
    assert len(seen) == 2
    assert all(kwargs["unused_threshold"] == expected for kwargs in seen)
    assert result.info["unused_threshold"] == expected


@pytest.mark.parametrize("stats", [[], [{}]])
def test_fit_model_imbalance_nan_without_iteration_stats(monkeypatch: pytest.MonkeyPatch, stats: list) -> None:
    class NoStats(_backend.NumpyEmeans):  # type: ignore[misc,name-defined]
        def train(self, x: np.ndarray, init_centroids: np.ndarray | None = None) -> object:
            result = super().train(x, init_centroids=init_centroids)
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


class _FakeDeterministicTorch:
    def __init__(self, hip: str | None) -> None:
        self.version = types.SimpleNamespace(hip=hip)
        self.enabled = False
        self.calls: list[tuple[bool, bool]] = []

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.enabled

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return False

    def use_deterministic_algorithms(self, mode: bool, *, warn_only: bool = False) -> None:
        self.enabled = mode
        self.calls.append((mode, warn_only))


@pytest.mark.parametrize("hip", [None, "10.1"])
def test_deterministic_restores_mode_and_env(monkeypatch: pytest.MonkeyPatch, hip: str | None) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = _FakeDeterministicTorch(hip)
    with pytest.raises(KeyError), _backend._deterministic(fake, enabled=True):
        assert fake.enabled is True
        assert ("CUBLAS_WORKSPACE_CONFIG" in os.environ) == (hip is None)
        raise KeyError
    assert fake.enabled is False
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_deterministic_keeps_user_cublas_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with _backend._deterministic(_FakeDeterministicTorch(None), enabled=True):
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def _direct_engine_centroids(init: np.ndarray | None) -> np.ndarray:
    em = _backend.NumpyEmeans(  # type: ignore[attr-defined]
        8, 4, nepochs=2.0, seed=1, spherical=True, max_points_per_centroid=None, capture_assignments=False
    )
    centroids, _ = em.train(unit_rows(), init_centroids=init)
    return np.asarray(centroids)


def test_fit_init_centroids_matches_direct_engine() -> None:
    init = unit_rows()[:4].copy()
    np.testing.assert_array_equal(fit(init_centroids=init).centroids, _direct_engine_centroids(init))


def test_fit_init_centroids_changes_result() -> None:
    x = unit_rows()
    first = fit(init_centroids=x[:4].copy()).centroids
    second = fit(init_centroids=x[10:14].copy()).centroids
    assert not np.array_equal(first, second)


def test_fit_default_init_unchanged() -> None:
    default = fit().centroids
    np.testing.assert_array_equal(default, fit(init_centroids=None).centroids)
    np.testing.assert_array_equal(default, _direct_engine_centroids(None))
