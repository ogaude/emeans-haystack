# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end fits on a real GPU (local only; needs PyTorch with a visible GPU)."""

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from haystack import Document

from haystack_integrations.components.preprocessors.emeans import EmeansFitter, _backend
from haystack_integrations.components.preprocessors.emeans._backend import (
    detect_emeans_source,
    detect_environment,
    select_backend,
)
from haystack_integrations.components.preprocessors.emeans._model import UNIT_NORM_TOLERANCE, load_model

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU visible to PyTorch"),
]

MakeDocs = Callable[..., list[Document]]


def _fitter(tmp_path: Path, **kwargs: Any) -> EmeansFitter:
    params: dict[str, Any] = {"model_path": str(tmp_path / "m.npz"), "k": 16, "nepochs": 2.0, "device": "cuda"}
    params.update(kwargs)
    return EmeansFitter(**params)


def _require_rocm_upstream() -> None:
    if torch.version.hip is None:
        pytest.skip("ROCm expectation")
    if detect_emeans_source() != "faiss-cpu":
        pytest.skip("needs upstream faiss-cpu e-means")


def _patch_train_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    base = importlib.import_module("faiss.contrib.e_means").Emeans

    msg = "fake"

    def _oom(*_args: Any, **_kwargs: Any) -> Any:
        raise torch.OutOfMemoryError(msg)

    monkeypatch.setattr(base, "train", _oom)


def test_fit_cuda_runs_on_gpu(tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture) -> None:
    if torch.version.hip is None:
        pytest.skip("ROCm expectation")
    docs = make_docs(2000, d=64, n_clusters=16)
    with caplog.at_level(logging.DEBUG):
        result = _fitter(tmp_path).run(documents=docs)
    stats = result["stats"]
    assert stats["backend"].startswith("gpu-")
    if detect_emeans_source() == "faiss-cpu":
        assert stats["backend"] == "gpu-pytorch"
        assert stats["assignment_backend"] == "native"
    assert stats["device_name"] == torch.cuda.get_device_name(0)
    assert stats["fallback_reason"] is None
    assert stats["emeans_source"] == detect_emeans_source()
    assert load_model(result["model_path"]).metadata["backend"] == stats["backend"]
    infos = [r for r in caplog.records if r.levelname == "INFO" and r.name.endswith("fitter")]
    assert len(infos) == 1
    assert stats["backend"] in infos[0].getMessage()


def test_auto_above_threshold_uses_gpu(tmp_path: Path, make_docs: MakeDocs) -> None:
    fitter = _fitter(tmp_path, device="auto", gpu_threshold=1)
    stats = fitter.run(documents=make_docs(2000, d=64, n_clusters=16))["stats"]
    assert stats["backend"].startswith("gpu-")
    assert stats["fallback_reason"] is None


def test_spherical_centroids_unit_norm_on_gpu(tmp_path: Path, make_docs: MakeDocs) -> None:
    result = _fitter(tmp_path).run(documents=make_docs(2000, d=64, n_clusters=16))
    centroids = load_model(result["model_path"]).centroids
    norms = np.linalg.norm(centroids, axis=1)
    assert np.all(np.abs(norms - 1.0) <= UNIT_NORM_TOLERANCE)


def test_deterministic_repeats_and_restores(tmp_path: Path, make_docs: MakeDocs) -> None:
    docs = make_docs(2000, d=64, n_clusters=16)
    runs = []
    for name in ("a", "b"):
        result = _fitter(tmp_path, model_path=str(tmp_path / f"{name}.npz"), deterministic=True, seed=3).run(
            documents=docs
        )
        runs.append(load_model(result["model_path"]).centroids)
    assert np.array_equal(runs[0], runs[1])
    assert torch.are_deterministic_algorithms_enabled() is False
    torch.use_deterministic_algorithms(True)
    try:
        _fitter(tmp_path, model_path=str(tmp_path / "c.npz"), deterministic=True, seed=3).run(documents=docs)
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        torch.use_deterministic_algorithms(False)


def test_real_training_oom_propagates(tmp_path: Path, make_docs: MakeDocs, monkeypatch: pytest.MonkeyPatch) -> None:
    select_backend(device="cuda", n=2000, k=16, gpu_threshold=1)  # memoises the probe before train is patched
    _patch_train_oom(monkeypatch)
    fitter = _fitter(tmp_path, precision="fp32")
    with pytest.raises(torch.OutOfMemoryError):
        fitter.run(documents=make_docs(2000, d=64, n_clusters=16))
    assert not (tmp_path / "m.npz").exists()


def test_probe_oom_propagates_and_is_not_memoised(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_train_oom(monkeypatch)
    with pytest.raises(torch.OutOfMemoryError):
        select_backend(device="cuda", n=2000, k=16, gpu_threshold=1)
    assert _backend._PROBE_CACHE == {}


def test_explicit_bf16_on_native_degrades_once(
    tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture
) -> None:
    _require_rocm_upstream()
    docs = make_docs(2000, d=64, n_clusters=16)
    with caplog.at_level(logging.DEBUG):
        stats = _fitter(tmp_path, precision="bf16").run(documents=docs)["stats"]
    assert stats["precision_used"] == "fp32"
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        stats = _fitter(tmp_path, model_path=str(tmp_path / "auto.npz"), precision="auto").run(documents=docs)["stats"]
    assert stats["precision_used"] == "fp32"
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_detect_environment_real() -> None:
    env = detect_environment()
    assert env.gpu_available is True
    assert env.device_name
    assert env.is_rocm == (torch.version.hip is not None)
    assert env.arch is not None
