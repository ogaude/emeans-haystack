# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end fits on a real GPU (local only; needs PyTorch with a visible GPU, otherwise every test skips).

Only some tests are gated on AMD; the rest run on any GPU that PyTorch sees, NVIDIA included:

- ROCm only (skip when `torch.version.hip` is None): `test_fit_cuda_runs_on_gpu`, the emeans-rocm tests
  (`test_emeans_rocm_reduced_precision`, `test_fp16_out_of_range_embeddings_train_in_fp32`, which also need
  emeans-rocm installed) and the upstream-AMD tests (`test_explicit_bf16_on_native_degrades_once`,
  `test_upstream_amd_*`, which also need emeans-rocm absent).
- Any GPU: every other test. They assert vendor-neutral behaviour (GPU backend chosen, unit-norm centroids,
  determinism, OOM propagation, GPU and CPU assignment agree, memory estimate). With emeans-rocm installed they run
  its engine, whose NVIDIA path has not been re-run on hardware, so a failure there on NVIDIA is a real finding
  about emeans-rocm, not a test bug. The memory-estimate tests are gated on the engine having `memory_estimate`.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from haystack import Document

from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter, _backend
from haystack_integrations.components.preprocessors.emeans._backend import (
    detect_emeans_source,
    detect_environment,
    emeans_rocm_importable,
    gpu_engine,
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
    if emeans_rocm_importable():
        pytest.skip("needs stock faiss-cpu e-means (emeans-rocm installed)")


def _patch_train_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    base = gpu_engine().Emeans

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
    if not emeans_rocm_importable():
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


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelname == "WARNING"]


@pytest.mark.parametrize(
    ("precision", "expected"),
    [("auto", "bf16"), ("bf16", "bf16"), ("fp16", "fp16"), ("fp32", "fp32")],
)
def test_emeans_rocm_reduced_precision(
    tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture, precision: str, expected: str
) -> None:
    if torch.version.hip is None or not emeans_rocm_importable():
        pytest.skip("needs emeans-rocm on ROCm")
    docs = make_docs(2000, d=64, n_clusters=16)
    with caplog.at_level(logging.DEBUG):
        stats = _fitter(tmp_path, precision=precision).run(documents=docs)["stats"]
    assert stats["backend"] == "gpu-triton"
    assert stats["precision_used"] == expected
    assert stats["emeans_source"].startswith("emeans-rocm")
    assert _warnings(caplog) == []


def test_fp16_out_of_range_embeddings_train_in_fp32(
    tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture
) -> None:
    if torch.version.hip is None or not emeans_rocm_importable():
        pytest.skip("needs emeans-rocm on ROCm")
    docs = make_docs(2000, d=64, n_clusters=16, unit=False)
    docs = [Document(content=d.content, embedding=[v * 1e5 for v in d.embedding or []]) for d in docs]
    with caplog.at_level(logging.DEBUG):
        stats = _fitter(tmp_path, precision="fp16", spherical=False).run(documents=docs)["stats"]
    assert stats["precision_used"] == "fp32"
    assert any("fp16 range" in r.getMessage() for r in _warnings(caplog))
    assert np.isfinite(load_model(str(tmp_path / "m.npz")).centroids).all()


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_upstream_amd_reduced_precision_degrades(
    tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture, precision: str
) -> None:
    _require_rocm_upstream()
    docs = make_docs(2000, d=64, n_clusters=16)
    with caplog.at_level(logging.DEBUG):
        stats = _fitter(tmp_path, precision=precision).run(documents=docs)["stats"]
    assert stats["precision_used"] == "fp32"
    assert stats["backend"] == "gpu-pytorch"
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert precision in warnings[0].getMessage()


def test_upstream_amd_auto_is_silent_fp32(
    tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture
) -> None:
    _require_rocm_upstream()
    with caplog.at_level(logging.DEBUG):
        stats = _fitter(tmp_path, precision="auto").run(documents=make_docs(2000, d=64, n_clusters=16))["stats"]
    assert stats["precision_used"] == "fp32"
    assert _warnings(caplog) == []


def test_detect_environment_real() -> None:
    env = detect_environment()
    assert env.gpu_available is True
    assert env.device_name
    assert env.is_rocm == (torch.version.hip is not None)
    assert env.arch is not None


@pytest.mark.parametrize("spherical", [True, False])
def test_gpu_assignment_matches_cpu(tmp_path: Path, make_docs: MakeDocs, *, spherical: bool) -> None:
    docs = make_docs(4000, d=64, n_clusters=16, unit=spherical)
    model = str(tmp_path / "m.npz")
    EmeansFitter(model_path=model, k=16, nepochs=2.0, spherical=spherical, device="cpu").run(documents=docs)
    outs = {}
    for device in ("cuda", "cpu"):
        comp = EmeansDocumentClusterer(model_path=model, device=device, return_distance=True, batch_size=1000)
        outs[device] = comp.run(documents=docs)["documents"]
    key = "cluster_similarity" if spherical else "cluster_distance"
    assert [d.meta["cluster_id"] for d in outs["cuda"]] == [d.meta["cluster_id"] for d in outs["cpu"]]
    diff = max(abs(a.meta[key] - b.meta[key]) for a, b in zip(outs["cuda"], outs["cpu"], strict=True))
    assert diff <= 1e-4


def test_clusterer_auto_above_threshold_uses_gpu(
    tmp_path: Path, make_docs: MakeDocs, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = make_docs(2000, d=64, n_clusters=16)
    model = str(tmp_path / "m.npz")
    EmeansFitter(model_path=model, k=16, nepochs=2.0, device="cpu").run(documents=docs)
    original = _backend._GpuAssigner.assign
    calls: list[int] = []

    def _spy(self: Any, x: np.ndarray) -> Any:
        calls.append(len(x))
        return original(self, x)

    monkeypatch.setattr(_backend._GpuAssigner, "assign", _spy)
    EmeansDocumentClusterer(model_path=model, device="auto", gpu_threshold=1).run(documents=docs)
    assert len(calls) >= 1


def _has_memory_estimate() -> bool:
    return hasattr(gpu_engine(), "memory_estimate")


def test_memory_estimate_recorded_with_emeans_rocm(tmp_path: Path, make_docs: MakeDocs) -> None:
    if not _has_memory_estimate():
        pytest.skip("needs an e-means with memory_estimate (emeans-rocm)")
    stats = _fitter(tmp_path).run(documents=make_docs(2000, d=64, n_clusters=16))["stats"]
    assert stats["backend"].startswith("gpu-")
    assert isinstance(stats["memory_estimate_bytes"], int)
    assert stats["memory_estimate_bytes"] > 0
    assert stats["fallback_reason"] is None


def test_auto_falls_back_to_cpu_when_estimate_exceeds_free(
    tmp_path: Path, make_docs: MakeDocs, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    if not _has_memory_estimate():
        pytest.skip("needs an e-means with memory_estimate (emeans-rocm)")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a, **_k: (1, 16 * 2**30))
    fitter = _fitter(tmp_path, device="auto", gpu_threshold=0)
    with caplog.at_level(logging.DEBUG):
        result = fitter.run(documents=make_docs(2000, d=64, n_clusters=16))
    stats = result["stats"]
    assert stats["backend"] == "cpu"
    assert stats["device_name"] == "cpu"
    assert stats["precision_used"] == "fp32"
    assert "GiB" in stats["fallback_reason"]
    assert stats["memory_estimate_bytes"] > 0
    assert [r.getMessage() for r in _warnings(caplog)] == [stats["fallback_reason"]]
    assert load_model(result["model_path"]).metadata["backend"] == "cpu"


def test_upstream_records_no_memory_estimate(tmp_path: Path, make_docs: MakeDocs) -> None:
    if _has_memory_estimate():
        pytest.skip("needs upstream e-means without memory_estimate")
    stats = _fitter(tmp_path).run(documents=make_docs(2000, d=64, n_clusters=16))["stats"]
    assert stats["backend"].startswith("gpu-")
    assert stats["memory_estimate_bytes"] is None


def test_fit_init_centroids_on_gpu() -> None:
    x = np.random.default_rng(0).normal(size=(2000, 32)).astype(np.float32)
    backend = select_backend(device="cuda", n=2000, k=16, gpu_threshold=0)
    kwargs: dict[str, Any] = {
        "k": 16,
        "spherical": False,
        "nepochs": 2.0,
        "seed": 1,
        "max_points_per_centroid": None,
        "backend": backend,
        "precision": "fp32",
        "deterministic": True,
    }
    first = _backend.fit_model(x, init_centroids=x[:16].copy(), **kwargs).centroids
    again = _backend.fit_model(x, init_centroids=x[:16].copy(), **kwargs).centroids
    other = _backend.fit_model(x, init_centroids=x[100:116].copy(), **kwargs).centroids
    assert first.dtype == np.float32
    assert first.shape == (16, 32)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
