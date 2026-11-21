# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""The only module that touches the e-means engine. It never imports PyTorch."""

import importlib.util
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install \"faiss-cpu>=1.16.0\"'") as faiss_import:
    import faiss
    from faiss.contrib.e_means import NumpyEmeans


@dataclass(frozen=True)
class EnvInfo:
    """What the current environment offers, detected without importing PyTorch."""

    torch_importable: bool
    gpu_available: bool
    is_rocm: bool
    fork_detected: bool
    device_name: str | None
    torch_version: str | None
    hip_version: str | None


@dataclass(frozen=True)
class BackendChoice:
    """The backend picked for one fit or assignment."""

    name: str
    device_name: str
    fallback_reason: str | None


class Assigner(Protocol):
    """Assigns embeddings to the nearest centroid of a saved model."""

    def assign(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `(score, labels)`: float32 scores and int64 cluster indices for each row of `x`."""


@dataclass(frozen=True)
class FitResult:
    """Centroids from a fit, plus the training report."""

    centroids: np.ndarray
    info: dict[str, Any]


def detect_environment() -> EnvInfo:
    """Inspect the environment without importing PyTorch."""
    try:
        fork_detected = importlib.util.find_spec("faiss.contrib.e_means.device_profile") is not None
    except (ImportError, ValueError):
        fork_detected = False
    return EnvInfo(
        torch_importable=importlib.util.find_spec("torch") is not None,
        gpu_available=False,
        is_rocm=False,
        fork_detected=fork_detected,
        device_name=None,
        torch_version=None,
        hip_version=None,
    )


def select_backend(*, device: str, n: int, k: int, gpu_threshold: float, env: EnvInfo | None = None) -> BackendChoice:
    """
    Pick the backend for a run.

    :param device: `"cpu"`, `"cuda"` or `"auto"`.
    :param n: Number of embeddings.
    :param k: Number of clusters.
    :param gpu_threshold: With `device="auto"`, a GPU is considered only when `n * k` reaches this value.
    :param env: Detected environment; detected here when omitted.
    :raises RuntimeError: If `device="cuda"` and no GPU backend is usable.
    :raises ValueError: If `device` is not a known value.
    """
    del env  # GPU backends are not available yet, so nothing in the environment changes the choice.
    if device == "cpu":
        return BackendChoice("cpu", "cpu", None)
    if device == "cuda":
        msg = "device='cuda' requested but no GPU backend usable: none available"
        raise RuntimeError(msg)
    if device == "auto":
        reason = "no GPU backend usable" if n * k >= gpu_threshold else None
        return BackendChoice("cpu", "cpu", reason)
    msg = f"device must be 'auto', 'cpu' or 'cuda', got {device!r}"
    raise ValueError(msg)


def fit_model(
    x: np.ndarray,
    *,
    k: int,
    spherical: bool,
    nepochs: float,
    seed: int,
    max_points_per_centroid: int | None,
    backend: BackendChoice,
) -> FitResult:
    """
    Train e-means on `x` and return the centroids with a report of what ran.

    :param x: float32 matrix of shape `(n, d)`; already normalised when `spherical` is true.
    :param k: Number of clusters.
    :param spherical: Cluster by cosine similarity.
    :param nepochs: Passes over the data.
    :param seed: Random seed.
    :param max_points_per_centroid: Train on a random subset when `n` exceeds this many points per centroid.
    :param backend: The backend chosen by `select_backend`.
    """
    faiss_import.check()
    n, d = x.shape
    em = NumpyEmeans(
        d,
        k,
        nepochs=nepochs,
        seed=seed,
        spherical=spherical,
        max_points_per_centroid=max_points_per_centroid,
        capture_assignments=False,
    )
    start = time.perf_counter()
    centroids, _ = em.train(x)
    train_seconds = time.perf_counter() - start
    recipe = em.recipe
    n_train = max_points_per_centroid * k if max_points_per_centroid and n > max_points_per_centroid * k else n
    info: dict[str, Any] = {
        "backend": backend.name,
        "device_name": backend.device_name,
        "precision_used": "fp32",
        "assignment_backend": str(recipe.get("assignment_backend", recipe["backend"])),
        "recipe": recipe,
        "n_train": int(n_train),
        "train_seconds": float(train_seconds),
        "emeans_source": "e-means-rocm" if detect_environment().fork_detected else "faiss-cpu",
        "faiss_version": str(faiss.__version__),
    }
    return FitResult(centroids=np.ascontiguousarray(centroids, dtype=np.float32), info=info)


class _CpuAssigner:
    def __init__(self, centroids: np.ndarray, *, spherical: bool) -> None:
        faiss_import.check()
        k, d = centroids.shape
        self._em = NumpyEmeans(d, k, spherical=spherical)
        self._em.centroids = centroids

    def assign(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score, labels = self._em.assign(x)
        return np.asarray(score, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def cpu_assigner(centroids: np.ndarray, *, spherical: bool) -> Assigner:
    """
    Build an assigner for saved centroids on CPU.

    :param centroids: float32 centroids of shape `(k, d)`.
    :param spherical: Whether the model clusters by cosine similarity (scores are then cosine similarities;
        otherwise squared L2 distances).
    """
    return _CpuAssigner(centroids, spherical=spherical)
