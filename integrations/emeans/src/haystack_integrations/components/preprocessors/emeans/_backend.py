# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""The only module that touches the e-means engine. It never imports PyTorch."""

import importlib.util
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from haystack import logging
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install \"faiss-cpu>=1.16.0\"'") as faiss_import:
    import faiss
    from faiss.contrib.e_means import NumpyEmeans


logger = logging.getLogger(__name__)

_DEVICES = ("auto", "cpu", "cuda")


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


class GpuRung(Protocol):
    """One GPU backend option that `select_backend` can consider; none are registered yet."""

    name: str

    def usable(self, env: EnvInfo) -> str | None:
        """Return `None` when this backend can run in `env`, otherwise the reason it cannot."""


class Assigner(Protocol):
    """Assigns embeddings to the nearest centroid of a saved model."""

    def assign(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `(score, labels)`: float32 scores and int64 cluster indices for each row of `x`."""


@dataclass(frozen=True)
class FitResult:
    """Centroids from a fit, plus the training report."""

    centroids: np.ndarray
    info: dict[str, Any]


def detect_emeans_source() -> str:
    """Return `"e-means-rocm"` when the ROCm-capable e-means fork is installed, else `"faiss-cpu"`."""
    try:
        fork = importlib.util.find_spec("faiss.contrib.e_means.device_profile") is not None
    except (ImportError, ValueError):
        fork = False
    return "e-means-rocm" if fork else "faiss-cpu"


def detect_environment() -> EnvInfo:
    """Inspect the environment without importing PyTorch."""
    return EnvInfo(
        torch_importable=importlib.util.find_spec("torch") is not None,
        gpu_available=False,
        is_rocm=False,
        fork_detected=detect_emeans_source() == "e-means-rocm",
        device_name=None,
        torch_version=None,
        hip_version=None,
    )


def select_backend(
    *,
    device: str,
    n: int,
    k: int,
    gpu_threshold: float,
    env: EnvInfo | None = None,
    gpu_rungs: Sequence[GpuRung] = (),
) -> BackendChoice:
    """
    Pick the backend for a run.

    :param device: `"cpu"`, `"cuda"` or `"auto"`.
    :param n: Number of embeddings.
    :param k: Number of clusters.
    :param gpu_threshold: With `device="auto"`, a GPU is considered only when `n * k` reaches this value.
    :param env: Detected environment; detected here when omitted and GPU backends are registered.
    :param gpu_rungs: GPU backends to try in order. None are registered yet, so the default runs on CPU.
    :raises RuntimeError: If `device="cuda"` and no GPU backend is usable.
    :raises ValueError: If `device` is not a known value.
    """
    if device not in _DEVICES:
        msg = f"device must be 'auto', 'cpu' or 'cuda', got {device!r}"
        raise ValueError(msg)
    if device == "cpu":
        return BackendChoice("cpu", "cpu", None)
    if device == "auto" and n * k < gpu_threshold:
        return BackendChoice("cpu", "cpu", None)
    reasons: list[str] = []
    if gpu_rungs:
        environment = env if env is not None else detect_environment()
        for rung in gpu_rungs:
            reason = rung.usable(environment)
            if reason is None:
                return BackendChoice(rung.name, rung.name, None)
            reasons.append(f"{rung.name}: {reason}")
    if device == "cuda":
        msg = f"device='cuda' requested but no GPU backend usable: {'; '.join(reasons) or 'none available'}"
        raise RuntimeError(msg)
    return BackendChoice("cpu", "cpu", "no GPU backend usable")


def fit_model(
    x: np.ndarray,
    *,
    k: int,
    spherical: bool,
    nepochs: float,
    seed: int,
    max_points_per_centroid: int | None,
    backend: BackendChoice,
    precision: str = "fp32",
    deterministic: bool = False,
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
    :param precision: Requested precision. The CPU backend always runs FP32 and warns once for `bf16` or `fp16`.
    :param deterministic: Request deterministic algorithms; has no effect on CPU.
    """
    del deterministic  # CPU training is deterministic for a fixed seed already.
    faiss_import.check()
    if precision in ("bf16", "fp16"):
        logger.warning("precision '{precision}' is not supported on the CPU backend; using fp32", precision=precision)
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
    iteration_stats = em.iteration_stats or [{}]
    info: dict[str, Any] = {
        "backend": backend.name,  # the backend that ran, not e-means' own "faiss" label
        "device_name": backend.device_name,
        "precision_used": "fp32",
        "assignment_backend": str(recipe.get("assignment_backend", recipe["backend"])),
        "recipe": recipe,
        "n_train": int(
            max_points_per_centroid * k if max_points_per_centroid and n > max_points_per_centroid * k else n
        ),
        "train_seconds": float(train_seconds),
        "emeans_source": detect_emeans_source(),
        "faiss_version": str(faiss.__version__),
        "nepochs": float(recipe["nepochs"]),
        "batch_size": int(recipe["batch_size"]),
        "alpha0": float(recipe["alpha0"]),
        "n_splits": int(em.n_splits),
        "imbalance": float(iteration_stats[-1]["imbalance_factor"]),
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
