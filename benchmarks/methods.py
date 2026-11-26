# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Training methods under comparison: e-means through the package, and two Lloyd's k-means baselines."""

import importlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrainResult:
    """Centroids, passes actually run and a method-specific info dict."""

    centroids: np.ndarray
    passes_used: float
    info: dict[str, Any]


@dataclass(frozen=True)
class MethodSpec:
    """A training method: its engine, device, default pass count and trainer."""

    name: str
    engine: str
    device: str
    default_passes: float
    train: Callable[..., TrainResult]
    available: Callable[[], str | None]


_INFO_KEYS = (
    "backend",
    "precision_used",
    "n_train",
    "train_seconds",
    "emeans_source",
    "faiss_version",
    "nepochs",
    "batch_size",
    "recipe",
)
_GPU_INFO_KEYS = (*_INFO_KEYS, "device_name", "assignment_backend")


def _train_emeans_cpu(
    x: np.ndarray, init: np.ndarray, *, k: int, spherical: bool, seed: int, passes: float | None
) -> TrainResult:
    from haystack_integrations.components.preprocessors.emeans._backend import (  # noqa: PLC0415
        fit_model,
        select_backend,
    )

    n = len(x)
    choice = select_backend(device="cpu", n=n, k=k, gpu_threshold=0)
    result = fit_model(
        x,
        k=k,
        spherical=spherical,
        nepochs=passes or 5.0,
        seed=seed,
        max_points_per_centroid=None,
        backend=choice,
        precision="fp32",
        deterministic=False,
        init_centroids=init,
    )
    info = {key: result.info[key] for key in _INFO_KEYS if key in result.info}
    return TrainResult(result.centroids, float(result.info["nepochs"]), info)


def _train_faiss_lloyd(
    x: np.ndarray, init: np.ndarray, *, k: int, spherical: bool, seed: int, passes: float | None
) -> TrainResult:
    import faiss  # noqa: PLC0415

    niter = int(passes or 25)
    km = faiss.Kmeans(
        x.shape[1],
        k,
        niter=niter,
        nredo=1,
        seed=seed,
        spherical=spherical,
        max_points_per_centroid=2**30,
        verbose=False,
    )
    km.train(x, init_centroids=init)
    # faiss stops early once assignments stop changing; the pass count actually run is what gets reported.
    used = len(km.iteration_stats or [])
    if used > niter or used == 0:
        msg = f"faiss.Kmeans ran {used} iterations, expected at most {niter}"
        raise RuntimeError(msg)
    return TrainResult(np.asarray(km.centroids, dtype=np.float32), float(used), {"niter": niter})


def _train_sklearn_lloyd(
    x: np.ndarray, init: np.ndarray, *, k: int, spherical: bool, seed: int, passes: float | None
) -> TrainResult:
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.exceptions import ConvergenceWarning  # noqa: PLC0415

    info: dict[str, Any] = {}
    if passes is None:
        max_iter, tol = 300, 1e-4
    else:
        max_iter, tol = int(passes), 0.0
    info["tol"] = tol
    info["max_iter"] = max_iter
    km = KMeans(n_clusters=k, init=init, n_init=1, algorithm="lloyd", max_iter=max_iter, tol=tol, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        km.fit(x)
    centroids = np.asarray(km.cluster_centers_, dtype=np.float32)
    if spherical:
        # Plain means of unit rows sit inside the sphere and always score a lower squared L2 than unit centroids,
        # so they are projected back to unit length to be scored like the spherical e-means and faiss centroids.
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.maximum(norms, np.float32(1e-12))
    info["centroids_normalised"] = spherical
    return TrainResult(centroids, float(km.n_iter_), info)


def _emeans_gpu(expected_backend: str, precision: str) -> Callable[..., TrainResult]:
    """Build a GPU e-means trainer that fails when the backend or precision actually used differs from the label."""

    def train(
        x: np.ndarray, init: np.ndarray, *, k: int, spherical: bool, seed: int, passes: float | None
    ) -> TrainResult:
        from haystack_integrations.components.preprocessors.emeans._backend import (  # noqa: PLC0415
            fit_model,
            select_backend,
        )

        # Selection runs inside the timed call so the cold time includes the backend probe and the JIT.
        choice = select_backend(device="cuda", n=len(x), k=k, gpu_threshold=0)
        result = fit_model(
            x,
            k=k,
            spherical=spherical,
            nepochs=passes or 5.0,
            seed=seed,
            max_points_per_centroid=None,
            backend=choice,
            precision=precision,
            deterministic=False,
            init_centroids=init,
        )
        used_backend = result.info.get("backend")
        used_precision = result.info.get("precision_used")
        if used_backend != expected_backend or used_precision != precision:
            msg = (
                f"expected backend {expected_backend} with precision {precision}, "
                f"got backend {used_backend} with precision {used_precision}"
            )
            raise RuntimeError(msg)
        info = {key: result.info[key] for key in _GPU_INFO_KEYS if key in result.info}
        return TrainResult(result.centroids, float(result.info["nepochs"]), info)

    return train


def _gpu_reason(*, want_emeans_rocm: bool) -> str | None:
    """Return why a GPU method cannot run in this env, or `None` when it can."""
    from haystack_integrations.components.preprocessors.emeans._backend import emeans_rocm_enabled  # noqa: PLC0415

    env_name = "bench-gpu-fork env (emeans-rocm installed)" if want_emeans_rocm else "bench-gpu env (stock e-means)"
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return f"torch is not installed; needs the {env_name}"
    if not torch.cuda.is_available():
        return f"torch sees no GPU; needs the {env_name}"
    if emeans_rocm_enabled() != want_emeans_rocm:
        return f"needs the {env_name}"
    return None


def _native_available() -> str | None:
    return _gpu_reason(want_emeans_rocm=False)


def _triton_available() -> str | None:
    return _gpu_reason(want_emeans_rocm=True)


def _always() -> str | None:
    return None


METHODS: dict[str, MethodSpec] = {
    "emeans-cpu": MethodSpec("emeans-cpu", "e-means", "cpu", 5.0, _train_emeans_cpu, _always),
    "emeans-gpu-native-fp32": MethodSpec(
        "emeans-gpu-native-fp32", "e-means", "gpu", 5.0, _emeans_gpu("gpu-pytorch", "fp32"), _native_available
    ),
    "emeans-gpu-triton-fp32": MethodSpec(
        "emeans-gpu-triton-fp32", "e-means", "gpu", 5.0, _emeans_gpu("gpu-triton", "fp32"), _triton_available
    ),
    "emeans-gpu-triton-bf16": MethodSpec(
        "emeans-gpu-triton-bf16", "e-means", "gpu", 5.0, _emeans_gpu("gpu-triton", "bf16"), _triton_available
    ),
    "faiss-lloyd": MethodSpec("faiss-lloyd", "lloyd", "cpu", 25.0, _train_faiss_lloyd, _always),
    "sklearn-lloyd": MethodSpec("sklearn-lloyd", "lloyd", "cpu", 300.0, _train_sklearn_lloyd, _always),
}
