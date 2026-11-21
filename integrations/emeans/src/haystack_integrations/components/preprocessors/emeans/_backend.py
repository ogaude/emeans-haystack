# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""The only module that touches the e-means engine; PyTorch is imported only inside functions, when a GPU is used."""

import contextlib
import importlib
import importlib.util
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from haystack import logging
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install \"faiss-cpu>=1.16.0\"'") as faiss_import:
    import faiss
    from faiss.contrib.e_means import NumpyEmeans


logger = logging.getLogger(__name__)

DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")


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
    cuda_version: str | None = None
    arch: str | None = None


@dataclass(frozen=True)
class BackendChoice:
    """The backend picked for one fit or assignment."""

    name: str
    device_name: str
    fallback_reason: str | None
    requested: str | None = None  # the e-means `assignment_backend` of the chosen GPU rung; None means CPU


class GpuRung(Protocol):
    """One GPU backend option that `select_backend` can consider."""

    @property
    def name(self) -> str:
        """Short name of the rung, used in reports."""

    @property
    def requested(self) -> str:
        """The e-means `assignment_backend` this rung asks for."""

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


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of the tiny training run that checks a GPU rung works."""

    ok: bool
    assignment_backend: str | None
    reason: str | None


ProbeFn = Callable[[str, str, str], ProbeResult]  # (rung name, requested assignment backend, precision)

_PROBE_CACHE: dict[tuple[str, str], ProbeResult] = {}
_PROBE_LOCK = threading.Lock()

_PROBE_ROWS = 1024
_PROBE_DIM = 64
_PROBE_K = 16


def _run_probe(requested: str, precision: str) -> ProbeResult:
    torch = importlib.import_module("torch")
    e_means = importlib.import_module("faiss.contrib.e_means")
    x = None
    em = None
    try:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        x = torch.randn(_PROBE_ROWS, _PROBE_DIM, device="cuda", generator=generator)
        x = x / x.norm(dim=1, keepdim=True)
        kwargs: dict[str, Any] = {
            "nepochs": 1,
            "seed": 0,
            "spherical": True,
            "capture_assignments": False,
            "assignment_backend": requested,
        }
        if precision == "fp32":
            em = e_means.QuantizedEmeans(_PROBE_DIM, _PROBE_K, quantizer=None, **kwargs)
        else:
            em = e_means.Emeans(_PROBE_DIM, _PROBE_K, **kwargs)
            if precision == "fp16":
                x = x.half()
        centroids, _ = em.train(x)
        if tuple(centroids.shape) != (_PROBE_K, _PROBE_DIM) or not bool(torch.isfinite(centroids).all()):
            return ProbeResult(False, None, "the probe produced non-finite or misshapen centroids")
        return ProbeResult(True, str(em.recipe["assignment_backend"]), None)
    except torch.OutOfMemoryError:
        raise
    except Exception as error:
        return ProbeResult(False, None, f"{type(error).__name__}: {error}")
    finally:
        del x, em
        torch.cuda.empty_cache()


def probe(rung_name: str, requested: str, precision: str) -> ProbeResult:
    """
    Run the tiny training probe for a rung once per process and precision, and remember the outcome.

    :param rung_name: Name of the rung being probed.
    :param requested: The e-means `assignment_backend` to probe.
    :param precision: `"fp32"`, `"bf16"` or `"fp16"`.
    :raises torch.OutOfMemoryError: If the GPU is out of memory; this is never remembered.
    """
    key = (rung_name, precision)
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(key)
        if cached is not None:
            return cached
        result = _run_probe(requested, precision)
        _PROBE_CACHE[key] = result
    logger.debug(
        "GPU probe for rung {rung} at {precision}: ok={ok} ({reason})",
        rung=rung_name,
        precision=precision,
        ok=result.ok,
        reason=result.reason,
    )
    return result


def clear_probe_cache() -> None:
    """Forget every remembered probe outcome."""
    with _PROBE_LOCK:
        _PROBE_CACHE.clear()


@dataclass(frozen=True)
class _Attempt:
    name: str
    requested: str
    gate: Callable[[EnvInfo], str | None]
    probe_fn: ProbeFn

    def usable(self, env: EnvInfo) -> str | None:
        reason = self.gate(env)
        if reason is not None:
            return reason
        result = self.probe_fn(self.name, self.requested, "fp32")
        return None if result.ok else result.reason


def _needs_fork(env: EnvInfo) -> str | None:
    if env.is_rocm and env.fork_detected:
        return None
    return "needs the e-means-rocm fork on a ROCm build of PyTorch"


def _needs_nvidia(env: EnvInfo) -> str | None:
    return "needs an NVIDIA GPU" if env.is_rocm else None


def _always(_env: EnvInfo) -> str | None:
    return None


def default_rungs(probe_fn: ProbeFn | None = None) -> tuple[GpuRung, ...]:
    """
    Return the GPU ladder in the order it is tried.

    :param probe_fn: Probe to use; the memoised `probe` when omitted.
    """
    fn = probe_fn if probe_fn is not None else probe
    return (
        _Attempt("fork-triton", "auto", _needs_fork, fn),
        _Attempt("nvidia-auto", "auto", _needs_nvidia, fn),
        _Attempt("nvidia-triton", "triton", _needs_nvidia, fn),
        _Attempt("native", "native", _always, fn),
    )


def detect_environment() -> EnvInfo:
    """Inspect the environment; PyTorch is imported only when it is installed."""
    fork = detect_emeans_source() == "e-means-rocm"
    try:
        torch_findable = importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        torch_findable = False
    unavailable = EnvInfo(False, False, False, fork, None, None, None)
    if not torch_findable:
        return unavailable
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return unavailable
    hip_version = torch.version.hip
    gpu_available = bool(torch.cuda.is_available())
    device_name: str | None = None
    arch: str | None = None
    if gpu_available:
        device_name = str(torch.cuda.get_device_name(0))
        if hip_version is not None:
            arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        else:
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"sm_{major}{minor}"
    return EnvInfo(
        torch_importable=True,
        gpu_available=gpu_available,
        is_rocm=hip_version is not None,
        fork_detected=fork,
        device_name=device_name,
        torch_version=str(torch.__version__),
        hip_version=hip_version,
        cuda_version=torch.version.cuda,
        arch=arch,
    )


def select_backend(
    *,
    device: str,
    n: int,
    k: int,
    gpu_threshold: float,
    env: EnvInfo | None = None,
    gpu_rungs: Sequence[GpuRung] | None = None,
) -> BackendChoice:
    """
    Pick the backend for a run.

    :param device: `"cpu"`, `"cuda"` or `"auto"`.
    :param n: Number of embeddings.
    :param k: Number of clusters.
    :param gpu_threshold: With `device="auto"`, a GPU is considered only when `n * k` reaches this value.
    :param env: Detected environment; detected here when omitted and a GPU is considered.
    :param gpu_rungs: GPU backends to try in order; `None` means the default ladder.
    :raises RuntimeError: If `device="cuda"` and no GPU backend is usable.
    :raises ValueError: If `device` is not a known value.
    """
    if device not in DEVICES:
        msg = f"device must be 'auto', 'cpu' or 'cuda', got {device!r}"
        raise ValueError(msg)
    if device == "cpu":
        return BackendChoice("cpu", "cpu", None)
    if device == "auto" and n * k < gpu_threshold:
        return BackendChoice("cpu", "cpu", None)
    environment = env if env is not None else detect_environment()
    if not environment.torch_importable:
        detail = "PyTorch is not installed"
    elif not environment.gpu_available:
        detail = "no GPU visible to PyTorch"
    else:
        rungs = gpu_rungs if gpu_rungs is not None else default_rungs()
        reasons: list[str] = []
        for rung in rungs:
            reason = rung.usable(environment)
            if reason is None:
                return BackendChoice(rung.name, environment.device_name or "cuda", None, requested=rung.requested)
            reasons.append(f"{rung.name}: {reason}")
        detail = "; ".join(reasons) or "no GPU backend registered"
    if device == "cuda":
        msg = f"device='cuda' requested but no GPU backend usable: {detail}"
        raise RuntimeError(msg)
    return BackendChoice("cpu", "cpu", f"no GPU backend usable: {detail}")


_BACKEND_NAMES = {"native": "gpu-pytorch", "triton": "gpu-triton", "cute": "gpu-cute"}
_PRECISION_NAMES = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}


@contextlib.contextmanager
def _deterministic(torch: Any, enabled: bool) -> Iterator[None]:
    """Run the body with PyTorch's deterministic algorithms on, then restore the previous setting."""
    if not enabled:
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if torch.version.hip is None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


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
    :param deterministic: Request deterministic algorithms on the GPU; CPU training is deterministic already.
    """
    faiss_import.check()
    if backend.requested is not None:
        return _fit_gpu(
            x,
            k=k,
            spherical=spherical,
            nepochs=nepochs,
            seed=seed,
            max_points_per_centroid=max_points_per_centroid,
            backend=backend,
            precision=precision,
            deterministic=deterministic,
        )
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
    iteration_stats = em.iteration_stats
    imbalance = float(iteration_stats[-1].get("imbalance_factor", float("nan"))) if iteration_stats else float("nan")
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
        "imbalance": imbalance,
    }
    return FitResult(centroids=np.ascontiguousarray(centroids, dtype=np.float32), info=info)


def _n_train(n: int, k: int, max_points_per_centroid: int | None) -> int:
    return int(max_points_per_centroid * k if max_points_per_centroid and n > max_points_per_centroid * k else n)


def _fit_gpu(
    x: np.ndarray,
    *,
    k: int,
    spherical: bool,
    nepochs: float,
    seed: int,
    max_points_per_centroid: int | None,
    backend: BackendChoice,
    precision: str,
    deterministic: bool,
) -> FitResult:
    torch = importlib.import_module("torch")
    e_means = importlib.import_module("faiss.contrib.e_means")
    if precision in ("bf16", "fp16"):
        logger.warning(
            "precision '{precision}' is not supported on the {backend} GPU backend; using fp32",
            precision=precision,
            backend=backend.name,
        )
    n, d = x.shape
    x_gpu = None
    em = None
    try:
        em = e_means.QuantizedEmeans(
            d,
            k,
            nepochs=nepochs,
            seed=seed,
            spherical=spherical,
            max_points_per_centroid=max_points_per_centroid,
            capture_assignments=False,
            quantizer=None,
            assignment_backend=backend.requested,
        )
        x_gpu = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to("cuda")
        with _deterministic(torch, deterministic):
            start = time.perf_counter()
            centroids, _ = em.train(x_gpu)
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - start
        result = np.ascontiguousarray(centroids.detach().float().cpu().numpy(), dtype=np.float32)
        recipe = em.recipe
        label = str(recipe["assignment_backend"])  # read after training: e-means resolves "auto" itself
        precision_label = str(recipe["representation"]).removeprefix("torch.")
        iteration_stats = em.iteration_stats
        imbalance = (
            float(iteration_stats[-1].get("imbalance_factor", float("nan"))) if iteration_stats else float("nan")
        )
        info: dict[str, Any] = {
            "backend": _BACKEND_NAMES.get(label, f"gpu-{label}"),
            "device_name": backend.device_name,
            "precision_used": _PRECISION_NAMES.get(precision_label, precision_label),
            "assignment_backend": label,
            "recipe": recipe,
            "n_train": _n_train(n, k, max_points_per_centroid),
            "train_seconds": float(train_seconds),
            "emeans_source": detect_emeans_source(),
            "faiss_version": str(faiss.__version__),
            "nepochs": float(recipe["nepochs"]),
            "batch_size": int(recipe["batch_size"]),
            "alpha0": float(recipe["alpha0"]),
            "n_splits": int(em.n_splits),
            "imbalance": imbalance,
        }
        return FitResult(centroids=result, info=info)
    finally:
        del x_gpu, em
        torch.cuda.empty_cache()


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
