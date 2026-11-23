# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""The only module that touches the e-means engine; PyTorch is imported only inside functions, when a GPU is used."""

import contextlib
import importlib
import importlib.metadata
import importlib.util
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
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
    emeans_rocm: bool
    device_name: str | None
    torch_version: str | None
    hip_version: str | None
    cuda_version: str | None = None
    arch: str | None = None
    mismatch_reason: str | None = None


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


def emeans_rocm_importable() -> bool:
    """Return whether the `emeans_rocm` package is installed, without importing it (so without importing torch)."""
    try:
        return importlib.util.find_spec("emeans_rocm") is not None
    except (ImportError, ValueError):
        return False


def gpu_engine() -> Any:
    """
    Return the module that provides the GPU `Emeans` and `QuantizedEmeans` classes.

    `emeans_rocm` is used on any GPU when it is importable, otherwise `faiss.contrib.e_means`. CPU paths must not
    call this: they use the stock `NumpyEmeans`.
    """
    if emeans_rocm_importable():
        return importlib.import_module("emeans_rocm")
    return importlib.import_module("faiss.contrib.e_means")


def _dist_version(dist: str) -> str | None:
    """Return the installed version of distribution `dist`, or `None` when it is not installed."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cpu_source() -> str:
    """Return the engine label for CPU fits, e.g. `"faiss-cpu 1.16.0"`."""
    return f"faiss-cpu {faiss.__version__}"


def detect_emeans_source() -> str:
    """Return the GPU engine label: `"emeans-rocm <version>"` when it is installed, else the faiss-cpu label."""
    if emeans_rocm_importable():
        version = _dist_version("emeans-rocm")
        return f"emeans-rocm {version}" if version else "emeans-rocm"
    return _cpu_source()


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of the tiny training run that checks a GPU rung works."""

    ok: bool
    assignment_backend: str | None
    reason: str | None


ProbeFn = Callable[[str, str, str], ProbeResult]  # (rung name, requested assignment backend, precision)

_PROBE_CACHE: dict[tuple[str, str, str], ProbeResult] = {}  # (rung name, requested, precision) -> outcome
_PROBE_LOCK = threading.Lock()  # guards the two dicts only; never held while a probe runs
_PROBE_KEY_LOCKS: dict[tuple[str, str, str], threading.Lock] = {}  # serialises probes of one key
_TRANSIENT_MARKERS = ("out of memory", "busy")


def _is_transient_failure(reason: str | None) -> bool:
    """Return whether a probe failure looks transient (memory pressure, busy device), so it must not be memoised."""
    text = (reason or "").lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


_PROBE_ROWS = 1024
_PROBE_DIM = 64
_PROBE_K = 16


def _run_probe(requested: str, precision: str) -> ProbeResult:
    # Both spherical modes are probed: e-means takes mode-dependent code paths, and the verdict is reused for
    # every later fit and assignment on this rung, whatever its mode.
    torch = importlib.import_module("torch")
    e_means = gpu_engine()
    x = None
    em = None
    centroids = None
    try:
        label = ""
        for spherical in (True, False):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(0)
            x = torch.randn(_PROBE_ROWS, _PROBE_DIM, device="cuda", generator=generator)
            if spherical:
                x = x / x.norm(dim=1, keepdim=True)
            kwargs: dict[str, Any] = {
                "nepochs": 1,
                "seed": 0,
                "spherical": spherical,
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
                mode = "spherical" if spherical else "non-spherical"
                return ProbeResult(False, None, f"the {mode} probe produced non-finite or misshapen centroids")
            if spherical:
                label = str(em.recipe["assignment_backend"])
        return ProbeResult(True, label, None)
    except torch.OutOfMemoryError:
        raise
    except Exception as error:
        return ProbeResult(False, None, f"{type(error).__name__}: {error}")
    finally:
        del x, em, centroids
        torch.cuda.empty_cache()


def probe(rung_name: str, requested: str, precision: str) -> ProbeResult:
    """
    Run the tiny training probe once per process for each rung, assignment backend and precision; remember it.

    :param rung_name: Name of the rung being probed.
    :param requested: The e-means `assignment_backend` to probe.
    :param precision: `"fp32"`, `"bf16"` or `"fp16"`.
    :raises torch.OutOfMemoryError: If the GPU is out of memory; this is never remembered.

    Probes for the same key run once at a time; probes for different keys do not wait for each other. A failure
    whose reason mentions running out of memory or a busy device is treated as transient and is not remembered.
    """
    key = (rung_name, requested, precision)
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(key)
        if cached is not None:
            return cached
        key_lock = _PROBE_KEY_LOCKS.setdefault(key, threading.Lock())
    with key_lock:
        with _PROBE_LOCK:
            cached = _PROBE_CACHE.get(key)
        if cached is not None:
            return cached  # another thread probed this key while we waited
        result = _run_probe(requested, precision)
        if result.ok or not _is_transient_failure(result.reason):
            with _PROBE_LOCK:
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


def _needs_emeans_rocm(env: EnvInfo) -> str | None:
    if env.is_rocm and env.emeans_rocm:
        return None
    return "needs the emeans-rocm package (pip install emeans-rocm) on a ROCm build of PyTorch"


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
        _Attempt("rocm-triton", "auto", _needs_emeans_rocm, fn),
        _Attempt("nvidia-auto", "auto", _needs_nvidia, fn),
        _Attempt("nvidia-triton", "triton", _needs_nvidia, fn),
        _Attempt("native", "native", _always, fn),
    )


def _device_nodes() -> tuple[bool, bool]:
    """Return `(amd_node, nvidia_node)`: whether a GPU device node exists, checked without importing PyTorch."""
    return (
        os.path.exists("/dev/kfd"),
        os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl"),
    )


def mismatch_reason(
    *,
    amd_node: bool,
    nvidia_node: bool,
    torch_version: str | None,
    hip_version: str | None,
    cuda_version: str | None,
    gpu_available: bool,
) -> str | None:
    """
    Explain why a GPU is present but PyTorch cannot use it because the installed build is the wrong one.

    :param amd_node: Whether `/dev/kfd` exists.
    :param nvidia_node: Whether an NVIDIA device node exists.
    :param torch_version: Installed PyTorch version.
    :param hip_version: `torch.version.hip` (`None` for builds without ROCm).
    :param cuda_version: `torch.version.cuda` (`None` for builds without CUDA).
    :param gpu_available: `torch.cuda.is_available()`.
    :returns: The explanation, or `None` when there is no mismatch.
    """
    if gpu_available:
        return None
    if nvidia_node and not amd_node and hip_version is not None:
        return (
            f"NVIDIA GPU found but PyTorch {torch_version} is a ROCm build; "
            "install a CUDA build of PyTorch, for example with the [gpu] extra"
        )
    # NVIDIA first: an AMD APU or iGPU exposes /dev/kfd next to an NVIDIA dGPU even when ROCm does not support it.
    if nvidia_node and cuda_version is None and hip_version is None:
        return (
            f"NVIDIA GPU found but PyTorch {torch_version} is a CPU-only build; "
            "install a CUDA build of PyTorch, for example with the [gpu] extra"
        )
    if amd_node and hip_version is None and not (nvidia_node and cuda_version is not None):
        return (
            f"AMD GPU found (/dev/kfd) but PyTorch {torch_version} is not a ROCm build; "
            "if this GPU is supported by ROCm, install the ROCm build of PyTorch and do not use the [gpu] extra"
        )
    return None


def _visible_devices_clause(name: str, value: str) -> str:
    return f"{name}={value!r} is set and may hide it; unset it, or use device='cpu' to run on CPU"


def hidden_gpu_reason(
    *,
    amd_node: bool,
    nvidia_node: bool,
    torch_version: str | None,
    hip_version: str | None,
    cuda_version: str | None,
    gpu_available: bool,
    environ: Mapping[str, str],
) -> str | None:
    """
    Explain why a GPU is present and PyTorch has the matching build but still sees no GPU.

    :param amd_node: Whether `/dev/kfd` exists.
    :param nvidia_node: Whether an NVIDIA device node exists.
    :param torch_version: Installed PyTorch version.
    :param hip_version: `torch.version.hip` (`None` for builds without ROCm).
    :param cuda_version: `torch.version.cuda` (`None` for builds without CUDA).
    :param gpu_available: `torch.cuda.is_available()`.
    :param environ: Process environment, scanned for `*_VISIBLE_DEVICES` variables relevant to the vendor.
    :returns: The explanation, or `None` when PyTorch sees a GPU or the build does not match the device.
    """
    if gpu_available:
        return None
    if amd_node and hip_version is not None:
        text = f"AMD GPU found (/dev/kfd) and PyTorch {torch_version} is a ROCm build, but PyTorch sees no GPU; "
        for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            if name in environ:
                return text + _visible_devices_clause(name, environ[name])
        return text + (
            "check that the amdgpu driver is loaded and that this user can open /dev/kfd and /dev/dri "
            "(render and video groups)"
        )
    if nvidia_node and cuda_version is not None:
        text = f"NVIDIA GPU found and PyTorch {torch_version} is a CUDA {cuda_version} build, but PyTorch sees no GPU; "
        if "CUDA_VISIBLE_DEVICES" in environ:
            return text + _visible_devices_clause("CUDA_VISIBLE_DEVICES", environ["CUDA_VISIBLE_DEVICES"])
        return text + (
            f"check that the NVIDIA driver is loaded and supports CUDA {cuda_version}, "
            "and that this user can open /dev/nvidia*"
        )
    return None


def detect_environment() -> EnvInfo:
    """Inspect the environment; PyTorch is imported only when it is installed."""
    emeans_rocm = emeans_rocm_importable()
    try:
        torch_findable = importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        torch_findable = False
    unavailable = EnvInfo(False, False, False, emeans_rocm, None, None, None)
    if not torch_findable:
        return unavailable
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        # find_spec found torch, so it is installed and any import failure means a broken install. A ROCm/CUDA build
        # whose shared libraries cannot be loaded usually raises ImportError, sometimes OSError or RuntimeError.
        reason = f"PyTorch is installed but could not be imported: {type(error).__name__}: {error}"
        return EnvInfo(False, False, False, emeans_rocm, None, None, None, mismatch_reason=reason)
    hip_version = torch.version.hip
    device_name: str | None = None
    arch: str | None = None
    init_error: str | None = None
    try:
        gpu_available = bool(torch.cuda.is_available())
        if gpu_available:
            device_name = str(torch.cuda.get_device_name(0))
            if hip_version is not None:
                arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
            else:
                major, minor = torch.cuda.get_device_capability(0)
                arch = f"sm_{major}{minor}"
    except (OSError, RuntimeError) as error:
        gpu_available, device_name, arch = False, None, None
        init_error = f"PyTorch could not initialise the GPU: {type(error).__name__}: {error}"
    amd_node, nvidia_node = _device_nodes()
    probe_args = {
        "amd_node": amd_node,
        "nvidia_node": nvidia_node,
        "torch_version": str(torch.__version__),
        "hip_version": hip_version,
        "cuda_version": torch.version.cuda,
        "gpu_available": gpu_available,
    }
    mismatch = init_error or mismatch_reason(**probe_args) or hidden_gpu_reason(**probe_args, environ=os.environ)
    return EnvInfo(
        torch_importable=True,
        gpu_available=gpu_available,
        is_rocm=hip_version is not None,
        emeans_rocm=emeans_rocm,
        device_name=device_name,
        torch_version=str(torch.__version__),
        hip_version=hip_version,
        cuda_version=torch.version.cuda,
        arch=arch,
        mismatch_reason=mismatch,
    )


def _is_out_of_memory(error: BaseException) -> bool:
    """Return whether `error` is a PyTorch out-of-memory error, without importing PyTorch (it may be absent)."""
    torch_module = sys.modules.get("torch")
    oom_type = getattr(torch_module, "OutOfMemoryError", None)
    return isinstance(oom_type, type) and isinstance(error, oom_type)


def select_backend(
    *,
    device: str,
    n: int,
    k: int,
    gpu_threshold: float,
    env: EnvInfo | None = None,
    gpu_rungs: Sequence[GpuRung] | None = None,
    warn: bool = True,
) -> BackendChoice:
    """
    Pick the backend for a run.

    A GPU rung is accepted when a tiny training probe (both spherical modes, fixed small `d` and `k`, without
    deterministic mode) succeeds on it, once per process. The probe does not cover every shape or mode, so a rung
    that passes it can still fail on the real data; such an error propagates and the next rung is not tried.
    With `device="auto"`, a GPU out-of-memory error during a rung's probe counts as that rung failing (it is not
    remembered, so later runs probe again); with `device="cuda"` it propagates. Errors from the real fit propagate.

    :param device: `"cpu"`, `"cuda"` or `"auto"`.
    :param n: Number of embeddings.
    :param k: Number of clusters.
    :param gpu_threshold: With `device="auto"`, a GPU is considered only when `n * k` reaches this value.
    :param env: Detected environment; detected here when omitted and a GPU is considered.
    :param gpu_rungs: GPU backends to try in order; `None` means the default ladder.
    :param warn: Log fallback warnings. Pass `False` when the same run has already logged them, for example
        after an auto-fit, so each run warns once.
    :raises RuntimeError: If `device="cuda"` and no GPU backend is usable.
    :raises ValueError: If `device` is not a known value.
    :raises torch.OutOfMemoryError: If `device="cuda"` and a rung's probe runs out of GPU memory.
    """
    if device not in DEVICES:
        msg = f"device must be 'auto', 'cpu' or 'cuda', got {device!r}"
        raise ValueError(msg)
    if device == "cpu":
        return BackendChoice("cpu", "cpu", None)
    if device == "auto" and n * k < gpu_threshold:
        return BackendChoice("cpu", "cpu", None)
    environment = env if env is not None else detect_environment()
    should_warn = environment.mismatch_reason is not None
    if environment.mismatch_reason is not None:
        detail = environment.mismatch_reason
    elif not environment.torch_importable:
        detail = "PyTorch is not installed"
    elif not environment.gpu_available:
        detail = "no GPU visible to PyTorch"
    else:
        rungs = gpu_rungs if gpu_rungs is not None else default_rungs()
        reasons: list[str] = []
        oom_rungs: list[str] = []
        for rung in rungs:
            try:
                reason = rung.usable(environment)
            except Exception as error:
                if device != "auto" or not _is_out_of_memory(error):
                    raise
                reason = f"probe ran out of GPU memory ({type(error).__name__}: {error})"
                oom_rungs.append(rung.name)
            if reason is None:
                if oom_rungs and warn:
                    logger.warning(
                        "GPU probe ran out of memory on {oom}; using {rung}", oom=", ".join(oom_rungs), rung=rung.name
                    )
                return BackendChoice(rung.name, environment.device_name or "cuda", None, requested=rung.requested)
            reasons.append(f"{rung.name}: {reason}")
        detail = "; ".join(reasons) or "no GPU backend registered"
        should_warn = True
    if device == "cuda":
        msg = f"device='cuda' requested but no GPU backend usable: {detail}"
        raise RuntimeError(msg)
    fallback_reason = f"no GPU backend usable: {detail}"
    if should_warn and warn:
        logger.warning("{reason}", reason=fallback_reason)
    return BackendChoice("cpu", "cpu", fallback_reason)


_EMEANS_ROCM_HINT = "(pip install emeans-rocm for Triton kernels and BF16/FP16 on RX 9000 GPUs)"


def _backend_hint(*, is_rocm: bool, assignment_backend: str, emeans_rocm: bool) -> str | None:
    """Return a one-line hint for AMD users running the native rung without emeans-rocm, otherwise `None`."""
    if is_rocm and assignment_backend == "native" and not emeans_rocm:
        return _EMEANS_ROCM_HINT
    return None


_BACKEND_NAMES = {"native": "gpu-pytorch", "triton": "gpu-triton", "cute": "gpu-cute"}
_PRECISION_NAMES = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}


_DETERMINISTIC_LOCK = threading.Lock()


@contextlib.contextmanager
def _deterministic(torch: Any, enabled: bool) -> Iterator[None]:
    """
    Run the body with PyTorch's deterministic algorithms on, then restore the previous setting.

    The setting is process-global, so deterministic bodies are serialised with a lock. On CUDA builds,
    `CUBLAS_WORKSPACE_CONFIG` is set for the body only when unset and removed afterwards; PyTorch may already have
    chosen its cuBLAS workspace by then, so users who need guaranteed reproducibility on NVIDIA should export it
    before starting the process.

    Known limitation: the lock serialises deterministic bodies against each other only. While one holds it, the
    process-global setting and the temporary `CUBLAS_WORKSPACE_CONFIG` also apply to GPU work in other threads
    (a non-deterministic fit, `_GpuAssigner.assign`, `_run_probe`). Taking the lock around that work as well would
    serialise all GPU work in the process, so it is documented instead.
    """
    if not enabled:
        yield
        return
    with _DETERMINISTIC_LOCK:
        previous = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        set_env = torch.version.hip is None and "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        if set_env:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
            yield
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)
            if set_env:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


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
    allow_cpu_fallback: bool = False,
    init_centroids: np.ndarray | None = None,
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
        The setting is process-wide while the fit trains, so GPU work in other threads runs with it too.
    :param allow_cpu_fallback: With True, a GPU fit whose memory estimate exceeds free GPU memory runs on the CPU
        backend instead; the fitter passes True for `device="auto"`.
    :param init_centroids: Optional starting centroids of shape `(k, d)`, float32 (private; used by the benchmarks).
        Default None keeps e-means' own seeded init.
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
            allow_cpu_fallback=allow_cpu_fallback,
            init_centroids=init_centroids,
        )
    return _fit_cpu(
        x,
        k=k,
        spherical=spherical,
        nepochs=nepochs,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        backend=backend,
        precision=precision,
        init_centroids=init_centroids,
    )


def _fit_cpu(
    x: np.ndarray,
    *,
    k: int,
    spherical: bool,
    nepochs: float,
    seed: int,
    max_points_per_centroid: int | None,
    backend: BackendChoice,
    precision: str,
    init_centroids: np.ndarray | None = None,
) -> FitResult:
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
    centroids, _ = em.train(x, init_centroids=init_centroids)
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
        "n_train": _n_train(n, k, max_points_per_centroid),
        "train_seconds": float(train_seconds),
        "emeans_source": _cpu_source(),
        "faiss_version": str(faiss.__version__),
        "nepochs": float(recipe["nepochs"]),
        "batch_size": int(recipe["batch_size"]),
        "alpha0": float(recipe["alpha0"]),
        "n_splits": int(em.n_splits),
        "imbalance": imbalance,
        "backend_hint": None,
        "memory_estimate_bytes": None,
        "fallback_reason": backend.fallback_reason,
    }
    return FitResult(centroids=np.ascontiguousarray(centroids, dtype=np.float32), info=info)


def resolve_precision(choice: BackendChoice, precision: str, probe_fn: ProbeFn | None = None) -> tuple[str, str | None]:
    """
    Decide the precision a GPU fit trains in, using a memoised probe at that precision on the chosen rung.

    :param choice: The chosen GPU backend; its `requested` assignment backend is what gets probed.
    :param precision: `"auto"`, `"fp32"`, `"bf16"` or `"fp16"` as requested by the user.
    :param probe_fn: Probe to run; defaults to the memoised `probe`.
    :returns: The precision to train with (`"fp32"`, `"bf16"` or `"fp16"`) and a warning text, or None when
        nothing needs reporting. `"auto"` falls back to FP32 silently; an explicit `"bf16"`/`"fp16"` that the rung
        cannot run falls back to FP32 with a warning.
    :raises ValueError: If `choice` is a CPU choice.
    """
    if choice.requested is None:
        msg = "resolve_precision applies to GPU backends only"
        raise ValueError(msg)
    if precision == "fp32":
        return "fp32", None
    if choice.requested == "native":
        # The native rung is the FP32 fallback by definition; explicit native assignment would accept BF16/FP16
        # tensors, but its reduced-precision behaviour is not part of the contract.
        return "fp32", None if precision == "auto" else _unsupported(precision, choice)
    run_probe = probe_fn or probe
    wanted = "bf16" if precision == "auto" else precision
    if run_probe(choice.name, choice.requested, wanted).ok:
        return wanted, None
    if precision == "auto":
        return "fp32", None
    return "fp32", _unsupported(precision, choice)


def _unsupported(precision: str, choice: BackendChoice) -> str:
    return f"precision '{precision}' is not supported on the {choice.name} GPU backend; using fp32"


def _n_train(n: int, k: int, max_points_per_centroid: int | None) -> int:
    return int(max_points_per_centroid * k if max_points_per_centroid and n > max_points_per_centroid * k else n)


class _FallbackToCpu(Exception):  # noqa: N818
    """Raised inside `_fit_gpu` to leave the GPU block, release its memory, and fit on the CPU instead."""


def _gib(size: int) -> str:
    return f"{size / 2**30:.2f} GiB"


def _memory_preflight(
    e_means: Any, torch: Any, em: Any, *, n: int, dtype: Any, subsampled: bool
) -> tuple[int | None, int | None]:
    """
    Ask e-means how much GPU memory training needs and how much is free.

    :returns: `(estimate, free)` in bytes, or `(None, None)` when e-means has no `memory_estimate` (upstream
        faiss), when training runs on a subset the estimate does not cover, or when the estimate fails: the
        pre-flight never breaks a fit that would have run.
    """
    estimate_fn = getattr(e_means, "memory_estimate", None)
    if estimate_fn is None:
        return None, None
    if subsampled:
        logger.debug("GPU memory estimate skipped: it does not cover the max_points_per_centroid subset copy")
        return None, None
    try:
        # include_input stays at its default (True): the host-to-GPU copy of the rows has not happened yet.
        # The fp16 path briefly holds the float32 copy plus the half copy, which the estimate does not count.
        estimate = int(estimate_fn(em, n, dtype, "cuda"))
        free = int(torch.cuda.mem_get_info()[0])
    except Exception as error:
        logger.debug("GPU memory estimate unavailable: {error}", error=error)
        return None, None
    return estimate, free


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
    allow_cpu_fallback: bool = False,
    init_centroids: np.ndarray | None = None,
) -> FitResult:
    torch = importlib.import_module("torch")
    e_means = gpu_engine()
    train_precision, warning = resolve_precision(backend, precision)
    if warning is not None:
        logger.warning("{warning}", warning=warning)
    if train_precision == "fp16" and x.size and float(np.abs(x).max()) > float(np.finfo(np.float16).max):
        # e-means does not check GPU data for infinity, so a value outside the FP16 range would train silently.
        logger.warning("embeddings exceed the fp16 range; using fp32")
        train_precision = "fp32"
    n, d = x.shape
    x_gpu = None
    init_gpu = None
    em = None
    centroids = None
    fallback_reason: str | None = None
    estimate: int | None = None
    try:
        engine = e_means.QuantizedEmeans if train_precision == "fp32" else e_means.Emeans
        extra: dict[str, Any] = {"quantizer": None} if train_precision == "fp32" else {}
        em = engine(
            d,
            k,
            nepochs=nepochs,
            seed=seed,
            spherical=spherical,
            max_points_per_centroid=max_points_per_centroid,
            capture_assignments=False,
            assignment_backend=backend.requested,
            **extra,
        )
        # bf16 runs pass float32 rows to an automatic Emeans, which memory_estimate costs as the BF16 cast.
        estimate, free = _memory_preflight(
            e_means,
            torch,
            em,
            n=n,
            dtype=torch.float16 if train_precision == "fp16" else torch.float32,
            subsampled=_n_train(n, k, max_points_per_centroid) < n,
        )
        if estimate is not None and free is not None and estimate > free:
            if allow_cpu_fallback:
                fallback_reason = (
                    f"GPU memory estimate for training ({_gib(estimate)}) exceeds free GPU memory "
                    f"({_gib(free)}); using the CPU backend"
                )
                logger.warning("{reason}", reason=fallback_reason)
            else:
                logger.info(
                    "GPU memory estimate for training ({estimate}) exceeds free GPU memory ({free}); "
                    "training on the GPU anyway because device='cuda'",
                    estimate=_gib(estimate),
                    free=_gib(free),
                )
        elif estimate is not None and free is not None:
            logger.debug(
                "GPU memory estimate for training {estimate}, free {free}", estimate=_gib(estimate), free=_gib(free)
            )
        if fallback_reason is not None:
            raise _FallbackToCpu
        x_gpu = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to("cuda")
        if train_precision == "fp16":
            x_gpu = x_gpu.half()
        if init_centroids is not None:
            # Starting centroids stay FP32 even when the data trains in FP16.
            init_gpu = torch.from_numpy(np.ascontiguousarray(init_centroids, dtype=np.float32)).to("cuda")
        with _deterministic(torch, deterministic):
            start = time.perf_counter()
            centroids, _ = em.train(x_gpu, init_centroids=init_gpu)
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
            "backend_hint": _backend_hint(
                is_rocm=torch.version.hip is not None,
                assignment_backend=label,
                emeans_rocm=emeans_rocm_importable(),
            ),
            "memory_estimate_bytes": estimate,
            "fallback_reason": backend.fallback_reason,
        }
        return FitResult(centroids=result, info=info)
    except _FallbackToCpu:
        pass
    finally:
        # Drop every GPU tensor before empty_cache(), or the centroids' memory stays allocated.
        del x_gpu, em, centroids, init_gpu
        torch.cuda.empty_cache()
    # The GPU memory is released by now, so the CPU fit does not compete with the dropped engine for it.
    fit = _fit_cpu(
        x,
        k=k,
        spherical=spherical,
        nepochs=nepochs,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        backend=BackendChoice("cpu", "cpu", fallback_reason),
        precision="fp32",
        init_centroids=init_centroids,
    )
    fit.info["memory_estimate_bytes"] = estimate
    return fit


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


class _GpuAssigner:
    """FP32 assignment on the GPU; reduced precision is never used for cluster ids."""

    def __init__(self, centroids: np.ndarray, *, spherical: bool, requested: str) -> None:
        faiss_import.check()
        self._torch = importlib.import_module("torch")
        e_means = gpu_engine()
        k, d = centroids.shape
        self._em = e_means.QuantizedEmeans(d, k, spherical=spherical, quantizer=None, assignment_backend=requested)
        self._em.centroids = self._torch.from_numpy(np.ascontiguousarray(centroids, dtype=np.float32)).to("cuda")

    def assign(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chunk = self._torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to("cuda")
        try:
            score, labels = self._em.assign(chunk)
            return (
                score.float().cpu().numpy().astype(np.float32),
                labels.cpu().numpy().astype(np.int64),
            )
        finally:
            del chunk


def assigner_for(backend: BackendChoice, centroids: np.ndarray, *, spherical: bool) -> Assigner:
    """
    Build an assigner for saved centroids on the chosen backend.

    :param backend: Backend chosen by `select_backend`; a CPU choice (`requested` is None) gives the CPU assigner.
    :param centroids: float32 centroids of shape `(k, d)`.
    :param spherical: Whether the model clusters by cosine similarity (otherwise squared L2 distances).
    :returns: An assigner whose `assign` returns `(score, labels)`; GPU assignment is always FP32.
    """
    if backend.requested is None:
        return cpu_assigner(centroids, spherical=spherical)
    return _GpuAssigner(centroids, spherical=spherical, requested=backend.requested)
