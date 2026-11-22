# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Run one benchmark cell in a fresh process and print one JSON record."""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from data import DATASETS, load_dataset, starts
from methods import METHODS


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _gpu_versions() -> dict[str, str | None]:
    import torch  # noqa: PLC0415

    return {"torch": str(torch.__version__), "hip": torch.version.hip}


def _gpu_name() -> str:
    import torch  # noqa: PLC0415

    return str(torch.cuda.get_device_name(0))


def _transfer_seconds(x: np.ndarray) -> float:
    """Time one host-to-GPU copy of `x`; the model copies the data itself on every call."""
    import torch  # noqa: PLC0415

    torch.cuda.synchronize()
    begin = time.perf_counter()
    tensor = torch.from_numpy(np.ascontiguousarray(x)).to("cuda")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    del tensor
    torch.cuda.empty_cache()
    return elapsed


def _versions() -> dict[str, str | None]:
    import faiss  # noqa: PLC0415

    try:
        import sklearn  # noqa: PLC0415

        sk: str | None = sklearn.__version__
    except ImportError:
        sk = None
    try:
        package: str | None = importlib.metadata.version("emeans-haystack")
    except importlib.metadata.PackageNotFoundError:
        package = None
    from haystack_integrations.components.preprocessors.emeans._backend import detect_emeans_source  # noqa: PLC0415

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": str(faiss.__version__),
        "sklearn": sk,
        "emeans_haystack": package,
        "emeans_source": detect_emeans_source(),
    }


def _host() -> dict[str, object]:
    return {
        "label": os.environ.get("BENCH_HOST_LABEL", "unlabelled"),
        "cpu": _cpu_model(),
        "cores": os.cpu_count(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def main() -> int:
    """Train one cell, time it cold (and warm), save the centroids and print the record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--data", required=True, choices=sorted(DATASETS))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, default=768)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--passes", type=float, default=None)
    parser.add_argument("--warm", choices=["yes", "no"], default="no")
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--clear-triton-cache", action="store_true")
    args = parser.parse_args()

    method = METHODS[args.method]
    is_gpu = method.device == "gpu"
    triton_cache = "persistent"
    if is_gpu and args.clear_triton_cache:
        # A fresh cache directory must be set before torch or Triton is imported.
        Path(args.scratch).mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-cache-", dir=args.scratch)
        triton_cache = "cleared"
    reason = method.available()
    if reason is not None:
        sys.stderr.write(f"method {args.method} unavailable: {reason}\n")
        return 1
    spec = DATASETS[args.data]
    x, data_info = load_dataset(args.data, args.n, args.d)
    init = starts(x, args.k, args.seed)
    kwargs = {"k": args.k, "spherical": spec.spherical, "seed": args.seed, "passes": args.passes}

    begin = time.perf_counter()
    result = method.train(x, init, **kwargs)
    cold_s = time.perf_counter() - begin
    # Taken after the cold call so torch initialisation and allocator warm-up stay in the cold time.
    transfer_s = _transfer_seconds(x) if is_gpu else None
    warm_s: float | None = None
    warm_status = "skipped (cost)"
    warm_result = None
    if args.warm == "yes":
        begin = time.perf_counter()
        warm_result = method.train(x, init, **kwargs)
        warm_s = time.perf_counter() - begin
        warm_status = "measured"
    info = dict(result.info)
    if transfer_s is not None:
        # fit_model copies the data to the GPU on every call, so the separately timed copy is subtracted.
        cold_s -= transfer_s
        if warm_s is not None:
            warm_s -= transfer_s
        info["train_seconds_cold"] = result.info.get("train_seconds")
        info["train_seconds_warm"] = warm_result.info.get("train_seconds") if warm_result is not None else None
        info["triton_cache"] = triton_cache

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    passes_tag = "default" if args.passes is None else f"{args.passes:g}"
    path = scratch / f"{args.method}-{args.data}-n{args.n}-k{args.k}-s{args.seed}-p{passes_tag}.npy"
    np.save(path, result.centroids, allow_pickle=False)

    versions = _versions()
    host = _host()
    if is_gpu:
        versions.update(_gpu_versions())
        host["gpu"] = _gpu_name()
    record = {
        "method": args.method,
        "data": args.data,
        "n": args.n,
        "d": args.d,
        "k": args.k,
        "seed": args.seed,
        "passes_requested": args.passes if args.passes is not None else method.default_passes,
        "passes_used": result.passes_used,
        "cold_s": cold_s,
        "warm_s": warm_s,
        "warm_status": warm_status,
        "transfer_s": transfer_s,
        "backend": result.info.get("backend", method.engine),
        "precision": result.info.get("precision_used", "fp32"),
        "centroids_path": str(path),
        "info": info,
        "versions": versions,
        "host": host,
        "data_info": data_info,
    }
    sys.stdout.write(json.dumps(record, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
