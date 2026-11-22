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
    args = parser.parse_args()

    method = METHODS[args.method]
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
    warm_s: float | None = None
    warm_status = "skipped (cost)"
    if args.warm == "yes":
        begin = time.perf_counter()
        method.train(x, init, **kwargs)
        warm_s = time.perf_counter() - begin
        warm_status = "measured"

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    passes_tag = "default" if args.passes is None else f"{args.passes:g}"
    path = scratch / f"{args.method}-{args.data}-n{args.n}-k{args.k}-s{args.seed}-p{passes_tag}.npy"
    np.save(path, result.centroids, allow_pickle=False)

    versions = _versions()
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
        "transfer_s": None,
        "backend": result.info.get("backend", method.engine),
        "precision": result.info.get("precision_used", "fp32"),
        "centroids_path": str(path),
        "info": result.info,
        "versions": versions,
        "host": _host(),
        "data_info": data_info,
    }
    sys.stdout.write(json.dumps(record, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
