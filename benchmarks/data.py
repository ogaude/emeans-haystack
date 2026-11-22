# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Datasets and the shared starting centroids for the benchmarks."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import fetch_data

RANDOM_DATA_SEED = 20261122
COHERE_SUBSET_SEED = 20261105
COHERE_ROWS = 1_000_000
COHERE_D = 768
_UNIT_CHECK_ROWS = 10_000
_UNIT_TOLERANCE = 1e-3
_CHUNK = 65536


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset: whether it clusters by cosine, how to load it, and whether it is available here."""

    name: str
    spherical: bool
    load: Callable[[int, int], tuple[np.ndarray, dict]]
    available: Callable[[], str | None]


def _load_random(n: int, d: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(RANDOM_DATA_SEED + n)
    x = np.empty((n, d), dtype=np.float32)
    for lo in range(0, n, _CHUNK):
        hi = min(lo + _CHUNK, n)
        x[lo:hi] = rng.standard_normal((hi - lo, d), dtype=np.float32)
    return x, {"data_id": "random-normal", "data_seed": RANDOM_DATA_SEED + n, "numpy": np.__version__}


def _cohere_path() -> Path:
    return fetch_data.CACHE_DIR / fetch_data.FILENAME


def _load_f32(path: Path, rows_total: int, d: int, n: int, normalise: bool = False) -> tuple[np.ndarray, dict]:
    """
    Read `n` rows of a headerless float32 matrix, as a seeded sorted subset unless `n` is every row.

    :param path: The `.f32` file.
    :param rows_total: Rows the file must hold.
    :param d: Row width.
    :param n: Rows to return.
    :param normalise: Scale every row to unit norm after reading (the raw norm range is recorded). Without it the
        rows must already be unit-norm.
    :returns: A contiguous float32 matrix and a provenance dict recording the subset rule.
    :raises ValueError: If the file size is wrong, `n` is out of range or the rows are not unit-norm.
    """
    expected = rows_total * d * 4
    actual = path.stat().st_size
    if actual != expected:
        msg = f"{path} has {actual} bytes, expected {expected} ({rows_total} rows x {d} x 4)"
        raise ValueError(msg)
    if not 0 < n <= rows_total:
        msg = f"n must be in 1..{rows_total} rows, got {n}"
        raise ValueError(msg)
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(rows_total, d))
    rows_sha: str | None = None
    seed: int | None = None
    if n < rows_total:
        seed = COHERE_SUBSET_SEED
        rows = np.sort(np.random.default_rng(seed).choice(rows_total, n, replace=False))
        x = np.ascontiguousarray(mm[rows], dtype=np.float32)
        rows_sha = hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest()
    else:
        x = np.array(mm, dtype=np.float32)
    raw_norms = np.linalg.norm(x[: min(_UNIT_CHECK_ROWS, n)], axis=1)
    if normalise:
        for lo in range(0, n, _CHUNK):
            x[lo : lo + _CHUNK] = normalise_rows(x[lo : lo + _CHUNK])
    head = x[: min(_UNIT_CHECK_ROWS, n)]
    if np.any(np.abs(np.linalg.norm(head, axis=1) - 1.0) > _UNIT_TOLERANCE):
        msg = f"{path}: rows are not unit-norm (spherical mode needs unit vectors)"
        raise ValueError(msg)
    info = {
        "data_id": f"{fetch_data.DATASET}@{fetch_data.REVISION}",
        "sha256": fetch_data.SHA256,
        "subset_seed": seed,
        "subset_rows_sha256": rows_sha,
        "n": n,
        "rows_normalised": normalise,
        "raw_norm_min": float(raw_norms.min()),
        "raw_norm_max": float(raw_norms.max()),
    }
    return x, info


def _load_cohere(n: int, d: int) -> tuple[np.ndarray, dict]:
    if d != COHERE_D:
        msg = f"the cohere dataset is {COHERE_D}-dimensional, got d={d}"
        raise ValueError(msg)
    return _load_f32(_cohere_path(), COHERE_ROWS, COHERE_D, n, normalise=True)


def _cohere_available() -> str | None:
    path = _cohere_path()
    expected = COHERE_ROWS * COHERE_D * 4
    if not path.is_file():
        return f"{path} is missing; run `hatch run bench-cpu:run fetch_data.py`"
    if path.stat().st_size != expected:
        return f"{path} has the wrong size (expected {expected}); re-run fetch_data.py"
    return None


DATASETS: dict[str, DatasetSpec] = {
    "random": DatasetSpec(name="random", spherical=False, load=_load_random, available=lambda: None),
    "cohere": DatasetSpec("cohere", spherical=True, load=_load_cohere, available=_cohere_available),
}


def load_dataset(name: str, n: int, d: int = 768) -> tuple[np.ndarray, dict]:
    """
    Load `n` rows of dataset `name` as float32.

    :param name: A key of `DATASETS`.
    :param n: Number of rows.
    :param d: Dimension (used by synthetic datasets).
    :returns: The matrix and a provenance dict.
    """
    return DATASETS[name].load(n, d)


def starts(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Pick `k` distinct data rows as starting centroids, identical for every method.

    :param x: Data matrix.
    :param k: Number of starting centroids.
    :param seed: Seed of the draw.
    :returns: Float32 copy of shape `(k, d)`.
    """
    idx = np.sort(np.random.default_rng(seed).choice(len(x), k, replace=False))
    return np.ascontiguousarray(x[idx], dtype=np.float32).copy()


def normalise_rows(x: np.ndarray) -> np.ndarray:
    """
    Scale every row to unit norm.

    :param x: Data matrix.
    :returns: Float32 matrix with unit-norm rows.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)
