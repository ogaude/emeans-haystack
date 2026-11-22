# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Datasets and the shared starting centroids for the benchmarks."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

RANDOM_DATA_SEED = 20261122
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


DATASETS: dict[str, DatasetSpec] = {
    "random": DatasetSpec(name="random", spherical=False, load=_load_random, available=lambda: None),
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
