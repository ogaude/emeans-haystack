# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from haystack import Document


def check_positive_int(name: str, value: object, *, allow_none: bool = False) -> None:
    """
    Raise `ValueError` unless `value` is an integer of at least 1 (or `None` when allowed).

    Booleans are rejected even though they are integers in Python.

    :param name: Parameter name used in the error message.
    :param value: Value to check.
    :param allow_none: Whether `None` is an accepted value.
    """
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer{' or None' if allow_none else ''}, got {value!r}"
        raise ValueError(msg)


def resolve_k(n_usable: int, k: int | None, target_cluster_size: int | None) -> int:
    """
    Return the number of clusters to fit.

    An explicit `k` is returned as is. Otherwise `k = max(1, min(round(n / target_cluster_size), n))`.

    :param n_usable: Number of documents with an embedding.
    :param k: Explicit number of clusters, or `None`.
    :param target_cluster_size: Average number of documents per cluster, used when `k` is `None`.
    """
    if k is not None:
        return k
    if target_cluster_size is None:
        msg = "Either k or target_cluster_size must be set"
        raise ValueError(msg)
    return max(1, min(round(n_usable / target_cluster_size), n_usable))


def extract_embeddings(
    documents: Sequence[Document], *, expected_dim: int | None = None
) -> tuple[list[int], np.ndarray]:
    """
    Collect the embeddings of the documents that have one into a new float32 matrix.

    :param documents: Input documents.
    :param expected_dim: If given, every embedding must have this dimension.
    :returns: The positions of the documents that have an embedding, and a C-contiguous float32 array of
        shape `(m, d)` holding their embeddings.
    :raises ValueError: On mixed or unexpected dimensions, or on NaN or infinite values.
    """
    positions: list[int] = []
    rows: list[Sequence[float]] = []
    dim: int | None = expected_dim
    first_dim: int | None = None
    for pos, doc in enumerate(documents):
        embedding = doc.embedding
        if embedding is None:
            continue
        length = len(embedding)
        if first_dim is None:
            first_dim = length
        elif length != first_dim:
            msg = (
                f"Embedding dimension mismatch: document {doc.id} has dimension {length}, "
                f"but other documents have dimension {first_dim}"
            )
            raise ValueError(msg)
        if expected_dim is not None and length != expected_dim:
            msg = f"Embedding dimension mismatch: document {doc.id} has dimension {length}, expected {expected_dim}"
            raise ValueError(msg)
        positions.append(pos)
        rows.append(embedding)

    if not rows:
        return positions, np.zeros((0, dim or 0), dtype=np.float32)

    x64 = np.asarray(rows, dtype=np.float64)
    _check_finite(x64, documents, positions)
    x32 = np.ascontiguousarray(x64.astype(np.float32))
    _check_finite(x32, documents, positions)
    return positions, x32


def _check_finite(x: np.ndarray, documents: Sequence[Document], positions: list[int]) -> None:
    bad_rows = np.flatnonzero(~np.isfinite(x).all(axis=1))
    if bad_rows.size:
        doc_id = documents[positions[int(bad_rows[0])]].id
        msg = f"Embedding of document {doc_id} contains NaN or infinite values"
        raise ValueError(msg)


def normalize_rows(x: np.ndarray, doc_ids: Sequence[str]) -> np.ndarray:
    """
    Return a new float32 array whose rows have unit L2 norm.

    Each row is first divided by its largest absolute value so that very large or very small values do not
    overflow or underflow in the norm.

    :param x: Matrix of shape `(m, d)`. It is not modified.
    :param doc_ids: Document id of every row, used in error messages.
    :raises ValueError: If a row is all zeros.
    """
    x64 = np.asarray(x, dtype=np.float64)
    scale = np.abs(x64).max(axis=1) if x64.shape[1] else np.zeros(x64.shape[0])
    zero_rows = np.flatnonzero(scale == 0)
    if zero_rows.size:
        msg = f"Document {doc_ids[int(zero_rows[0])]} has a zero embedding, which cannot be used with spherical=True"
        raise ValueError(msg)
    scaled = x64 / scale[:, None]
    scaled /= np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.ascontiguousarray(scaled.astype(np.float32))
