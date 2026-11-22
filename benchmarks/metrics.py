# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Centroid drift between two centroid sets after one-to-one (Hungarian) matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment

_BLOCK = 1024
BIG_GAP = 0.1


def pairwise_sqdist(a: np.ndarray, b: np.ndarray, block: int = _BLOCK) -> np.ndarray:
    """
    Squared Euclidean distances between every row of `a` and every row of `b`, in float64.

    Uses the norm expansion in row blocks, so no `(len(a), len(b), d)` array is ever built.

    :param a: Matrix of shape `(p, d)`.
    :param b: Matrix of shape `(q, d)`.
    :param block: Rows of `a` per matrix product.
    :returns: Float64 matrix of shape `(p, q)`, clipped at 0.
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    a_norm = (a64 * a64).sum(axis=1)
    b_norm = (b64 * b64).sum(axis=1)
    out = np.empty((len(a64), len(b64)), dtype=np.float64)
    for lo in range(0, len(a64), block):
        part = a64[lo : lo + block]
        out[lo : lo + block] = np.maximum(a_norm[lo : lo + block, None] + b_norm[None, :] - 2.0 * (part @ b64.T), 0.0)
    return out


def drift_metrics(
    c: np.ndarray, ref: np.ndarray, labels_c: np.ndarray, labels_ref: np.ndarray
) -> dict[str, float | int]:
    """
    Measure how far the centroids `c` moved from the reference `ref`, after matching them one to one.

    `drift_gap` divides each centroid's move by the distance from its matched reference centroid to the nearest other
    reference centroid, so 1.0 means it moved as far as the next cluster. `drift_rel` divides by the reference norm.
    Reference centroids with a duplicate (nearest distance 0) are left out of the gap statistics and counted.

    :param c: Centroids under test, shape `(k, d)`.
    :param ref: Reference centroids, shape `(k, d)`.
    :param labels_c: Per-point labels from the shared scorer for `c`.
    :param labels_ref: Per-point labels from the shared scorer for `ref`.
    :returns: The drift statistics and the share of points whose cluster is the same after matching.
    """
    c64 = np.asarray(c, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    rows, cols = linear_sum_assignment(pairwise_sqdist(c64, ref64))
    m = np.empty(len(c64), dtype=np.int64)
    m[rows] = cols  # c[i] <-> ref[m[i]]
    ref_m = ref64[m]
    diff = np.linalg.norm(c64 - ref_m, axis=1)
    rel = diff / np.maximum(np.linalg.norm(ref_m, axis=1), 1e-300)
    rd = np.sqrt(pairwise_sqdist(ref64, ref64))
    np.fill_diagonal(rd, np.inf)
    nearest = rd.min(axis=1)[m]
    usable = nearest > 0
    gap = diff[usable] / nearest[usable]
    share = float(np.mean(m[np.asarray(labels_c)] == np.asarray(labels_ref)))
    return {
        "drift_gap_median": float(np.median(gap)) if len(gap) else 0.0,
        "drift_gap_max": float(gap.max()) if len(gap) else 0.0,
        "drift_rel_median": float(np.median(rel)),
        "drift_rel_max": float(rel.max()),
        "share_same_cluster": share,
        "n_gap_gt_0p1": int((gap > BIG_GAP).sum()),
        "n_zero_gap_divisors": int((~usable).sum()),
    }
