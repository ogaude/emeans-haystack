# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import sys
import time

import numpy as np

from metrics import drift_metrics, pairwise_sqdist


def _centroids(k: int = 12, d: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((k, d)) * 5.0


def test_identity() -> None:
    c = _centroids()
    labels = np.arange(100) % len(c)
    out = drift_metrics(c, c, labels, labels)
    assert out["drift_gap_median"] == 0
    assert out["drift_gap_max"] == 0
    assert out["drift_rel_max"] == 0
    assert out["share_same_cluster"] == 1.0
    assert out["n_gap_gt_0p1"] == 0


def test_permutation_invariant() -> None:
    ref = _centroids()
    perm = np.random.default_rng(1).permutation(len(ref))
    c = ref[perm]  # c[i] is ref[perm[i]]
    labels_ref = np.arange(120) % len(ref)
    inverse = np.argsort(perm)  # ref id j is c id inverse[j]
    labels_c = inverse[labels_ref]
    out = drift_metrics(c, ref, labels_c, labels_ref)
    assert out["drift_gap_max"] == 0
    assert out["share_same_cluster"] == 1.0


def test_known_shift() -> None:
    ref = _centroids()
    rd = np.sqrt(pairwise_sqdist(ref, ref))
    np.fill_diagonal(rd, np.inf)
    nearest = rd.min(1)
    c = ref.copy()
    direction = np.zeros(ref.shape[1])
    direction[0] = 1.0
    c[3] = ref[3] + direction * 0.5 * nearest[3]
    labels = np.arange(50) % len(ref)
    out = drift_metrics(c, ref, labels, labels)
    assert abs(out["drift_gap_max"] - 0.5) < 1e-9
    assert out["n_gap_gt_0p1"] == 1


def test_label_agreement_partial() -> None:
    c = _centroids()
    labels_ref = np.arange(1000) % len(c)
    labels_c = labels_ref.copy()
    labels_c[:100] = (labels_c[:100] + 1) % len(c)
    out = drift_metrics(c, c, labels_c, labels_ref)
    assert abs(out["share_same_cluster"] - 0.9) < 1e-12


def test_duplicate_reference_guard() -> None:
    ref = _centroids()
    ref[1] = ref[0]
    labels = np.arange(60) % len(ref)
    out = drift_metrics(ref.copy(), ref, labels, labels)
    for key in ("drift_gap_median", "drift_gap_max", "drift_rel_median", "drift_rel_max"):
        assert np.isfinite(out[key])
    assert out["n_zero_gap_divisors"] == 2


def test_pairwise_blocks() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((300, 16))
    b = rng.standard_normal((200, 16))
    expected = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    got = pairwise_sqdist(a, b, block=64)
    assert got.dtype == np.float64
    assert np.allclose(got, expected, atol=1e-9)
    assert (got >= 0).all()


def test_k4096_runtime() -> None:
    rng = np.random.default_rng(4)
    ref = rng.standard_normal((4096, 768)).astype(np.float32)
    c = (ref + 0.01 * rng.standard_normal(ref.shape)).astype(np.float32)
    labels = rng.integers(0, 4096, 5000)
    begin = time.perf_counter()
    drift_metrics(c, ref, labels, labels)
    elapsed = time.perf_counter() - begin
    sys.stderr.write(f"k=4096 drift_metrics seconds: {elapsed:.1f}\n")
    assert elapsed < 120
