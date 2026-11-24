# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from methods import METHODS


def _unit_rows(n: int, d: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_sklearn_spherical_centroids_are_unit_norm() -> None:
    x = _unit_rows(400, 16, 0)
    result = METHODS["sklearn-lloyd"].train(x, x[:8].copy(), k=8, spherical=True, seed=0, passes=3)
    np.testing.assert_allclose(np.linalg.norm(result.centroids, axis=1), 1.0, rtol=1e-5)
    assert result.info["centroids_normalised"] is True


def test_sklearn_euclidean_centroids_are_plain_means() -> None:
    x = _unit_rows(400, 16, 1)
    result = METHODS["sklearn-lloyd"].train(x, x[:8].copy(), k=8, spherical=False, seed=0, passes=3)
    assert np.all(np.linalg.norm(result.centroids, axis=1) < 1.0)
    assert result.info["centroids_normalised"] is False
