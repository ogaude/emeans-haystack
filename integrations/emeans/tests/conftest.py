# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import numpy as np
import pytest
from haystack import Document


@pytest.fixture
def make_docs() -> Callable[..., list[Document]]:
    """Return a factory for documents whose embeddings form well separated clusters."""

    def _make(n: int, d: int = 16, n_clusters: int = 4, seed: int = 0, unit: bool = True) -> list[Document]:
        rng = np.random.default_rng(seed)
        centres = rng.normal(size=(n_clusters, d))
        rows = centres[np.arange(n) % n_clusters] + 0.05 * rng.normal(size=(n, d))
        if unit:
            rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        return [Document(content=f"doc {i}", embedding=row.tolist()) for i, row in enumerate(rows)]

    return _make
