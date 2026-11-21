# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterator

import numpy as np
import pytest
from haystack import Document

from haystack_integrations.components.preprocessors.emeans import _backend
from haystack_integrations.components.preprocessors.emeans._backend import EnvInfo

NO_GPU = EnvInfo(
    torch_importable=False,
    gpu_available=False,
    is_rocm=False,
    fork_detected=False,
    device_name=None,
    torch_version=None,
    hip_version=None,
)


@pytest.fixture(autouse=True)
def _isolate_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start every test with an empty probe cache and, unless it is a gpu test, an environment without a GPU."""
    _backend.clear_probe_cache()
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(_backend, "detect_environment", lambda: NO_GPU)
    yield
    _backend.clear_probe_cache()


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
