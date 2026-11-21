# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np
    from haystack import Document  # haystack itself probes for torch when it is imported, so it comes first

    before = "torch" in sys.modules
    ATTEMPTS = []

    class BlockTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                ATTEMPTS.append(name)
                raise ImportError("torch is blocked in this process")
            return None

    sys.meta_path.insert(0, BlockTorch())

    from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter
    from haystack_integrations.components.rankers.emeans import EmeansClusterRanker

    rng = np.random.default_rng(0)
    centres = rng.normal(size=(2, 8))
    rows = centres[np.arange(40) % 2] + 0.05 * rng.normal(size=(40, 8))
    docs = [Document(content=str(i), embedding=row.tolist()) for i, row in enumerate(rows)]

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "m.npz")
        stats = EmeansFitter(model_path=path, k=2).run(documents=docs)["stats"]
        assert stats["k"] == 2
        stats = EmeansFitter(model_path=path, k=2, device="cpu", overwrite=True).run(documents=docs)["stats"]
        assert stats["backend"] == "cpu"
        stats = EmeansFitter(model_path=path, k=2, device="auto", overwrite=True).run(documents=docs)["stats"]
        assert stats["backend"] == "cpu"
        out = EmeansDocumentClusterer(model_path=path).run(documents=docs)["documents"]
        assert all("cluster_id" in d.meta for d in out)
        fresh = [Document(content=d.content, embedding=d.embedding, score=0.5) for d in docs]
        ranked = EmeansClusterRanker(model_path=path).run(documents=fresh)["documents"]
        assert len(ranked) == 2

    assert ATTEMPTS == [], ATTEMPTS
    assert "torch" not in sys.modules or before, "torch was imported"
    """
)

SCRIPT_UNBLOCKED = textwrap.dedent(
    """
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np
    from haystack import Document

    before = "torch" in sys.modules

    import numpy as np
    from haystack import Document

    from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter
    from haystack_integrations.components.rankers.emeans import EmeansClusterRanker

    rng = np.random.default_rng(0)
    centres = rng.normal(size=(2, 8))
    rows = centres[np.arange(40) % 2] + 0.05 * rng.normal(size=(40, 8))
    docs = [Document(content=str(i), embedding=row.tolist()) for i, row in enumerate(rows)]

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "m.npz")
        stats = EmeansFitter(model_path=path, k=2).run(documents=docs)["stats"]
        assert stats["k"] == 2
        stats = EmeansFitter(model_path=path, k=2, device="cpu", overwrite=True).run(documents=docs)["stats"]
        assert stats["backend"] == "cpu"
        stats = EmeansFitter(model_path=path, k=2, device="auto", overwrite=True).run(documents=docs)["stats"]
        assert stats["backend"] == "cpu"
        out = EmeansDocumentClusterer(model_path=path).run(documents=docs)["documents"]
        assert all("cluster_id" in d.meta for d in out)
        fresh = [Document(content=d.content, embedding=d.embedding, score=0.5) for d in docs]
        ranked = EmeansClusterRanker(model_path=path).run(documents=fresh)["documents"]
        assert len(ranked) == 2

    assert "torch" not in sys.modules or before, "our code imported torch"
    """
)


def _run(script: str, cwd: Path) -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=cwd, check=False, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpu_fit_and_assign_never_import_torch(tmp_path: Path) -> None:
    _run(SCRIPT, tmp_path)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_cpu_paths_never_import_installed_torch(tmp_path: Path) -> None:
    _run(SCRIPT_UNBLOCKED, tmp_path)
