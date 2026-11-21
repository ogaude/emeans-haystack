# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys
    import tempfile
    from pathlib import Path

    class BlockTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is blocked in this process")
            return None

    sys.meta_path.insert(0, BlockTorch())

    import numpy as np
    from haystack import Document

    from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter
    from haystack_integrations.components.rankers.emeans import EmeansClusterRanker  # noqa: F401

    rng = np.random.default_rng(0)
    centres = rng.normal(size=(2, 8))
    rows = centres[np.arange(40) % 2] + 0.05 * rng.normal(size=(40, 8))
    docs = [Document(content=str(i), embedding=row.tolist()) for i, row in enumerate(rows)]

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "m.npz")
        stats = EmeansFitter(model_path=path, k=2).run(documents=docs)["stats"]
        assert stats["k"] == 2
        out = EmeansDocumentClusterer(model_path=path).run(documents=docs)["documents"]
        assert all("cluster_id" in d.meta for d in out)

    assert "torch" not in sys.modules, "torch was imported"
    """
)


def test_cpu_fit_and_assign_never_import_torch(tmp_path: Path) -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, cwd=tmp_path, check=False, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
