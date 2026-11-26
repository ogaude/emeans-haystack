# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"
BENCHMARKS_SHA256 = "637e816498520e6d9fafdfb966e657c73f8a0513c4edb5ca731b2a013e7bf36c"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _quickstart_block() -> str:
    section = _readme().split("## Quick Start", 1)[1]
    match = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert match, "no python block under ## Quick Start"
    return match.group(1)


def test_readme_quickstart_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {}
    exec(compile(_quickstart_block(), "README.md", "exec"), namespace)  # noqa: S102  # our own tracked README
    assert namespace["stats"]["backend"] == "cpu"
    assert all(isinstance(doc.meta["cluster_id"], int) for doc in namespace["clustered"])
    diverse = namespace["diverse"]
    assert len(diverse) == 5
    assert len({doc.meta["cluster_id"] for doc in diverse}) == 5
    scores = [doc.score for doc in diverse]
    assert scores == sorted(scores, reverse=True)


def test_readme_quickstart_is_cpu_only() -> None:
    block = _quickstart_block()
    assert "torch" not in block
    assert "sentence_transformers" not in block
    assert "SentenceTransformers" not in block


def test_readme_citation_and_license() -> None:
    text = _readme()
    for needle in (
        "## Citation",
        "https://arxiv.org/abs/2611.09501",
        "bessa2026emeans",
        "## License",
        "Apache-2.0",
        "MIT",
        "emeans-rocm",
    ):
        assert needle in text, f"README lacks {needle!r}"


def test_readme_benchmarks_unchanged() -> None:
    text = _readme()
    start = text.index("## Benchmarks")
    end = text.index("## Contributing")
    assert hashlib.sha256(text[start:end].encode("utf-8")).hexdigest() == BENCHMARKS_SHA256


def test_readme_numpy_requirement() -> None:
    text = _readme()
    assert "numpy>=2.0" in text
    assert "numpy>=1.25" not in text
