# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

INTEGRATION_ROOT = Path(__file__).resolve().parent.parent
SCANNED_DIRS = [
    INTEGRATION_ROOT / "src",
    INTEGRATION_ROOT / "tests",
    INTEGRATION_ROOT.parent.parent / "examples",
]
SPDX_HEADER = [
    "# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>",
    "#",
    "# SPDX-License-Identifier: Apache-2.0",
]
NAMESPACE_ROOT = INTEGRATION_ROOT / "src" / "haystack_integrations"
# Anchored on the first character so the scan skips its own pattern literal.
NAMING_RE = re.compile("k-?means", re.IGNORECASE)


def _python_files() -> Iterator[Path]:
    for directory in SCANNED_DIRS:
        for path in sorted(directory.rglob("*.py")):
            if ".output" in path.parts or ".venv" in path.parts:
                continue
            yield path


def test_scan_finds_files() -> None:
    files = list(_python_files())
    assert any(p.parent.name == "examples" for p in files)
    assert any("src" in p.parts for p in files)
    assert any(p.name == "test_conformance.py" for p in files)


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: p.name)
def test_spdx_headers(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:3] == SPDX_HEADER, f"{path} does not start with the SPDX header"


def test_emeans_naming_rule() -> None:
    offenders = []
    for path in _python_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if path.name == "test_conformance.py" and "NAMING_RE" in line:
                continue
            if NAMING_RE.search(line) and "Lloyd" not in line:
                offenders.append(f"{path}:{number}: {line.strip()}")
    message = "e-means is not Lloyd's algorithm; only Lloyd's may be named with the k- prefix:\n"
    assert not offenders, message + "\n".join(offenders)


def test_public_import_paths() -> None:
    preprocessors = importlib.import_module("haystack_integrations.components.preprocessors.emeans")
    rankers = importlib.import_module("haystack_integrations.components.rankers.emeans")
    assert sorted(preprocessors.__all__) == ["EmeansDocumentClusterer", "EmeansFitter"]
    assert sorted(rankers.__all__) == ["EmeansClusterRanker"]
    assert preprocessors.EmeansFitter.__name__ == "EmeansFitter"
    assert preprocessors.EmeansDocumentClusterer.__name__ == "EmeansDocumentClusterer"
    assert rankers.EmeansClusterRanker.__name__ == "EmeansClusterRanker"


def test_namespace_layout() -> None:
    components = NAMESPACE_ROOT / "components"
    for directory in (NAMESPACE_ROOT, components, components / "preprocessors", components / "rankers"):
        assert directory.is_dir(), directory
        assert not (directory / "__init__.py").exists(), f"{directory} must stay a namespace package"
    assert (components / "preprocessors" / "py.typed").is_file()
    assert (components / "rankers" / "py.typed").is_file()
