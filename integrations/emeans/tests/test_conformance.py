# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter
from haystack_integrations.components.rankers.emeans import EmeansClusterRanker

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


ALLOWED_ENV_VARS = {
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
}
ENV_METHODS = {"get", "pop", "setdefault"}


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _env_var_literals() -> set[str]:
    found: set[str] = set()
    for path in sorted((INTEGRATION_ROOT / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            arg: ast.AST | None = None
            if isinstance(node, ast.Subscript) and _is_os_environ(node.value):
                arg = node.slice
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                is_environ_method = _is_os_environ(node.func.value) and node.func.attr in ENV_METHODS
                is_getenv = (
                    node.func.attr == "getenv" and isinstance(node.func.value, ast.Name) and node.func.value.id == "os"
                )
                if (is_environ_method or is_getenv) and node.args:
                    arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.add(arg.value)
    return found


def _pyproject() -> dict:
    return tomllib.loads((INTEGRATION_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_no_own_env_vars() -> None:
    found = _env_var_literals()
    assert found, "the environment scan found nothing, so its patterns are dead"
    unexpected = sorted(found - ALLOWED_ENV_VARS)
    assert not unexpected, f"unexpected environment variables read or set: {unexpected}"


def test_no_run_async() -> None:
    for component in (EmeansFitter, EmeansDocumentClusterer, EmeansClusterRanker):
        assert not hasattr(component, "run_async"), f"{component.__name__} must stay synchronous"


def test_no_console_scripts() -> None:
    project = _pyproject()["project"]
    assert "scripts" not in project
    assert "gui-scripts" not in project
    assert not list((INTEGRATION_ROOT / "src").rglob("__main__.py"))


def test_numpy_floor() -> None:
    assert "numpy>=2.0" in _pyproject()["project"]["dependencies"]


def test_license_copyright() -> None:
    integration_license = (INTEGRATION_ROOT / "LICENSE.txt").read_bytes()
    root_license = (INTEGRATION_ROOT.parent.parent / "LICENSE").read_bytes()
    assert integration_license == root_license
    text = integration_license.decode("utf-8")
    assert "[yyyy]" not in text
    assert "[name of copyright owner]" not in text
