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
# Methods of the environment mapping whose first argument is one variable name.
ENV_KEY_METHODS = {"get", "pop", "setdefault", "__getitem__", "__contains__", "__delitem__"}
# Functions of `os` whose first argument is one variable name.
OS_KEY_FUNCTIONS = {"getenv", "putenv", "unsetenv"}
# Names that expose the process environment when imported from `os`.
OS_ENV_NAMES = {"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"}
# The parameter name under which the code injects `os.environ` (`hidden_gpu_reason(..., environ=os.environ)`); every
# `environ` name is scanned as the environment.
INJECTED_ENVIRON = "environ"


class _EnvScanner(ast.NodeVisitor):
    """
    Collect every environment variable the scanned code reads or writes, and every access it cannot name.

    The environment is `os.environ` (or `os.environb`) and any name `environ` (the injected mapping). A key that is a
    loop variable over a literal tuple of strings resolves to those strings; any other non-literal key is an offender,
    as are bulk or unknown uses of the environment and `from os import environ` style imports.
    """

    def __init__(self, where: str) -> None:
        self.where = where
        self.found: set[str] = set()
        self.offenders: list[str] = []
        self._loop_literals: list[dict[str, set[str]]] = [{}]
        self._handled: set[int] = set()

    def _offend(self, node: ast.AST, what: str) -> None:
        self.offenders.append(f"{self.where}:{getattr(node, 'lineno', '?')}: {what}")

    @staticmethod
    def _is_environ(node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "environb"}:
            return isinstance(node.value, ast.Name) and node.value.id == "os"
        return isinstance(node, ast.Name) and node.id == INJECTED_ENVIRON

    def _key(self, node: ast.AST | None, owner: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            self.found.add(node.value)
            return
        if isinstance(node, ast.Name):
            for scope in reversed(self._loop_literals):
                if node.id in scope:
                    self.found.update(scope[node.id])
                    return
        self._offend(owner, f"environment key is not a literal: {ast.unparse(node) if node else '<none>'}")

    def visit_For(self, node: ast.For) -> None:
        literals: dict[str, set[str]] = {}
        if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Tuple | ast.List):
            values = [e.value for e in node.iter.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if len(values) == len(node.iter.elts):
                literals[node.target.id] = set(values)
        self.visit(node.target)
        self.visit(node.iter)
        self._loop_literals.append(literals)
        for child in node.body:
            self.visit(child)
        self._loop_literals.pop()
        for child in node.orelse:
            self.visit(child)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "os":
            for alias in node.names:
                if alias.name in OS_ENV_NAMES or alias.name == "*":
                    self._offend(node, f"`from os import {alias.name}` hides environment access from this scan")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "os" and alias.asname not in {None, "os"}:
                self._offend(node, f"`import os as {alias.asname}` hides environment access from this scan")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._is_environ(node.value):
            self._handled.add(id(node.value))
            self._key(node.slice, node)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        left = node.left
        for op, right in zip(node.ops, node.comparators, strict=True):
            if isinstance(op, ast.In | ast.NotIn) and self._is_environ(right):
                self._handled.add(id(right))
                self._key(left, node)
            left = right
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and self._is_environ(func.value):
            self._handled.add(id(func.value))
            if func.attr in ENV_KEY_METHODS:
                self._key(node.args[0] if node.args else None, node)
            else:
                self._offend(node, f"bulk or unknown environment access: {ast.unparse(node)}")
        elif (
            isinstance(func, ast.Attribute)
            and func.attr in OS_KEY_FUNCTIONS
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            self._key(node.args[0] if node.args else None, node)
        for keyword in node.keywords:
            if keyword.arg == INJECTED_ENVIRON and self._is_environ(keyword.value):
                self._handled.add(id(keyword.value))  # read through the `environ` parameter, scanned there
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._is_environ(node) and id(node) not in self._handled:
            self._offend(node, f"unscanned use of the environment: {ast.unparse(node)}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if self._is_environ(node) and id(node) not in self._handled:
            self._offend(node, f"unscanned use of the environment: {node.id}")


def _scan_env(source: str, where: str = "<source>") -> tuple[set[str], list[str]]:
    scanner = _EnvScanner(where)
    scanner.visit(ast.parse(source))
    return scanner.found, scanner.offenders


def _env_var_literals() -> tuple[set[str], list[str]]:
    found: set[str] = set()
    offenders: list[str] = []
    for path in sorted((INTEGRATION_ROOT / "src").rglob("*.py")):
        names, problems = _scan_env(path.read_text(encoding="utf-8"), str(path))
        found |= names
        offenders += problems
    return found, offenders


def _pyproject() -> dict:
    return tomllib.loads((INTEGRATION_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_no_own_env_vars() -> None:
    found, offenders = _env_var_literals()
    assert not offenders, "environment accesses the scan cannot name:\n" + "\n".join(offenders)
    unexpected = sorted(found - ALLOWED_ENV_VARS)
    assert not unexpected, f"unexpected environment variables read or set: {unexpected}"
    stale = sorted(ALLOWED_ENV_VARS - found)
    assert not stale, f"allowlisted environment variables no longer used, drop them: {stale}"


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ('import os\nos.environ["X"]', "X"),
        ('import os\nos.environ.get("X")', "X"),
        ('import os\nos.environ.pop("X", None)', "X"),
        ('import os\nos.environ.setdefault("X", "1")', "X"),
        ('import os\nos.getenv("X")', "X"),
        ('import os\nos.putenv("X", "1")', "X"),
        ('import os\nif "X" in os.environ: pass', "X"),
        ('import os\nif "X" not in os.environ: pass', "X"),
        ('def f(environ):\n    return "X" in environ', "X"),
        ('def f(environ):\n    return environ["X"]', "X"),
        ('def f(environ):\n    for n in ("X",):\n        environ[n]', "X"),
    ],
)
def test_env_scan_finds(source: str, name: str) -> None:
    found, offenders = _scan_env(source)
    assert found == {name}
    assert not offenders


@pytest.mark.parametrize(
    "source",
    [
        "import os\nNAME = 'X'\nos.environ.get(NAME)",
        "import os\nos.environ.update(X='1')",
        "import os\ndict(os.environ)",
        "from os import environ",
        "from os import getenv",
        "import os as system",
    ],
)
def test_env_scan_flags(source: str) -> None:
    _, offenders = _scan_env(source)
    assert offenders


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
