# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import re
import tomllib
from pathlib import Path

# Reads only files inside integrations/emeans/, so the test moves upstream unchanged.
ROOT = Path(__file__).resolve().parents[1]


def test_gpu_fork_env_pins_emeans_rocm_0_2_9_exactly() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = config["tool"]["hatch"]["envs"]["gpu-fork"]["extra-dependencies"]
    rocm = [d for d in deps if re.match(r"emeans-rocm\b", d)]
    assert rocm == ["emeans-rocm==0.2.9"]


def test_changelog_documents_the_0_1_1_release() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## 0\.1\.1\s*$(.*?)(?=^## |\Z)", changelog, re.MULTILINE | re.DOTALL)
    assert match is not None, "CHANGELOG.md has no '## 0.1.1' heading"
    body = match.group(1)
    assert "emeans-rocm" in body
    assert "unused_threshold" in body
    # newest first: 0.1.1 is listed above 0.1.0
    assert changelog.index("## 0.1.1") < changelog.index("## 0.1.0")


def test_changelog_documents_the_0_1_2_release() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## 0\.1\.2\s*$(.*?)(?=^## |\Z)", changelog, re.MULTILINE | re.DOTALL)
    assert match is not None, "CHANGELOG.md has no '## 0.1.2' heading"
    body = match.group(1)
    assert "emeans-rocm" in body
    assert "unused_threshold" in body
    assert "PyPI" in body
    # newest first: 0.1.2 is listed above 0.1.1
    assert changelog.index("## 0.1.2") < changelog.index("## 0.1.1")
