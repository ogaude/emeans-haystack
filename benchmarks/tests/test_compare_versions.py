# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest

import compare_versions as cv

METHOD = "emeans-gpu-triton-fp32"


def test_centroid_path() -> None:
    path = cv.centroid_path(Path("d"), METHOD, "cohere", 100000, 1024, 2)
    assert path == Path("d") / "emeans-gpu-triton-fp32-cohere-n100000-k1024-s2-pdefault.npy"


def test_gate_pass() -> None:
    g = cv.gate_setting([1.0, 1.2, 1.1], [1.05, 1.15, 1.3], [0.7, 0.8, 0.75], [0.6, 0.65, 0.7])
    assert g["new_objective_median"] == 1.15
    assert g["ok_objective"] is True
    assert g["ok_ari"] is True
    assert g["ok"] is True


def test_gate_objective_outside() -> None:
    g = cv.gate_setting([1.0, 1.2], [1.25, 1.25, 1.25], [0.9], [0.5])
    assert g["ok_objective"] is False
    assert g["ok"] is False


def test_gate_ari_below() -> None:
    g = cv.gate_setting([1.0, 1.2], [1.1], [0.6], [0.65])
    assert g["ok_objective"] is True
    assert g["ok_ari"] is False
    assert g["ok"] is False


def test_gate_boundaries_inclusive() -> None:
    assert cv.gate_setting([1.0, 1.2], [1.0], [0.5], [0.5])["ok"] is True
    assert cv.gate_setting([1.0, 1.2], [1.2], [0.5], [0.5])["ok"] is True


def _write_records(directory: Path, source: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"method": METHOD, "versions": {"emeans_source": source}})
    (directory / "records.jsonl").write_text(line + "\n")


def test_load_run_records_rejects_other_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_records(tmp_path, "emeans-rocm 0.2.6")
    with pytest.raises(SystemExit) as exc:
        cv.load_run_records(tmp_path, cv.OLD_SOURCE)
    assert exc.value.code == 2
    assert str(tmp_path / "records.jsonl") in capsys.readouterr().err
    assert cv.load_run_records(tmp_path, cv.NEW_SOURCE)[0]["method"] == METHOD


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_shift: float) -> list[str]:
    n, d, k = 200, 8, 4
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d)).astype(np.float32)
    monkeypatch.setattr(cv, "load_dataset", lambda *_args: (x, {}))
    monkeypatch.setattr(cv, "GATE_N", n)
    monkeypatch.setattr(cv, "D", d)
    monkeypatch.setattr(cv, "GATE_KS", (k,))
    monkeypatch.setattr(cv, "GATE_METHODS", (METHOD,))
    monkeypatch.setattr(cv, "GATE_DATA", ("random",))
    old, rerun, new = tmp_path / "old", tmp_path / "rerun", tmp_path / "new"
    _write_records(old, cv.OLD_SOURCE)
    _write_records(rerun, cv.OLD_SOURCE)
    _write_records(new, cv.NEW_SOURCE)
    grid_lines = []
    for seed in cv.SEEDS:
        cent = x[rng.choice(n, k, replace=False)]
        np.save(cv.centroid_path(old, METHOD, "random", n, k, seed), cent)
        np.save(cv.centroid_path(new, METHOD, "random", n, k, seed), cent + new_shift)
        if seed == cv.RERUN_SEED:
            np.save(cv.centroid_path(rerun, METHOD, "random", n, k, seed), cent)
        obj = cv.score(x, cent)[0]
        record = {
            "timing": "cold", "kind": "main", "passes_requested": None, "status": "ok", "method": METHOD,
            "data": "random", "n": n, "k": k, "seed": seed, "objective": obj,
            "versions": {"emeans_source": cv.OLD_SOURCE},
        }  # fmt: skip
        grid_lines.append(json.dumps(record))
    records = tmp_path / "speed-old.jsonl"
    records.write_text("\n".join(grid_lines) + "\n")
    return [
        "--old", str(old), "--rerun", str(rerun), "--new", str(new),
        "--old-records", str(records), "--out", str(tmp_path / "out.jsonl"),
    ]  # fmt: skip


def test_main_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _setup(tmp_path, monkeypatch, 0.0)
    monkeypatch.setattr("sys.argv", ["compare_versions", *argv])
    assert cv.main() == 0
    lines = (tmp_path / "out.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ok"] is True
    assert len(record["old_objectives"]) == 7
    assert "STOP" not in capsys.readouterr().out


def test_main_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _setup(tmp_path, monkeypatch, 50.0)
    monkeypatch.setattr("sys.argv", ["compare_versions", *argv])
    assert cv.main() == 3
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 1
    assert "STOP" in capsys.readouterr().out
