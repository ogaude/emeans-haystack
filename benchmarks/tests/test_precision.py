# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import precision
from precision import PCell, precision_grid, precision_smoke_grid, prepare, run, run_cell


def test_precision_grid_full() -> None:
    cells = precision_grid(("random", "cohere"), seeds=(0, 1, 2))
    assert len(cells) == 48
    assert {c.n for c in cells} == {100_000, 1_000_000}
    assert {c.k for c in cells} == {1024, 4096}
    assert {c.mode for c in cells} == {"spherical", "euclidean"}
    assert len({c.key for c in cells}) == 48
    assert PCell(100_000, 1024, "random", "spherical", 2).key == "100000|1024|random|spherical|2"


def test_precision_grid_smoke() -> None:
    cells = precision_smoke_grid(("random", "cohere"))
    assert len(cells) == 4
    assert {(c.n, c.k, c.seed) for c in cells} == {(10_000, 256, 0)}
    assert {c.mode for c in cells} == {"spherical", "euclidean"}


def test_cell_key_parse_validates() -> None:
    cell = PCell(10_000, 256, "random", "euclidean", 1)
    assert PCell.parse(cell.key) == cell
    for bad in (
        "1|2|3",
        "10|2|nope|euclidean|0",
        "10|2|random|weird|0",
        "x|2|random|euclidean|0",
        "10|20|random|euclidean|0",
    ):
        with pytest.raises(ValueError):
            PCell.parse(bad)


def test_cell_data_prep(monkeypatch: pytest.MonkeyPatch) -> None:
    x, _, prep = prepare("random", 500, "spherical", 16)
    assert prep == "row-normalised"
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-5
    x, _, prep = prepare("random", 500, "euclidean", 16)
    assert prep == "raw"
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() > 0.5
    unit = np.random.default_rng(0).standard_normal((300, 16)).astype(np.float32)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    monkeypatch.setattr(precision.data_mod, "load_dataset", lambda _n, _s, _d=768: (unit, {"data_id": "fake"}))
    x, _, prep = prepare("cohere", 300, "euclidean")
    assert prep == "row-normalised (loader)"
    assert np.array_equal(x, unit)


def _fake_fit(*, fallback: bool = False, nonfinite: bool = False) -> Any:
    def fit(x: np.ndarray, init: np.ndarray, *, k: int, spherical: bool, seed: int, precision: str) -> Any:  # noqa: ARG001
        c = init.astype(np.float32) + (0.001 * (seed + 1)) * np.ones_like(init)
        if precision != "fp32":
            c = c + 1e-3
        used = "fp32" if fallback and precision == "bf16" else precision
        if nonfinite and precision == "fp16":
            c = c.copy()
            c[0, 0] = np.inf
        return c, {"precision_used": used, "backend": "fake", "train_seconds": 0.5}

    return fit


def _cell() -> PCell:
    return PCell(2000, 8, "random", "euclidean", 0)


def test_variant_assertions() -> None:
    record = run_cell(_cell(), fit=_fake_fit(fallback=True), d=16)
    assert record["variants"]["bf16"]["status"] == "precision_fallback"
    assert record["variants"]["fp16"]["status"] == "ok"
    record = run_cell(_cell(), fit=_fake_fit(nonfinite=True), d=16)
    assert record["variants"]["fp16"]["nonfinite"] is True
    assert record["variants"]["fp16"]["status"] == "nonfinite"
    assert record["variants"]["bf16"]["nonfinite"] is False


def test_record_shape_and_baselines() -> None:
    record = run_cell(_cell(), fit=_fake_fit(), d=16)
    assert set(record["variants"]) == {"fp32_rerun", "fp32_other_seed", "fp16", "bf16", "round16", "roundbf"}
    for entry in record["variants"].values():
        for key in ("drift_gap_median", "drift_gap_max", "share_same_cluster", "objective_delta_rel", "objective"):
            assert key in entry
    assert record["variants"]["fp32_rerun"]["drift_gap_max"] == 0.0
    assert record["variants"]["fp32_rerun"]["share_same_cluster"] == 1.0
    assert "_labels" not in record["variants"]["fp16"]
    assert record["prep"] == "raw"


def test_record_subset_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    record = run_cell(_cell(), fit=_fake_fit(), d=16)
    assert record["subset_seed"] is None
    assert record["subset_rows_sha256"] is None
    unit = np.random.default_rng(0).standard_normal((2000, 16)).astype(np.float32)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    info = {
        "data_id": "cohere@rev",
        "subset_seed": 20261105,
        "subset_rows_sha256": "cd" * 32,
        "rows_normalised": True,
        "raw_norm_min": 0.5,
        "raw_norm_max": 2.0,
    }
    monkeypatch.setattr(precision.data_mod, "load_dataset", lambda _n, _s, _d=768: (unit, info))
    record = run_cell(PCell(2000, 8, "cohere", "spherical", 0), fit=_fake_fit())
    assert record["data_id"] == "cohere@rev"
    assert record["subset_seed"] == 20261105
    assert record["subset_rows_sha256"] == "cd" * 32
    assert record["rows_normalised"] is True
    assert (record["raw_norm_min"], record["raw_norm_max"]) == (0.5, 2.0)


def test_bf16_rounding() -> None:
    values = np.array([1.0, 1.00390625, 3.14159265, -2.5e-3, 1e30], dtype=np.float32)
    rounded = precision._round_bf16(values)
    assert rounded[0] == 1.0
    assert np.all(np.abs(rounded - values) <= np.abs(values) * 2.0**-8)
    assert np.array_equal(precision._round_bf16(rounded), rounded)


class _FakeRunner:
    def __init__(self, behaviour: dict[str, Any]) -> None:
        self.behaviour = behaviour
        self.ran: list[str] = []

    def run_cell(self, cell: PCell) -> dict[str, Any]:
        self.ran.append(cell.key)
        what = self.behaviour.get(cell.key)
        if isinstance(what, Exception):
            raise what
        return {"key": cell.key, "status": "ok", "variants": {}}


def test_resume_and_failure_records(tmp_path: Path) -> None:
    cells = [PCell(100, 4, "random", "euclidean", s) for s in range(3)]
    results = tmp_path / "precision.jsonl"
    behaviour = {
        cells[1].key: precision.CellError("boom tail"),
        cells[2].key: subprocess.TimeoutExpired(["x"], 7200),
    }
    runner = _FakeRunner(behaviour)
    run(cells, results=results, runner=runner, clock=lambda: "t")
    lines = [__import__("json").loads(line) for line in results.read_text().splitlines()]
    assert [r["status"] for r in lines] == ["ok", "failed", "timed_out"]
    assert "boom tail" in lines[1]["reason"]
    assert "2.0 h" in lines[2]["reason"]
    again = _FakeRunner({})
    run(cells, results=results, runner=again, clock=lambda: "t")
    assert again.ran == []
    assert len(results.read_text().splitlines()) == 3


@pytest.mark.gpu
def test_precision_cell_gpu() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    if precision._gpu_refusal() is not None:
        pytest.skip("needs the bench-gpu-fork env")
    record = run_cell(PCell(20_000, 64, "random", "euclidean", 0), d=64)
    assert set(record["variants"]) == {"fp32_rerun", "fp32_other_seed", "fp16", "bf16", "round16", "roundbf"}
    for entry in record["variants"].values():
        for key in ("drift_gap_median", "drift_gap_max", "share_same_cluster", "objective_delta_rel"):
            assert key in entry
    assert record["variants"]["fp32_rerun"]["drift_gap_max"] == 0.0
    assert record["variants"]["fp32_rerun"]["share_same_cluster"] == 1.0
