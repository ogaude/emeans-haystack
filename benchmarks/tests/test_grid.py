# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path
from typing import Any

import run_grid
from run_grid import Cell, WorkerError, full_grid, run, smoke_grid

METHODS = ["emeans-cpu", "faiss-lloyd", "sklearn-lloyd"]
SCHEMA_KEYS = {
    "schema", "key", "timing", "method", "engine", "device", "backend", "precision", "env", "n", "k", "n_x_k", "d",
    "data", "data_id", "mode", "seed", "kind", "passes_requested", "passes_used", "seconds", "transfer_s",
    "objective", "objective_sum", "status", "skipped_reason", "estimate_s", "estimate_basis", "host", "versions",
    "triton_cache", "timestamp",
}  # fmt: skip


class FakeRunner:
    def __init__(self, probe_s: float = 1.0, fail: Exception | None = None) -> None:
        self.probe_s = probe_s
        self.fail = fail
        self.probes: list[Cell] = []
        self.trains: list[tuple[Cell, bool]] = []

    def probe(self, cell: Cell) -> float:
        self.probes.append(cell)
        return self.probe_s

    def train(self, cell: Cell, *, warm: bool) -> dict[str, Any]:
        self.trains.append((cell, warm))
        if self.fail is not None:
            raise self.fail
        return {
            "cold_s": 2.0,
            "warm_s": 1.0 if warm else None,
            "warm_status": "measured" if warm else "skipped (cost)",
            "transfer_s": None,
            "backend": "fake",
            "precision": "fp32",
            "passes_used": 5.0,
            "centroids_path": "/nonexistent/c.npy",
            "info": {},
            "versions": {},
            "host": {},
            "data_info": {"data_id": "random-normal"},
        }

    def score(self, cell: Cell, _path: str) -> dict[str, Any]:
        return {"objective": 1.5, "objective_sum": 1.5 * cell.n, "n": cell.n, "k": cell.k}


def _run(cells: list[Cell], tmp_path: Path, runner: FakeRunner, cap_s: float | None = 3600.0) -> list[dict]:
    results = tmp_path / "speed.jsonl"
    run(cells, results=results, scratch=tmp_path / "scratch", cap_s=cap_s, runner=runner, clock=lambda: "t")
    return [json.loads(line) for line in results.read_text().splitlines()]


def _cell(method: str = "emeans-cpu", **kw: Any) -> Cell:
    base: dict[str, Any] = {"data": "random", "n": 10_000, "k": 256, "seed": 0, "kind": "main", "passes": None}
    base.update(kw)
    return Cell(method, **base)


def test_full_grid_shape() -> None:
    cells = [c for c in full_grid(METHODS, ["random"], seeds=(0, 1, 2)) if c.kind == "main"]
    assert len(cells) == 2 * 1 * 2 * 4 * 3 + 1 * 1 * 1 * 4 * 3
    assert all(c.passes is None for c in cells)
    sizes = {c.n * c.k for c in cells}
    assert min(sizes) == 1_024 * 100_000
    assert max(sizes) == 65_536 * 1_000_000
    assert any(s < 1e10 for s in sizes)
    assert any(s > 1e10 for s in sizes)


def test_sklearn_runs_at_100k_only() -> None:
    cells = full_grid(METHODS, ["random", "cohere"], seeds=(0, 1, 2))
    assert {c.n for c in cells if c.method == "sklearn-lloyd"} == {100_000}
    assert {c.n for c in cells if c.method == "faiss-lloyd"} == {100_000, 1_000_000}
    assert ("cohere", 1_000_000, 16384) in {(c.data, c.n, c.k) for c in cells if c.method == "faiss-lloyd"}


def test_sweep_cells() -> None:
    cells = [c for c in full_grid(METHODS, ["random"], seeds=(0, 1, 2)) if c.kind == "sweep"]
    assert {(c.data, c.n, c.k) for c in cells} == {("random", 100_000, 4096)}
    assert {c.passes for c in cells} == {1.0, 2.0, 5.0, 10.0, 25.0}
    assert len(cells) == 3 * 3 * 5


def test_smoke_grid() -> None:
    cells = smoke_grid(METHODS, ["random"])
    assert {(c.n, c.k, c.seed) for c in cells} == {(10_000, 256, 0)}
    assert len([c for c in cells if c.kind == "main"]) == 3
    assert {c.passes for c in cells if c.kind == "sweep"} == {1.0, 2.0}


def test_resume_skips_finished(tmp_path: Path) -> None:
    cells = [_cell("emeans-cpu"), _cell("faiss-lloyd"), _cell("sklearn-lloyd")]
    results = tmp_path / "speed.jsonl"
    first = cells[0]
    stale = {"key": f"{cells[1].key}|cold"}
    lines = [{"key": f"{first.key}|cold"}, {"key": f"{first.key}|warm"}, stale]
    results.write_text("".join(json.dumps(r) + "\n" for r in lines) + "{torn")
    runner = FakeRunner()
    run(cells, results=results, scratch=tmp_path / "s", cap_s=3600.0, runner=runner, clock=lambda: "t")
    assert [c.method for c, _ in runner.trains] == ["faiss-lloyd", "sklearn-lloyd"]


def test_cap_skips_with_record(tmp_path: Path) -> None:
    cell = _cell("faiss-lloyd")  # default 25 passes
    runner = FakeRunner(probe_s=120.0)  # 25 passes: 3000 s, over a 2400 s cap
    records = _run([cell], tmp_path, runner, cap_s=2400.0)
    assert runner.trains == []
    assert [r["timing"] for r in records] == ["cold", "warm"]
    assert {r["status"] for r in records} == {"skipped"}
    assert records[0]["skipped_reason"] == "not run: est. 0.8 h"
    runner2 = FakeRunner(probe_s=120.0)
    records = _run([cell], tmp_path / "b", runner2, cap_s=None)
    assert len(runner2.trains) == 1
    assert records[0]["status"] == "ok"


def test_extrapolated_skip(tmp_path: Path) -> None:
    small = _cell("faiss-lloyd", n=10_000, k=256)
    big = _cell("faiss-lloyd", n=1_000_000, k=65536)
    runner = FakeRunner(probe_s=1.0)
    records = _run([small, big], tmp_path, runner, cap_s=3600.0)
    assert runner.probes == [small]
    last = records[-1]
    assert last["status"] == "skipped"
    assert last["estimate_basis"].startswith("extrapolated")


def test_sklearn_estimate_uses_expected_iters(tmp_path: Path) -> None:
    runner = FakeRunner(probe_s=2.0)
    records = _run([_cell("sklearn-lloyd")], tmp_path, runner)
    assert records[0]["estimate_s"] == 200.0
    assert records[0]["estimate_basis"] == "probe x 100 expected iterations"


def _no_train(*_args: Any, **_kwargs: Any) -> Any:
    return None


def test_warm_rule(tmp_path: Path, monkeypatch: Any) -> None:
    gpu = run_grid.METHODS["emeans-cpu"].__class__("fake-gpu", "e-means", "gpu", 5.0, _no_train, lambda: None)
    monkeypatch.setitem(run_grid.METHODS, "fake-gpu", gpu)
    runner = FakeRunner(probe_s=1000.0)  # 5000 s estimate for the GPU cell
    _run([_cell("fake-gpu")], tmp_path / "g", runner, cap_s=None)
    assert runner.trains[0][1] is True

    runner = FakeRunner(probe_s=100.0)  # 500 s > WARM_MAX_S
    records = _run([_cell("emeans-cpu")], tmp_path / "e", runner, cap_s=None)
    assert runner.trains[0][1] is False
    assert records[1]["status"] == "skipped"
    assert records[1]["skipped_reason"] == "skipped (cost)"

    runner = FakeRunner(probe_s=1.0)
    records = _run([_cell("emeans-cpu")], tmp_path / "c", runner, cap_s=None)
    assert runner.trains[0][1] is True
    assert records[1]["status"] == "ok"


def test_timeout_record(tmp_path: Path) -> None:
    runner = FakeRunner(fail=subprocess.TimeoutExpired(["x"], 7200))
    records = _run([_cell()], tmp_path, runner)
    assert {r["status"] for r in records} == {"timed_out"}
    assert records[0]["skipped_reason"] == "timed out after 2.0 h"


def test_failed_record(tmp_path: Path) -> None:
    runner = FakeRunner(fail=WorkerError("boom " * 200))
    cells = [_cell(), _cell("faiss-lloyd")]
    records = _run(cells, tmp_path, runner)
    assert len(records) == 4
    assert records[0]["status"] == "failed"
    assert len(records[0]["skipped_reason"]) <= 500


def test_record_schema(tmp_path: Path) -> None:
    records = _run([_cell()], tmp_path, FakeRunner())
    assert len(records) == 2
    for record in records:
        assert set(record) == SCHEMA_KEYS
    assert records[0]["seconds"] == 2.0
    assert records[1]["seconds"] == 1.0
    assert records[0]["n_x_k"] == 10_000 * 256
    assert records[0]["mode"] == "euclidean"
