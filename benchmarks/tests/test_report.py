# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import report
from report import aggregate, check, marks, render
from run_grid import full_grid

DEVICE = {"emeans-cpu": "cpu", "faiss-lloyd": "cpu", "sklearn-lloyd": "cpu", "gpu-a": "gpu"}
ENGINE = {"emeans-cpu": "e-means", "faiss-lloyd": "lloyd", "sklearn-lloyd": "lloyd", "gpu-a": "e-means"}


def rec(
    method: str, seed: int, seconds: float | None, objective: float | None, *, timing: str = "warm", **kw: Any
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": f"{method}|random|100000|1024|euclidean|{seed}|main|default|{timing}",
        "timing": timing, "method": method, "engine": ENGINE[method], "device": DEVICE[method],
        "backend": "b", "precision": "fp32", "n": 100_000, "k": 1024, "n_x_k": 102_400_000, "d": 768,
        "data": "random", "mode": "euclidean", "seed": seed, "kind": "main", "passes_requested": None,
        "passes_used": 5.0, "seconds": seconds, "transfer_s": 0.0, "objective": objective,
        "status": "ok", "skipped_reason": None, "host": {"label": "t"}, "versions": {"faiss": "1"},
    }  # fmt: skip
    out.update(kw)
    return out


def test_aggregate_median_minmax() -> None:
    records = [rec("emeans-cpu", s, v, o) for s, v, o in [(0, 1.0, 10.0), (1, 3.0, 12.0), (2, 2.0, 11.0)]]
    (row,) = aggregate(records)
    assert (row.seconds, row.seconds_min, row.seconds_max) == (2.0, 1.0, 3.0)
    assert (row.objective, row.objective_min, row.objective_max) == (11.0, 10.0, 12.0)
    assert row.n_seeds == 3


def _cell_rows(gpu_seconds: float, gpu_transfer: float = 0.0, cpu: float = 10.0, timing: str = "warm") -> list:
    records = [
        rec("emeans-cpu", 0, cpu, 100.0, timing=timing),
        rec("faiss-lloyd", 0, 20.0, 100.0, timing=timing),
        rec("gpu-a", 0, gpu_seconds, 100.0, timing=timing, transfer_s=gpu_transfer),
    ]
    return aggregate(records)


def test_gpu_does_not_help_mark() -> None:
    assert "GPU does not help" in marks(_cell_rows(gpu_seconds=9.0, gpu_transfer=1.0))  # tie counts
    assert "GPU does not help" not in marks(_cell_rows(gpu_seconds=2.0, gpu_transfer=1.0))
    cold = marks(_cell_rows(gpu_seconds=12.0, timing="cold"))
    assert "GPU does not help (cold)" in cold


def test_cpu_emeans_wins_mark() -> None:
    assert "CPU e-means wins" in marks(_cell_rows(gpu_seconds=50.0))
    assert "CPU e-means wins" not in marks(_cell_rows(gpu_seconds=2.0))
    worse = aggregate(
        [
            rec("emeans-cpu", 0, 10.0, 101.0),  # 1 % worse than Lloyd: outside the 0.5 % tolerance
            rec("faiss-lloyd", 0, 20.0, 100.0),
            rec("gpu-a", 0, 50.0, 100.0),
        ]
    )
    assert "CPU e-means wins" not in marks(worse)


def test_marks_need_data() -> None:
    no_gpu = aggregate([rec("emeans-cpu", 0, 1.0, 1.0), rec("faiss-lloyd", 0, 2.0, 1.0)])
    assert marks(no_gpu) == ["not comparable"]
    no_cpu = aggregate([rec("faiss-lloyd", 0, 2.0, 1.0), rec("gpu-a", 0, 1.0, 1.0)])
    assert marks(no_cpu) == ["not comparable"]
    skipped = aggregate([rec("emeans-cpu", 0, 1.0, 1.0), rec("gpu-a", 0, None, None, status="skipped")])
    assert marks(skipped) == ["not comparable"]


def test_skipped_rows_listed() -> None:
    records = [
        rec("emeans-cpu", 0, 1.0, 1.0),
        rec("faiss-lloyd", 0, None, None, status="skipped", skipped_reason="not run: est. 9.0 h"),
        rec("sklearn-lloyd", 0, None, None, status="timed_out", skipped_reason="timed out after 2.0 h"),
    ]
    text = render(records)
    not_run = text.split("## Not run")[1]
    assert "not run: est. 9.0 h" in not_run
    assert "timed out after 2.0 h" in not_run


def test_render_columns() -> None:
    records = [rec("emeans-cpu", 0, 1.0, 1.0), rec("emeans-cpu", 0, 2.0, 1.0, timing="cold")]
    records.append(rec("emeans-cpu", 0, 1.0, 1.0, n=1_000_000, k=16384, key="x|cold", timing="cold"))
    text = render(records, smoke=True)
    header = (
        "| data | mode | n | k | n x k | method | backend | precision | passes used | cold s | warm s | "
        "transfer s | objective | vs faiss | marks |"
    )
    assert header in text
    assert "1.0e8" in text
    assert "(>= 1e10)" in text
    assert "SMOKE RUN" in text
    assert "OBJECTIVE_TOLERANCE = 0.005" in text
    for line in text.splitlines():
        if "k-means" in line:
            assert "Lloyd" in line or "not k-means" in line


def test_check_completeness() -> None:
    cells = full_grid(["emeans-cpu"], ["random"], seeds=(0, 1, 2))
    records = []
    for c in cells:
        for timing in ("cold", "warm"):
            records.append(rec("emeans-cpu", c.seed, 1.0, 1.0, timing=timing, key=f"{c.key}|{timing}", n=c.n, k=c.k,
                               kind=c.kind, passes_requested=c.passes))  # fmt: skip
    assert check(records, cells, min_seeds=3) == []
    missing = [r for r in records if r["key"] != f"{cells[0].key}|warm"]
    assert any("missing" in p for p in check(missing, cells))
    one_seed = [r for r in records if r["seed"] == 0]
    problems = check(one_seed, [c for c in cells if c.seed == 0], min_seeds=3)
    assert any("fewer than 3 seeds" in p for p in problems)


def test_sweep_table() -> None:
    records = []
    for passes in (1.0, 2.0):
        for timing in ("cold", "warm"):
            records.append(rec("emeans-cpu", 0, passes, 100.0 / passes, timing=timing, kind="sweep",
                               passes_requested=passes))  # fmt: skip
            records.append(rec("faiss-lloyd", 0, passes, 90.0 / passes, timing=timing, kind="sweep",
                               passes_requested=passes))  # fmt: skip
    text = render(records)
    sweep = text.split("## Pass sweep")[1].split("## Not run")[0]
    assert sweep.count("| emeans-cpu |") == 2
    assert sweep.count("| faiss-lloyd |") == 2
    assert report.OBJECTIVE_TOLERANCE == 0.005
