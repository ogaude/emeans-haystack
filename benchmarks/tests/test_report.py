# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest

import report
from precision import precision_grid
from report import aggregate, check, marks, render
from run_grid import full_grid

TRITON = "emeans-gpu-triton-fp32"
DEVICE = {"emeans-cpu": "cpu", "faiss-lloyd": "cpu", "sklearn-lloyd": "cpu", "gpu-a": "gpu", TRITON: "gpu"}
ENGINE = {
    "emeans-cpu": "e-means", "faiss-lloyd": "lloyd", "sklearn-lloyd": "lloyd", "gpu-a": "e-means", TRITON: "e-means"
}  # fmt: skip


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


def _variant(gap: float, *, gap_max: float | None = None, **kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "ok", "nonfinite": False, "precision_used": "fp32", "drift_gap_median": gap,
        "drift_gap_max": gap if gap_max is None else gap_max, "share_same_cluster": 0.9 - gap / 10,
        "objective_delta_rel": gap / 1000, "objective": 1.0,
    }  # fmt: skip
    out.update(kw)
    return out


def prec(seed: int, *, reduced_gap: float = 0.2, other_gap: float = 0.8, **overrides: Any) -> dict[str, Any]:
    variants = {
        "fp32_rerun": _variant(0.0),
        "fp32_other_seed": _variant(other_gap),
        "fp16": _variant(reduced_gap, precision_requested="fp16", precision_used="fp16"),
        "bf16": _variant(reduced_gap, precision_requested="bf16", precision_used="bf16"),
        "round16": _variant(0.001),
        "roundbf": _variant(0.01),
    }
    variants.update(overrides)
    return {
        "key": f"100000|1024|random|euclidean|{seed}", "status": "ok", "n": 100_000, "k": 1024, "data": "random",
        "mode": "euclidean", "seed": seed, "variants": variants,
    }  # fmt: skip


def test_precision_section() -> None:
    text = render([rec("emeans-cpu", 0, 1.0, 1.0)], precision_records=[prec(s) for s in range(3)])
    section = text.split("## Precision")[1]
    for name in ("fp16", "bf16", "fp32_rerun", "fp32_other_seed"):
        assert f"| {name} |" in section
    floor = section.split("### Storage floor")[1]
    assert "| round16 |" in floor
    assert "| roundbf |" in floor
    assert "| drift_gap median | drift_gap max |" in section
    assert "| 3 |" in section


def test_spike_contradictions_none() -> None:
    assert report.spike_contradictions([prec(s) for s in range(3)]) == []


def test_spike_contradictions_flags() -> None:
    slower = report.spike_contradictions([prec(s, reduced_gap=0.9, other_gap=0.8) for s in range(3)])
    assert any("bf16" in m and "0.9" in m and "0.8" in m and "random euclidean n=100000 k=1024" in m for m in slower)
    nonfinite = report.spike_contradictions([prec(0, fp16=_variant(0.1, nonfinite=True, status="nonfinite"))])
    assert any("non-finite" in m for m in nonfinite)
    rerun = report.spike_contradictions([prec(0, fp32_rerun=_variant(0.0, gap_max=0.02))])
    assert any("rerun" in m for m in rerun)
    fallback = report.spike_contradictions(
        [prec(0, bf16=_variant(0.1, status="precision_fallback", precision_requested="bf16"))]
    )
    assert any("fallback" in m for m in fallback)


def test_fallback_excluded_from_headline() -> None:
    bad = _variant(0.0, status="precision_fallback", precision_requested="bf16")
    stats = report.aggregate_precision([prec(0), prec(1, bf16=bad)])
    entry = stats[("random", "euclidean", 100_000, 1024)]["bf16"]
    assert entry["seeds"] == 1
    assert entry["excluded"] == 1


def test_check_precision_completeness() -> None:
    cells = precision_grid(("random",), seeds=(0, 1, 2))
    assert report.check_precision([], cells) == ["precision study not run"]
    one = [prec(0)]
    problems = report.check_precision(one, [c for c in cells if (c.n, c.k, c.mode) == (100_000, 1024, "euclidean")])
    assert any("missing precision cell" in p for p in problems)
    assert any("fewer than 3 seeds" in p for p in problems)
    full = [prec(s) for s in range(3)]
    only = [c for c in cells if (c.n, c.k, c.mode) == (100_000, 1024, "euclidean")]
    assert report.check_precision(full, only) == []


def test_render_includes_spike_check_line() -> None:
    ok = render([rec("emeans-cpu", 0, 1.0, 1.0)], precision_records=[prec(0)])
    assert "Spike check: consistent with the Phase 2 spike" in ok
    bad = render([rec("emeans-cpu", 0, 1.0, 1.0)], precision_records=[prec(0, reduced_gap=0.9)])
    assert "Spike check: results contradict the Phase 2 spike" in bad
    missing = render([rec("emeans-cpu", 0, 1.0, 1.0)], precision_records=[])
    assert "Precision study not run" in missing


def test_protocol_states_cohere_normalisation() -> None:
    assert "normalises every row" in render([rec("emeans-cpu", 0, 1.0, 1.0)])


def test_header_lists_engine_per_method() -> None:
    records = [
        # Old records store a bare label; new ones carry the version. Both render as stored.
        rec("emeans-cpu", 0, 1.0, 1.0, versions={"faiss": "1", "emeans_source": "faiss-cpu"}),
        rec("emeans-cpu", 1, 1.0, 1.0, versions={"faiss": "1", "emeans_source": "faiss-cpu 1.16.0"}),
        rec("gpu-a", 0, 1.0, 1.0, versions={"faiss": "1", "emeans_source": "emeans-rocm 0.2.5"}),
        # A Lloyd cell run in the fork env must not be reported as an e-means engine.
        rec("faiss-lloyd", 0, 1.0, 1.0, versions={"faiss": "1", "emeans_source": "emeans-rocm 0.2.5"}),
    ]
    text = render(records)
    assert "- e-means engine per method:" in text
    assert "  - emeans-cpu: faiss-cpu, faiss-cpu 1.16.0" in text
    assert "  - gpu-a: emeans-rocm 0.2.5" in text
    assert "  - faiss-lloyd:" not in text
    assert "- e-means source:" not in text


def _sweep(method: str, objectives: dict[float, float], *, data: str = "cohere", n: int = 100_000) -> list:
    mode = "spherical" if data == "cohere" else "euclidean"
    return [
        rec(method, 0, 2.0 * p, obj, timing="cold", kind="sweep", passes_requested=p, passes_used=p, data=data,
            mode=mode, n=n, k=4096, key=f"{method}|{data}|{n}|4096|{mode}|0|sweep|{p:g}|cold")
        for p, obj in objectives.items()
    ]  # fmt: skip


EMEANS_SWEEP = {1.0: 110.0, 2.0: 100.0, 5.0: 90.0, 10.0: 89.0, 25.0: 88.0}
LLOYD_SWEEP = {1.0: 120.0, 2.0: 110.0, 5.0: 100.0, 10.0: 95.0, 25.0: 93.0}


def test_passes_to_objective() -> None:
    rows = aggregate(_sweep("emeans-cpu", EMEANS_SWEEP) + _sweep("faiss-lloyd", LLOYD_SWEEP))
    (reach,) = report.passes_to_objective(rows)
    assert reach.lloyd.objective == 93.0
    assert reach.emeans is not None and reach.emeans.passes_used == 5.0
    assert reach.earlier is not None and reach.earlier.passes_used == 2.0
    assert reach.pass_ratio == "5.0-12.5x"
    assert reach.lloyd_to_default is None  # Lloyd never reaches e-means' 5-pass objective within 25 passes
    first = report.passes_to_objective(
        aggregate(_sweep("emeans-cpu", {1.0: 90.0}) + _sweep("faiss-lloyd", LLOYD_SWEEP))
    )
    assert first[0].pass_ratio == ">= 25.0x"
    never = report.passes_to_objective(
        aggregate(_sweep("emeans-cpu", {5.0: 99.0}) + _sweep("faiss-lloyd", LLOYD_SWEEP))
    )
    assert never[0].emeans is None
    assert never[0].pass_ratio == "-"


def _main(method: str, seconds: float, objective: float, **kw: Any) -> dict[str, Any]:
    return rec(method, 0, seconds, objective, timing="cold", passes_used=kw.pop("passes_used", 5.0), **kw)


def test_main_comparison_cpu_only_mark() -> None:
    rows = aggregate([_main("emeans-cpu", 10.0, 99.0), _main("faiss-lloyd", 30.0, 100.0, passes_used=25.0)])
    (compare,) = report.main_comparison(rows)
    assert compare.cpu_only_win
    assert compare.gap is not None and round(compare.gap, 4) == -0.01
    worse = report.main_comparison(aggregate([_main("emeans-cpu", 10.0, 110.0), _main("faiss-lloyd", 30.0, 100.0)]))
    assert not worse[0].cpu_only_win
    slower = report.main_comparison(aggregate([_main("emeans-cpu", 40.0, 99.0), _main("faiss-lloyd", 30.0, 100.0)]))
    assert not slower[0].cpu_only_win
    alone = report.main_comparison(aggregate([_main("emeans-cpu", 40.0, 99.0)]))
    assert alone[0].lloyd is None and not alone[0].cpu_only_win


def test_lloyd_stop_labels() -> None:
    rows = aggregate(
        [_main("sklearn-lloyd", 1.0, 1.0, passes_used=35.0), _main("faiss-lloyd", 1.0, 1.0, passes_used=20.0)]
    )
    labels = {r.method: report._stop(r) for r in rows}
    assert labels["sklearn-lloyd"] == "converged (tol 1e-4) after 35"
    assert labels["faiss-lloyd"] == "stopped after 20 of 25"


def _full() -> list[dict[str, Any]]:
    records = _sweep("emeans-cpu", EMEANS_SWEEP) + _sweep("faiss-lloyd", LLOYD_SWEEP)
    records += [
        _main("emeans-cpu", 10.0, 99.0),
        _main("faiss-lloyd", 30.0, 100.0, passes_used=25.0),
        _main("gpu-a", 1.0, 99.0, transfer_s=0.5, host={"label": "t", "gpu": "Test GPU"}),
        rec("faiss-lloyd", 0, None, None, timing="cold", status="skipped", skipped_reason="not run: est. 3.1 h",
            n=1_000_000, k=65536, key="capped|cold"),
    ]  # fmt: skip
    return records


def test_render_new_sections() -> None:
    text = render(_full(), precision_records=[prec(s) for s in range(3)])
    order = [
        "## Headline", "## Protocol", "## Passes to the same objective", "## e-means at 5 passes vs Lloyd's k-means",
        "## Upstream claims, checked", "## Main results", "## Where the GPU does not help", "## GPU engines",
        "## Pass sweep", "## Cuts and choices", "## Not run", "## Precision",
    ]  # fmt: skip
    positions = [text.index(title) for title in order]
    assert positions == sorted(positions)
    headline = text.split("## Headline")[1].split("## Protocol")[0]
    assert "e-means needs fewer passes (epochs) than Lloyd's k-means" in headline
    assert "within 5 passes (90)" in headline
    assert "Test GPU" in headline
    engines = text.split("## GPU engines")[1]
    assert "0.2.6" in engines
    assert "unused_threshold" in engines
    assert "upcoming" not in engines
    for line in text.splitlines():
        if "k-means" in line:
            assert "Lloyd" in line or "not k-means" in line, line


def test_where_section_explains_missing_marks() -> None:
    where = render(_full()).split("## Where the GPU does not help")[1].split("## GPU engines")[0]
    assert "No cell carries `GPU does not help` or `CPU e-means wins`." in where
    assert "The GPU helps in every main cell measured" in where
    assert "`CPU e-means wins (CPU only)` in 1 of 1 comparable main cells (0 have no Lloyd run)" in where
    assert "finishes sooner than every CPU Lloyd run" in where
    assert "(cold in all 1 cells)" in where


def test_cuts_table_names_constants_and_capped_cells() -> None:
    cuts = render(_full()).split("## Cuts and choices")[1].split("## Not run")[0]
    assert f"{report.DEFAULT_CAP_MIN}-minute cap" in cuts
    assert f"over {report.WARM_MAX_S} s (`WARM_MAX_S`)" in cuts
    assert "`METHOD_NS`" in cuts
    assert "`SWEEP_CELLS`" in cuts
    assert "`faiss-lloyd` random 1M x 65536 (est. 3.1 h)" in cuts
    assert "Slower cell-seeds kept" in cuts  # the kept cell-seeds are not in these records
    assert "60 warm records" not in cuts
    assert "A setting is one (data, n, k)" in cuts
    assert "option B" in cuts
    assert cuts.count("\n| ") >= len(report.notes.STATIC_CUTS) + 7


def test_precision_summary_ratio_and_caution() -> None:
    lines = report.precision_summary([prec(s, reduced_gap=0.4, other_gap=0.8) for s in range(3)])
    text = "\n".join(lines)
    assert "| random | euclidean | 100k | 1024 | 0.50 | 0.50 |" in text
    assert "- random: ratio 0.50-0.50." in text
    assert report.notes.OPTION_B_CAUTION in text


def test_fork_comparison() -> None:
    old = [_main("gpu-a", 2.0, 100.0, key="a|cold"), rec("gpu-a", 0, 1.0, 100.0, key="a|warm")]
    new = [_main("gpu-a", 2.5, 100.0, key="a|cold"), rec("gpu-a", 0, 1.1, 100.0, key="a|warm")]
    text = "\n".join(report.fork_comparison(new, old, [prec(0)], [prec(0)]))
    assert "2 matching records" in text
    assert "median 1.10" in text
    assert "+0.500 s" in text
    assert "0 of 4 variant drift_gap medians differ" in text
    assert "drift is 0 in 1 of 1 cells" in text


def test_kept_slow_row_from_records() -> None:
    (kept_a, ref_a, _), (kept_b, _ref_b, _), (kept_c, ref_c, _) = report.notes.KEPT_SLOW
    records = [
        rec("emeans-cpu", 0, 106.8, 1.0, key=kept_a, passes_used=1.0),
        rec("emeans-cpu", 1, 105.8, 1.0, key=kept_b, passes_used=1.0),
        rec("emeans-cpu", 2, 100.0, 1.0, key=ref_a, passes_used=1.0),
        rec("faiss-lloyd", 1, 104.1, 1.0, key=kept_c, passes_used=1.0),
        rec("faiss-lloyd", 1, 2500.0, 1.0, key=ref_c, passes_used=25.0),
    ]
    cut, _reason, hides = report._kept_slow_row(records)
    assert cut.startswith("Cell-seeds 4-7 % slower")
    assert "(+6.8 %)" in cut and "(+5.8 %)" in cut and "(+4.1 %)" in cut
    assert "vary by 4-7 %" in hides


def test_warm_skip_counts_main_and_sweep() -> None:
    records = [
        *_full(),
        rec("emeans-cpu", 0, None, None, status="skipped", skipped_reason="skipped (cost)", key="w1|warm"),
        rec("emeans-cpu", 0, None, None, status="skipped", skipped_reason="skipped (cost)", kind="sweep",
            passes_requested=5.0, key="w2|warm"),
    ]  # fmt: skip
    cuts = render(records).split("## Cuts and choices")[1]
    assert "2 warm records (1 main cell-seeds, 1 sweep points; emeans-cpu)" in cuts


def test_coverage_and_gpu_lead() -> None:
    rows = aggregate(
        [
            _main("emeans-cpu", 100.0, 1.0),
            _main("gpu-a", 4.0, 1.0, precision="fp32"),
            _main("gpu-a", 1.0, 1.0, precision="bf16", backend="t", key="bf|cold"),
            rec("emeans-cpu", 0, 100.0, 1.0),
            rec("gpu-a", 0, 200.0, 1.0),
        ]
    )
    assert (
        report._coverage(report.gpu_speedups(rows))
        == "cold in all 1 cells, warm in 0 of 1 cells that have a CPU warm time"
    )
    lead = report._gpu_lead(rows)
    assert lead.index("25.0x") < lead.index("100.0x")
    assert "BF16 on the GPU against FP32 on the CPU" in lead


def test_lost_text_degenerate_and_found() -> None:
    def cmp(data: str, n: int, k: int, gap: float) -> report.MainCompare:
        rows = aggregate(
            [
                _main("emeans-cpu", 1.0, 1.0 + gap, data=data, n=n, k=k),
                _main("faiss-lloyd", 2.0, 1.0, data=data, n=n, k=k),
            ]
        )
        return report.main_comparison(rows)[0]

    degenerate = cmp("cohere", 100_000, 65536, 0.2)
    found = cmp("random", 100_000, 16384, 0.079)
    twin = cmp("cohere", 100_000, 16384, -0.0069)
    assert "degenerate: 1.5 rows per centroid" in report._lost_text(degenerate, [degenerate])
    text = report._lost_text(found, [found, twin])
    assert "reported as found; cohere at the same size: -0.69 %" in text
    assert "rows per centroid" not in text


def test_spike_values_unrounded() -> None:
    records = [prec(s, reduced_gap=1.205, other_gap=1.19) for s in range(3)]
    assert report._spike_values(records) == "random euclidean 100k x 1024: fp16 1.205, bf16 1.205 vs 1.190"
    assert "equal to or slightly above" in "\n".join(report.render_precision(records))


def test_per_pass_summary_ranges() -> None:
    rows = aggregate(_sweep("emeans-cpu", EMEANS_SWEEP) + _sweep("faiss-lloyd", LLOYD_SWEEP))
    lines = report._per_pass_summary(report.passes_to_objective(rows), [])
    assert any("25-pass points" in line and "+0.0 %" in line for line in lines)


def test_fork_comparison_off_by_default() -> None:
    assert "fork build" not in render(_full(), precision_records=[prec(0)])


def _version_records(new_objective: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old: list[dict[str, Any]] = []
    new: list[dict[str, Any]] = []
    for seed in range(3):
        for timing, o_s, n_s in (("cold", 2.0, 2.2), ("warm", 1.0, 1.1)):
            old.append(rec(TRITON, seed, o_s, 100.0 + seed, timing=timing))
            new.append(rec(TRITON, seed, n_s, new_objective + seed, timing=timing))
    return old, new


def _ari(ok: bool = True, ok_objective: bool = True) -> dict[str, Any]:
    return {
        "method": TRITON, "data": "random", "n": 100_000, "k": 1024, "ari_new_old": [0.91, 0.92, 0.93],
        "ari_old_seeds": [0.8, 0.85, 0.9], "ari_old_rerun": 0.99, "ari_new_old_median": 0.92,
        "ari_old_seeds_median": 0.85, "ok_objective": ok_objective, "ok_ari": ok, "ok": ok and ok_objective,
    }  # fmt: skip


def test_version_comparison() -> None:
    old, new = _version_records(100.0)
    text = "\n".join(report.version_comparison(new, old, [prec(0)], [prec(0)], [_ari()]))
    assert "### emeans-rocm 0.2.5 vs 0.2.6" in text
    assert report.notes.VERSION_NOTE in text
    assert "6 matching records" in text
    assert "1.10 [1.10-1.10]" in text
    assert "| 0.9200 |" in text
    assert "| yes | yes |" in text
    assert "matches 0.2.5 within seed noise" in text
    assert "0 of 4 variant drift_gap medians differ" in text


def test_version_comparison_missing_ari_shows_dash() -> None:
    old, new = _version_records(100.0)
    text = "\n".join(report.version_comparison(new, old, [], [], []))
    assert "| - |" in text
    assert "ARI between versions is at least" not in text
    assert "matches 0.2.5 within seed noise" not in text
    assert "no ARI records were given" in text


def test_load_version_ari_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no such file"):
        report.load_version_ari(tmp_path / "typo.jsonl")


def test_version_comparison_differs() -> None:
    old, new = _version_records(130.0)
    text = "\n".join(report.version_comparison(new, old, [], [], [_ari(ok=False)]))
    assert "outside the min-max of the 3 seeds of 0.2.5 of the same cell in 1 cells" in text
    two = "\n".join(report.version_comparison(new[:4], old[:4], [], [], [_ari(ok=False)]))
    assert "outside the min-max of the 2 seeds of 0.2.5" in two
    assert "its ARI part in 0 of 1" in text
    assert "largest main-cell objective difference is fp32 " in text


def test_version_gate_note_only_with_matching_counts() -> None:
    old, new = _version_records(130.0)
    differs = "\n".join(report.version_comparison(new, old, [], [], [_ari(ok=False)]))
    assert report.notes.VERSION_GATE_NOTE not in differs
    assert report.notes.VERSION_GATE_UNMATCHED in differs
    recorded = [_ari(ok_objective=i < 5) for i in range(8)]
    matched = "\n".join(report.version_comparison(new, old, [], [], recorded))
    assert "objective part in 5 of 8 settings and its ARI part in 8 of 8" in matched
    assert report.notes.VERSION_GATE_NOTE in matched
    assert report.notes.VERSION_GATE_UNMATCHED not in matched


def test_spike_acceptance_only_with_matching_count(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [prec(0, reduced_gap=0.9)]
    found = len(report.spike_contradictions(records))
    other = "\n".join(report.render_precision(records))
    assert f"These {found} contradictions have not been reviewed by the owner" in other
    assert "were accepted by the owner" not in other
    monkeypatch.setattr(report.notes, "ACCEPTED_CONTRADICTIONS_026", found)
    same = "\n".join(report.render_precision(records))
    assert f"These {found} contradictions of the 0.2.6 records were accepted by the owner on 2026-11-27" in same


def test_version_comparison_objective_failure_is_not_an_ari_failure() -> None:
    old, new = _version_records(130.0)
    text = "\n".join(report.version_comparison(new, old, [], [], [_ari(ok=True, ok_objective=False)]))
    assert "objective part in 0 of 1 settings and its ARI part in 1 of 1" in text
    assert "ARI settings fail" not in text


def test_ari_baseline_uses_same_seed_rerun() -> None:
    bf16 = "emeans-gpu-triton-bf16"
    records = [
        {**_ari(), "method": bf16, "data": "cohere", "ari_new_old_median": 0.27, "ari_old_seeds_median": 0.25,
         "ari_old_rerun": 1.0},
        {**_ari(), "method": bf16, "data": "cohere", "k": 4096, "ari_new_old_median": 0.40,
         "ari_old_seeds_median": 0.39, "ari_old_rerun": 1.0},
        {**_ari(), "method": TRITON, "data": "cohere", "ari_old_rerun": 0.45},
        {**_ari(), "data": "random", "ari_new_old_median": 0.0005, "ari_old_seeds_median": 0.0004,
         "ari_old_rerun": 0.0005},
    ]  # fmt: skip
    text = report._ari_baseline(records)
    assert "bf16 0.2.5 reruns are deterministic (ARI 1.0)" in text
    assert "fp32 0.2.5 reruns are not reproducible (same-seed ARI 0.45 on cohere)" in text
    assert "On cohere, 0.2.6 against 0.2.5 at the same seed gives an ARI of 0.27-0.92" in text
    assert "On random data (no cluster structure) the ARI between versions and between seeds is 0.0004-0.0005" in text
    assert "kept as run" in text
    old, new = _version_records(100.0)
    section = "\n".join(report.version_comparison(new, old, [], [], records))
    assert text in section


def test_version_comparison_objective_summary() -> None:
    old, new = _version_records(99.0)
    text = "\n".join(report.version_comparison(new, old, [], [], [_ari()]))
    assert "0.2.6 lower than 0.2.5 in 3 of 3 records; largest difference -1.000 % (" in text
    assert "random 100k x 1024, 97.7 rows per centroid)" in text


def test_version_comparison_off_by_default() -> None:
    text = render(_full(), precision_records=[prec(0)])
    assert "0.2.5 vs 0.2.6" not in text
    assert report.notes.VERSION_ARI_CUT[0] not in text


def test_render_version_comparison_lists_ari_cut() -> None:
    old, new = _version_records(100.0)
    text = render(
        _full() + new, precision_records=[prec(0)], prev_speed=old, prev_precision=[prec(0)], version_ari=[_ari()]
    )
    assert "### emeans-rocm 0.2.5 vs 0.2.6" in text
    assert report.notes.VERSION_ARI_CUT[0] in text


def test_main_warns_on_partial_version_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    speed = tmp_path / "speed.jsonl"
    speed.write_text(json.dumps(rec("emeans-cpu", 0, 1.0, 1.0)) + "\n")
    out = tmp_path / "RESULTS.md"
    argv = ["report", "--results", str(speed), "--out", str(out), "--precision", str(tmp_path / "none.jsonl")]
    monkeypatch.setattr("sys.argv", [*argv, "--prev-speed", str(speed)])
    assert report.main() == 0
    err = capsys.readouterr().err
    assert "--prev-speed given without --prev-precision, --version-ari" in err
    assert "0.2.5 vs 0.2.6" not in out.read_text()
