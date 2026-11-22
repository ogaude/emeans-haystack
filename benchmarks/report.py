# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Turn benchmark JSONL records into RESULTS.md, with honesty marks computed from the data."""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from precision import PCell, precision_grid
from run_grid import Cell, full_grid

BENCH_DIR = Path(__file__).resolve().parent
OBJECTIVE_TOLERANCE = 0.005
ABOVE_1E10 = 1e10
GPU_NOT_HELP = "GPU does not help"
CPU_WINS = "CPU e-means wins"
NOT_COMPARABLE = "not comparable"
BASELINE = "faiss-lloyd"
PRECISION_VARIANTS = ("fp16", "bf16", "fp32_rerun", "fp32_other_seed")
FLOOR_VARIANTS = ("round16", "roundbf")
REDUCED = ("fp16", "bf16")
SPIKE_CONSISTENT = "consistent with the Phase 2 spike"
_BIG = 1000


@dataclass(frozen=True)
class Row:
    """Median and range over seeds of one (cell, method, timing) group of ok records."""

    data: str
    mode: str
    n: int
    k: int
    kind: str
    passes_requested: float | None
    method: str
    engine: str
    device: str
    backend: str
    precision: str
    timing: str
    seconds: float
    seconds_min: float
    seconds_max: float
    transfer_s: float
    objective: float
    objective_min: float
    objective_max: float
    passes_used: float | None
    n_seeds: int

    @property
    def cell(self) -> tuple[str, str, int, int, str, float | None]:
        """The cell this row belongs to, independent of the method."""
        return (self.data, self.mode, self.n, self.k, self.kind, self.passes_requested)

    @property
    def end_to_end(self) -> float:
        """Seconds including the host-to-device transfer."""
        return self.seconds + self.transfer_s


def load_records(path: Path) -> list[dict[str, Any]]:
    """
    Read records, ignoring malformed lines and keeping the last record of a repeated key.

    :param path: JSONL file.
    :returns: The records in file order.
    """
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"{path}:{number}: malformed line ignored\n")
            continue
        key = record.get("key")
        if key is None:
            continue
        if key not in by_key:
            order.append(key)
        by_key[key] = record
    return [by_key[key] for key in order]


def aggregate(records: Iterable[dict[str, Any]]) -> list[Row]:
    """
    Group ok records over seeds.

    :param records: Records as written by `run_grid`.
    :returns: One row per (data, mode, n, k, kind, passes, method, backend, precision, timing).
    """
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("status") != "ok" or r.get("seconds") is None or r.get("objective") is None:
            continue
        key = (
            r["data"], r["mode"], r["n"], r["k"], r["kind"], r.get("passes_requested"),
            r["method"], r.get("backend") or "", r.get("precision") or "", r["timing"],
        )  # fmt: skip
        groups[key].append(r)
    rows = []
    for key, recs in groups.items():
        seconds = [float(r["seconds"]) for r in recs]
        objective = [float(r["objective"]) for r in recs]
        used = [float(r["passes_used"]) for r in recs if r.get("passes_used") is not None]
        rows.append(
            Row(
                *key[:6],
                method=key[6],
                engine=recs[0].get("engine") or "",
                device=recs[0].get("device") or "",
                backend=key[7],
                precision=key[8],
                timing=key[9],
                seconds=statistics.median(seconds),
                seconds_min=min(seconds),
                seconds_max=max(seconds),
                transfer_s=statistics.median([float(r.get("transfer_s") or 0.0) for r in recs]),
                objective=statistics.median(objective),
                objective_min=min(objective),
                objective_max=max(objective),
                passes_used=statistics.median(used) if used else None,
                n_seeds=len(recs),
            )
        )
    rows.sort(key=lambda r: (r.data, r.n, r.k, r.kind, r.passes_requested or 0, r.method, r.timing))
    return rows


def _timing_marks(rows: list[Row], timing: str) -> tuple[bool, list[str]]:
    suffix = "" if timing == "warm" else " (cold)"
    sel = [r for r in rows if r.timing == timing]
    cpu = [r for r in sel if r.method == "emeans-cpu"]
    gpu = [r for r in sel if r.engine == "e-means" and r.device == "gpu"]
    if not cpu or not gpu:
        return False, []
    out = []
    best_gpu = min(r.end_to_end for r in gpu)
    if best_gpu >= cpu[0].seconds:
        out.append(GPU_NOT_HELP + suffix)
    others = [r for r in sel if r is not cpu[0]]
    lloyd = [r for r in sel if r.engine == "lloyd"]
    methods_in_cell = {r.method for r in rows}
    if lloyd and {r.method for r in sel} == methods_in_cell:
        faster = all(cpu[0].seconds < (r.end_to_end if r.device == "gpu" else r.seconds) for r in others)
        best_lloyd = min(r.objective for r in lloyd)
        if faster and cpu[0].objective <= best_lloyd * (1 + OBJECTIVE_TOLERANCE):
            out.append(CPU_WINS + suffix)
    return True, out


def marks(cell_rows: Sequence[Row]) -> list[str]:
    """
    Compute the honesty marks of one cell from its rows (both timings).

    :param cell_rows: All rows of one cell.
    :returns: `GPU does not help` and `CPU e-means wins` (with a `(cold)` variant), or `not comparable`.
    """
    rows = list(cell_rows)
    out: list[str] = []
    comparable = False
    for timing in ("warm", "cold"):
        ok, found = _timing_marks(rows, timing)
        comparable = comparable or ok
        out += found
    return out if comparable else [NOT_COMPARABLE]


def _num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3g}" if abs(value) < _BIG else f"{value:.0f}"


def _span(median: float, low: float, high: float) -> str:
    return f"{_num(median)} [{_num(low)}-{_num(high)}]"


def _nxk(n: int, k: int) -> str:
    value = n * k
    text = f"{value:.1e}".replace("e+0", "e").replace("e+", "e")
    return f"{text} (>= 1e10)" if value >= ABOVE_1E10 else text


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h"


def _protocol() -> list[str]:
    return [
        "## Protocol",
        "",
        "- Every method trains the same k from the same starting centroids: a seeded sample of k distinct data rows.",
        (
            "- Passes per method: e-means `nepochs` 5 (the `EmeansFitter` default), `faiss.Kmeans` `niter` 25, "
            "scikit-learn `KMeans` `max_iter` 300, `tol` 1e-4, `n_init` 1. One e-means pass costs about the same "
            "distance evaluations as one Lloyd iteration, so the pass sweep compares equal work."
        ),
        "- No subsampling: `faiss.Kmeans` `max_points_per_centroid` 2**30, e-means `None`.",
        (
            "- Final objective: mean squared distance to the nearest centroid, from one exact FP32 CPU scorer "
            "(`faiss.IndexFlatL2`), never a library's own objective. The `vs faiss` column is the relative "
            "difference to the Lloyd `faiss-lloyd` median objective in the same cell."
        ),
        (
            "- Real data is spherical (scikit-learn runs on unit-norm input, its centroids are not renormalised). "
            "The Cohere file's rows are not unit-norm: the harness normalises every row to unit length when it "
            "loads them (the raw norm range is recorded in each run's data provenance). "
            "Random data is Euclidean and is for speed only: Gaussian noise has no cluster structure."
        ),
        (
            "- Cold: the first run in a fresh process, including the backend probe and Triton JIT with the "
            "persistent Triton cache (recorded per row). Warm: a repeat in the same process. Host-to-device "
            "transfer is timed separately and subtracted; GPU end-to-end below adds it back."
        ),
        (
            "- Time cap: a cell whose probe estimate exceeds the cap is listed as `not run: est. X h`, never dropped; "
            "a worker past its hard timeout is listed as `timed out after X h`."
        ),
        "- 100k x 65536 is degenerate: about 1.5 points per centroid.",
        "- Lloyd's k-means is the baseline here; e-means is not k-means.",
        (
            f"- Marks are computed from the data. `{GPU_NOT_HELP}`: the best GPU e-means end-to-end median is not "
            f"below the CPU e-means median. `{CPU_WINS}`: CPU e-means is strictly faster than every other method "
            f"(GPU end-to-end) and its objective is within OBJECTIVE_TOLERANCE = {OBJECTIVE_TOLERANCE} "
            "(0.5 %) of the best Lloyd median. A `(cold)` suffix means the cold timings were compared."
        ),
        "",
    ]


def _header(records: list[dict[str, Any]], title: str, smoke: bool) -> list[str]:
    ok = [r for r in records if r.get("status") == "ok"]
    host: dict[str, Any] = {}
    versions: dict[str, Any] = {}
    for r in ok:
        host = r.get("host") or host
        versions = r.get("versions") or versions
        if host and versions:
            break
    out = [f"# {title}", ""]
    if smoke:
        out += ["> SMOKE RUN: not results. A tiny CPU grid that only checks the pipeline.", ""]
    out.append(f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}.")
    fields = [
        ("host", host.get("label")),
        ("CPU", host.get("cpu")),
        ("cores", host.get("cores")),
        ("GPU", host.get("gpu")),
        ("torch", versions.get("torch")),
        ("HIP", versions.get("hip")),
        ("faiss", versions.get("faiss")),
        ("scikit-learn", versions.get("sklearn")),
        ("e-means source", versions.get("emeans_source")),
    ]
    out += [f"- {name}: {value}" for name, value in fields if value is not None]
    out.append("")
    return out


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in body]
    out.append("")
    return out


def _main_table(rows: list[Row]) -> tuple[list[str], list[str]]:
    cells: dict[Any, list[Row]] = defaultdict(list)
    for r in rows:
        cells[r.cell].append(r)
    header = [
        "data", "mode", "n", "k", "n x k", "method", "backend", "precision", "passes used",
        "cold s", "warm s", "transfer s", "objective", "vs faiss", "marks",
    ]  # fmt: skip
    body: list[list[str]] = []
    summary: list[str] = []
    for cell in sorted(cells, key=lambda c: (c[0], c[2], c[3], c[4], c[5] or 0)):
        data, mode, n, k, kind, _passes = cell
        if kind != "main":
            continue
        cell_rows = cells[cell]
        cell_marks = marks(cell_rows)
        base = [r.objective for r in cell_rows if r.method == BASELINE and r.timing == "cold"]
        label = ", ".join(cell_marks)
        if cell_marks and cell_marks != [NOT_COMPARABLE]:
            summary.append(f"- {data} {mode} n={n} k={k}: {label}")
        keys = sorted({(r.method, r.backend, r.precision) for r in cell_rows})
        for method, backend, precision in keys:
            mine = {
                r.timing: r for r in cell_rows if (r.method, r.backend, r.precision) == (method, backend, precision)
            }
            cold, warm = mine.get("cold"), mine.get("warm")
            ref = cold or warm
            assert ref is not None  # noqa: S101
            vs = f"{(ref.objective / base[0] - 1) * 100:+.2f} %" if base else "-"
            body.append(
                [
                    data, mode, str(n), str(k), _nxk(n, k), method, backend or "-", precision or "-",
                    _num(ref.passes_used),
                    _span(cold.seconds, cold.seconds_min, cold.seconds_max) if cold else "-",
                    _span(warm.seconds, warm.seconds_min, warm.seconds_max) if warm else "-",
                    _num(ref.transfer_s) if ref.device == "gpu" else "-",
                    f"{ref.objective:.5g} (n={ref.n_seeds})",
                    vs,
                    label,
                ]
            )  # fmt: skip
    return _table(header, body), summary


def _sweep_tables(rows: list[Row]) -> list[str]:
    cells: dict[Any, list[Row]] = defaultdict(list)
    for r in rows:
        if r.kind == "sweep":
            cells[(r.data, r.mode, r.n, r.k)].append(r)
    out: list[str] = []
    for (data, mode, n, k), cell_rows in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3])):
        out += [f"### {data} ({mode}), n={n}, k={k}, n x k = {_nxk(n, k)}", ""]
        keyed: dict[Any, dict[str, Row]] = defaultdict(dict)
        for r in cell_rows:
            keyed[(r.method, r.passes_requested or 0.0)][r.timing] = r
        body = []
        for (method, passes), by in sorted(keyed.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            ref = by.get("cold") or by["warm"]
            cold, warm = by.get("cold"), by.get("warm")
            body.append(
                [
                    method, _num(passes), _num(ref.passes_used), f"{ref.objective:.5g}",
                    _num(cold.seconds) if cold else "-", _num(warm.seconds) if warm else "-", str(ref.n_seeds),
                ]
            )  # fmt: skip
        out += _table(["method", "passes", "passes used", "objective", "cold s", "warm s", "seeds"], body)
    return out


def _not_run(records: list[dict[str, Any]]) -> list[str]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in records:
        if r.get("status") == "ok":
            continue
        key = (
            r["data"], r["n"], r["k"], r["kind"], r.get("passes_requested"), r["method"], r["timing"],
            r.get("status"), r.get("skipped_reason"),
        )  # fmt: skip
        groups[key].append(r["seed"])
    body = []
    for key in sorted(groups, key=lambda g: (g[0], g[1], g[2], g[3], g[4] or 0, g[5], g[6])):
        data, n, k, kind, passes, method, timing, status, reason = key
        seeds = ",".join(str(s) for s in sorted(groups[key]))
        body.append([data, str(n), str(k), kind, _num(passes), method, timing, str(status), str(reason), seeds])
    if not body:
        return ["Nothing was skipped, timed out or failed.", ""]
    return _table(["data", "n", "k", "kind", "passes", "method", "timing", "status", "reason", "seeds"], body)


def load_precision(path: Path) -> list[dict[str, Any]]:
    """
    Read precision records, or return none when the study has not been run.

    :param path: JSONL file written by `precision.py`.
    :returns: The records (last record of a repeated key wins); empty when the file is missing.
    """
    return load_records(path) if path.exists() else []


PrecisionGroup = tuple[str, str, int, int]


def _group(record: dict[str, Any]) -> PrecisionGroup:
    return (record["data"], record["mode"], record["n"], record["k"])


def _label(group: PrecisionGroup) -> str:
    data, mode, n, k = group
    return f"{data} {mode} n={n} k={k}"


def aggregate_precision(records: Iterable[dict[str, Any]]) -> dict[PrecisionGroup, dict[str, dict[str, Any]]]:
    """
    Aggregate precision records over seeds.

    Variants that fell back to another precision or produced non-finite centroids are excluded from the statistics
    and counted in `excluded`.

    :param records: Records as written by `precision.py`.
    :returns: For each (data, mode, n, k) and variant: seed count, drift_gap median of medians and max of maxima,
        share of same cluster and objective delta (percent) as (median, min, max), and the excluded count.
    """
    gathered: dict[PrecisionGroup, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get("status") != "ok":
            continue
        for name, entry in (record.get("variants") or {}).items():
            gathered[_group(record)][name].append(entry)
    out: dict[PrecisionGroup, dict[str, dict[str, Any]]] = {}
    for group, variants in gathered.items():
        out[group] = {}
        for name, entries in variants.items():
            good = [e for e in entries if e.get("status") == "ok" and "drift_gap_median" in e]
            stats: dict[str, Any] = {"seeds": len(good), "excluded": len(entries) - len(good)}
            if good:
                gap = [e["drift_gap_median"] for e in good]
                share = [e["share_same_cluster"] for e in good]
                delta = [100.0 * e["objective_delta_rel"] for e in good]
                stats.update(
                    drift_gap_median=statistics.median(gap),
                    drift_gap_max=max(e["drift_gap_max"] for e in good),
                    share=(statistics.median(share), min(share), max(share)),
                    delta=(statistics.median(delta), min(delta), max(delta)),
                )
            out[group][name] = stats
    return out


def spike_contradictions(records: Iterable[dict[str, Any]]) -> list[str]:
    """
    List results that contradict the Phase 2 precision spike (D-23).

    The spike found no overflow, deterministic reruns that agree exactly, and reduced precision that drifts less
    from FP32 than a change of seed does. A contradiction is reported as measured; nothing here changes guidance.

    :param records: Records as written by `precision.py`.
    :returns: One message per problem, naming the cell and the numbers; empty when nothing contradicts the spike.
    """
    records = [r for r in records if r.get("status") == "ok"]
    messages: list[str] = []
    for record in records:
        cell = f"{_label(_group(record))} seed {record.get('seed')}"
        for name, entry in (record.get("variants") or {}).items():
            if entry.get("nonfinite"):
                messages.append(f"{cell}: {name} produced non-finite centroids")
            if entry.get("status") == "precision_fallback":
                messages.append(
                    f"{cell}: {name} ran as {entry.get('precision_used')} instead of "
                    f"{entry.get('precision_requested')} (silent fallback)"
                )
        rerun = (record.get("variants") or {}).get("fp32_rerun") or {}
        if rerun.get("drift_gap_max", 0.0) > 0:
            messages.append(
                f"{cell}: deterministic FP32 rerun drifted (drift_gap max {rerun['drift_gap_max']:.3g}, expected 0)"
            )
    for group, variants in sorted(aggregate_precision(records).items()):
        baseline = variants.get("fp32_other_seed", {}).get("drift_gap_median")
        if baseline is None:
            continue
        for name in REDUCED:
            value = variants.get(name, {}).get("drift_gap_median")
            if value is not None and value >= baseline:
                messages.append(
                    f"{_label(group)}: {name} drift_gap median {value:.3g} is not below the "
                    f"FP32 other-seed baseline {baseline:.3g}"
                )
    return messages


def _precision_rows(
    group_stats: dict[str, dict[str, Any]], names: Sequence[str], group: PrecisionGroup
) -> list[list[str]]:
    data, mode, n, k = group
    body = []
    for name in names:
        stats = group_stats.get(name)
        if stats is None:
            continue
        if "drift_gap_median" not in stats:
            body.append(
                [data, mode, str(n), str(k), name, str(stats["seeds"]), "-", "-", "-", "-", str(stats["excluded"])]
            )
            continue
        body.append(
            [
                data, mode, str(n), str(k), name, str(stats["seeds"]),
                _num(stats["drift_gap_median"]), _num(stats["drift_gap_max"]),
                _span(*stats["share"]), _span(*stats["delta"]), str(stats["excluded"]),
            ]
        )  # fmt: skip
    return body


_PRECISION_HEADER = [
    "data", "mode", "n", "k", "variant", "seeds", "drift_gap median", "drift_gap max",
    "share same cluster", "objective vs FP32 (%)", "excluded",
]  # fmt: skip


def render_precision(records: list[dict[str, Any]]) -> list[str]:
    """
    Render the Precision section.

    :param records: Records as written by `precision.py`.
    :returns: Markdown lines.
    """
    out = ["## Precision", ""]
    if not any(r.get("status") == "ok" for r in records):
        return [*out, "Precision study not run.", "", "Spike check: not evaluated (no precision records).", ""]
    out += [
        (
            "All fits are deterministic from the same shared start; the reference is FP32. The baselines are an FP32 "
            "deterministic rerun and an FP32 run with another seed and another start. FP16 and BF16 run on the "
            "e-means-rocm fork's Triton rung. Centroids are matched one to one (Hungarian) before measuring. "
            "`drift_gap` 1.0 means a centroid moved as far as its nearest other reference centroid; the table shows "
            "the median over seeds of the per-run median and the maximum over seeds of the per-run maximum. Labels "
            "and objective come from the shared exact FP32 scorer. Runs that fell back to another precision or "
            "produced non-finite centroids are left out of the statistics and counted under `excluded`. "
            "Cells show `median [min-max]` over seeds."
        ),
        "",
    ]
    aggregated = aggregate_precision(records)
    body: list[list[str]] = []
    floor: list[list[str]] = []
    for group in sorted(aggregated):
        body += _precision_rows(aggregated[group], PRECISION_VARIANTS, group)
        floor += _precision_rows(aggregated[group], FLOOR_VARIANTS, group)
    out += _table(_PRECISION_HEADER, body)
    out += [
        "### Storage floor",
        "",
        "FP32 reference centroids rounded to FP16 or BF16 and back: the least drift possible.",
        "",
    ]
    out += _table(_PRECISION_HEADER, floor)
    problems = [f"{r['key']}: {r.get('status')}: {r.get('reason')}" for r in records if r.get("status") != "ok"]
    if problems:
        out += ["Cells without a result:", "", *[f"- {p}" for p in problems], ""]
    contradictions = spike_contradictions(records)
    if contradictions:
        out += ["Spike check: results contradict the Phase 2 spike; reported as measured, guidance not changed:", ""]
        out += [f"- {m}" for m in contradictions]
        out.append("")
    else:
        out += [f"Spike check: {SPIKE_CONSISTENT}.", ""]
    return out


def render(
    records: list[dict[str, Any]],
    *,
    title: str = "Benchmark results",
    smoke: bool = False,
    precision_records: list[dict[str, Any]] | None = None,
) -> str:
    """
    Render RESULTS.md.

    :param records: Records as written by `run_grid`.
    :param title: Page title.
    :param smoke: Mark the page as a smoke run, not results.
    :param precision_records: Records of the precision study; None leaves the Precision section out, an empty list
        says the study was not run.
    :returns: Markdown text.
    """
    rows = aggregate(records)
    table, summary = _main_table(rows)
    lines = _header(records, title, smoke) + _protocol()
    lines += ["## Main results", "", "Cold and warm are `median [min-max]` seconds over seeds.", ""]
    lines += table
    lines += ["## Where the GPU does not help / where CPU e-means wins", ""]
    lines += [*summary, ""] if summary else ["No cell carries a mark in these records.", ""]
    lines += ["## Pass sweep", ""]
    lines += _sweep_tables(rows) or ["No sweep records.", ""]
    lines += ["## Not run", ""]
    lines += _not_run(records)
    if precision_records is not None:
        lines += render_precision(precision_records)
    return "\n".join(lines)


def check(records: list[dict[str, Any]], expected_cells: Iterable[Cell], min_seeds: int = 3) -> list[str]:
    """
    Check a run for completeness.

    :param records: Records as written by `run_grid`.
    :param expected_cells: Cells the full grid should have covered.
    :param min_seeds: Minimum ok seeds for any (cell, method) that has ok records.
    :returns: Problems found; empty when complete.
    """
    have = {r.get("key") for r in records}
    problems = []
    ok_seeds: dict[tuple, set[int]] = defaultdict(set)
    for r in records:
        if r.get("status") == "ok" and r.get("timing") == "cold":
            ok_seeds[(r["method"], r["data"], r["n"], r["k"], r["kind"], r.get("passes_requested"))].add(r["seed"])
    for cell in expected_cells:
        for timing in ("cold", "warm"):
            if f"{cell.key}|{timing}" not in have:
                problems.append(f"missing {timing} record: {cell.key}")
    for key, seeds in sorted(ok_seeds.items(), key=lambda kv: str(kv[0])):
        if len(seeds) < min_seeds:
            problems.append(f"fewer than {min_seeds} seeds ({len(seeds)}) for ok row: {key}")
    return problems


def check_precision(records: list[dict[str, Any]], expected_cells: Iterable[PCell], min_seeds: int = 3) -> list[str]:
    """
    Check the precision study for completeness.

    :param records: Records as written by `precision.py`; empty means the study was not run.
    :param expected_cells: Cells the full precision grid should have covered.
    :param min_seeds: Minimum ok seeds per (data, mode, n, k).
    :returns: Problems found; empty when complete.
    """
    if not records:
        return ["precision study not run"]
    by_key = {r.get("key"): r for r in records}
    problems = []
    for cell in expected_cells:
        found = by_key.get(cell.key)
        if found is None:
            problems.append(f"missing precision cell: {cell.key}")
        elif found.get("status") != "ok":
            problems.append(f"precision cell {cell.key}: {found.get('status')}: {found.get('reason')}")
    seeds: dict[PrecisionGroup, set[int]] = defaultdict(set)
    for r in records:
        if r.get("status") == "ok":
            seeds[_group(r)].add(r["seed"])
    for group, have in sorted(seeds.items()):
        if len(have) < min_seeds:
            problems.append(f"fewer than {min_seeds} seeds ({len(have)}) for precision cell: {_label(group)}")
    return problems


def main() -> int:
    """Write RESULTS.md from a records file, or check a full run for completeness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--precision", type=Path, default=None)
    args = parser.parse_args()
    smoke_dir = BENCH_DIR / "scratch" / "smoke"
    results = args.results or (smoke_dir / "speed.jsonl" if args.smoke else BENCH_DIR / "results" / "speed.jsonl")
    out = args.out or (smoke_dir / "RESULTS.md" if args.smoke else BENCH_DIR / "RESULTS.md")
    if not results.exists():
        sys.stderr.write(f"no records at {results}\n")
        return 2
    records = load_records(results)
    precision_path = args.precision or (
        smoke_dir / "precision.jsonl" if args.smoke else BENCH_DIR / "results" / "precision.jsonl"
    )
    precision_records = load_precision(precision_path)
    if args.check:
        methods = sorted({r["method"] for r in records})
        datasets = sorted({r["data"] for r in records})
        problems = check(records, full_grid(methods, datasets))
        precision_datasets = sorted({r["data"] for r in precision_records if "data" in r}) or datasets
        problems += check_precision(precision_records, precision_grid(precision_datasets))
        contradictions = spike_contradictions(precision_records)
        for problem in problems:
            sys.stdout.write(problem + "\n")
        if contradictions:
            sys.stdout.write("Spike contradictions (D-23: stop and ask the owner)\n")
            for message in contradictions:
                sys.stdout.write(message + "\n")
        return 1 if problems or contradictions else 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(records, smoke=args.smoke, precision_records=precision_records), encoding="utf-8")
    sys.stderr.write(f"wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
