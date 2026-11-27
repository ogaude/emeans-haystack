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

import report_notes as notes
from data import COHERE_SUBSET_ROWS_SHA256, COHERE_SUBSET_SEED
from methods import METHODS
from precision import KS as PRECISION_KS
from precision import PCell, precision_grid
from run_grid import DEFAULT_CAP_MIN, METHOD_NS, NS, SEEDS, SWEEP_CELLS, SWEEP_PASSES, WARM_MAX_S, Cell, full_grid

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
CPU_ONLY_WINS = "CPU e-means wins (CPU only)"
EMEANS_CPU = "emeans-cpu"
LLOYD_METHODS = ("faiss-lloyd", "sklearn-lloyd")
SKLEARN_MAX_ITER = 300
FAISS_NITER = 25
DEFAULT_PASSES = 5.0
DRIFT_DIFF = 0.005
PER_PASS_SAME = 0.05
DEGENERATE_ROWS_PER_CENTROID = 2
SAME_OBJECTIVE = 1e-4
RATIO_AS_TIMES = 2
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
            "- scikit-learn runs at 100k rows only, to save time: `faiss-lloyd` is the Lloyd baseline at every "
            "size, so the 1M rows have no scikit-learn entries."
        ),
        (
            "- Final objective: mean squared distance to the nearest centroid, from one exact FP32 CPU scorer "
            "(`faiss.IndexFlatL2`), never a library's own objective. The `vs faiss` column is the relative "
            "difference to the Lloyd `faiss-lloyd` median objective in the same cell."
        ),
        (
            "- Real data is spherical. `faiss-lloyd` (`faiss.Kmeans(spherical=True)`) normalises its centroids "
            "to unit length after every iteration: it is the true spherical Lloyd baseline, and spherical "
            "comparisons should be read against it. scikit-learn has no spherical mode: `sklearn-lloyd` runs "
            "Euclidean Lloyd on the unit-norm rows and projects its final centroids to unit length once, after "
            "training, so it is scored on the sphere but is not spherical Lloyd (no projection per iteration). "
            "The Cohere file's rows are not unit-norm (raw L2 norms about 12 to 17.5): the harness normalises "
            "every row to unit length when it loads them. "
            "Random data is Euclidean and is for speed only: Gaussian noise has no cluster structure."
        ),
        (
            "- The real 100k set is a seeded, sorted random subset of the 1M Cohere rows: seed "
            f"{COHERE_SUBSET_SEED} (`COHERE_SUBSET_SEED` in `data.py`), row-index sha256 "
            f"`{COHERE_SUBSET_ROWS_SHA256}`. "
            "The records published here do not carry the subset seed, row-index sha256 or raw norm range per "
            "record; records written by later runs do."
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
    ]
    out += [f"- {name}: {value}" for name, value in fields if value is not None]
    out += _engine_lines(ok)
    out.append("")
    return out


def _engine_lines(ok: list[dict[str, Any]]) -> list[str]:
    """
    List the e-means engine of every e-means method, from each record's own `versions.emeans_source`.

    One results file is written by several envs, so one header label cannot name them all. Old records store a bare
    `faiss-cpu`, new ones `faiss-cpu <version>` or `emeans-rocm <version>`; both are shown as stored. Lloyd
    baselines run no e-means engine and are left out.
    """
    sources: dict[str, set[str]] = defaultdict(set)
    for r in ok:
        method = r.get("method")
        if not method:
            continue
        engine = r.get("engine") or (METHODS[method].engine if method in METHODS else "")
        if engine != "e-means":
            continue
        source = (r.get("versions") or {}).get("emeans_source")
        sources[method].add(str(source) if source else "not recorded")
    if not sources:
        return []
    out = ["- e-means engine per method:"]
    out += [f"  - {method}: {', '.join(sorted(found))}" for method, found in sorted(sources.items())]
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


def _spike_values(records: list[dict[str, Any]]) -> str:
    """Unrounded drift medians of every reduced variant not below its other-seed baseline."""
    parts = []
    for group, variants in sorted(aggregate_precision(records).items()):
        base = variants.get("fp32_other_seed", {}).get("drift_gap_median")
        if base is None:
            continue
        hits = [
            f"{name} {variants[name]['drift_gap_median']:.3f}"
            for name in REDUCED
            if variants.get(name, {}).get("drift_gap_median", -1.0) >= base
        ]
        if hits:
            data, mode, n, k = group
            parts.append(f"{data} {mode} {_n_label(n)} x {k}: {', '.join(hits)} vs {base:.3f}")
    return "; ".join(parts)


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
            "emeans-rocm package's Triton rung. Centroids are matched one to one (Hungarian) before measuring. "
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
    out += precision_summary(records)
    problems = [f"{r['key']}: {r.get('status')}: {r.get('reason')}" for r in records if r.get("status") != "ok"]
    if problems:
        out += ["Cells without a result:", "", *[f"- {p}" for p in problems], ""]
    contradictions = spike_contradictions(records)
    if contradictions:
        out += ["Spike check: results contradict the Phase 2 spike; reported as measured, guidance not changed:", ""]
        out += [f"- {m}" for m in contradictions]
        if len(contradictions) == notes.ACCEPTED_CONTRADICTIONS_026:
            history = (
                f"These {len(contradictions)} contradictions of the 0.2.6 records were accepted by the owner on "
                "2026-11-27 (those of the 0.2.5 records on 2026-11-23), who kept option B: the caution above, no "
                "other change to the precision guidance."
            )
        else:
            history = (
                f"These {len(contradictions)} contradictions have not been reviewed by the owner: the acceptance of "
                f"2026-11-27 covered {notes.ACCEPTED_CONTRADICTIONS_026} contradictions of the 0.2.6 records. The "
                "guidance is still option B (owner, 2026-11-23): the caution above."
            )
        out += [
            "",
            (
                "Each contradiction is a reduced-precision drift equal to or slightly above the FP32 other-seed drift "
                f"({_spike_values(records)}). {history}"
            ),
            "",
        ]
    else:
        out += [f"Spike check: {SPIKE_CONSISTENT}.", ""]
    return out


def _n_label(n: int) -> str:
    return f"{n // 1_000_000}M" if n % 1_000_000 == 0 else f"{n // 1000}k"


def _cell_label(data: str, n: int, k: int) -> str:
    return f"{data} {_n_label(n)} x {k}"


def _pct(value: float) -> str:
    return f"{value * 100:+.2f} %"


def _obj(row: Row) -> str:
    return f"{row.objective:.5g}"


def _passes(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:g} pass" if value == 1 else f"{value:g} passes"


def _per_pass_cost(emeans: Row, lloyd: Row) -> str:
    """How e-means' cost per pass compares with Lloyd's, in words, with both values."""
    mine, theirs = _per_pass(emeans), _per_pass(lloyd)
    if not mine or not theirs:
        return "per-pass cost not recorded"
    rel = mine / theirs - 1
    values = f"{_num(mine)} vs {_num(theirs)} s/pass"
    if abs(rel) <= PER_PASS_SAME:
        return f"a pass costs about the same ({values})"
    return f"an e-means pass costs {rel * 100:+.0f} % ({values})"


def _per_pass(row: Row) -> float | None:
    """Seconds per pass of a row: its median time over its median passes used."""
    return row.seconds / row.passes_used if row.passes_used else None


SweepCell = tuple[str, str, int, int]


def _sweep_points(rows: Iterable[Row]) -> dict[SweepCell, dict[str, dict[float, Row]]]:
    """Sweep rows by cell, method and requested passes; the cold row when there is one, else the warm row."""
    out: dict[SweepCell, dict[str, dict[float, Row]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.kind != "sweep" or r.passes_requested is None:
            continue
        slot = out[(r.data, r.mode, r.n, r.k)][r.method]
        if r.passes_requested not in slot or r.timing == "cold":
            slot[r.passes_requested] = r
    return out


def _main_points(rows: Iterable[Row]) -> dict[SweepCell, dict[tuple[str, str, str], Row]]:
    """Main rows by cell and (method, backend, precision); the cold row when there is one, else the warm row."""
    out: dict[SweepCell, dict[tuple[str, str, str], Row]] = defaultdict(dict)
    for r in rows:
        if r.kind != "main":
            continue
        slot = out[(r.data, r.mode, r.n, r.k)]
        key = (r.method, r.backend, r.precision)
        if key not in slot or r.timing == "cold":
            slot[key] = r
    return out


@dataclass(frozen=True)
class PassReach:
    """How many swept e-means passes reach one Lloyd method's objective at its largest swept pass count."""

    cell: SweepCell
    lloyd: Row
    emeans: Row | None
    earlier: Row | None
    emeans_default: Row | None
    lloyd_to_default: Row | None
    lloyd_points: tuple[Row, ...]
    emeans_points: tuple[Row, ...]

    @property
    def emeans_last(self) -> Row:
        """The e-means point with the most passes (its s/pass is the least affected by start-up)."""
        return self.emeans_points[-1]

    @property
    def pass_ratio(self) -> str:
        """Lloyd passes over e-means passes, as a range: e-means reached the target between two swept points."""
        if self.emeans is None or not self.lloyd.passes_used or not self.emeans.passes_used:
            return "-"
        low = self.lloyd.passes_used / self.emeans.passes_used
        if self.earlier is None or not self.earlier.passes_used:
            return f">= {low:.1f}x"
        return f"{low:.1f}-{self.lloyd.passes_used / self.earlier.passes_used:.1f}x"

    @property
    def time_ratio(self) -> float | None:
        """Lloyd seconds over the e-means seconds that reached the same objective (both CPU, cold)."""
        return self.lloyd.seconds / self.emeans.seconds if self.emeans is not None and self.emeans.seconds else None


def passes_to_objective(rows: Iterable[Row], emeans: str = EMEANS_CPU) -> list[PassReach]:
    """
    Find, per sweep cell and Lloyd method, the fewest swept e-means passes that reach Lloyd's objective.

    The target is Lloyd's median objective at its largest swept pass count. "Reach" means a median objective at or
    below the target (lower is better); nothing is interpolated between swept points.

    :param rows: Aggregated rows.
    :param emeans: The e-means method to compare.
    :returns: One entry per (sweep cell, Lloyd method) that has both methods.
    """
    out = []
    for cell, methods in sorted(_sweep_points(rows).items()):
        mine = methods.get(emeans)
        if not mine:
            continue
        points = [mine[p] for p in sorted(mine)]
        for lloyd in LLOYD_METHODS:
            theirs = methods.get(lloyd)
            if not theirs:
                continue
            target = theirs[max(theirs)]
            index = next((i for i, r in enumerate(points) if r.objective <= target.objective), None)
            default = mine.get(DEFAULT_PASSES)
            match = (
                next((theirs[p] for p in sorted(theirs) if theirs[p].objective <= default.objective), None)
                if default is not None
                else None
            )
            out.append(
                PassReach(
                    cell=cell,
                    lloyd=target,
                    emeans=points[index] if index is not None else None,
                    earlier=points[index - 1] if index else None,
                    emeans_default=default,
                    lloyd_to_default=match,
                    lloyd_points=tuple(theirs[p] for p in sorted(theirs)),
                    emeans_points=tuple(points),
                )
            )
    return out


def _per_pass_summary(reaches: list[PassReach], compares: "list[MainCompare]") -> list[str]:
    """What one CPU e-means pass costs relative to one Lloyd iteration, measured, as a range."""
    out = ["Per-pass cost of CPU e-means relative to Lloyd's k-means (same CPU, cold medians):", ""]
    for r in reaches:
        mine, theirs = _per_pass(r.emeans_last), _per_pass(r.lloyd)
        if mine and theirs:
            rel = f"{(mine / theirs - 1) * 100:+.1f} %" if mine < RATIO_AS_TIMES * theirs else f"{mine / theirs:.1f}x"
            out.append(
                f"- sweep, 25-pass points, {_cell_label(r.cell[0], r.cell[2], r.cell[3])} vs `{r.lloyd.method}`: "
                f"{rel} ({_num(mine)} vs {_num(theirs)} s/pass)"
            )
    by_k: dict[tuple[str, int], list[float]] = defaultdict(list)
    for c in compares:
        mine = _per_pass(c.emeans)
        for lloyd in c.cpu_lloyd:
            theirs = _per_pass(lloyd)
            if mine and theirs:
                by_k[(lloyd.method, c.cell[3])].append(mine / theirs)
    for (method, k), values in sorted(by_k.items()):
        if max(values) < RATIO_AS_TIMES:
            low, high = f"{(min(values) - 1) * 100:+.0f} %", f"{(max(values) - 1) * 100:+.0f} %"
        else:
            low, high = f"{min(values):.1f}x", f"{max(values):.1f}x"
        text = low if low == high else f"{low} to {high}"
        out.append(f"- main cells, k = {k}, vs `{method}`: {text} ({len(values)} cells)")
    return out


def _render_passes(reaches: list[PassReach], compares: "list[MainCompare]") -> list[str]:
    out = [
        "## Passes to the same objective",
        "",
        (
            "The measured advantage of e-means is fewer passes (epochs) to reach the same objective, not faster "
            "passes (per-pass cost below). Target: the Lloyd's "
            "k-means median objective at its largest swept pass count (25 requested; passes actually used shown). "
            "`e-means passes`: the fewest swept CPU e-means passes whose median objective is at or below the target. "
            f"Swept points are {', '.join(f'{p:g}' for p in SWEEP_PASSES)}, nothing is interpolated, so the pass "
            "ratio is a range: e-means got there after the previous point and by this one. `one step earlier` is "
            "the previous point's distance to the target. `Lloyd to match e-means at 5`: the fewest swept Lloyd "
            "passes reaching CPU e-means' objective at its default 5 passes. Seconds are cold medians on the same "
            "CPU; s/pass is taken from each method's 25-pass point."
        ),
        "",
    ]
    if not reaches:
        return [*out, "No sweep records.", ""]
    out += [*_per_pass_summary(reaches, compares), ""]
    body = []
    for r in reaches:
        data, _mode, n, k = r.cell
        earlier = "-"
        if r.earlier is not None:
            earlier = f"{_passes(r.earlier.passes_used)}: {_pct(r.earlier.objective / r.lloyd.objective - 1)}"
        match = "-"
        if r.emeans_default is not None:
            match = f"{r.lloyd_to_default.passes_used:g}" if r.lloyd_to_default else f"not within {max_passes(r)}"
        ratio = r.time_ratio
        body.append(
            [
                data, _n_label(n), str(k), r.lloyd.method, _num(r.lloyd.passes_used), _obj(r.lloyd),
                _num(_per_pass(r.lloyd)), _num(r.lloyd.seconds),
                _num(r.emeans.passes_used) if r.emeans else f"not within {max_passes(r)}",
                _obj(r.emeans) if r.emeans else "-", _num(_per_pass(r.emeans_last)),
                _num(r.emeans.seconds) if r.emeans else "-", r.pass_ratio,
                f"{ratio:.1f}x" if ratio else "-", earlier, match,
            ]
        )  # fmt: skip
    header = [
        "data", "n", "k", "Lloyd method", "Lloyd passes used", "Lloyd objective", "Lloyd s/pass", "Lloyd s",
        "e-means passes", "e-means objective", "e-means s/pass", "e-means s", "pass ratio", "time ratio",
        "one step earlier", "Lloyd to match e-means at 5",
    ]  # fmt: skip
    return out + _table(header, body)


def max_passes(reach: PassReach) -> str:
    """The largest swept pass count of a comparison, for `not within N` cells."""
    return f"{max(reach.lloyd.passes_requested or 0, reach.emeans_last.passes_requested or 0):g}"


@dataclass(frozen=True)
class MainCompare:
    """CPU e-means at its default passes against the best Lloyd's k-means run of one main cell."""

    cell: SweepCell
    emeans: Row
    lloyd: Row | None
    cpu_lloyd: tuple[Row, ...]

    def gap_to(self, method: str) -> float | None:
        """Relative objective difference of e-means to one Lloyd method's row, or None when it did not run."""
        row = next((r for r in self.cpu_lloyd if r.method == method), None)
        return self.emeans.objective / row.objective - 1 if row else None

    @property
    def gap(self) -> float | None:
        """Relative objective difference of e-means to the best Lloyd (negative: e-means lower, better)."""
        return self.emeans.objective / self.lloyd.objective - 1 if self.lloyd else None

    @property
    def cpu_only_win(self) -> bool:
        """CPU e-means is faster than every CPU Lloyd and within the objective tolerance of the best one."""
        if not self.cpu_lloyd or self.lloyd is None:
            return False
        faster = all(self.emeans.seconds < r.seconds for r in self.cpu_lloyd)
        return faster and self.emeans.objective <= self.lloyd.objective * (1 + OBJECTIVE_TOLERANCE)


def main_comparison(rows: Iterable[Row]) -> list[MainCompare]:
    """
    Compare CPU e-means with the best-objective Lloyd's k-means run in every main cell (cold timings).

    :param rows: Aggregated rows.
    :returns: One entry per main cell that has a CPU e-means row.
    """
    out = []
    for cell, by in sorted(_main_points(rows).items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3])):
        mine = [r for (method, _b, _p), r in by.items() if method == EMEANS_CPU]
        if not mine:
            continue
        lloyd = [r for r in by.values() if r.engine == "lloyd" or r.method in LLOYD_METHODS]
        best = min(lloyd, key=lambda r: r.objective) if lloyd else None
        out.append(MainCompare(cell, mine[0], best, tuple(r for r in lloyd if r.device != "gpu")))
    return out


def _stop(row: Row) -> str:
    used = row.passes_used
    if used is None:
        return "-"
    if row.method == "sklearn-lloyd":
        return f"converged (tol 1e-4) after {used:g}" if used < SKLEARN_MAX_ITER else f"max_iter {SKLEARN_MAX_ITER}"
    cap = row.passes_requested or FAISS_NITER
    return f"ran all {cap:g}" if used >= cap else f"stopped after {used:g} of {cap:g}"


def _render_main_comparison(compares: list[MainCompare]) -> list[str]:
    out = [
        "## e-means at 5 passes vs Lloyd's k-means, per main cell",
        "",
        (
            "CPU e-means at its default 5 passes against the Lloyd's k-means run (`faiss.Kmeans`, scikit-learn "
            "`KMeans`) with the lowest median objective in the cell, at the passes it used (medians over seeds). "
            "scikit-learn stops on `tol` 1e-4 (converged); `faiss.Kmeans` runs at most 25 passes and may stop "
            "earlier. Cold medians on the same CPU. "
            f"`{CPU_ONLY_WINS}`: without a GPU, CPU e-means is faster than every CPU Lloyd run and its objective is "
            f"within {OBJECTIVE_TOLERANCE * 100:g} % of the best Lloyd median."
        ),
        "",
    ]
    body = []
    for c in compares:
        data, mode, n, k = c.cell
        lloyd = c.lloyd
        body.append(
            [
                data, mode, _n_label(n), str(k), f"{n / k:.1f}", _obj(c.emeans), _num(c.emeans.seconds),
                _num(_per_pass(c.emeans)),
                lloyd.method if lloyd else "not run", _stop(lloyd) if lloyd else "-", _obj(lloyd) if lloyd else "-",
                _num(lloyd.seconds) if lloyd else "-", _num(_per_pass(lloyd)) if lloyd else "-",
                _pct(c.gap) if c.gap is not None else "-", CPU_ONLY_WINS if c.cpu_only_win else "",
            ]
        )  # fmt: skip
    header = [
        "data", "mode", "n", "k", "rows per centroid", "e-means objective", "e-means s", "e-means s/pass",
        "best Lloyd", "Lloyd passes", "Lloyd objective", "Lloyd s", "Lloyd s/pass", "e-means vs Lloyd", "mark",
    ]  # fmt: skip
    return out + _table(header, body)


@dataclass(frozen=True)
class GpuSpeedup:
    """Best GPU e-means end-to-end time against CPU e-means in one main cell and timing."""

    cell: SweepCell
    timing: str
    cpu: Row
    gpu: Row

    @property
    def speedup(self) -> float:
        """CPU e-means seconds over the best GPU end-to-end seconds."""
        return self.cpu.seconds / self.gpu.end_to_end


def gpu_speedups(rows: Iterable[Row]) -> list[GpuSpeedup]:
    """
    List, per main cell and timing, the best GPU e-means row against CPU e-means.

    :param rows: Aggregated rows.
    :returns: Entries for every main cell and timing that has both.
    """
    by: dict[tuple[SweepCell, str], list[Row]] = defaultdict(list)
    for r in rows:
        if r.kind == "main":
            by[((r.data, r.mode, r.n, r.k), r.timing)].append(r)
    out = []
    for (cell, timing), sel in sorted(by.items(), key=lambda kv: (kv[0][0][0], kv[0][0][2], kv[0][0][3], kv[0][1])):
        cpu = [r for r in sel if r.method == EMEANS_CPU]
        gpu = [r for r in sel if r.engine == "e-means" and r.device == "gpu"]
        if cpu and gpu:
            out.append(GpuSpeedup(cell, timing, cpu[0], min(gpu, key=lambda r: r.end_to_end)))
    return out


def _speedup_line(s: GpuSpeedup) -> str:
    data, _mode, n, k = s.cell
    return (
        f"{s.speedup:.1f}x at {_cell_label(data, n, k)} ({s.timing}: `{s.gpu.method}` {_num(s.gpu.end_to_end)} s "
        f"end-to-end vs {_num(s.cpu.seconds)} s)"
    )


def _render_where(
    summary: list[str], speedups: list[GpuSpeedup], compares: list[MainCompare], rows: list[Row]
) -> list[str]:
    out = ["## Where the GPU does not help / where CPU e-means wins", ""]
    out += [*summary, ""] if summary else [f"No cell carries `{GPU_NOT_HELP}` or `{CPU_WINS}`.", ""]
    if speedups:
        slower = [s for s in speedups if s.speedup <= 1]
        cells = {s.cell for s in speedups}
        if not slower:
            small = min(cells, key=lambda c: c[2] * c[3])
            out.append(
                f"- The GPU helps in every main cell measured: the best GPU e-means run (end-to-end, transfer "
                f"included) is faster than CPU e-means ({_coverage(speedups)}), including the smallest n x k "
                f"({_nxk(small[2], small[3])}). Cells without a CPU warm time have no warm comparison."
            )
        low = min(speedups, key=lambda s: s.speedup)
        high = max(speedups, key=lambda s: s.speedup)
        out.append(f"- Smallest GPU speedup over CPU e-means: {_speedup_line(low)}.")
        lead = _gpu_lead(rows)
        out.append(
            f"- Largest: {_speedup_line(high)}"
            + (", BF16 on the GPU against FP32 on the CPU" if high.gpu.precision != "fp32" else "")
            + "."
        )
        if lead:
            out.append(f"- Per GPU method {lead}.")
        out.append(
            "- CPU e-means is never the fastest method here, because a GPU e-means row is in every cell; "
            f"`{CPU_WINS}` therefore never applies on this host."
        )
    wins = [c for c in compares if c.cpu_only_win]
    comparable = [c for c in compares if c.lloyd is not None]
    if compares:
        labels = ", ".join(_cell_label(c.cell[0], c.cell[2], c.cell[3]) for c in wins) or "none"
        out.append(
            f"- Without a GPU, `{CPU_ONLY_WINS}` in {len(wins)} of {len(comparable)} comparable main cells "
            f"({len(compares) - len(comparable)} have no Lloyd run): CPU e-means finishes sooner than every CPU "
            f"Lloyd run with an objective within {OBJECTIVE_TOLERANCE * 100:g} % of the best one: {labels}."
        )
        for data in sorted({c.cell[0] for c in comparable}):
            mine = [c for c in comparable if c.cell[0] == data]
            out.append(f"  - {data}: {sum(1 for c in mine if c.cpu_only_win)} of {len(mine)} comparable main cells.")
        lost = [c for c in compares if c.gap is not None and c.gap > OBJECTIVE_TOLERANCE]
        if lost:
            text = "; ".join(_lost_text(c, compares) for c in lost)
            out.append(f"- Lloyd's k-means has the better objective at: {text}.")
        none = [c for c in compares if c.lloyd is None]
        if none:
            labels = ", ".join(_cell_label(c.cell[0], c.cell[2], c.cell[3]) for c in none)
            out.append(f"- No Lloyd run, so no comparison: {labels} (see Not run).")
    out.append("")
    return out


def _coverage(speedups: list[GpuSpeedup]) -> str:
    """Which cells and timings the GPU wins, e.g. `cold in all 16 cells; warm in all 12 cells ...`."""
    parts = []
    for timing, scope in (("cold", "cells"), ("warm", "cells that have a CPU warm time")):
        sel = [s for s in speedups if s.timing == timing]
        if not sel:
            continue
        won = sum(1 for s in sel if s.speedup > 1)
        count = f"all {won}" if won == len(sel) else f"{won} of {len(sel)}"
        parts.append(f"{timing} in {count} {scope}")
    return ", ".join(parts)


def _largest_cell_speedups(rows: list[Row]) -> tuple[SweepCell, Row, list[tuple[Row, float]]] | None:
    """CPU e-means and every GPU e-means row (cold, end-to-end) at the largest cell, real data first."""
    main = _main_points(rows)
    cells = [
        c for c, by in main.items()
        if any(r.device == "gpu" for r in by.values()) and any(m == EMEANS_CPU for m, _b, _p in by)
    ]  # fmt: skip
    if not cells:
        return None
    cell = max(cells, key=lambda c: (c[2] * c[3], c[0] == "cohere"))
    cpu = next(r for (m, _b, _p), r in main[cell].items() if m == EMEANS_CPU)
    gpu = [(r, cpu.seconds / r.end_to_end) for r in main[cell].values() if r.device == "gpu" and r.engine == "e-means"]
    return cell, cpu, sorted(gpu, key=lambda pair: (pair[0].precision != "fp32", -pair[1]))


def _gpu_lead(rows: list[Row]) -> str:
    """FP32-vs-FP32 GPU speedup at the largest cell first, then BF16 vs CPU FP32."""
    found = _largest_cell_speedups(rows)
    if found is None:
        return ""
    cell, cpu, gpu = found
    fp32 = [(r, s) for r, s in gpu if r.precision == "fp32"]
    other = [(r, s) for r, s in gpu if r.precision != "fp32"]
    label = _cell_label(cell[0], cell[2], cell[3])
    text = f"at {label} (cold, end-to-end, CPU e-means {_num(cpu.seconds)} s), FP32 against FP32: " + ", ".join(
        f"`{r.method}` {s:.1f}x ({_num(r.end_to_end)} s)" for r, s in fp32
    )
    if other:
        text += "; " + ", ".join(
            f"`{r.method}` {s:.1f}x ({_num(r.end_to_end)} s), {r.precision.upper()} on the GPU against FP32 on the CPU"
            for r, s in other
        )
    return text


def _lost_text(c: MainCompare, compares: list[MainCompare]) -> str:
    """One cell where Lloyd's objective is better: degenerate cells say so, others are reported as found."""
    data, _mode, n, k = c.cell
    text = f"{_cell_label(data, n, k)} ({_pct(c.gap or 0.0)}"
    if n / k < DEGENERATE_ROWS_PER_CENTROID:
        return text + f", degenerate: {n / k:.1f} rows per centroid)"
    twins = [o for o in compares if (o.cell[2], o.cell[3]) == (n, k) and o.cell[0] != data and o.gap is not None]
    twin = "".join(f"; {o.cell[0]} at the same size: {_pct(o.gap or 0.0)}" for o in twins)
    return text + f", reported as found{twin})"


def _render_headline(
    reaches: list[PassReach], compares: list[MainCompare], speedups: list[GpuSpeedup], host: str, rows: list[Row]
) -> list[str]:
    out = ["## Headline", ""]
    reached = [r for r in reaches if r.emeans is not None and r.lloyd.passes_used]
    lost = [c for c in compares if c.gap is not None and c.gap > OBJECTIVE_TOLERANCE]
    if reached and len(reached) == len(reaches):
        fewer = all((r.emeans.passes_used or 0) < (r.lloyd.passes_used or 0) for r in reached if r.emeans)
        ks = " and ".join(str(k) for k in sorted({r.cell[3] for r in reaches}))
        if fewer:
            out.append(
                "- e-means needs fewer passes (epochs) than Lloyd's k-means (`faiss.Kmeans`, scikit-learn `KMeans`) "
                f"to reach the same objective in all {len(reached)} sweep comparisons (k = {ks}). It is not faster "
                "per pass on CPU; the wall-time gain comes from the fewer passes."
                + (" Exceptions at 5 passes in the main cells: see the line below." if lost else "")
            )
    by_cell = {c.cell: c for c in compares}
    faiss = [r for r in reaches if r.lloyd.method == "faiss-lloyd" and r.emeans is not None]
    for r in faiss:
        data, _mode, n, k = r.cell
        assert r.emeans is not None  # noqa: S101
        quoted, lloyd_pp = _per_pass(r.emeans), _per_pass(r.lloyd)
        rel = quoted / lloyd_pp - 1 if quoted and lloyd_pp else None
        main = by_cell.get(r.cell)
        lloyd_main = next((x for x in main.cpu_lloyd if x.method == "faiss-lloyd"), None) if main else None
        main_text = (
            f" (pass-sweep medians; the main-cell runs give {_num(main.emeans.seconds)} s vs "
            f"{_num(lloyd_main.seconds)} s)"
            if main and lloyd_main and main.emeans.passes_used == r.emeans.passes_used
            else " (pass-sweep medians)"
        )
        out.append(
            f"- {_cell_label(data, n, k)}: CPU e-means reaches the objective `faiss.Kmeans` has after "
            f"{_passes(r.lloyd.passes_used)} ({_obj(r.lloyd)}) within {_passes(r.emeans.passes_used)} "
            f"({_obj(r.emeans)}), pass ratio {r.pass_ratio}; at the 25-pass points "
            f"{_per_pass_cost(r.emeans_last, r.lloyd)}"
            + (
                f", at the {_passes(r.emeans.passes_used)} quoted e-means takes {_num(quoted)} s/pass "
                f"({rel * 100:+.0f} % vs `faiss.Kmeans`)"
                if rel is not None
                else ""
            )
            + f"; {_num(r.emeans.seconds)} s vs {_num(r.lloyd.seconds)} s on the same CPU{main_text}."
        )
    sk = [r for r in reaches if r.lloyd.method == "sklearn-lloyd"]
    if sk:
        text = "; ".join(
            f"{_cell_label(r.cell[0], r.cell[2], r.cell[3])}: {_num(_per_pass(r.emeans_last))} vs "
            f"{_num(_per_pass(r.lloyd))} s/pass"
            for r in sk
        )
        out.append(
            f"- Per pass, CPU e-means is not faster than Lloyd: scikit-learn `KMeans` is cheaper per pass at 100k "
            f"({text}, 25-pass points), so the wall-time gain there is smaller than the pass gain."
        )
    for r in reaches:
        data, _mode, n, k = r.cell
        if r.lloyd.method != "faiss-lloyd" or n < max(NS) or len(r.lloyd_points) <= 1:
            continue
        before = r.lloyd_points[-2]
        if r.lloyd.objective < before.objective and r.lloyd.passes_used == r.lloyd.passes_requested:
            out.append(
                f"- At {_cell_label(data, n, k)}, `faiss.Kmeans` used all {r.lloyd.passes_used:g} passes and was "
                f"still improving ({before.passes_used:g} -> {r.lloyd.passes_used:g} passes: {_obj(before)} -> "
                f"{_obj(r.lloyd)}), so 1M comparisons are against Lloyd at 25 passes, not at convergence."
            )
    seen: set[SweepCell] = set()
    for r in reaches:
        data, _mode, n, k = r.cell
        if r.cell in seen or r.emeans_default is None:
            continue
        seen.add(r.cell)
        after = [p for p in r.emeans_points if (p.passes_requested or 0) > DEFAULT_PASSES]
        if any(p.objective > r.emeans_default.objective for p in after):
            values = " / ".join(_obj(p) for p in (r.emeans_default, *after))
            passes = "/".join(f"{p.passes_used:g}" for p in (r.emeans_default, *after))
            out.append(
                f"- {_cell_label(data, n, k)}: after 5 passes the CPU e-means objective is flat and not monotone "
                f"({passes} passes: {values}); reported as found."
            )
    beat = [c for c in compares if c.gap is not None and c.gap <= 0]
    total = [c for c in compares if c.gap is not None]
    if total:
        out.append(
            f"- At its default 5 passes CPU e-means has a lower objective than the best Lloyd's k-means run in "
            f"{len(beat)} of {len(total)} comparable main cells"
            + (f"; Lloyd is better at {', '.join(_lost_text(c, compares) for c in lost)}." if lost else ".")
        )
    if speedups:
        low = min(speedups, key=lambda s: s.speedup)
        lead = _gpu_lead(rows)
        out.append(
            f"- GPU ({host}): the best GPU e-means run is faster than CPU e-means ({_coverage(speedups)})"
            + (f"; {lead}" if lead else "")
            + f"; smallest gain {_speedup_line(low)}."
        )
    out.append("")
    return out


def _render_claims(reaches: list[PassReach], compares: list[MainCompare], speedups: list[GpuSpeedup]) -> list[str]:
    out = [
        "## Upstream claims, checked",
        "",
        (
            f"Quoted from {notes.UPSTREAM_SOURCE} (square brackets mark words added here); the measured lines are "
            "computed from the tables on this page."
        ),
        "",
        f"1. > {notes.CLAIM_EPOCHS}",
        "",
    ]
    for r in reaches:
        if r.emeans is None:
            out.append(f"   - {_cell_label(r.cell[0], r.cell[2], r.cell[3])} vs `{r.lloyd.method}`: not reached.")
            continue
        out.append(
            f"   - {_cell_label(r.cell[0], r.cell[2], r.cell[3])} vs `{r.lloyd.method}` at "
            f"{_passes(r.lloyd.passes_used)}: e-means within {_passes(r.emeans.passes_used)}, pass ratio "
            f"{r.pass_ratio}."
        )
    out += ["", f"2. > {notes.CLAIM_FIVE}", ""]
    for method, label in (
        ("sklearn-lloyd", "against scikit-learn `KMeans` run to convergence (tol 1e-4; 100k rows)"),
        ("faiss-lloyd", "against `faiss.Kmeans` at up to 25 passes"),
    ):
        group = [(c, c.gap_to(method)) for c in compares if c.gap_to(method) is not None]
        if not group:
            continue
        worse = [(c, g) for c, g in group if (g or 0.0) > 0]
        text = ", ".join(f"{_cell_label(c.cell[0], c.cell[2], c.cell[3])} {_pct(g or 0.0)}" for c, g in worse)
        out.append(
            f"   - {label}: e-means at 5 passes is lower in {len(group) - len(worse)} of {len(group)} cells"
            + (f"; higher at {text}." if worse else ".")
        )
    none = [c for c in compares if c.lloyd is None]
    if none:
        out.append(
            "   - Not checked (no Lloyd run, see Not run): "
            + ", ".join(_cell_label(c.cell[0], c.cell[2], c.cell[3]) for c in none)
            + "."
        )
    out += ["", f"3. > {notes.CLAIM_WALL}", ""]
    ratios = [(r, r.time_ratio) for r in reaches if r.time_ratio]
    for r, ratio in ratios:
        out.append(
            f"   - CPU, {_cell_label(r.cell[0], r.cell[2], r.cell[3])}: e-means reaches `{r.lloyd.method}`'s "
            f"{r.lloyd.passes_used:g}-pass objective {ratio:.1f}x faster in wall time."
        )
    out.append(
        "   - Per pass, CPU e-means is not faster than Lloyd (see the s/pass columns); the wall-time gain comes "
        "from fewer passes."
    )
    out += ["", f"4. > {notes.CLAIM_GPU}", ""]
    if speedups:
        below = [s for s in speedups if s.cell[2] * s.cell[3] < ABOVE_1E10]
        if below:
            low = min(below, key=lambda s: s.speedup)
            out.append(
                f"   - On this host the GPU already helps below 1e10: smallest gain below 1e10 is {_speedup_line(low)}."
            )
    out.append("")
    return out


_CAP_PREFIX = "not run: est. "


def _capped(records: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], list[float]]:
    """Estimated hours of the cells the time cap left out, by (method, data, n, k)."""
    found: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    for r in records:
        reason = r.get("skipped_reason") or ""
        if r.get("status") != "ok" and reason.startswith(_CAP_PREFIX) and r.get("timing") == "cold":
            try:
                hours = float(reason.removeprefix(_CAP_PREFIX).split()[0])
            except (IndexError, ValueError):
                continue
            found[(r["method"], r["data"], r["n"], r["k"])].append(hours)
    return found


def _render_cuts(
    records: list[dict[str, Any]],
    rows: list[Row],
    reaches: list[PassReach],
    extra_cuts: tuple[tuple[str, str, str], ...] = (),
) -> list[str]:
    capped = _capped(records)
    main = _main_points(rows)
    hidden = []
    for (method, data, n, k), estimates in sorted(capped.items()):
        by = next((v for c, v in main.items() if (c[0], c[2], c[3]) == (data, n, k)), {})
        cpu = [r for (m, _b, _p), r in by.items() if m == EMEANS_CPU]
        gpu = [r for r in by.values() if r.device == "gpu"]
        extra = ""
        if cpu:
            extra = f"; CPU e-means ran it in {_num(cpu[0].seconds)} s ({cpu[0].seconds / 60:.0f} min)"
        if gpu:
            extra += f", GPU e-means in {_num(min(r.end_to_end for r in gpu))}-{_num(max(r.end_to_end for r in gpu))} s"
        low, high = min(estimates), max(estimates)
        span = f"{low:.1f} h" if low == high else f"{low:.1f}-{high:.1f} h"
        hidden.append(f"`{method}` {_cell_label(data, n, k)} (est. {span}){extra}")
    warm_skipped = [r for r in records if r.get("timing") == "warm" and r.get("skipped_reason") == "skipped (cost)"]
    cpu_warm_methods = sorted({r["method"] for r in warm_skipped})
    still = [
        r for r in reaches
        if r.lloyd.method == "faiss-lloyd" and r.cell[2] == max(NS) and len(r.lloyd_points) > 1
        and r.lloyd.objective < r.lloyd_points[-2].objective
    ]  # fmt: skip
    improving = "; ".join(
        f"{_cell_label(r.cell[0], r.cell[2], r.cell[3])} {r.lloyd_points[-2].passes_used:g} -> "
        f"{r.lloyd.passes_used:g} passes: {_obj(r.lloyd_points[-2])} -> {_obj(r.lloyd)}"
        for r in still
    )
    sweep_cells = ", ".join(_cell_label(d, n, k) for d, n, k in SWEEP_CELLS)
    sk_ns = ", ".join(_n_label(n) for n in METHOD_NS.get("sklearn-lloyd", NS))
    computed = [
        (
            f"scikit-learn runs at {sk_ns} rows only (`METHOD_NS` in `run_grid.py`)",
            "at 1M it would cost most of a 30-40 h CPU grid for a reference point `faiss-lloyd` already gives",
            "scikit-learn speed and objective at 1M; there `faiss-lloyd` is the only Lloyd baseline",
        ),
        (
            (
                f"{DEFAULT_CAP_MIN}-minute cap per cell-seed (`DEFAULT_CAP_MIN`); a cell-seed whose one-pass probe "
                "estimate exceeds it is not run"
            ),
            "one Lloyd cell-seed at 1M x 65536 would take about 3 h",
            ("not run: " + "; ".join(hidden) + ". No Lloyd objective there, so no e-means vs Lloyd comparison")
            if hidden
            else "nothing: no cell was capped",
        ),
        (
            f"Warm repeat skipped for CPU cell-seeds estimated over {WARM_MAX_S} s (`WARM_MAX_S`)",
            "a warm repeat would double the cost of the largest CPU cell-seeds",
            (
                f"{len(warm_skipped)} warm records ({sum(1 for r in warm_skipped if r.get('kind') == 'main')} main "
                f"cell-seeds, {sum(1 for r in warm_skipped if r.get('kind') == 'sweep')} sweep points; "
                f"{', '.join(cpu_warm_methods)}); their cold times are shown"
            ),
        ),
        (
            (
                f"Pass sweep ({', '.join(f'{p:g}' for p in SWEEP_PASSES)} passes) on {len(SWEEP_CELLS)} settings only "
                f"(`SWEEP_CELLS`): {sweep_cells}"
            ),
            "each sweep point is a full run per method and seed",
            "passes-to-objective at other settings; the main cells run fixed passes",
        ),
        (
            f"`faiss.Kmeans` capped at `niter` {FAISS_NITER}; e-means at its default 5 passes",
            "fixed, stated passes per method; a Lloyd pass at 1M x 16384 costs about 2 min on CPU",
            f"Lloyd at convergence at 1M: it was still improving ({improving})" if improving else "nothing measured",
        ),
        (
            f"{len(SEEDS)} seeds per cell (`SEEDS`)",
            "cost",
            "the spread is a min-max over 3 seeds, not a confidence interval",
        ),
        (
            f"Precision study at k = {', '.join(str(k) for k in PRECISION_KS)} only",
            "cost (6 fits per precision cell-seed)",
            "precision effects at larger k",
        ),
    ]
    out = [
        "## Cuts and choices",
        "",
        (
            "Every cut made to save time or memory, with its reason and what it hides. A setting is one "
            "(data, n, k); a cell is a setting run by one method; a cell-seed is one seed of a cell, with a cold "
            "and a warm record; a sweep point is one pass count of a sweep cell-seed."
        ),
        "",
    ]
    return out + _table(
        ["cut or choice", "reason", "what it hides"],
        [list(row) for row in [*computed, _kept_slow_row(records), *notes.STATIC_CUTS, *extra_cuts]],
    )


def _kept_slow_row(records: list[dict[str, Any]]) -> tuple[str, str, str]:
    """The kept slower cell-seeds, each against its clean 2026-11-26 rerun, in s/pass from the records."""
    by_key = {r.get("key"): r for r in records if r.get("status") == "ok"}
    found = []
    for kept, reference, label in notes.KEPT_SLOW:
        a, b = by_key.get(kept), by_key.get(reference)
        if not a or not b or not a.get("passes_used") or not b.get("passes_used"):
            continue
        mine, theirs = a["seconds"] / a["passes_used"], b["seconds"] / b["passes_used"]
        found.append((label, mine, theirs, mine / theirs - 1))
    if not found:
        return ("Slower cell-seeds kept", "-", "not in these records")
    span = f"{min(f[3] for f in found) * 100:.0f}-{max(f[3] for f in found) * 100:.0f} %"
    items = "; ".join(
        f"{label} {mine:.1f} vs {theirs:.1f} s/pass ({rel * 100:+.1f} %)" for label, mine, theirs, rel in found
    )
    return (
        f"Cell-seeds {span} slower than the clean 2026-11-26 reruns were kept, not rerun (owner, 2026-11-26): {items}",
        "the gap matches day-to-day variation; a rerun would cost hours",
        f"per-pass timings vary by {span} between days and machine states; objectives and pass counts are not affected",
    )


def _render_gpu_engines(rows: list[Row]) -> list[str]:
    out = ["## GPU engines", "", notes.GPU_ENGINE_NOTE, ""]
    main = _main_points(rows)
    cells = [c for c, by in main.items() if any(r.device == "gpu" for r in by.values())]
    if not cells:
        return out
    cell = max(cells, key=lambda c: (c[2] * c[3], c[0] == "cohere"))
    data, mode, n, k = cell
    body = []
    for (method, backend, precision), r in sorted(main[cell].items()):
        if r.engine != "e-means":
            continue
        body.append(
            [method, backend, precision, _num(r.passes_used), _num(r.seconds), _num(r.end_to_end),
             f"{r.objective:.5g} [{r.objective_min:.5g}-{r.objective_max:.5g}]"]
        )  # fmt: skip
    out += [f"Largest cell, {data} ({mode}) {_n_label(n)} x {k}, cold medians over seeds:", ""]
    out += _table(["method", "backend", "precision", "passes", "cold s", "end-to-end s", "objective [min-max]"], body)
    warm: dict[SweepCell, dict[str, Row]] = defaultdict(dict)
    for r in rows:
        if r.kind == "main" and r.timing == "warm" and r.device == "gpu":
            warm[(r.data, r.mode, r.n, r.k)][r.method] = r
    pairs = [
        (c, by["emeans-gpu-native-fp32"], by["emeans-gpu-triton-fp32"])
        for c, by in sorted(warm.items())
        if "emeans-gpu-native-fp32" in by and "emeans-gpu-triton-fp32" in by
    ]
    if pairs:
        native = [c for c, a, b in pairs if a.end_to_end < b.end_to_end]
        out.append(
            f"Native FP32 is faster than Triton FP32 (warm, end-to-end) in {len(native)} of {len(pairs)} main cells"
            + (": " + ", ".join(_cell_label(c[0], c[2], c[3]) for c in native) + "." if native else ".")
        )
        out.append("")
    return out


def fork_comparison(
    speed: list[dict[str, Any]],
    fork_speed: list[dict[str, Any]],
    precision: list[dict[str, Any]],
    fork_precision: list[dict[str, Any]],
) -> list[str]:
    """
    Compare emeans-rocm 0.2.5 records with the earlier fork build's records of the same cells.

    :param speed: Current speed records.
    :param fork_speed: Fork-build speed records (same keys).
    :param precision: Current precision records.
    :param fork_precision: Fork-build precision records (same keys).
    :returns: Markdown lines; one line per finding.
    """
    now = {r["key"]: r for r in speed if r.get("status") == "ok"}
    pairs = [(r, now[r["key"]]) for r in fork_speed if r.get("status") == "ok" and r.get("key") in now]
    out = ["### emeans-rocm 0.2.5 vs the earlier fork build", "", notes.FORK_NOTE, ""]
    if pairs:
        warm = [b["seconds"] / a["seconds"] for a, b in pairs if a["timing"] == "warm" and a["seconds"]]
        cold = [b["seconds"] - a["seconds"] for a, b in pairs if a["timing"] == "cold"]
        out.append(
            f"- Speed: {len(pairs)} matching records. Warm time ratio 0.2.5 / fork, median "
            f"{statistics.median(warm):.2f} [{min(warm):.2f}-{max(warm):.2f}]; cold time 0.2.5 - fork, median "
            f"{statistics.median(cold):+.3f} s."
        )
        for precision_name in sorted({a.get("precision") or "" for a, _ in pairs}):
            sel = [(a, b) for a, b in pairs if a.get("precision") == precision_name and a["timing"] == "cold"]
            deltas = [abs(b["objective"] / a["objective"] - 1) for a, b in sel if a["kind"] == "main"]
            same = sum(1 for d in deltas if d <= SAME_OBJECTIVE)
            if deltas:
                out.append(
                    f"- {precision_name} main-cell objectives: largest difference {max(deltas) * 100:.3f} %, "
                    f"{same} of {len(deltas)} records within 0.01 %."
                )
        groups: dict[tuple, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
        for a, b in pairs:
            if a["timing"] == "cold" and a["kind"] == "main" and a.get("precision") == "fp32":
                cell = (a["method"], a["data"], a["n"], a["k"])
                groups[cell][0].append(a["objective"])
                groups[cell][1].append(b["objective"])
        if groups:
            inside = sum(1 for old, new in groups.values() if min(old) <= statistics.median(new) <= max(old))
            out.append(
                f"- fp32 main cells: the 0.2.5 median objective lies inside the fork build's min-max over seeds in "
                f"{inside} of {len(groups)} cells."
            )
    now_p = {r["key"]: r for r in precision if r.get("status") == "ok"}
    p_pairs = [(r, now_p[r["key"]]) for r in fork_precision if r.get("status") == "ok" and r.get("key") in now_p]
    if p_pairs:
        diffs = [
            abs(b["variants"][v]["drift_gap_median"] - a["variants"][v]["drift_gap_median"])
            for a, b in p_pairs
            for v in PRECISION_VARIANTS
            if "drift_gap_median" in a["variants"].get(v, {}) and "drift_gap_median" in b["variants"].get(v, {})
        ]
        zero = sum(1 for _, b in p_pairs if b["variants"].get("fp32_rerun", {}).get("drift_gap_max") == 0)
        out.append(
            f"- Precision: {sum(1 for d in diffs if d > DRIFT_DIFF)} of {len(diffs)} variant drift_gap medians differ "
            f"by more than {DRIFT_DIFF} between the builds; the 0.2.5 FP32 rerun drift is 0 in {zero} of "
            f"{len(p_pairs)} cells."
        )
    out.append("")
    return out


def load_version_ari(path: Path) -> list[dict[str, Any]]:
    """
    Read the ARI records written by `compare_versions.py`.

    :param path: JSONL file, one record per line.
    :returns: The records.
    :raises SystemExit: When the file is missing (a mistyped path must not read as "no ARI data").
    """
    if not path.exists():
        msg = f"{path}: no such file (written by compare_versions.py)"
        raise SystemExit(msg)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


TRITON_PREFIX = "emeans-gpu-triton-"
VERSION_LISTED = 10


def _spread(values: list[float]) -> str:
    return f"{statistics.median(values):.2f} [{min(values):.2f}-{max(values):.2f}]"


RESEED_LEVEL = 0.05  # ARI between versions within this of the ARI between seeds reads as "about a reseed"


def _range(values: list[float], digits: int = 2) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.{digits}f}" if f"{lo:.{digits}f}" == f"{hi:.{digits}f}" else f"{lo:.{digits}f}-{hi:.{digits}f}"


def _ari_baseline(ari: list[dict[str, Any]]) -> str:
    """
    Explain the weak baseline of the R-02 ARI part and set the ARI between versions against the same-seed rerun.

    :param ari: Records of `compare_versions.py`.
    :returns: One paragraph; the gate rule and its result are reported as run, not re-evaluated.
    """
    text = (
        "Baseline of the ARI part: the gate compared the ARI between versions at the same seed with the ARI between "
        "different 0.2.5 seeds. That baseline is weak: fits from one seed share their starting centroids, so the "
        "comparison leans towards passing, and a version change that moves assignments as much as a reseed still "
        "passes. The fair baseline is the same-seed 0.2.5 rerun (`ARI 0.2.5 rerun`)."
    )
    structured = [a for a in ari if a["data"] != "random"]
    names = ", ".join(sorted({a["data"] for a in structured}))
    for precision_name in sorted({str(a["method"]).removeprefix(TRITON_PREFIX) for a in structured}):
        reruns = [
            a["ari_old_rerun"] for a in structured if str(a["method"]).removeprefix(TRITON_PREFIX) == precision_name
        ]
        if min(reruns) == 1.0:
            text += f" {precision_name} 0.2.5 reruns are deterministic (ARI 1.0)."
        else:
            text += f" {precision_name} 0.2.5 reruns are not reproducible (same-seed ARI {_range(reruns)} on {names})."
    for data in sorted({a["data"] for a in structured}):
        sel = [a for a in structured if a["data"] == data]
        between = [a["ari_new_old_median"] for a in sel]
        seeds = [a["ari_old_seeds_median"] for a in sel]
        level = abs(statistics.median(between) - statistics.median(seeds)) <= RESEED_LEVEL
        text += (
            f" On {data}, 0.2.6 against 0.2.5 at the same seed gives an ARI of {_range(between)}, "
            + ("about the agreement of a reseed" if level else "against")
            + f" ({_range(seeds)} between 0.2.5 seeds)."
        )
    noise = [v for a in ari if a["data"] == "random" for v in (a["ari_new_old_median"], a["ari_old_seeds_median"])]
    if noise:
        text += (
            f" On random data (no cluster structure) the ARI between versions and between seeds is {_range(noise, 4)}, "
            "chance level, so the ARI part compares noise there."
        )
    return text + " The gate rule and its committed result (R-02) are kept as run."


def version_comparison(
    speed: list[dict[str, Any]],
    prev_speed: list[dict[str, Any]],
    precision: list[dict[str, Any]],
    prev_precision: list[dict[str, Any]],
    ari: list[dict[str, Any]],
) -> list[str]:
    """
    Compare the Triton records of emeans-rocm 0.2.5 with those of 0.2.6 for the same cells.

    :param speed: 0.2.6 speed records.
    :param prev_speed: 0.2.5 speed records (same keys).
    :param precision: 0.2.6 precision records.
    :param prev_precision: 0.2.5 precision records (same keys).
    :param ari: Records of `compare_versions.py` (ARI on the 100k subset and the smoke gate).
    :returns: Markdown lines.
    """
    now = {r["key"]: r for r in speed if r.get("status") == "ok" and str(r.get("method", "")).startswith(TRITON_PREFIX)}
    pairs = [(r, now[r["key"]]) for r in prev_speed if r.get("status") == "ok" and r.get("key") in now]
    out = ["### emeans-rocm 0.2.5 vs 0.2.6", "", notes.VERSION_NOTE, ""]
    ari_by_cell: dict[tuple, dict[str, Any]] = {(a["method"], a["data"], a["n"], a["k"]): a for a in ari}
    outside: list[str] = []
    largest: dict[str, float] = {}
    seed_counts: set[int] = set()
    if pairs:
        warm = [b["seconds"] / a["seconds"] for a, b in pairs if a["timing"] == "warm" and a["seconds"]]
        cold = [b["seconds"] / a["seconds"] for a, b in pairs if a["timing"] == "cold" and a["seconds"]]
        line = f"- Speed: {len(pairs)} matching records."
        if warm:
            line += f" Warm time ratio 0.2.6 / 0.2.5, median {_spread(warm)}."
        if cold:
            line += f" Cold time ratio, median {_spread(cold)}."
        out.append(line)
        for precision_name in sorted({a.get("precision") or "" for a, _ in pairs}):
            sel = [(a, b) for a, b in pairs if a.get("precision") == precision_name and a["timing"] == "cold"]
            deltas = [abs(b["objective"] / a["objective"] - 1) for a, b in sel if a["kind"] == "main"]
            if deltas:
                largest[precision_name] = max(deltas) * 100
                same = sum(1 for d in deltas if d <= SAME_OBJECTIVE)
                out.append(
                    f"- {precision_name} main-cell objectives: largest difference {max(deltas) * 100:.3f} %, "
                    f"{same} of {len(deltas)} records within 0.01 %."
                )
        signed = [
            (b["objective"] / a["objective"] - 1, a) for a, b in pairs if a["timing"] == "cold" and a["kind"] == "main"
        ]
        if signed:
            worst_delta, worst = max(signed, key=lambda t: abs(t[0]))
            lower = sum(1 for d, _ in signed if d < 0)
            out.append(
                f"- Main-cell objectives, all precisions: 0.2.6 lower than 0.2.5 in {lower} of {len(signed)} records; "
                f"largest difference {worst_delta * 100:+.3f} % ({worst['method']} "
                f"{_cell_label(worst['data'], worst['n'], worst['k'])}, {worst['n'] / worst['k']:.1f} rows per "
                "centroid)."
            )
        out.append("")
        groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: {"old": [], "new": [], "w": [], "c": []})
        for a, b in pairs:
            cell = (a["method"], a["data"], a["n"], a["k"], a["kind"], a.get("passes_requested"))
            if a["timing"] == "cold":
                groups[cell]["old"].append(a["objective"])
                groups[cell]["new"].append(b["objective"])
            if a["seconds"]:
                groups[cell]["w" if a["timing"] == "warm" else "c"].append(b["seconds"] / a["seconds"])
        body = []
        seed_counts = {len(g["old"]) for g in groups.values() if g["old"]}
        for cell, g in sorted(groups.items(), key=lambda kv: str(kv[0])):
            method, data, n, k, kind, passes = cell
            if not g["old"]:
                continue
            old_med, new_med = statistics.median(g["old"]), statistics.median(g["new"])
            inside = min(g["old"]) <= new_med <= max(g["old"])
            if not inside:
                outside.append(f"{method} {_cell_label(data, n, k)} ({kind}, {passes or 'default'} passes)")
            match = ari_by_cell.get((method, data, n, k)) if kind == "main" and passes is None else None
            body.append(
                [
                    method, _cell_label(data, n, k), kind, _num(passes) if passes else "default",
                    f"{old_med:.5g}", f"{new_med:.5g}", f"{(new_med / old_med - 1) * 100:+.3f} %",
                    "yes" if inside else "no",
                    f"{statistics.median(g['w']):.2f}" if g["w"] else "-",
                    f"{statistics.median(g['c']):.2f}" if g["c"] else "-",
                    f"{statistics.median(match['ari_new_old']):.4f}" if match else "-",
                ]
            )  # fmt: skip
        out += _table(
            ["method", "cell", "kind", "passes", "objective 0.2.5", "objective 0.2.6", "difference",
             "inside 0.2.5 seed range", "warm ratio", "cold ratio", "ARI 0.2.6 vs 0.2.5"],
            body,
        )  # fmt: skip
    else:
        out += ["No matching Triton records.", ""]
    if ari:
        out += ["Assignment agreement (100k subset), median over seeds:", ""]
        out += _table(
            ["method", "cell", "ARI 0.2.6 vs 0.2.5", "ARI between 0.2.5 seeds", "ARI 0.2.5 rerun",
             "objective in 0.2.5 range", "ARI at least seed-to-seed"],
            [
                [
                    a["method"], _cell_label(a["data"], a["n"], a["k"]), f"{a['ari_new_old_median']:.4f}",
                    f"{a['ari_old_seeds_median']:.4f}", f"{a['ari_old_rerun']:.4f}",
                    "yes" if a["ok_objective"] else "no", "yes" if a["ok_ari"] else "no",
                ]
                for a in ari
            ],
        )  # fmt: skip
        out += [_ari_baseline(ari), ""]
    now_p = {r["key"]: r for r in precision if r.get("status") == "ok"}
    p_pairs = [(r, now_p[r["key"]]) for r in prev_precision if r.get("status") == "ok" and r.get("key") in now_p]
    if p_pairs:
        diffs = [
            abs(b["variants"][v]["drift_gap_median"] - a["variants"][v]["drift_gap_median"])
            for a, b in p_pairs
            for v in PRECISION_VARIANTS
            if "drift_gap_median" in a["variants"].get(v, {}) and "drift_gap_median" in b["variants"].get(v, {})
        ]
        out.append(
            f"- Precision: {len(p_pairs)} matching records; {sum(1 for d in diffs if d > DRIFT_DIFF)} of {len(diffs)} "
            f"variant drift_gap medians differ by more than {DRIFT_DIFF} between the versions."
        )
        for v in REDUCED:
            old_s = [a["variants"][v]["share_same_cluster"] for a, _ in p_pairs if v in a["variants"]]
            new_s = [b["variants"][v]["share_same_cluster"] for _, b in p_pairs if v in b["variants"]]
            if old_s and new_s:
                out.append(
                    f"- {v}: median share of documents in the same cluster as FP32, 0.2.5 "
                    f"{statistics.median(old_s):.3f}, 0.2.6 {statistics.median(new_s):.3f}."
                )
        refs = [
            abs(b["reference"]["objective"] / a["reference"]["objective"] - 1)
            for a, b in p_pairs
            if a.get("reference", {}).get("objective") and b.get("reference", {}).get("objective")
        ]
        if refs:
            median_ref = statistics.median(refs) * 100
            out.append(f"- Precision reference objective: median relative difference {median_ref:.3f} %.")
        out.append("")
    gates_ok = bool(ari) and all(a.get("ok") for a in ari)
    if pairs and gates_ok and not outside:
        out.append(
            "On these cells emeans-rocm 0.2.6 matches 0.2.5 within seed noise: every median objective lies inside "
            "the 0.2.5 range over seeds and the ARI between versions is at least that between 0.2.5 seeds."
        )
    elif pairs and not ari and not outside:
        out.append(
            "Every median objective of emeans-rocm 0.2.6 lies inside the 0.2.5 range over seeds; no ARI records were "
            "given, so assignment agreement is not evaluated."
        )
    else:
        seeds = f"{next(iter(seed_counts))} seeds" if len(seed_counts) == 1 else "seeds"
        listed = "; ".join(outside[:VERSION_LISTED]) + (" and more" if len(outside) > VERSION_LISTED else "")
        n_obj = sum(1 for a in ari if a.get("ok_objective"))
        n_ari = sum(1 for a in ari if a.get("ok_ari"))
        spread = ", ".join(f"{v} {x:.3f} %" for v, x in sorted(largest.items()))
        out.append(
            f"emeans-rocm 0.2.6 has a median objective outside the min-max of the {seeds} of 0.2.5 of the same cell "
            f"in {len(outside)} cells"
            + (f" ({listed})" if outside else "")
            + (f"; the largest main-cell objective difference is {spread}" if spread else "")
            + f". The smoke gate passed its objective part in {n_obj} of {len(ari)} settings and its ARI part in "
            f"{n_ari} of {len(ari)}."
        )
    counts = (len(ari), sum(1 for a in ari if a.get("ok_objective")), sum(1 for a in ari if a.get("ok_ari")))
    gate_note = notes.VERSION_GATE_NOTE if counts == notes.VERSION_GATE_RECORDED else notes.VERSION_GATE_UNMATCHED
    out += ["", gate_note, "", notes.VERSION_CENTROIDS_NOTE, "", notes.VERSION_TIME_NOTE, ""]
    return out


def precision_summary(records: list[dict[str, Any]]) -> list[str]:
    """
    Summarise reduced-precision drift against the FP32 other-seed baseline, per cell.

    :param records: Records as written by `precision.py`.
    :returns: Markdown lines: a ratio table, the objective range and the precision caution.
    """
    aggregated = aggregate_precision(records)
    body = []
    ratios: dict[str, list[float]] = defaultdict(list)
    deltas: list[float] = []
    for group in sorted(aggregated):
        stats = aggregated[group]
        other = stats.get("fp32_other_seed", {})
        if "drift_gap_median" not in other:
            continue
        row = [group[0], group[1], _n_label(group[2]), str(group[3])]
        for name in REDUCED:
            entry = stats.get(name, {})
            if "drift_gap_median" not in entry or not other["drift_gap_median"]:
                row.append("-")
                continue
            ratio = entry["drift_gap_median"] / other["drift_gap_median"]
            ratios[group[0]].append(ratio)
            row.append(f"{ratio:.2f}")
            if group[0] != "random":
                deltas += [entry["delta"][1], entry["delta"][2]]
        row += [
            f"{stats[name]['share'][0]:.3g}" if "share" in stats.get(name, {}) else "-"
            for name in (*REDUCED, "fp32_other_seed")
        ]
        body.append(row)
    if not body:
        return []
    out = [
        "### Reduced precision against reseeding",
        "",
        (
            "Ratio: FP16 or BF16 drift_gap median over the FP32 other-seed drift_gap median (1.00: reduced precision "
            "moves the centroids as much as a new seed does). Share: median share of documents in the same cluster "
            "as under the FP32 reference."
        ),
        "",
    ]
    out += _table(
        ["data", "mode", "n", "k", "FP16 / reseed", "BF16 / reseed", "share FP16", "share BF16", "share reseed"], body
    )
    for data in sorted(ratios):
        out.append(f"- {data}: ratio {min(ratios[data]):.2f}-{max(ratios[data]):.2f}.")
    if deltas:
        out.append(
            f"- Real data: FP16/BF16 objective vs FP32 between {min(deltas):+.3f} % and {max(deltas):+.3f} % "
            "over seeds."
        )
    out += ["", notes.OPTION_B_CAUTION, ""]
    return out


def render(
    records: list[dict[str, Any]],
    *,
    title: str = "Benchmark results",
    smoke: bool = False,
    precision_records: list[dict[str, Any]] | None = None,
    fork_speed: list[dict[str, Any]] | None = None,
    fork_precision: list[dict[str, Any]] | None = None,
    prev_speed: list[dict[str, Any]] | None = None,
    prev_precision: list[dict[str, Any]] | None = None,
    version_ari: list[dict[str, Any]] | None = None,
) -> str:
    """
    Render RESULTS.md.

    :param records: Records as written by `run_grid`.
    :param title: Page title.
    :param smoke: Mark the page as a smoke run, not results.
    :param precision_records: Records of the precision study; None leaves the Precision section out, an empty list
        says the study was not run.
    :param fork_speed: Speed records of the earlier fork build; with `fork_precision`, adds the build comparison.
    :param fork_precision: Precision records of the earlier fork build.
    :param prev_speed: Speed records of emeans-rocm 0.2.5; with `prev_precision` and `version_ari`, adds the 0.2.5
        vs 0.2.6 comparison and its cut.
    :param prev_precision: Precision records of emeans-rocm 0.2.5.
    :param version_ari: Records written by `compare_versions.py`.
    :returns: Markdown text.
    """
    rows = aggregate(records)
    table, summary = _main_table(rows)
    reaches = passes_to_objective(rows)
    compares = main_comparison(rows)
    speedups = gpu_speedups(rows)
    host = next(((r.get("host") or {}).get("gpu") for r in records if (r.get("host") or {}).get("gpu")), None)
    lines = _header(records, title, smoke)
    lines += _render_headline(reaches, compares, speedups, host or "GPU", rows)
    lines += _protocol()
    lines += _render_passes(reaches, compares)
    lines += _render_main_comparison(compares)
    lines += _render_claims(reaches, compares, speedups)
    lines += ["## Main results", "", "Cold and warm are `median [min-max]` seconds over seeds.", ""]
    lines += table
    lines += _render_where(summary, speedups, compares, rows)
    lines += _render_gpu_engines(rows)
    if fork_speed is not None and fork_precision is not None:
        lines += fork_comparison(records, fork_speed, precision_records or [], fork_precision)
    compare_versions = prev_speed is not None and prev_precision is not None and version_ari is not None
    if compare_versions:
        lines += version_comparison(
            records, prev_speed or [], precision_records or [], prev_precision or [], version_ari or []
        )
    lines += ["## Pass sweep", ""]
    lines += _sweep_tables(rows) or ["No sweep records.", ""]
    lines += _render_cuts(records, rows, reaches, extra_cuts=(notes.VERSION_ARI_CUT,) if compare_versions else ())
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
    parser.add_argument("--fork-speed", type=Path, default=None, help="speed records of the earlier fork build")
    parser.add_argument("--fork-precision", type=Path, default=None, help="precision records of the fork build")
    parser.add_argument(
        "--prev-speed", type=Path, default=None, help="emeans-rocm 0.2.5 speed records (speed-emeans-rocm-0.2.5.jsonl)"
    )
    parser.add_argument(
        "--prev-precision",
        type=Path,
        default=None,
        help="emeans-rocm 0.2.5 precision records (precision-emeans-rocm-0.2.5.jsonl)",
    )
    parser.add_argument(
        "--version-ari", type=Path, default=None, help="records written by compare_versions.py (ARI and gate)"
    )
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
        methods = sorted(METHODS)
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
    version_flags = {"--prev-speed": args.prev_speed, "--prev-precision": args.prev_precision}
    version_flags["--version-ari"] = args.version_ari
    given = [flag for flag, value in version_flags.items() if value]
    if given and len(given) < len(version_flags):
        missing = ", ".join(flag for flag in version_flags if flag not in given)
        sys.stderr.write(
            f"warning: {', '.join(given)} given without {missing}; the 0.2.5 vs 0.2.6 comparison is left out\n"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fork_speed = load_records(args.fork_speed) if args.fork_speed else None
    fork_precision = load_records(args.fork_precision) if args.fork_precision else None
    prev_speed = load_records(args.prev_speed) if args.prev_speed else None
    prev_precision = load_records(args.prev_precision) if args.prev_precision else None
    version_ari = load_version_ari(args.version_ari) if args.version_ari else None
    text = render(
        records,
        smoke=args.smoke,
        precision_records=precision_records,
        fork_speed=fork_speed,
        fork_precision=fork_precision,
        prev_speed=prev_speed,
        prev_precision=prev_precision,
        version_ari=version_ari,
    )
    out.write_text(text, encoding="utf-8")
    sys.stderr.write(f"wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
