# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Walk the benchmark grid: one fresh worker process per cell, resumable, with an explicit time cap."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import DATASETS
from methods import METHODS

BENCH_DIR = Path(__file__).resolve().parent

NS = (100_000, 1_000_000)
KS = (1024, 4096, 16384, 65536)
SEEDS = (0, 1, 2)
SWEEP_PASSES = (1, 2, 5, 10, 25)
# Narrowed scope (owner, 2026-11-23): scikit-learn runs at 100k only. faiss is the Lloyd baseline at every size;
# scikit-learn at 1M would cost most of the CPU grid's 30-40 h for a reference point that adds little.
METHOD_NS: dict[str, tuple[int, ...]] = {"sklearn-lloyd": (100_000,)}
# (data, n, k): 1M x 16384 is above 1e10, so Lloyd gets sweep points there even when its full run is capped.
SWEEP_CELLS = (("random", 100_000, 4096), ("cohere", 100_000, 4096), ("cohere", 1_000_000, 16384))
DEFAULT_CAP_MIN = 60
WARM_MAX_S = 300
SKLEARN_EXPECTED_ITERS = 100
SKLEARN_METHODS = frozenset({"sklearn-lloyd"})
EXTRAPOLATION_FACTOR = 4
TAIL_CHARS = 500
DIM = 768
SCHEMA = 1
SMOKE_N = 10_000
SMOKE_K = 256
SMOKE_PASSES = (1, 2)


@dataclass(frozen=True)
class Cell:
    """One unit of work: a method on a dataset at a size, seed and (optionally) a fixed pass count."""

    method: str
    data: str
    n: int
    k: int
    seed: int
    kind: str
    passes: float | None

    @property
    def mode(self) -> str:
        """Clustering mode of the dataset: `spherical` (cosine) or `euclidean`."""
        return "spherical" if DATASETS[self.data].spherical else "euclidean"

    @property
    def key(self) -> str:
        """Stable identity of the cell, used for resume."""
        passes = "default" if self.passes is None else f"{self.passes:g}"
        return f"{self.method}|{self.data}|{self.n}|{self.k}|{self.mode}|{self.seed}|{self.kind}|{passes}"

    @property
    def probe_key(self) -> str:
        """Identity of the one-pass probe of this cell (independent of the pass count)."""
        return f"{self.method}|{self.data}|{self.n}|{self.k}|{self.seed}"


class WorkerError(Exception):
    """A worker or scorer process exited with an error; the message is the stderr tail."""


class Runner(Protocol):
    """How a cell is probed, trained and scored (swapped for a fake in tests)."""

    def probe(self, cell: Cell) -> float:
        """Return the seconds one pass of this cell takes."""

    def train(self, cell: Cell, *, warm: bool) -> dict[str, Any]:
        """Train the cell in a fresh process and return the worker record."""

    def score(self, cell: Cell, centroids_path: str) -> dict[str, Any]:
        """Score saved centroids with the shared scorer."""


def _order(cells: list[Cell], datasets: Sequence[str], methods: Sequence[str]) -> list[Cell]:
    return sorted(
        cells,
        key=lambda c: (
            datasets.index(c.data),
            c.n,
            c.k,
            methods.index(c.method),
            c.seed,
            c.kind != "main",
            c.passes or 0,
        ),
    )


def full_grid(methods: Sequence[str], datasets: Sequence[str], seeds: Sequence[int] = SEEDS) -> list[Cell]:
    """
    Build the full grid: main cells plus the pass-sweep cells, minus the sizes ``METHOD_NS`` leaves out.

    :param methods: Method names.
    :param datasets: Dataset names (those not registered are left out of the sweep, never invented).
    :param seeds: Seeds per cell.
    :returns: Cells ordered by dataset, n, k, then method and seed, so cheap cells run first.
    """
    cells = [
        Cell(m, d, n, k, s, "main", None) for d in datasets for n in NS for k in KS for m in methods for s in seeds
    ]
    for data, n, k in SWEEP_CELLS:
        if data not in datasets:
            continue
        cells += [Cell(m, data, n, k, s, "sweep", float(p)) for m in methods for s in seeds for p in SWEEP_PASSES]
    cells = [c for c in cells if c.n in METHOD_NS.get(c.method, NS)]
    return _order(cells, list(datasets), list(methods))


def smoke_grid(methods: Sequence[str], datasets: Sequence[str]) -> list[Cell]:
    """
    Build the tiny CPU smoke grid.

    :param methods: Method names.
    :param datasets: Dataset names; the first one also gets the small pass sweep.
    :returns: One seed of a 10k x 256 cell, plus passes 1 and 2 on the first dataset.
    """
    cells = [Cell(m, d, SMOKE_N, SMOKE_K, 0, "main", None) for d in datasets for m in methods]
    cells += [Cell(m, datasets[0], SMOKE_N, SMOKE_K, 0, "sweep", float(p)) for m in methods for p in SMOKE_PASSES]
    return _order(cells, list(datasets), list(methods))


def _tail(text: str) -> str:
    return text.strip()[-TAIL_CHARS:]


class SubprocessRunner:
    """Run the worker and the scorer as separate fresh processes."""

    def __init__(self, scratch: Path, timeout_s: float | None) -> None:
        self.scratch = scratch
        self.timeout_s = timeout_s

    def _call(self, script: str, args: list[str]) -> dict[str, Any]:
        command = [sys.executable, "-I", str(BENCH_DIR / script), *args]
        # Argument list, no shell; every value comes from the registries or ints.
        done = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, timeout=self.timeout_s, check=False
        )
        if done.returncode != 0:
            raise WorkerError(_tail(done.stderr))
        lines = done.stdout.strip().splitlines()
        if not lines:
            raise WorkerError("no output: " + _tail(done.stderr))
        record: dict[str, Any] = json.loads(lines[-1])
        return record

    def _worker_args(self, cell: Cell, passes: float | None, warm: bool) -> list[str]:
        args = [
            "--method",
            cell.method,
            "--data",
            cell.data,
            "--n",
            str(cell.n),
            "--d",
            str(DIM),
            "--k",
            str(cell.k),
            "--seed",
            str(cell.seed),
            "--warm",
            "yes" if warm else "no",
            "--scratch",
            str(self.scratch),
        ]
        if passes is not None:
            args += ["--passes", f"{passes:g}"]
        return args

    def probe(self, cell: Cell) -> float:
        """Return the cold seconds one pass of this cell takes, as measured inside the worker."""
        record = self._call("worker.py", self._worker_args(cell, 1.0, warm=False))
        Path(record["centroids_path"]).unlink(missing_ok=True)
        return float(record["cold_s"])

    def train(self, cell: Cell, *, warm: bool) -> dict[str, Any]:
        """Train the cell in a fresh process."""
        return self._call("worker.py", self._worker_args(cell, cell.passes, warm))

    def score(self, cell: Cell, centroids_path: str) -> dict[str, Any]:
        """Score saved centroids with the shared scorer."""
        return self._call(
            "score.py", ["--data", cell.data, "--n", str(cell.n), "--d", str(DIM), "--centroids", centroids_path]
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            sys.stderr.write(f"{path}:{number}: malformed line ignored\n")
    return out


def _append(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, default=str) + "\n" for r in records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _passes_and_basis(cell: Cell) -> tuple[float, str]:
    if cell.passes is not None:
        return cell.passes, f"probe x {cell.passes:g} passes"
    if cell.method in SKLEARN_METHODS:
        return float(SKLEARN_EXPECTED_ITERS), f"probe x {SKLEARN_EXPECTED_ITERS} expected iterations"
    passes = METHODS[cell.method].default_passes
    return passes, f"probe x {passes:g} passes"


class _Rates:
    """Per (method, data): the per-pass rate from the largest probed cell, used to extrapolate."""

    def __init__(self) -> None:
        self.best: dict[tuple[str, str], tuple[int, float, Cell]] = {}

    def add(self, cell: Cell, probe_s: float) -> None:
        size = cell.n * cell.k
        key = (cell.method, cell.data)
        if key not in self.best or size > self.best[key][0]:
            self.best[key] = (size, probe_s / size, cell)

    def extrapolate(self, cell: Cell) -> tuple[float, str] | None:
        found = self.best.get((cell.method, cell.data))
        if found is None:
            return None
        _, rate, source = found
        passes, _ = _passes_and_basis(cell)
        return rate * cell.n * cell.k * passes, f"extrapolated from n={source.n},k={source.k}"


def _records(
    cell: Cell,
    *,
    status: str,
    reason: str | None,
    worker: dict[str, Any] | None,
    scored: dict[str, Any] | None,
    estimate_s: float | None,
    basis: str | None,
    clock: Callable[[], str],
) -> list[dict[str, Any]]:
    method = METHODS[cell.method]
    out = []
    for timing in ("cold", "warm"):
        w = worker or {}
        seconds = w.get(f"{timing}_s") if status == "ok" else None
        rec_status, rec_reason = status, reason
        if status == "ok" and timing == "warm" and seconds is None:
            rec_status, rec_reason = "skipped", w.get("warm_status", "skipped (cost)")
        info = w.get("info") or {}
        out.append(
            {
                "schema": SCHEMA,
                "key": f"{cell.key}|{timing}",
                "timing": timing,
                "method": cell.method,
                "engine": method.engine,
                "device": method.device,
                "backend": w.get("backend"),
                "precision": w.get("precision"),
                "env": os.environ.get("HATCH_ENV_ACTIVE"),
                "n": cell.n,
                "k": cell.k,
                "n_x_k": cell.n * cell.k,
                "d": DIM,
                "data": cell.data,
                "data_id": (w.get("data_info") or {}).get("data_id"),
                "mode": cell.mode,
                "seed": cell.seed,
                "kind": cell.kind,
                "passes_requested": cell.passes,
                "passes_used": w.get("passes_used"),
                "seconds": seconds,
                "transfer_s": w.get("transfer_s"),
                "objective": (scored or {}).get("objective"),
                "objective_sum": (scored or {}).get("objective_sum"),
                "status": rec_status,
                "skipped_reason": rec_reason,
                "estimate_s": estimate_s,
                "estimate_basis": basis,
                "host": w.get("host"),
                "versions": w.get("versions"),
                "triton_cache": info.get("triton_cache"),
                "timestamp": clock(),
            }
        )
    return out


def _hours(seconds: float) -> float:
    return seconds / 3600


def run(
    cells: Iterable[Cell],
    *,
    results: Path,
    scratch: Path,
    cap_s: float | None,
    runner: Runner,
    clock: Callable[[], str] = _utc_now,
    keep_centroids: bool = False,
) -> int:
    """
    Run every cell not yet finished and append its cold and warm records.

    :param cells: Cells in execution order.
    :param results: JSONL file of records (appended to, resumed from).
    :param scratch: Directory for the probe cache and centroids.
    :param cap_s: Per-seed time cap in seconds; None disables the cap.
    :param runner: Probe, train and score implementation.
    :param clock: Timestamp source.
    :param keep_centroids: Keep saved centroids after scoring.
    :returns: 0 when the grid finished (skipped, timed-out and failed cells are records, not errors).
    """
    scratch.mkdir(parents=True, exist_ok=True)
    done = {r.get("key") for r in _read_jsonl(results)}
    probes_path = scratch / "probes.jsonl"
    probes = {r["key"]: float(r["seconds"]) for r in _read_jsonl(probes_path) if "key" in r}
    rates = _Rates()
    cell_list = list(cells)
    total = len(cell_list)
    for index, cell in enumerate(cell_list, start=1):
        if f"{cell.key}|cold" in done and f"{cell.key}|warm" in done:
            sys.stderr.write(f"[{index}/{total}] already done: {cell.key}\n")
            if cell.probe_key in probes:
                rates.add(cell, probes[cell.probe_key])
            continue
        records = _run_cell(cell, cap_s, runner, rates, probes, probes_path, clock, keep_centroids)
        _append(results, records)
        sys.stderr.write(f"[{index}/{total}] {records[0]['status']}: {cell.key}\n")
    return 0


def _run_cell(
    cell: Cell,
    cap_s: float | None,
    runner: Runner,
    rates: _Rates,
    probes: dict[str, float],
    probes_path: Path,
    clock: Callable[[], str],
    keep_centroids: bool,
) -> list[dict[str, Any]]:
    estimate_s: float | None = None
    basis: str | None = None

    def fail(status: str, reason: str) -> list[dict[str, Any]]:
        return _records(
            cell,
            status=status,
            reason=reason,
            worker=None,
            scored=None,
            estimate_s=estimate_s,
            basis=basis,
            clock=clock,
        )

    def skipped() -> list[dict[str, Any]]:
        assert estimate_s is not None  # noqa: S101
        return fail("skipped", f"not run: est. {_hours(estimate_s):.1f} h")

    try:
        guess = rates.extrapolate(cell)
        if cap_s is not None and guess is not None and guess[0] > EXTRAPOLATION_FACTOR * cap_s:
            estimate_s, basis = guess
            return skipped()
        if cell.probe_key not in probes:
            probes[cell.probe_key] = runner.probe(cell)
            _append(probes_path, [{"key": cell.probe_key, "seconds": probes[cell.probe_key]}])
        probe_s = probes[cell.probe_key]
        rates.add(cell, probe_s)
        passes, basis = _passes_and_basis(cell)
        estimate_s = probe_s * passes
        if cap_s is not None and estimate_s > cap_s:
            return skipped()
        warm = METHODS[cell.method].device == "gpu" or estimate_s <= WARM_MAX_S
        worker = runner.train(cell, warm=warm)
        path = worker["centroids_path"]
        try:
            scored = runner.score(cell, path)
        finally:
            if not keep_centroids:
                Path(path).unlink(missing_ok=True)
        return _records(
            cell,
            status="ok",
            reason=None,
            worker=worker,
            scored=scored,
            estimate_s=estimate_s,
            basis=basis,
            clock=clock,
        )
    except subprocess.TimeoutExpired as exc:
        hours = _hours(float(exc.timeout)) if exc.timeout is not None else 0.0
        return fail("timed_out", f"timed out after {hours:.1f} h")
    except WorkerError as exc:
        return fail("failed", str(exc)[-TAIL_CHARS:])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU grid, results under scratch/smoke/")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--scratch", type=Path, default=BENCH_DIR / "scratch")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--data", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--cap-min", type=float, default=DEFAULT_CAP_MIN)
    parser.add_argument("--no-cap", action="store_true")
    parser.add_argument("--timeout-min", type=float, default=None)
    parser.add_argument("--keep-centroids", action="store_true")
    parser.add_argument("--kind", choices=["main", "sweep", "all"], default="all")
    return parser


def main() -> int:
    """Run the grid from the command line."""
    args = _parser().parse_args()
    if args.methods is None:
        methods = []
        for name, spec in METHODS.items():
            reason = spec.available()
            if reason is None:
                methods.append(name)
            else:
                sys.stderr.write(f"method {name} unavailable: {reason}\n")
    else:
        methods = list(args.methods)
    datasets = list(args.data) if args.data else list(DATASETS)
    for name in datasets:
        if name not in DATASETS:
            sys.stderr.write(f"unknown dataset {name}\n")
            return 2
        reason = DATASETS[name].available()
        if reason is not None and args.data:
            sys.stderr.write(f"dataset {name} unavailable: {reason}\n")
            return 2
    datasets = [d for d in datasets if DATASETS[d].available() is None]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        sys.stderr.write(f"unknown methods: {unknown}\n")
        return 2

    cells = smoke_grid(methods, datasets) if args.smoke else full_grid(methods, datasets, args.seeds)
    if args.kind != "all":
        cells = [c for c in cells if c.kind == args.kind]
    scratch: Path = args.scratch / "smoke" if args.smoke else args.scratch
    results: Path = args.results or (scratch / "speed.jsonl" if args.smoke else BENCH_DIR / "results" / "speed.jsonl")
    cap_s = None if args.no_cap else args.cap_min * 60
    if args.timeout_min is not None:
        timeout_s: float | None = args.timeout_min * 60
    else:
        timeout_s = None if cap_s is None else 2 * cap_s
    start = time.perf_counter()
    code = run(
        cells,
        results=results,
        scratch=scratch,
        cap_s=cap_s,
        runner=SubprocessRunner(scratch, timeout_s),
        keep_centroids=args.keep_centroids,
    )
    sys.stderr.write(f"grid finished in {time.perf_counter() - start:.1f} s, results in {results}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
