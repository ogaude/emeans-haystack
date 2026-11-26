# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""
Precision study: how far GPU FP16 and BF16 e-means centroids move from FP32, against honest baselines.

Every cell trains an FP32 reference, an FP32 deterministic rerun, an FP32 run with another seed and start, and FP16
and BF16 runs from the same shared start. Every centroid set is scored by the shared exact FP32 scorer and compared
with the reference after one-to-one matching. FP16 and BF16 exist only on emeans-rocm's Triton rung, so the
orchestrator runs in the `bench-gpu-fork` env.
"""

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

import numpy as np

import data as data_mod
import score as score_mod
from data import DATASETS
from metrics import drift_metrics

BENCH_DIR = Path(__file__).resolve().parent
NS = (100_000, 1_000_000)
KS = (1024, 4096)
SEEDS = (0, 1, 2)
MODES = ("spherical", "euclidean")
SMOKE_N = 10_000
SMOKE_K = 256
NEPOCHS = 5.0
DIM = 768
SCHEMA = 1
DEFAULT_TIMEOUT_MIN = 60
TAIL_CHARS = 500
# (variant name, requested precision, seed offset): the reference is the first row.
FITS = (
    ("reference", "fp32", 0),
    ("fp32_rerun", "fp32", 0),
    ("fp32_other_seed", "fp32", 1),
    ("fp16", "fp16", 0),
    ("bf16", "bf16", 0),
)
_BF16_LOW_BITS = 16
_BF16_ROUND = 0x7FFF


@dataclass(frozen=True)
class PCell:
    """One precision cell: a dataset at a size, k, mode and seed."""

    n: int
    k: int
    data: str
    mode: str
    seed: int

    @property
    def key(self) -> str:
        """Stable identity of the cell, used for resume and for the worker argument."""
        return f"{self.n}|{self.k}|{self.data}|{self.mode}|{self.seed}"

    @classmethod
    def parse(cls, key: str) -> "PCell":
        """
        Parse a cell key back into a validated cell.

        :param key: A key as produced by `PCell.key`.
        :returns: The cell.
        :raises ValueError: If a field is malformed, unknown or out of range.
        """
        parts = key.split("|")
        if len(parts) != 5:  # noqa: PLR2004
            msg = f"bad cell key: {key!r}"
            raise ValueError(msg)
        n, k, name, mode, seed = parts
        if name not in DATASETS or mode not in MODES:
            msg = f"bad cell key: {key!r}"
            raise ValueError(msg)
        cell = cls(int(n), int(k), name, mode, int(seed))
        if cell.n <= 0 or not 0 < cell.k <= cell.n or cell.seed < 0:
            msg = f"bad cell key: {key!r}"
            raise ValueError(msg)
        return cell


def _order(cells: list[PCell], datasets: Sequence[str]) -> list[PCell]:
    return sorted(cells, key=lambda c: (c.n, c.k, list(datasets).index(c.data), MODES.index(c.mode), c.seed))


def precision_grid(datasets: Sequence[str], seeds: Sequence[int] = SEEDS) -> list[PCell]:
    """
    Build the full precision grid.

    :param datasets: Dataset names.
    :param seeds: Seeds per cell.
    :returns: n in {100k, 1M} x k in {1024, 4096} x data x mode x seeds, cheapest first.
    """
    cells = [PCell(n, k, d, m, s) for n in NS for k in KS for d in datasets for m in MODES for s in seeds]
    return _order(cells, datasets)


def precision_smoke_grid(datasets: Sequence[str]) -> list[PCell]:
    """
    Build the tiny smoke grid.

    :param datasets: Dataset names.
    :returns: One seed of 10k x 256 per dataset and mode.
    """
    return _order([PCell(SMOKE_N, SMOKE_K, d, m, 0) for d in datasets for m in MODES], datasets)


def prepare(name: str, n: int, mode: str, d: int = DIM) -> tuple[np.ndarray, dict[str, Any], str]:
    """
    Load a dataset for a mode.

    Random rows are normalised for spherical mode and left raw for Euclidean mode. The Cohere loader already
    normalises its rows (the file is not unit-norm), and Euclidean mode keeps those unit-norm rows.

    :param name: Dataset name.
    :param n: Rows.
    :param mode: `spherical` or `euclidean`.
    :param d: Dimension for synthetic data.
    :returns: The matrix, its provenance and a description of the row preparation.
    """
    x, info = data_mod.load_dataset(name, n, d)
    if name == "cohere":
        return x, info, "row-normalised (loader)"
    if mode == "spherical":
        return data_mod.normalise_rows(x), info, "row-normalised"
    return x, info, "raw"


def _round_fp16(c: np.ndarray) -> np.ndarray:
    return c.astype(np.float16).astype(np.float32)


def _round_bf16(c: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (nearest, ties to even) and back, by bit manipulation."""
    wide = np.ascontiguousarray(c, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (wide + _BF16_ROUND + ((wide >> _BF16_LOW_BITS) & 1)) >> _BF16_LOW_BITS << _BF16_LOW_BITS
    return rounded.astype(np.uint32).view(np.float32)


FitFn = Callable[..., tuple[np.ndarray, dict[str, Any]]]


def _fit_model(x: np.ndarray, init: np.ndarray, *, k: int, spherical: bool, seed: int, precision: str) -> Any:
    from haystack_integrations.components.preprocessors.emeans._backend import (  # noqa: PLC0415
        fit_model,
        select_backend,
    )

    choice = select_backend(device="cuda", n=len(x), k=k, gpu_threshold=0)
    result = fit_model(
        x,
        k=k,
        spherical=spherical,
        nepochs=NEPOCHS,
        seed=seed,
        max_points_per_centroid=None,
        backend=choice,
        precision=precision,
        deterministic=True,
        init_centroids=init,
    )
    return result.centroids, dict(result.info)


def _variant(
    c: np.ndarray, info: dict[str, Any], requested: str | None, x: np.ndarray, ref: dict[str, Any] | None
) -> dict[str, Any]:
    used = info.get("precision_used")
    finite = bool(np.isfinite(c).all())
    status = "ok"
    if requested is not None and used != requested:
        status = "precision_fallback"
    if not finite:
        status = "nonfinite" if status == "ok" else status
    out: dict[str, Any] = {
        "status": status,
        "nonfinite": not finite,
        "precision_requested": requested,
        "precision_used": used,
        "backend": info.get("backend"),
        "emeans_source": info.get("emeans_source"),
        "train_seconds": info.get("train_seconds"),
    }
    if not finite:
        return out
    objective, _, labels = score_mod.score(x, c)
    out["objective"] = objective
    out["_labels"] = labels
    out["_centroids"] = c
    if ref is not None:
        out.update(drift_metrics(c, ref["_centroids"], labels, ref["_labels"]))
        out["objective_delta_rel"] = (objective - ref["objective"]) / ref["objective"]
    return out


def run_cell(
    cell: PCell, *, fit: FitFn | None = None, d: int = DIM, clock: Callable[[], str] | None = None
) -> dict[str, Any]:
    """
    Run one precision cell and return its record.

    The five fits run one after another because deterministic mode is process-wide.

    :param cell: The cell.
    :param fit: Trainer `(x, init, *, k, spherical, seed, precision) -> (centroids, info)`; defaults to `fit_model`.
    :param d: Dimension for synthetic data.
    :param clock: Timestamp source.
    :returns: The record with one entry per variant.
    """
    trainer = fit or _fit_model
    now = clock or _utc_now
    x, data_info, prep = prepare(cell.data, cell.n, cell.mode, d)
    spherical = cell.mode == "spherical"
    starts = {0: data_mod.starts(x, cell.k, cell.seed), 1: data_mod.starts(x, cell.k, cell.seed + 1)}
    variants: dict[str, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    for name, precision, offset in FITS:
        c, info = trainer(
            x, starts[offset], k=cell.k, spherical=spherical, seed=cell.seed + offset, precision=precision
        )
        c = np.asarray(c, dtype=np.float32)
        if name == "reference":
            reference = _variant(c, info, precision, x, None)
            if reference["status"] != "ok":
                msg = f"reference fit is not a clean FP32 run: {reference['status']}"
                raise RuntimeError(msg)
            continue
        variants[name] = _variant(c, info, precision, x, reference)
    assert reference is not None  # noqa: S101
    ref_c = reference["_centroids"]
    for name, rounded in (("round16", _round_fp16(ref_c)), ("roundbf", _round_bf16(ref_c))):
        variants[name] = _variant(rounded, {"precision_used": None, "backend": "rounding"}, None, x, reference)
    for entry in [reference, *variants.values()]:
        entry.pop("_labels", None)
        entry.pop("_centroids", None)
    return {
        "schema": SCHEMA,
        "key": cell.key,
        "status": "ok",
        "n": cell.n,
        "k": cell.k,
        "n_x_k": cell.n * cell.k,
        "data": cell.data,
        "data_id": data_info.get("data_id"),
        **data_mod.provenance(data_info),
        "mode": cell.mode,
        "prep": prep,
        "seed": cell.seed,
        "nepochs": NEPOCHS,
        "reference": {
            "objective": reference["objective"],
            "train_seconds": reference["train_seconds"],
            "backend": reference["backend"],
            "precision_used": reference["precision_used"],
        },
        "variants": variants,
        "timestamp": now(),
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CellRunner(Protocol):
    """How one cell is run (swapped for a fake in tests)."""

    def run_cell(self, cell: PCell) -> dict[str, Any]:
        """Run the cell and return its record, or raise `CellError`."""


class CellError(Exception):
    """A cell worker exited with an error; the message is the stderr tail."""


class SubprocessCellRunner:
    """Run every cell in a fresh process."""

    def __init__(self, timeout_s: float | None) -> None:
        self.timeout_s = timeout_s

    def run_cell(self, cell: PCell) -> dict[str, Any]:
        """Run `precision.py --cell <key>` and parse the record it prints."""
        command = [sys.executable, "-I", str(BENCH_DIR / "precision.py"), "--cell", cell.key]
        # Argument list, no shell; the key was built from validated ints and registry names.
        done = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, timeout=self.timeout_s, check=False
        )
        if done.returncode != 0:
            raise CellError(done.stderr.strip()[-TAIL_CHARS:])
        lines = done.stdout.strip().splitlines()
        if not lines:
            raise CellError("no output: " + done.stderr.strip()[-TAIL_CHARS:])
        record: dict[str, Any] = json.loads(lines[-1])
        return record


def _failure(cell: PCell, status: str, reason: str, clock: Callable[[], str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "key": cell.key,
        "status": status,
        "reason": reason,
        "n": cell.n,
        "k": cell.k,
        "n_x_k": cell.n * cell.k,
        "data": cell.data,
        "mode": cell.mode,
        "seed": cell.seed,
        "variants": {},
        "timestamp": clock(),
    }


def _read_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            keys.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError):
            continue
    return keys


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(cells: Iterable[PCell], *, results: Path, runner: CellRunner, clock: Callable[[], str] = _utc_now) -> int:
    """
    Run every cell not yet finished and append one record per cell.

    :param cells: Cells in execution order.
    :param results: JSONL file (appended to, resumed from by key).
    :param runner: Cell runner.
    :param clock: Timestamp source.
    :returns: 0 (failed and timed-out cells are records, not errors).
    """
    done = _read_keys(results)
    cell_list = list(cells)
    for index, cell in enumerate(cell_list, start=1):
        if cell.key in done:
            sys.stderr.write(f"[{index}/{len(cell_list)}] already done: {cell.key}\n")
            continue
        try:
            record = runner.run_cell(cell)
        except subprocess.TimeoutExpired as exc:
            hours = float(exc.timeout) / 3600 if exc.timeout is not None else 0.0
            record = _failure(cell, "timed_out", f"timed out after {hours:.1f} h", clock)
        except CellError as exc:
            record = _failure(cell, "failed", str(exc)[-TAIL_CHARS:], clock)
        _append(results, record)
        sys.stderr.write(f"[{index}/{len(cell_list)}] {record['status']}: {cell.key}\n")
    return 0


def _record_context(record: dict[str, Any]) -> dict[str, Any]:
    from worker import _gpu_name, _gpu_versions, _host, _versions  # noqa: PLC0415

    versions = _versions()
    versions.update(_gpu_versions())
    for entry in record.get("variants", {}).values():
        if entry.get("emeans_source"):
            versions["emeans_source"] = entry["emeans_source"]
            break
    host = _host()
    host["gpu"] = _gpu_name()
    record["host"] = host
    record["versions"] = versions
    return record


def _gpu_refusal() -> str | None:
    """Return why this env cannot run the study, or None."""
    msg = "the precision study needs the bench-gpu-fork env (emeans-rocm installed, GPU visible): "
    try:
        import torch  # noqa: PLC0415
        from haystack_integrations.components.preprocessors.emeans._backend import (  # noqa: PLC0415
            detect_emeans_source,
            emeans_rocm_enabled,
            emeans_rocm_unsupported_reason,
        )
    except ImportError as exc:
        return msg + f"import failed ({exc})"
    if not emeans_rocm_enabled():
        reason = emeans_rocm_unsupported_reason() or f"e-means source is {detect_emeans_source()}"
        return msg + f"{reason}; needs an installed emeans-rocm within the supported range"
    if not torch.cuda.is_available():
        return msg + "no GPU visible to torch"
    return None


def main() -> int:
    """Run one cell (`--cell`) or the whole precision grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny grid, results under scratch/smoke/")
    parser.add_argument("--cell", default=None, help="internal: run one cell by key and print its record")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--data", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--timeout-min", type=float, default=DEFAULT_TIMEOUT_MIN)
    args = parser.parse_args()
    refusal = _gpu_refusal()
    if refusal is not None:
        sys.stderr.write(refusal + "\n")
        return 2
    if args.cell is not None:
        try:
            cell = PCell.parse(args.cell)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        sys.stdout.write(json.dumps(_record_context(run_cell(cell)), default=str) + "\n")
        return 0
    names = list(args.data) if args.data else list(DATASETS)
    for name in names:
        if name not in DATASETS:
            sys.stderr.write(f"unknown dataset {name}\n")
            return 2
        reason = DATASETS[name].available()
        if reason is not None and args.data:
            sys.stderr.write(f"dataset {name} unavailable: {reason}\n")
            return 2
    names = [name for name in names if DATASETS[name].available() is None]
    cells = precision_smoke_grid(names) if args.smoke else precision_grid(names, args.seeds)
    smoke_dir = BENCH_DIR / "scratch" / "smoke"
    results: Path = args.results or (
        smoke_dir / "precision.jsonl" if args.smoke else BENCH_DIR / "results" / "precision.jsonl"
    )
    start = time.perf_counter()
    code = run(cells, results=results, runner=SubprocessCellRunner(args.timeout_min * 60))
    sys.stderr.write(f"precision study finished in {time.perf_counter() - start:.1f} s, results in {results}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
