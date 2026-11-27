# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Compare e-means centroids of emeans-rocm 0.2.5 and 0.2.6: objectives, ARI and the R-02 smoke gate."""

import json
import statistics
import sys
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import adjusted_rand_score

from data import DATASETS, load_dataset
from score import score

BENCH_DIR = Path(__file__).resolve().parent

# The gate tolerance is fixed here, before any 0.2.6 run, and is not tuned afterwards:
# a setting passes when the 0.2.6 median objective lies inside the min-max of the 0.2.5 objectives
# AND the median ARI(0.2.6, 0.2.5) is at least the median ARI between 0.2.5 seeds.
# Known weakness, kept as committed (R-02, owner 2026-11-27): ARI(0.2.6, 0.2.5) pairs fits from the same seed and
# starting centroids, the seed-to-seed ARI pairs different starts, so the ARI part leans towards passing. The
# like-for-like baseline, the same-seed 0.2.5 rerun (`ari_old_rerun`), is recorded and printed but does not gate.
GATE_METHODS = ("emeans-gpu-triton-fp32", "emeans-gpu-triton-bf16")
GATE_DATA = ("random", "cohere")
GATE_N = 100_000
GATE_KS = (1024, 4096)
SEEDS = (0, 1, 2)
RERUN_SEED = 0
D = 768
OLD_SOURCE = "emeans-rocm 0.2.5"
NEW_SOURCE = "emeans-rocm 0.2.6"


def centroid_path(directory: Path, method: str, data: str, n: int, k: int, seed: int) -> Path:
    """Return the file the worker writes for one cell run with the default pass count."""
    return directory / f"{method}-{data}-n{n}-k{k}-s{seed}-pdefault.npy"


def load_run_records(directory: Path, expected_source: str) -> list[dict[str, Any]]:
    """
    Read `records.jsonl` of a run directory and refuse records of another engine version.

    :param directory: Directory holding the worker output lines in `records.jsonl`.
    :param expected_source: The exact `emeans_source` label every record must carry.
    :returns: The records.
    :raises SystemExit: With code 2 when a record has another label.
    """
    path = directory / "records.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for record in records:
        source = record.get("versions", {}).get("emeans_source")
        if source != expected_source:
            sys.stderr.write(f"{path}: emeans_source is {source!r}, expected {expected_source!r}\n")
            raise SystemExit(2)
    return records


def old_grid_objectives(path: Path, method: str, data: str, n: int, k: int) -> list[float]:
    """Return the objectives of the 0.2.5 grid cells (cold, main, default passes) for the three seeds."""
    out: list[float] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (
            r.get("timing") == "cold"
            and r.get("kind") == "main"
            and r.get("passes_requested") in (None, "default")
            and r.get("status") == "ok"
            and r.get("method") == method
            and r.get("data") == data
            and r.get("n") == n
            and r.get("k") == k
            and r.get("seed") in SEEDS
            and r.get("versions", {}).get("emeans_source") == OLD_SOURCE
        ):
            out.append(float(r["objective"]))
    return out


def gate_setting(
    old_objectives: list[float], new_objectives: list[float], ari_new_old: list[float], ari_old_seeds: list[float]
) -> dict[str, Any]:
    """Apply the R-02 gate to one setting; both comparisons are inclusive. The same-seed rerun ARI does not gate."""
    lo, hi = min(old_objectives), max(old_objectives)
    new_median = statistics.median(new_objectives)
    ari_new_old_median = statistics.median(ari_new_old)
    ari_old_seeds_median = statistics.median(ari_old_seeds)
    ok_objective = lo <= new_median <= hi
    ok_ari = ari_new_old_median >= ari_old_seeds_median
    return {
        "objective_spread": [lo, hi],
        "new_objective_median": new_median,
        "ari_new_old_median": ari_new_old_median,
        "ari_old_seeds_median": ari_old_seeds_median,
        "ok_objective": ok_objective,
        "ok_ari": ok_ari,
        "ok": ok_objective and ok_ari,
    }


def _load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def compare_setting(
    x: np.ndarray,
    old_dir: Path,
    rerun_dir: Path,
    new_dir: Path,
    old_records_path: Path,
    method: str,
    data: str,
    k: int,
) -> dict[str, Any]:
    """Score the saved centroids of one setting and compute the objectives, ARI records and the gate."""
    n = GATE_N
    old_obj: list[float] = []
    new_obj: list[float] = []
    old_labels: dict[int, np.ndarray] = {}
    new_labels: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        o, _, ol = score(x, _load(centroid_path(old_dir, method, data, n, k, seed)))
        nw, _, nl = score(x, _load(centroid_path(new_dir, method, data, n, k, seed)))
        old_obj.append(o)
        new_obj.append(nw)
        old_labels[seed] = ol
        new_labels[seed] = nl
    rerun_obj, _, rerun_labels = score(x, _load(centroid_path(rerun_dir, method, data, n, k, RERUN_SEED)))
    ari_new_old = [float(adjusted_rand_score(new_labels[s], old_labels[s])) for s in SEEDS]
    ari_old_seeds = [float(adjusted_rand_score(old_labels[a], old_labels[b])) for a, b in combinations(SEEDS, 2)]
    ari_old_rerun = float(adjusted_rand_score(old_labels[RERUN_SEED], rerun_labels))
    # The 0.2.5 seed spread: the 3 grid objectives, the 3 objectives of the captured centroids and the rerun (7 values).
    grid = old_grid_objectives(old_records_path, method, data, n, k)
    spread = [*grid, *old_obj, rerun_obj]
    record: dict[str, Any] = {
        "schema": 1,
        "method": method,
        "data": data,
        "n": n,
        "k": k,
        "mode": "spherical" if DATASETS[data].spherical else "euclidean",
        "seeds": list(SEEDS),
        "old_source": OLD_SOURCE,
        "new_source": NEW_SOURCE,
        "old_objectives": spread,
        "new_objectives": new_obj,
        "ari_new_old": ari_new_old,
        "ari_old_seeds": ari_old_seeds,
        "ari_old_rerun": ari_old_rerun,
    }
    record.update(gate_setting(spread, new_obj, ari_new_old, ari_old_seeds))
    return record


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BENCH_DIR / p


def main() -> int:
    """Run the gate over all settings, write the records and print a table."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--old", default="scratch/cmp-025")
    parser.add_argument("--rerun", default="scratch/cmp-025-rerun")
    parser.add_argument("--new", default="scratch/cmp-026")
    parser.add_argument("--old-records", default="results/speed-emeans-rocm-0.2.5.jsonl")
    parser.add_argument("--out", default="results/version-ari-emeans-rocm-0.2.5-0.2.6.jsonl")
    args = parser.parse_args()
    old, rerun, new = _resolve(args.old), _resolve(args.rerun), _resolve(args.new)
    old_records = _resolve(args.old_records)
    out = _resolve(args.out)
    for directory, source in ((old, OLD_SOURCE), (rerun, OLD_SOURCE), (new, NEW_SOURCE)):
        load_run_records(directory, source)

    records: list[dict[str, Any]] = []
    rows: list[list[str]] = []
    for data in GATE_DATA:
        x, _ = load_dataset(data, GATE_N, D)
        for method in GATE_METHODS:
            for k in GATE_KS:
                try:
                    r = compare_setting(x, old, rerun, new, old_records, method, data, k)
                except FileNotFoundError as exc:
                    sys.stderr.write(f"missing centroid file: {exc.filename}\n")
                    return 2
                records.append(r)
                lo, hi = r["objective_spread"]
                rows.append(
                    [
                        method, data, str(k), f"{lo:.6g}-{hi:.6g}", f"{r['new_objective_median']:.6g}",
                        f"{r['ari_new_old_median']:.4f}", f"{r['ari_old_seeds_median']:.4f}",
                        f"{r['ari_old_rerun']:.4f}", "yes" if r["ok_objective"] else "no",
                        "yes" if r["ok_ari"] else "no",
                    ]
                )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    header = [
        "method", "data", "k", "0.2.5 objective range", "0.2.6 median", "ARI 0.2.6 vs 0.2.5",
        "ARI 0.2.5 seeds", "ARI 0.2.5 rerun (not gating)", "objective gate", "ARI gate",
    ]  # fmt: skip
    table = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    table += ["| " + " | ".join(row) + " |" for row in rows]
    sys.stdout.write("\n".join(table) + "\n")
    if not all(r["ok"] for r in records):
        sys.stdout.write("R-02 gate: STOP, report to the owner (D-07)\n")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
