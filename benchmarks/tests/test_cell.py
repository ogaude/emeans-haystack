# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from data import load_dataset, starts
from methods import METHODS
from score import score

HERE = Path(__file__).resolve().parent.parent
KEYS = {
    "method", "data", "n", "d", "k", "seed", "passes_requested", "passes_used", "cold_s", "warm_s",
    "warm_status", "transfer_s", "backend", "precision", "centroids_path", "info", "versions", "host", "data_info",
}  # fmt: skip


def run_worker(tmp: Path, method: str, warm: str = "yes", extra: tuple[str, ...] = ()) -> dict:
    cmd = [
        sys.executable, "-I", str(HERE / "worker.py"), "--method", method, "--data", "random", "--n", "2000",
        "--d", "16", "--k", "8", "--seed", "0", "--warm", warm, "--scratch", str(tmp), *extra,
    ]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=HERE)
    return json.loads(out.stdout.strip().splitlines()[-1])


def brute(x: np.ndarray, c: np.ndarray) -> float:
    d2 = ((x.astype(np.float64)[:, None, :] - c.astype(np.float64)[None]) ** 2).sum(-1)
    return float(d2.min(1).mean())


@pytest.mark.parametrize("method", ["emeans-cpu", "faiss-lloyd", "sklearn-lloyd"])
def test_cell_end_to_end(tmp_path: Path, method: str) -> None:
    rec = run_worker(tmp_path, method)
    assert set(rec) == KEYS
    assert rec["cold_s"] > 0
    assert rec["warm_s"] > 0
    assert rec["transfer_s"] is None
    cmd = [
        sys.executable, "-I", str(HERE / "score.py"), "--data", "random", "--n", "2000", "--d", "16",
        "--centroids", rec["centroids_path"],
    ]  # fmt: skip
    out = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=HERE).stdout)
    assert set(out) == {"objective", "objective_sum", "n", "k"}
    x, _ = load_dataset("random", 2000, 16)
    expected = brute(x, np.load(rec["centroids_path"], allow_pickle=False))
    assert out["objective"] == pytest.approx(expected, rel=1e-5)


def test_warm_no(tmp_path: Path) -> None:
    rec = run_worker(tmp_path, "faiss-lloyd", warm="no")
    assert rec["warm_s"] is None
    assert rec["warm_status"] == "skipped (cost)"


def test_starts_shared_and_distinct() -> None:
    x, _ = load_dataset("random", 500, 16)
    a = starts(x, 8, 3)
    assert a.dtype == np.float32
    assert len({row.tobytes() for row in a}) == 8
    assert all(any(np.array_equal(r, xr) for xr in x) for r in a)
    np.testing.assert_array_equal(a, starts(x, 8, 3))
    assert not np.array_equal(a, starts(x, 8, 4))


def test_faiss_no_subsample() -> None:
    x, _ = load_dataset("random", 3000, 16)
    init = starts(x, 4, 0)
    train = METHODS["faiss-lloyd"].train
    one = train(x, init, k=4, spherical=False, seed=1, passes=None)
    two = train(x, init, k=4, spherical=False, seed=2, passes=None)
    np.testing.assert_array_equal(one.centroids, two.centroids)
    assert one.passes_used == 25


def test_one_lloyd_step_matches() -> None:
    x, _ = load_dataset("random", 2000, 16)
    init = starts(x, 8, 0)
    a = METHODS["faiss-lloyd"].train(x, init, k=8, spherical=False, seed=0, passes=1)
    b = METHODS["sklearn-lloyd"].train(x, init, k=8, spherical=False, seed=0, passes=1)
    assert b.passes_used == 1
    np.testing.assert_allclose(a.centroids, b.centroids, atol=1e-4)


def test_emeans_uses_init_and_full_data() -> None:
    x, _ = load_dataset("random", 2000, 16)
    train = METHODS["emeans-cpu"].train
    first = train(x, starts(x, 8, 0), k=8, spherical=False, seed=0, passes=None)
    again = train(x, starts(x, 8, 0), k=8, spherical=False, seed=0, passes=None)
    other = train(x, starts(x, 8, 1), k=8, spherical=False, seed=0, passes=None)
    assert first.info["n_train"] == 2000
    np.testing.assert_array_equal(first.centroids, again.centroids)
    assert not np.array_equal(first.centroids, other.centroids)


def test_score_matches_bruteforce() -> None:
    x, _ = load_dataset("random", 1500, 16)
    c = starts(x, 6, 0) + 0.01
    obj, total, labels = score(x, c, chunk=400)
    d2 = ((x.astype(np.float64)[:, None, :] - c.astype(np.float64)[None]) ** 2).sum(-1)
    np.testing.assert_array_equal(labels, d2.argmin(1))
    assert obj == pytest.approx(d2.min(1).mean(), rel=1e-6)
    assert total == pytest.approx(d2.min(1).sum(), rel=1e-6)
