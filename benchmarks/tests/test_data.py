# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import data
import fetch_data

PAYLOAD = bytes(range(256)) * 8
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse:
    def __init__(self, data: bytes, status: int) -> None:
        self._data = data
        self._pos = 0
        self.status = status

    def read(self, n: int = -1) -> bytes:
        end = len(self._data) if n < 0 else self._pos + n
        chunk = self._data[self._pos : end]
        self._pos = min(end, len(self._data))
        return chunk

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeUrlopen:
    def __init__(self, data: bytes, honour_range: bool = True) -> None:
        self.data = data
        self.honour_range = honour_range
        self.ranges: list[str | None] = []
        self.calls = 0

    def __call__(self, request: object, **_: object) -> FakeResponse:
        self.calls += 1
        rng = request.get_header("Range")  # type: ignore[attr-defined]
        self.ranges.append(rng)
        if rng and self.honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            return FakeResponse(self.data[start:], 206)
        return FakeResponse(self.data, 200)


URL = "https://example.invalid/file.f32"


def test_verify_accepts_and_rejects(tmp_path: Path) -> None:
    p = tmp_path / "a.f32"
    p.write_bytes(PAYLOAD)
    assert fetch_data.verify(p, DIGEST, len(PAYLOAD)) is None
    assert fetch_data.verify(p, "0" * 64, len(PAYLOAD)) is not None
    assert fetch_data.verify(p, DIGEST, len(PAYLOAD) + 1) is not None
    assert fetch_data.verify(tmp_path / "missing.f32", DIGEST, 1) is not None


def test_download_resumes_with_range(tmp_path: Path) -> None:
    dest = tmp_path / "a.f32"
    (tmp_path / "a.f32.part").write_bytes(PAYLOAD[:10])
    fake = FakeUrlopen(PAYLOAD)
    assert fetch_data.fetch(URL, dest, DIGEST, len(PAYLOAD), urlopen=fake) == 0
    assert fake.ranges == ["bytes=10-"]
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "a.f32.part").exists()


def test_server_ignoring_range_restarts(tmp_path: Path) -> None:
    dest = tmp_path / "a.f32"
    (tmp_path / "a.f32.part").write_bytes(b"garbage!!!")
    fake = FakeUrlopen(PAYLOAD, honour_range=False)
    assert fetch_data.fetch(URL, dest, DIGEST, len(PAYLOAD), urlopen=fake) == 0
    assert dest.read_bytes() == PAYLOAD


def test_refuses_on_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = tmp_path / "a.f32"
    fake = FakeUrlopen(PAYLOAD)
    assert fetch_data.fetch(URL, dest, "0" * 64, len(PAYLOAD), urlopen=fake) == 1
    assert capsys.readouterr().err
    assert not dest.exists()
    assert (tmp_path / "a.f32.part").exists()


def test_already_present_is_verified_not_redownloaded(tmp_path: Path) -> None:
    dest = tmp_path / "a.f32"
    dest.write_bytes(PAYLOAD)
    fake = FakeUrlopen(PAYLOAD)
    assert fetch_data.fetch(URL, dest, DIGEST, len(PAYLOAD), urlopen=fake) == 0
    assert fake.calls == 0


def test_https_only(tmp_path: Path) -> None:
    fake = FakeUrlopen(PAYLOAD)
    assert fetch_data.fetch("http://example.invalid/x", tmp_path / "a.f32", DIGEST, len(PAYLOAD), urlopen=fake) == 2
    assert fake.calls == 0


def _write_fixture(path: Path, rows: int, d: int, unit: bool = True) -> np.ndarray:
    x = np.random.default_rng(1).standard_normal((rows, d)).astype(np.float32)
    if unit:
        x = data.normalise_rows(x)
    x.tofile(path)
    return x


def test_cohere_subset_rule(tmp_path: Path) -> None:
    p = tmp_path / "f.f32"
    x = _write_fixture(p, 1000, 8)
    a, info = data._load_f32(p, 1000, 8, 100)
    rows = np.sort(np.random.default_rng(data.COHERE_SUBSET_SEED).choice(1000, 100, replace=False))
    assert np.array_equal(a, x[rows])
    assert fetch_data.REVISION in info["data_id"]
    assert info["subset_seed"] == data.COHERE_SUBSET_SEED
    assert info["subset_rows_sha256"] == hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest()
    b, info2 = data._load_f32(p, 1000, 8, 100)
    assert np.array_equal(a, b)
    assert info == info2
    full, info_full = data._load_f32(p, 1000, 8, 1000)
    assert np.array_equal(full, x)
    assert info_full["subset_rows_sha256"] is None
    assert info_full["subset_seed"] is None
    with pytest.raises(ValueError, match="rows"):
        data._load_f32(p, 1000, 8, 1001)


def test_cohere_size_check(tmp_path: Path) -> None:
    p = tmp_path / "f.f32"
    _write_fixture(p, 999, 8)
    with pytest.raises(ValueError, match=str(1000 * 8 * 4)):
        data._load_f32(p, 1000, 8, 10)


def test_cohere_unit_norm_check(tmp_path: Path) -> None:
    p = tmp_path / "f.f32"
    _write_fixture(p, 1000, 8, unit=False)
    with pytest.raises(ValueError, match="unit"):
        data._load_f32(p, 1000, 8, 100)


def test_cohere_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = data.DATASETS["cohere"]
    assert spec.spherical is True
    monkeypatch.setattr(fetch_data, "CACHE_DIR", tmp_path)
    reason = spec.available()
    assert reason is not None
    assert "fetch_data.py" in reason
    p = tmp_path / fetch_data.FILENAME
    with p.open("wb") as f:
        f.truncate(data.COHERE_ROWS * data.COHERE_D * 4)
    assert spec.available() is None
    with p.open("wb") as f:
        f.truncate(100)
    assert spec.available() is not None


REAL = fetch_data.CACHE_DIR / fetch_data.FILENAME


@pytest.mark.skipif(
    not REAL.is_file() or REAL.stat().st_size != fetch_data.SIZE,
    reason="run fetch_data.py first",
)
@pytest.mark.parametrize("method", ["emeans-cpu", "faiss-lloyd", "sklearn-lloyd"])
def test_cohere_cell_end_to_end(tmp_path: Path, method: str) -> None:
    here = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable, "-I", str(here / "worker.py"), "--method", method, "--data", "cohere", "--n", "10000",
        "--k", "64", "--seed", "0", "--warm", "no", "--scratch", str(tmp_path),
    ]  # fmt: skip
    w = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=here)
    assert w.returncode == 0, w.stderr
    rec = json.loads(w.stdout.strip().splitlines()[-1])
    assert fetch_data.REVISION in rec["data_info"]["data_id"]
    assert rec["data_info"]["subset_rows_sha256"]
    cmd = [
        sys.executable, "-I", str(here / "score.py"), "--data", "cohere", "--n", "10000",
        "--centroids", rec["centroids_path"],
    ]  # fmt: skip
    s = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=here)
    assert s.returncode == 0, s.stderr
    objective = json.loads(s.stdout)["objective"]
    assert np.isfinite(objective)
    assert objective < 2.0


def test_cohere_normalises_non_unit_rows(tmp_path: Path) -> None:
    p = tmp_path / "f.f32"
    _write_fixture(p, 1000, 8, unit=False)
    x, info = data._load_f32(p, 1000, 8, 100, normalise=True)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)
    assert info["rows_normalised"] is True
    assert info["raw_norm_max"] > 1.0 + 1e-3
