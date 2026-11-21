# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from haystack_integrations.components.preprocessors.emeans import _model
from haystack_integrations.components.preprocessors.emeans._model import (
    FORMAT_VERSION,
    METADATA_KEYS,
    load_model,
    model_checksum,
    save_model,
)

CALLER_KEYS = {
    "spherical": True,
    "n_train": 10,
    "seed": 1,
    "nepochs": 5.0,
    "backend": "cpu",
    "assignment_backend": "faiss",
    "precision_used": "fp32",
    "emeans_source": "faiss-cpu",
    "faiss_version": "1.16.0",
    "recipe": {"nepochs": 5.0},
}


def unit_centroids(k: int = 3, d: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    c = rng.normal(size=(k, d))
    return (c / np.linalg.norm(c, axis=1, keepdims=True)).astype(np.float32)


def write_raw(path: Path, centroids: Any, meta_overrides: dict[str, Any] | None = None, **extra: Any) -> str:
    """Write an npz by hand so each test can break exactly one thing."""
    c = np.asarray(centroids)
    meta: dict[str, Any] = {
        **CALLER_KEYS,
        "format_version": FORMAT_VERSION,
        "k": int(c.shape[0]) if c.ndim == 2 else 1,
        "d": int(c.shape[1]) if c.ndim == 2 else 1,
        "created_at": datetime.now(UTC).isoformat(),
        "emeans_haystack_version": "x",
    }
    meta.update(meta_overrides or {})
    if "centroids_sha256" not in meta:
        meta["centroids_sha256"] = expected_checksum(c, bool(meta["spherical"]))
    np.savez(path, centroids=c, metadata=np.array(json.dumps(meta)), **extra)
    return str(path)


def expected_checksum(c: np.ndarray, spherical: bool) -> str:
    """The format-v1 checksum, written out by hand so that a change to its definition breaks these tests."""
    k, d = (c.shape[0], c.shape[1]) if c.ndim == 2 else (1, 1)
    header = f"spherical={1 if spherical else 0};k={k};d={d};".encode("ascii")
    return hashlib.sha256(header + c.tobytes()).hexdigest()


def good_file(tmp_path: Path, name: str = "m.npz") -> str:
    p = str(tmp_path / name)
    save_model(p, unit_centroids(), dict(CALLER_KEYS), overwrite=False)
    return p


def test_round_trip(tmp_path: Path) -> None:
    c = unit_centroids()
    p = str(tmp_path / "m.npz")
    save_model(p, c, dict(CALLER_KEYS), overwrite=False)
    with np.load(p, allow_pickle=False) as z:
        assert set(z.files) == {"centroids", "metadata"}
        assert z["metadata"].ndim == 0
        assert z["metadata"].dtype.kind == "U"
    model = load_model(p)
    assert model.centroids.dtype == np.float32
    np.testing.assert_array_equal(model.centroids, c)
    assert set(model.metadata) == set(METADATA_KEYS)
    assert len(METADATA_KEYS) == 16
    assert model.metadata["format_version"] == 1
    assert model.metadata["centroids_sha256"] == expected_checksum(c, spherical=True)
    assert model.metadata["centroids_sha256"] == model_checksum(c, spherical=True)
    created = datetime.fromisoformat(model.metadata["created_at"])
    assert created.utcoffset() is not None
    assert created.utcoffset().total_seconds() == 0
    assert (model.k, model.d, model.spherical) == (3, 4, True)


def test_save_exists_no_overwrite(tmp_path: Path) -> None:
    p = good_file(tmp_path)
    before = Path(p).read_bytes()
    with pytest.raises(FileExistsError):
        save_model(p, unit_centroids(2, 4), dict(CALLER_KEYS), overwrite=False)
    assert Path(p).read_bytes() == before
    assert [f.name for f in tmp_path.iterdir()] == ["m.npz"]


def test_save_overwrite(tmp_path: Path) -> None:
    p = good_file(tmp_path)
    save_model(p, unit_centroids(2, 4), dict(CALLER_KEYS), overwrite=True)
    assert load_model(p).k == 2
    assert [f.name for f in tmp_path.iterdir()] == ["m.npz"]


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_save_rejects_bad_metadata_keys(tmp_path: Path, change: str) -> None:
    meta = dict(CALLER_KEYS)
    if change == "missing":
        del meta["seed"]
    else:
        meta["surprise"] = 1
    with pytest.raises(ValueError, match="metadata keys"):
        save_model(str(tmp_path / "m.npz"), unit_centroids(), meta, overwrite=False)


def test_save_link_permission_error_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise PermissionError

    monkeypatch.setattr(os, "link", boom)
    p = str(tmp_path / "m.npz")
    save_model(p, unit_centroids(), dict(CALLER_KEYS), overwrite=False)
    assert load_model(p).k == 3
    with pytest.raises(FileExistsError):
        save_model(p, unit_centroids(), dict(CALLER_KEYS), overwrite=False)
    assert [f.name for f in tmp_path.iterdir()] == ["m.npz"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_save_applies_umask_without_changing_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    previous = os.umask(0o027)
    try:

        def no_umask(*_args: object) -> int:
            msg = "save_model must not touch the process umask"
            raise AssertionError(msg)

        monkeypatch.setattr(os, "umask", no_umask)
        p = good_file(tmp_path)
    finally:
        monkeypatch.undo()
        os.umask(previous)
    assert os.stat(p).st_mode & 0o777 == 0o640


def test_save_failure_leaves_no_tmp(tmp_path: Path) -> None:
    meta = dict(CALLER_KEYS)
    meta["recipe"] = object()  # not JSON serialisable
    with pytest.raises(TypeError):
        save_model(str(tmp_path / "m.npz"), unit_centroids(), meta, overwrite=False)
    assert list(tmp_path.iterdir()) == []


def test_load_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "nope.npz"))


def test_load_directory_is_file_check(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="file check failed"):
        load_model(str(tmp_path))


def check_fails(path: str, check: str) -> None:
    with pytest.raises(ValueError) as info:
        load_model(path)
    assert path in str(info.value)
    assert f"{check} check failed" in str(info.value)


def test_load_garbage_bytes(tmp_path: Path) -> None:
    p = tmp_path / "g.npz"
    p.write_bytes(b"this is definitely not a model" * 10)
    check_fails(str(p), "arrays")


def test_load_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "e.npz"
    p.write_bytes(b"")
    check_fails(str(p), "arrays")


def test_load_truncated_zip(tmp_path: Path) -> None:
    good = Path(good_file(tmp_path)).read_bytes()
    p = tmp_path / "t.npz"
    p.write_bytes(good[: len(good) // 2])
    check_fails(str(p), "arrays")


def test_load_bare_npy(tmp_path: Path) -> None:
    p = tmp_path / "b.npy"
    np.save(p, unit_centroids())
    check_fails(str(p), "arrays")


def test_load_object_array_refused(tmp_path: Path) -> None:
    p = tmp_path / "o.npz"
    np.savez(p, centroids=np.array([{"a": 1}], dtype=object), metadata=np.array("{}"))
    check_fails(str(p), "arrays")


def test_load_missing_metadata(tmp_path: Path) -> None:
    p = tmp_path / "mm.npz"
    np.savez(p, centroids=unit_centroids())
    check_fails(str(p), "arrays")


def test_load_extra_array(tmp_path: Path) -> None:
    p = write_raw(tmp_path / "x.npz", unit_centroids(), extra=np.zeros(2))
    check_fails(p, "arrays")


def test_load_metadata_not_json(tmp_path: Path) -> None:
    p = tmp_path / "nj.npz"
    np.savez(p, centroids=unit_centroids(), metadata=np.array("{not json"))
    check_fails(str(p), "arrays")


def test_load_metadata_not_object(tmp_path: Path) -> None:
    p = tmp_path / "no.npz"
    np.savez(p, centroids=unit_centroids(), metadata=np.array("[1]"))
    check_fails(str(p), "arrays")


def test_load_metadata_missing_required_key(tmp_path: Path) -> None:
    p = tmp_path / "rk.npz"
    np.savez(p, centroids=unit_centroids(), metadata=np.array(json.dumps({"k": 3})))
    check_fails(str(p), "arrays")


def test_load_newer_format(tmp_path: Path) -> None:
    p = write_raw(tmp_path / "v2.npz", unit_centroids(), {"format_version": 2})
    check_fails(p, "format_version")
    with pytest.raises(ValueError, match="upgrade emeans-haystack"):
        load_model(p)


@pytest.mark.parametrize("version", [0, "1", 1.0, True])
def test_load_bad_format_version(tmp_path: Path, version: object) -> None:
    p = write_raw(tmp_path / "v.npz", unit_centroids(), {"format_version": version})
    check_fails(p, "format_version")


@pytest.mark.parametrize("spherical", ["false", "0", 1, 0, None])
def test_load_spherical_must_be_bool(tmp_path: Path, spherical: object) -> None:
    p = write_raw(tmp_path / "sp.npz", unit_centroids() * 3, {"spherical": spherical})
    check_fails(p, "metadata")


@pytest.mark.parametrize("checksum", [None, 123, ["x"]])
def test_load_checksum_must_be_string(tmp_path: Path, checksum: object) -> None:
    p = write_raw(tmp_path / "cs.npz", unit_centroids(), {"centroids_sha256": checksum})
    check_fails(p, "metadata")


def test_check_order_metadata_before_shape(tmp_path: Path) -> None:
    check_fails(write_raw(tmp_path / "o0.npz", unit_centroids(), {"spherical": "true", "k": 9}), "metadata")


def test_load_wrong_shape(tmp_path: Path) -> None:
    p = write_raw(tmp_path / "s.npz", unit_centroids(3, 4), {"k": 5})
    check_fails(p, "shape")


def test_load_float64(tmp_path: Path) -> None:
    p = write_raw(tmp_path / "f64.npz", unit_centroids().astype(np.float64))
    check_fails(p, "shape")


def test_load_nan(tmp_path: Path) -> None:
    c = unit_centroids()
    c[1, 1] = np.nan
    check_fails(write_raw(tmp_path / "n.npz", c), "finite")


def test_load_non_unit_row(tmp_path: Path) -> None:
    c = unit_centroids()
    c[0] *= 0.5
    check_fails(write_raw(tmp_path / "u.npz", c), "unit_norm")


def test_load_zero_row_spherical(tmp_path: Path) -> None:
    c = unit_centroids()
    c[2] = 0
    check_fails(write_raw(tmp_path / "z.npz", c), "unit_norm")


def test_load_non_unit_euclidean_ok(tmp_path: Path) -> None:
    c = unit_centroids() * 3
    p = write_raw(tmp_path / "eu.npz", c, {"spherical": False})
    assert not load_model(p).spherical


def test_load_checksum_mismatch(tmp_path: Path) -> None:
    c = unit_centroids()
    p = write_raw(tmp_path / "c.npz", c, {"centroids_sha256": "0" * 64})
    check_fails(p, "checksum")


def test_load_tampered_centroids(tmp_path: Path) -> None:
    c = unit_centroids()
    other = unit_centroids()[::-1].copy()
    p = write_raw(tmp_path / "tp.npz", other, {"centroids_sha256": expected_checksum(c, spherical=True)})
    check_fails(p, "checksum")


def test_checksum_ignores_unrelated_metadata(tmp_path: Path) -> None:
    p = write_raw(tmp_path / "um.npz", unit_centroids(), {"seed": 99, "backend": "edited"})
    assert load_model(p).k == 3


def test_load_flipped_spherical_flag_fails_checksum(tmp_path: Path) -> None:
    # Unit-norm centroids pass the Euclidean load checks too; only the checksum catches the flipped mode.
    c = unit_centroids()
    p = write_raw(tmp_path / "flip.npz", c, {"spherical": False, "centroids_sha256": expected_checksum(c, True)})
    check_fails(p, "checksum")


def test_load_old_centroid_only_checksum_refused(tmp_path: Path) -> None:
    c = unit_centroids()
    p = write_raw(tmp_path / "old.npz", c, {"centroids_sha256": hashlib.sha256(c.tobytes()).hexdigest()})
    check_fails(p, "checksum")


def test_checksum_covers_mode_and_shape() -> None:
    c = unit_centroids(4, 6)
    base = model_checksum(c, spherical=True)
    assert model_checksum(c, spherical=False) != base
    assert model_checksum(c.reshape(6, 4), spherical=True) != base
    assert model_checksum(np.asfortranarray(c), spherical=True) == base


def test_save_rejects_non_bool_spherical(tmp_path: Path) -> None:
    meta = {**CALLER_KEYS, "spherical": 1}
    with pytest.raises(ValueError, match="spherical"):
        save_model(str(tmp_path / "m.npz"), unit_centroids(), meta, overwrite=False)


def test_check_order_version_before_finite(tmp_path: Path) -> None:
    c = unit_centroids()
    c[0, 0] = np.nan
    check_fails(write_raw(tmp_path / "o1.npz", c, {"format_version": 2}), "format_version")


def test_check_order_shape_before_finite(tmp_path: Path) -> None:
    c = unit_centroids()
    c[0, 0] = np.nan
    check_fails(write_raw(tmp_path / "o2.npz", c, {"k": 9}), "shape")


def test_check_order_finite_before_unit_norm_before_checksum(tmp_path: Path) -> None:
    c = unit_centroids()
    c[0] *= 0.5
    check_fails(write_raw(tmp_path / "o3.npz", c, {"centroids_sha256": "bad"}), "unit_norm")


def test_no_pickle_in_module() -> None:
    source = Path(_model.__file__).read_text()
    assert "allow_pickle=True" not in source
    assert "allow_pickle=False" in source
