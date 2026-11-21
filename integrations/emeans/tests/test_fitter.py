# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from haystack import Document

from haystack_integrations.components.preprocessors.emeans import EmeansFitter, fitter
from haystack_integrations.components.preprocessors.emeans._backend import FitResult
from haystack_integrations.components.preprocessors.emeans._model import METADATA_KEYS, load_model

MakeDocs = Callable[..., list[Document]]

STATS_TYPES: dict[str, type] = {
    "k": int,
    "d": int,
    "n_documents": int,
    "n_skipped": int,
    "n_train": int,
    "spherical": bool,
    "backend": str,
    "assignment_backend": str,
    "device_name": str,
    "precision_used": str,
    "emeans_source": str,
    "faiss_version": str,
    "nepochs": float,
    "batch_size": int,
    "alpha0": float,
    "n_splits": int,
    "imbalance": float,
    "train_seconds": float,
}
STATS_KEYS = {*STATS_TYPES, "fallback_reason"}


def warnings_in(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelname == "WARNING"]


def set_embedding(docs: list[Document], index: int, embedding: list[float] | None) -> None:
    docs[index] = dataclasses.replace(docs[index], embedding=embedding)


def make_fitter(tmp_path: Path, **kwargs: Any) -> EmeansFitter:
    params: dict[str, Any] = {"model_path": str(tmp_path / "m.npz"), "k": 4, "nepochs": 2.0}
    params.update(kwargs)
    return EmeansFitter(**params)


class TestInit:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"k": None},
            {"k": 2, "target_cluster_size": 5},
            {"k": 0},
            {"k": True},
            {"k": None, "target_cluster_size": 0},
            {"nepochs": 0},
            {"nepochs": -1},
            {"nepochs": True},
            {"nepochs": float("inf")},
            {"nepochs": float("nan")},
            {"nepochs": "5"},
            {"spherical": "no"},
            {"spherical": 1},
            {"deterministic": "yes"},
            {"overwrite": "no"},
            {"seed": "1"},
            {"seed": 1.0},
            {"seed": True},
            {"gpu_threshold": "big"},
            {"gpu_threshold": float("nan")},
            {"gpu_threshold": True},
            {"max_points_per_centroid": 0},
            {"max_points_per_centroid": True},
            {"device": "tpu"},
            {"precision": "int8"},
        ],
    )
    def test_invalid(self, tmp_path: Path, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            make_fitter(tmp_path, **kwargs)

    def test_numpy_integers_accepted_and_stored_as_int(self, tmp_path: Path) -> None:
        f = make_fitter(tmp_path, k=np.int64(3), seed=np.int32(7), max_points_per_centroid=np.int64(50))
        assert (type(f.k), type(f.seed), type(f.max_points_per_centroid)) == (int, int, int)
        assert (f.k, f.seed, f.max_points_per_centroid) == (3, 7, 50)

    def test_no_filesystem_access(self, tmp_path: Path) -> None:
        EmeansFitter(model_path=str(tmp_path / "missing" / "m.npz"), k=2)

    def test_to_from_dict_round_trip(self, tmp_path: Path) -> None:
        params: dict[str, Any] = {
            "model_path": str(tmp_path / "x.npz"),
            "k": None,
            "target_cluster_size": 7,
            "spherical": False,
            "nepochs": 3.5,
            "precision": "bf16",
            "device": "cpu",
            "gpu_threshold": 123.0,
            "seed": 99,
            "deterministic": True,
            "max_points_per_centroid": 50,
            "overwrite": True,
        }
        data = EmeansFitter(**params).to_dict()
        assert set(data["init_parameters"]) == set(params)
        assert data["init_parameters"] == params
        restored = EmeansFitter.from_dict(data)
        for name, value in params.items():
            assert getattr(restored, name) == value


class TestRunOverrides:
    def test_run_k_replaces_target_cluster_size(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        f = make_fitter(tmp_path, k=None, target_cluster_size=50)
        assert f.run(documents=make_docs(60), k=3)["stats"]["k"] == 3
        assert f.k is None
        assert f.target_cluster_size == 50

    def test_run_target_cluster_size_replaces_k(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        f = make_fitter(tmp_path, k=3)
        assert f.run(documents=make_docs(40), target_cluster_size=10)["stats"]["k"] == 4
        assert f.k == 3
        assert f.target_cluster_size is None

    def test_run_model_path_override(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        other = str(tmp_path / "other.npz")
        out = make_fitter(tmp_path).run(documents=make_docs(40), model_path=other)
        assert out["model_path"] == other
        assert Path(other).exists()
        assert not (tmp_path / "m.npz").exists()

    def test_run_invalid_override(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        with pytest.raises(ValueError):
            make_fitter(tmp_path).run(documents=make_docs(40), k=0)


class TestKResolution:
    @pytest.mark.parametrize(("n", "tcs", "expected"), [(100, 40, 2), (10, 50, 1), (5, 1, 5)])
    def test_target_cluster_size(self, tmp_path: Path, make_docs: MakeDocs, n: int, tcs: int, expected: int) -> None:
        f = make_fitter(tmp_path, k=None, target_cluster_size=tcs)
        assert f.run(documents=make_docs(n))["stats"]["k"] == expected


class TestStats:
    def test_keys_and_types(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        stats = make_fitter(tmp_path).run(documents=make_docs(50))["stats"]
        assert set(stats) == STATS_KEYS
        assert len(stats) == 19
        for key, expected in STATS_TYPES.items():
            assert type(stats[key]) is expected, key
        assert stats["fallback_reason"] is None
        assert stats["backend"] == "cpu"
        assert stats["device_name"] == "cpu"
        assert stats["precision_used"] == "fp32"
        assert stats["emeans_source"] == "faiss-cpu"
        assert stats["assignment_backend"] == "faiss"
        assert stats["k"] == 4
        assert stats["d"] == 16
        assert stats["n_documents"] == 50
        assert stats["n_skipped"] == 0

    def test_n_train_subsample(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        stats = make_fitter(tmp_path, max_points_per_centroid=5).run(documents=make_docs(100))["stats"]
        assert stats["n_train"] == 20

    def test_n_train_default(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        assert make_fitter(tmp_path).run(documents=make_docs(100))["stats"]["n_train"] == 100

    def test_metadata_matches_stats(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        f = make_fitter(tmp_path, max_points_per_centroid=5)
        stats = f.run(documents=make_docs(100))["stats"]
        meta = load_model(str(tmp_path / "m.npz")).metadata
        assert set(meta) == set(METADATA_KEYS)
        assert meta["backend"] == "cpu"
        assert meta["precision_used"] == "fp32"
        assert meta["spherical"] is True
        assert meta["n_train"] == stats["n_train"]
        assert meta["nepochs"] == 2.0
        assert meta["seed"] == 1234

    def test_info_log_names_backend(
        self, tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            make_fitter(tmp_path).run(documents=make_docs(40))
        infos = [r for r in caplog.records if r.levelname == "INFO" and r.name == fitter.__name__]
        assert len(infos) == 1
        assert "cpu" in infos[0].getMessage()


class TestEmbeddings:
    def test_skipped_documents(self, tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture) -> None:
        docs = make_docs(50)
        for i in range(3):
            set_embedding(docs, i, None)
        with caplog.at_level(logging.DEBUG):
            stats = make_fitter(tmp_path).run(documents=docs)["stats"]
        assert (stats["n_documents"], stats["n_skipped"]) == (50, 3)
        warnings = warnings_in(caplog)
        assert len(warnings) == 1
        assert "3" in warnings[0].getMessage()

    def test_mixed_dimensions(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        docs = make_docs(20)
        set_embedding(docs, 5, [0.1] * 7)
        with pytest.raises(ValueError) as info:
            make_fitter(tmp_path).run(documents=docs)
        msg = str(info.value)
        assert docs[5].id in msg
        assert "7" in msg
        assert "16" in msg

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1e39])
    def test_non_finite(self, tmp_path: Path, make_docs: MakeDocs, bad: float) -> None:
        docs = make_docs(20)
        set_embedding(docs, 2, [bad, *(docs[2].embedding or [])[1:]])
        with pytest.raises(ValueError, match=docs[2].id):
            make_fitter(tmp_path).run(documents=docs)

    def test_out_of_float32_range_has_own_message(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        docs = make_docs(20)
        set_embedding(docs, 2, [1e39, *(docs[2].embedding or [])[1:]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(ValueError, match="outside the float32 range") as info:
                make_fitter(tmp_path).run(documents=docs)
        assert docs[2].id in str(info.value)
        assert "NaN" not in str(info.value)

    def test_spherical_zero_vector(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        docs = make_docs(20)
        set_embedding(docs, 4, [0.0] * 16)
        with pytest.raises(ValueError, match=docs[4].id):
            make_fitter(tmp_path).run(documents=docs)

    def test_euclidean_zero_vector_ok(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        docs = make_docs(20)
        set_embedding(docs, 4, [0.0] * 16)
        stats = make_fitter(tmp_path, spherical=False).run(documents=docs)["stats"]
        assert stats["spherical"] is False

    def test_single_usable_document(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        with pytest.raises(ValueError, match="2"):
            make_fitter(tmp_path, k=1).run(documents=make_docs(1))

    def test_k_larger_than_usable(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        docs = make_docs(8)
        for i in range(2):
            set_embedding(docs, i, None)
        with pytest.raises(ValueError) as info:
            make_fitter(tmp_path, k=10).run(documents=docs)
        assert "10" in str(info.value)
        assert "6" in str(info.value)


class TestFiles:
    def test_exists_no_overwrite_before_training(
        self, tmp_path: Path, make_docs: MakeDocs, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "m.npz"
        path.write_bytes(b"precious")

        def spy(*_args: object, **_kwargs: object) -> None:
            pytest.fail("fit_model must not be called")

        monkeypatch.setattr(fitter, "fit_model", spy)
        with pytest.raises(FileExistsError):
            make_fitter(tmp_path).run(documents=make_docs(40))
        assert path.read_bytes() == b"precious"

    def test_overwrite(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        make_fitter(tmp_path).run(documents=make_docs(40))
        make_fitter(tmp_path, k=2, overwrite=True).run(documents=make_docs(40))
        assert load_model(str(tmp_path / "m.npz")).k == 2

    def test_directory_at_model_path_fails_before_training(
        self, tmp_path: Path, make_docs: MakeDocs, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def spy(*_args: object, **_kwargs: object) -> None:
            pytest.fail("fit_model must not be called")

        monkeypatch.setattr(fitter, "fit_model", spy)
        target = tmp_path / "a_dir"
        target.mkdir()
        with pytest.raises(IsADirectoryError, match="a_dir"):
            make_fitter(tmp_path, model_path=str(target), overwrite=True).run(documents=make_docs(40))
        with pytest.raises(FileExistsError):
            make_fitter(tmp_path, model_path=str(target)).run(documents=make_docs(40))

    def test_missing_parent_directory(
        self, tmp_path: Path, make_docs: MakeDocs, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def spy(*_args: object, **_kwargs: object) -> None:
            pytest.fail("fit_model must not be called")

        monkeypatch.setattr(fitter, "fit_model", spy)
        target = tmp_path / "no_such_dir" / "m.npz"
        with pytest.raises(FileNotFoundError, match="no_such_dir"):
            make_fitter(tmp_path, model_path=str(target)).run(documents=make_docs(40))

    def test_empty_centroid_writes_nothing(
        self, tmp_path: Path, make_docs: MakeDocs, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = fitter.fit_model

        def fake(*args: Any, **kwargs: Any) -> FitResult:
            result = real(*args, **kwargs)
            centroids = result.centroids.copy()
            centroids[1] = 0
            return FitResult(centroids=centroids, info=result.info)

        monkeypatch.setattr(fitter, "fit_model", fake)
        with pytest.raises(ValueError, match="1 empty centroids"):
            make_fitter(tmp_path).run(documents=make_docs(40))
        assert list(tmp_path.iterdir()) == []

    def test_non_unit_spherical_centroid_writes_nothing(
        self, tmp_path: Path, make_docs: MakeDocs, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = fitter.fit_model

        def fake(*args: Any, **kwargs: Any) -> FitResult:
            result = real(*args, **kwargs)
            centroids = result.centroids.copy()
            centroids[1] *= 0.9  # not empty, but outside the loader's unit-norm tolerance
            return FitResult(centroids=centroids, info=result.info)

        monkeypatch.setattr(fitter, "fit_model", fake)
        with pytest.raises(ValueError, match=r"1 spherical centroids .* unit norm"):
            make_fitter(tmp_path).run(documents=make_docs(40))
        assert list(tmp_path.iterdir()) == []


class TestSpherical:
    def test_unit_centroids_and_inputs_untouched(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        docs = make_docs(40, unit=False)
        docs = [dataclasses.replace(d, embedding=[v * 7 for v in d.embedding or []]) for d in docs]
        before = [(d.id, list(d.embedding), dict(d.meta)) for d in docs]  # type: ignore[arg-type]
        out = make_fitter(tmp_path).run(documents=docs)
        assert set(out) == {"model_path", "stats"}
        norms = np.linalg.norm(load_model(out["model_path"]).centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)
        assert [(d.id, list(d.embedding), dict(d.meta)) for d in docs] == before  # type: ignore[arg-type]


class TestPrecisionAndDevice:
    @pytest.mark.parametrize("precision", ["bf16", "fp16"])
    def test_unsupported_precision(
        self, tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture, precision: str
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            stats = make_fitter(tmp_path, precision=precision).run(documents=make_docs(40))["stats"]
        assert stats["precision_used"] == "fp32"
        assert len(warnings_in(caplog)) == 1

    @pytest.mark.parametrize("precision", ["auto", "fp32"])
    def test_supported_precision(
        self, tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture, precision: str
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            make_fitter(tmp_path, precision=precision).run(documents=make_docs(40))
        assert warnings_in(caplog) == []

    def test_auto_above_threshold_falls_back_silently(
        self, tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            stats = make_fitter(tmp_path, device="auto", gpu_threshold=1).run(documents=make_docs(40))["stats"]
        assert stats["fallback_reason"] == "no GPU backend usable: PyTorch is not installed"
        assert warnings_in(caplog) == []

    def test_cuda_raises(self, tmp_path: Path, make_docs: MakeDocs) -> None:
        with pytest.raises(RuntimeError, match="no GPU backend usable"):
            make_fitter(tmp_path, device="cuda").run(documents=make_docs(40))
        assert list(tmp_path.iterdir()) == []

    def test_deterministic_cpu(self, tmp_path: Path, make_docs: MakeDocs, caplog: pytest.LogCaptureFixture) -> None:
        docs = make_docs(40)
        with caplog.at_level(logging.DEBUG):
            make_fitter(tmp_path, deterministic=True).run(documents=docs, model_path=str(tmp_path / "a.npz"))
            make_fitter(tmp_path, deterministic=True).run(documents=docs, model_path=str(tmp_path / "b.npz"))
        assert warnings_in(caplog) == []
        np.testing.assert_array_equal(
            load_model(str(tmp_path / "a.npz")).centroids, load_model(str(tmp_path / "b.npz")).centroids
        )
