# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from haystack import Document

from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter
from haystack_integrations.components.preprocessors.emeans import clusterer as clusterer_module
from haystack_integrations.components.preprocessors.emeans._model import load_model

MakeDocs = Callable[..., list[Document]]


def _fit(path: Path, docs: list[Document], *, spherical: bool = True, k: int = 4, overwrite: bool = False) -> str:
    EmeansFitter(model_path=str(path), k=k, spherical=spherical, overwrite=overwrite).run(documents=docs)
    return str(path)


def _matrix(docs: list[Document]) -> np.ndarray:
    return np.asarray([d.embedding for d in docs], dtype=np.float64)


@pytest.fixture
def docs(make_docs: MakeDocs) -> list[Document]:
    return make_docs(200)


@pytest.fixture
def spherical_model(tmp_path: Path, docs: list[Document]) -> str:
    return _fit(tmp_path / "s.npz", docs)


@pytest.fixture
def euclidean_model(tmp_path: Path, make_docs: MakeDocs) -> str:
    return _fit(tmp_path / "e.npz", make_docs(200, unit=False), spherical=False)


def _assert_nearest(out: list[Document], x: np.ndarray, c: np.ndarray, *, spherical: bool) -> None:
    if spherical:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        scores = x @ c.T.astype(np.float64)
        order = np.sort(scores, axis=1)
        gap = order[:, -1] - order[:, -2]
        expected = scores.argmax(axis=1)
    else:
        scores = ((x[:, None, :] - c[None].astype(np.float64)) ** 2).sum(axis=2)
        order = np.sort(scores, axis=1)
        gap = order[:, 1] - order[:, 0]
        expected = scores.argmin(axis=1)
    for doc, exp, g in zip(out, expected, gap, strict=True):
        if g >= 1e-5:
            assert doc.meta["cluster_id"] == exp


class TestStableIds:
    def test_spherical_matches_brute_force(self, docs: list[Document], spherical_model: str) -> None:
        out = EmeansDocumentClusterer(model_path=spherical_model).run(documents=docs)["documents"]
        _assert_nearest(out, _matrix(docs), load_model(spherical_model).centroids, spherical=True)

    def test_euclidean_matches_brute_force(self, make_docs: MakeDocs, euclidean_model: str) -> None:
        docs = make_docs(200, unit=False)
        out = EmeansDocumentClusterer(model_path=euclidean_model).run(documents=docs)["documents"]
        _assert_nearest(out, _matrix(docs), load_model(euclidean_model).centroids, spherical=False)

    def test_scale_does_not_change_spherical_ids(self, docs: list[Document], spherical_model: str) -> None:
        scaled = [Document(content=d.content, embedding=[v * 7 for v in d.embedding or []]) for d in docs]
        comp = EmeansDocumentClusterer(model_path=spherical_model)
        a = comp.run(documents=docs)["documents"]
        b = comp.run(documents=scaled)["documents"]
        assert [d.meta["cluster_id"] for d in a] == [d.meta["cluster_id"] for d in b]


class TestMetaKeys:
    def test_types_and_ranges(self, docs: list[Document], spherical_model: str) -> None:
        scaled = [Document(content=d.content, embedding=[v * 7 for v in d.embedding or []]) for d in docs]
        out = EmeansDocumentClusterer(model_path=spherical_model, return_distance=True).run(documents=scaled)
        for doc in out["documents"]:
            assert type(doc.meta["cluster_id"]) is int
            assert 0 <= doc.meta["cluster_id"] < 4
            assert type(doc.meta["cluster_similarity"]) is float
            assert -1.0 <= doc.meta["cluster_similarity"] <= 1.0
            assert "cluster_distance" not in doc.meta

    def test_euclidean_distance(self, make_docs: MakeDocs, euclidean_model: str) -> None:
        docs = make_docs(50, unit=False)
        c = load_model(euclidean_model).centroids.astype(np.float64)
        out = EmeansDocumentClusterer(model_path=euclidean_model, return_distance=True).run(documents=docs)
        for doc in out["documents"]:
            assert type(doc.meta["cluster_distance"]) is float
            assert "cluster_similarity" not in doc.meta
            expected = ((np.asarray(doc.embedding) - c[doc.meta["cluster_id"]]) ** 2).sum()
            assert doc.meta["cluster_distance"] >= 0
            assert doc.meta["cluster_distance"] == pytest.approx(expected, abs=1e-4)

    def test_no_distance_keys_by_default(self, docs: list[Document], spherical_model: str) -> None:
        out = EmeansDocumentClusterer(model_path=spherical_model).run(documents=docs)["documents"]
        assert all(set(d.meta) == {"cluster_id"} for d in out)

    def test_existing_keys_overwritten_and_foreign_kept(self, docs: list[Document], spherical_model: str) -> None:
        doc = Document(
            content="x",
            embedding=docs[0].embedding,
            meta={"cluster_id": 99, "cluster_similarity": -5.0, "keep": "me", "cluster_distance": 3.0},
        )
        out = EmeansDocumentClusterer(model_path=spherical_model, return_distance=True).run(documents=[doc])
        meta = out["documents"][0].meta
        assert meta["cluster_id"] != 99
        assert -1.0 <= meta["cluster_similarity"] <= 1.0
        assert meta["keep"] == "me"
        assert meta["cluster_distance"] == 3.0  # stale opposite-mode key is left untouched


class TestImmutabilityAndOrder:
    def test_inputs_unchanged_outputs_new(self, docs: list[Document], spherical_model: str) -> None:
        before = copy.deepcopy(docs)
        out = EmeansDocumentClusterer(model_path=spherical_model).run(documents=docs)["documents"]
        assert [(d.meta, d.embedding) for d in docs] == [(d.meta, d.embedding) for d in before]
        for src, dst in zip(docs, out, strict=True):
            assert dst is not src
            assert dst.id == src.id

    def test_order_and_passthrough_with_single_warning(
        self, docs: list[Document], spherical_model: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        plain = [Document(content=f"plain {i}") for i in range(3)]
        mixed = [plain[0], docs[0], plain[1], docs[1], docs[2], plain[2]]
        with caplog.at_level(logging.WARNING):
            out = EmeansDocumentClusterer(model_path=spherical_model).run(documents=mixed)["documents"]
        assert [d.id for d in out] == [d.id for d in mixed]
        for i in (0, 2, 5):
            assert out[i] is mixed[i]
            assert "cluster_id" not in out[i].meta
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "3" in warnings[0].getMessage()

    def test_no_warning_when_all_have_embeddings(
        self, docs: list[Document], spherical_model: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            EmeansDocumentClusterer(model_path=spherical_model).run(documents=docs)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestErrors:
    def test_no_model_path(self, docs: list[Document]) -> None:
        with pytest.raises(ValueError, match="model path"):
            EmeansDocumentClusterer().run(documents=docs)

    def test_no_model_path_even_when_empty(self) -> None:
        with pytest.raises(ValueError, match="model path"):
            EmeansDocumentClusterer().run(documents=[])

    def test_missing_file(self, docs: list[Document], tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            EmeansDocumentClusterer(model_path=str(tmp_path / "nope.npz")).run(documents=docs)

    def test_dimension_mismatch(self, spherical_model: str) -> None:
        bad = [Document(content="x", embedding=[0.1] * 8)]
        with pytest.raises(ValueError, match=r"(?s)8.*16|16.*8"):
            EmeansDocumentClusterer(model_path=spherical_model).run(documents=bad)

    def test_nan(self, spherical_model: str) -> None:
        bad = Document(content="x", embedding=[float("nan")] + [0.1] * 15)
        with pytest.raises(ValueError, match=bad.id):
            EmeansDocumentClusterer(model_path=spherical_model).run(documents=[bad])

    def test_spherical_zero_vector(self, spherical_model: str) -> None:
        bad = Document(content="x", embedding=[0.0] * 16)
        with pytest.raises(ValueError, match=bad.id):
            EmeansDocumentClusterer(model_path=spherical_model).run(documents=[bad])

    def test_euclidean_zero_vector_is_assigned(self, euclidean_model: str) -> None:
        out = EmeansDocumentClusterer(model_path=euclidean_model).run(
            documents=[Document(content="x", embedding=[0.0] * 16)]
        )
        assert type(out["documents"][0].meta["cluster_id"]) is int


class TestPathAndCache:
    def test_run_path_wins(self, docs: list[Document], spherical_model: str, tmp_path: Path) -> None:
        comp = EmeansDocumentClusterer(model_path=str(tmp_path / "missing.npz"))
        assert len(comp.run(documents=docs, model_path=spherical_model)["documents"]) == len(docs)

    def test_warm_up_loads_once_and_reloads_on_change(
        self, docs: list[Document], spherical_model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        real = clusterer_module.load_model
        monkeypatch.setattr(clusterer_module, "load_model", lambda p: (calls.append(p), real(p))[1])
        comp = EmeansDocumentClusterer(model_path=spherical_model)
        comp.warm_up()
        comp.run(documents=docs)
        comp.run(documents=docs)
        assert calls == [spherical_model]
        other = _fit(tmp_path / "other.npz", docs, k=3)
        comp.run(documents=docs, model_path=other)
        assert calls == [spherical_model, other]

    def test_rewritten_file_is_reloaded(self, docs: list[Document], spherical_model: str) -> None:
        comp = EmeansDocumentClusterer(model_path=spherical_model)
        out = comp.run(documents=docs)["documents"]
        assert max(d.meta["cluster_id"] for d in out) <= 3
        _fit(Path(spherical_model), docs, k=2, overwrite=True)
        st = os.stat(spherical_model)
        os.utime(spherical_model, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
        out = comp.run(documents=docs)["documents"]
        assert {d.meta["cluster_id"] for d in out} <= {0, 1}

    def test_empty_input_does_not_load_model(self, tmp_path: Path) -> None:
        comp = EmeansDocumentClusterer(model_path=str(tmp_path / "missing.npz"))
        assert comp.run(documents=[]) == {"documents": []}

    def test_no_embeddings_does_not_load_model(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        plain = [Document(content="a"), Document(content="b")]
        comp = EmeansDocumentClusterer(model_path=str(tmp_path / "missing.npz"))
        with caplog.at_level(logging.WARNING):
            out = comp.run(documents=plain)["documents"]
        assert out == plain
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


class TestBatchAndDevice:
    def test_batch_size_does_not_change_results(self, docs: list[Document], spherical_model: str) -> None:
        a = EmeansDocumentClusterer(model_path=spherical_model, batch_size=7, return_distance=True)
        b = EmeansDocumentClusterer(model_path=spherical_model, batch_size=65536, return_distance=True)
        assert [d.meta for d in a.run(documents=docs)["documents"]] == [
            d.meta for d in b.run(documents=docs)["documents"]
        ]

    @pytest.mark.parametrize("bad", [0, True, -1, 1.5])
    def test_bad_batch_size(self, bad: object) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            EmeansDocumentClusterer(model_path="m.npz", batch_size=bad)  # type: ignore[arg-type]

    def test_cuda_without_gpu_backend(self, docs: list[Document], spherical_model: str) -> None:
        with pytest.raises(RuntimeError, match="no GPU backend usable"):
            EmeansDocumentClusterer(model_path=spherical_model, device="cuda").run(documents=docs)

    def test_auto_with_low_threshold_runs_on_cpu(
        self, docs: list[Document], spherical_model: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            out = EmeansDocumentClusterer(model_path=spherical_model, gpu_threshold=1).run(documents=docs)
        assert len(out["documents"]) == len(docs)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_invalid_device(self) -> None:
        with pytest.raises(ValueError, match="device"):
            EmeansDocumentClusterer(device="tpu")  # type: ignore[arg-type]


class TestSerialisation:
    def test_round_trip_all_params(self) -> None:
        comp = EmeansDocumentClusterer(
            model_path="m.npz",
            return_distance=True,
            batch_size=10,
            device="cpu",
            gpu_threshold=5.0,
            fit_if_missing=True,
            fit_params={"k": 3, "seed": 7},
        )
        data = comp.to_dict()
        assert data["init_parameters"] == {
            "model_path": "m.npz",
            "return_distance": True,
            "batch_size": 10,
            "device": "cpu",
            "gpu_threshold": 5.0,
            "fit_if_missing": True,
            "fit_params": {"k": 3, "seed": 7},
        }
        again = EmeansDocumentClusterer.from_dict(data)
        assert again.to_dict() == data

    def test_init_does_not_touch_filesystem(self, tmp_path: Path) -> None:
        path = tmp_path / "never.npz"
        EmeansDocumentClusterer(model_path=str(path))
        assert not path.exists()


class TestFitIfMissing:
    def test_missing_without_flag_raises(self, docs: list[Document], tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            EmeansDocumentClusterer(model_path=str(tmp_path / "m.npz")).run(documents=docs)

    def test_fits_on_missing_and_matches_saved_file(self, docs: list[Document], tmp_path: Path) -> None:
        path = str(tmp_path / "auto.npz")
        comp = EmeansDocumentClusterer(model_path=path, fit_if_missing=True, fit_params={"k": 3})
        out = comp.run(documents=docs)["documents"]
        assert load_model(path).k == 3
        assert all(d.meta["cluster_id"] in range(3) for d in out)
        again = EmeansDocumentClusterer(model_path=path).run(documents=docs)["documents"]
        assert [d.meta["cluster_id"] for d in out] == [d.meta["cluster_id"] for d in again]

    def test_existing_file_is_not_refitted(
        self, docs: list[Document], spherical_model: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = Path(spherical_model).read_bytes()
        runs: list[int] = []
        monkeypatch.setattr(EmeansFitter, "run", lambda _self, **_kw: runs.append(1))
        comp = EmeansDocumentClusterer(model_path=spherical_model, fit_if_missing=True, fit_params={"k": 2})
        comp.warm_up()
        comp.run(documents=docs)
        assert runs == []
        assert Path(spherical_model).read_bytes() == before

    def test_second_run_reuses_file(
        self, docs: list[Document], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        comp = EmeansDocumentClusterer(model_path=str(tmp_path / "a.npz"), fit_if_missing=True, fit_params={"k": 3})
        comp.run(documents=docs)
        runs: list[int] = []
        monkeypatch.setattr(EmeansFitter, "run", lambda _self, **_kw: runs.append(1))
        comp.run(documents=docs)
        assert runs == []

    def test_race_with_other_writer(
        self, docs: list[Document], spherical_model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "raced.npz"

        def losing_run(_self: EmeansFitter, **_kwargs: object) -> None:
            target.write_bytes(Path(spherical_model).read_bytes())
            raise FileExistsError(str(target))

        monkeypatch.setattr(EmeansFitter, "run", losing_run)
        comp = EmeansDocumentClusterer(model_path=str(target), fit_if_missing=True, fit_params={"k": 3})
        out = comp.run(documents=docs)["documents"]
        assert all(d.meta["cluster_id"] in range(4) for d in out)

    @pytest.mark.parametrize(
        "fit_params",
        [
            None,
            {},
            {"k": 0},
            {"k": 2, "target_cluster_size": 5},
            {"k": 2, "model_path": "x"},
            {"k": 2, "overwrite": True},
        ],
    )
    def test_invalid_fit_params(self, fit_params: dict | None) -> None:
        with pytest.raises(ValueError):
            EmeansDocumentClusterer(model_path="m.npz", fit_if_missing=True, fit_params=fit_params)

    def test_euclidean_auto_fit(self, make_docs: MakeDocs, tmp_path: Path) -> None:
        comp = EmeansDocumentClusterer(
            model_path=str(tmp_path / "e.npz"),
            fit_if_missing=True,
            fit_params={"k": 3, "spherical": False},
            return_distance=True,
        )
        out = comp.run(documents=make_docs(60, unit=False))["documents"]
        assert not load_model(str(tmp_path / "e.npz")).spherical
        assert all("cluster_distance" in d.meta for d in out)
