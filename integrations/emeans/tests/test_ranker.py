# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from haystack import Document

from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter
from haystack_integrations.components.preprocessors.emeans._model import EmeansModel, load_model
from haystack_integrations.components.rankers.emeans import EmeansClusterRanker
from haystack_integrations.components.rankers.emeans import ranker as ranker_module


def _doc(name: str, cluster: object, score: float | None) -> Document:
    meta = {} if cluster is None else {"cluster_id": cluster}
    return Document(id=name, content=name, meta=meta, score=score)


def _ids(result: dict) -> list[str]:
    return [d.id for d in result["documents"]]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.fixture
def basic_docs() -> list[Document]:
    return [_doc("A", 0, 0.9), _doc("B", 0, 0.95), _doc("C", 1, 0.5), _doc("D", 2, 0.7)]


def test_best_per_cluster(basic_docs: list[Document]) -> None:
    assert _ids(EmeansClusterRanker(top_k=10).run(basic_docs)) == ["B", "D", "C"]


def test_truncation(basic_docs: list[Document]) -> None:
    assert _ids(EmeansClusterRanker(top_k=10).run(basic_docs, top_k=2)) == ["B", "D"]


def test_fewer_clusters_than_top_k() -> None:
    docs = [_doc(f"d{i}", 0, 0.1 * i) for i in range(5)]
    assert len(EmeansClusterRanker().run(docs, top_k=3)["documents"]) == 1


def test_top_k_none_returns_one_per_cluster(basic_docs: list[Document]) -> None:
    assert _ids(EmeansClusterRanker(top_k=None).run(basic_docs)) == ["B", "D", "C"]


def test_run_top_k_none_uses_init_value(basic_docs: list[Document]) -> None:
    assert _ids(EmeansClusterRanker(top_k=1).run(basic_docs, top_k=None)) == ["B"]


def test_ties_keep_input_order() -> None:
    docs = [_doc("x", 0, 0.5), _doc("y", 1, 0.5), _doc("z", 0, 0.5)]
    assert _ids(EmeansClusterRanker().run(docs)) == ["x", "y"]


def test_score_none_ranks_last_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    docs = [_doc("X", 0, None), _doc("Y", 1, 0.1), _doc("Z", 2, None)]
    with caplog.at_level(logging.WARNING):
        result = EmeansClusterRanker().run(docs)
    assert _ids(result) == ["Y", "X", "Z"]
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "2 documents have no score" in warnings[0].getMessage()


def test_scored_beats_none_within_cluster() -> None:
    docs = [_doc("N", 0, None), _doc("S", 0, 0.01)]
    assert _ids(EmeansClusterRanker().run(docs)) == ["S"]


def test_no_warning_when_all_scored(basic_docs: list[Document], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        EmeansClusterRanker().run(basic_docs)
    assert _warnings(caplog) == []


def test_returns_input_objects_unchanged(basic_docs: list[Document]) -> None:
    metas = copy.deepcopy([d.meta for d in basic_docs])
    scores = [d.score for d in basic_docs]
    result = EmeansClusterRanker().run(basic_docs)["documents"]
    assert all(any(r is d for d in basic_docs) for r in result)
    assert result is not basic_docs
    assert [d.meta for d in basic_docs] == metas
    assert [d.score for d in basic_docs] == scores


@pytest.mark.parametrize("policy", ["keep", "drop"])
def test_missing_cluster_one_warning(policy: str, caplog: pytest.LogCaptureFixture) -> None:
    docs = [_doc("a", None, 0.9), _doc("b", None, 0.8), _doc("c", 1, 0.1)]
    with caplog.at_level(logging.WARNING):
        result = EmeansClusterRanker(missing_cluster=policy).run(docs)  # type: ignore[arg-type]
    expected = ["a", "b", "c"] if policy == "keep" else ["c"]
    assert _ids(result) == expected
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "2 documents have no usable cluster_id" in warnings[0].getMessage()


@pytest.mark.parametrize("value", [3, np.int64(3), 3.0, np.int32(3)])
def test_accepted_cluster_id_types(value: object) -> None:
    docs = [_doc("a", 3, 0.5), _doc("b", value, 0.9)]
    assert _ids(EmeansClusterRanker().run(docs)) == ["b"]


@pytest.mark.parametrize("value", [True, "3", 3.5, None])
def test_unusable_cluster_id_is_missing(value: object) -> None:
    docs = [_doc("a", value, 0.5), _doc("b", value, 0.9)]
    assert _ids(EmeansClusterRanker(missing_cluster="keep").run(docs)) == ["b", "a"]
    assert EmeansClusterRanker(missing_cluster="drop").run(docs)["documents"] == []


def test_empty_input_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert EmeansClusterRanker().run([]) == {"documents": []}
    assert _warnings(caplog) == []


@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_top_k_validation(bad: object, basic_docs: list[Document]) -> None:
    with pytest.raises(ValueError, match="top_k"):
        EmeansClusterRanker(top_k=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k"):
        EmeansClusterRanker().run(basic_docs, top_k=bad)  # type: ignore[arg-type]


def test_missing_cluster_literal_validation() -> None:
    with pytest.raises(ValueError, match="missing_cluster"):
        EmeansClusterRanker(missing_cluster="other")  # type: ignore[arg-type]


def test_serialisation_round_trip() -> None:
    ranker = EmeansClusterRanker(top_k=None, model_path="m.npz", missing_cluster="drop")
    restored = EmeansClusterRanker.from_dict(ranker.to_dict())
    assert (restored.top_k, restored.model_path, restored.missing_cluster) == (None, "m.npz", "drop")


# On-the-fly assignment with a model


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    ranker_module._MODEL_CACHE.clear()


@pytest.fixture
def fitted(tmp_path: Path, make_docs: Callable[..., list[Document]]) -> tuple[str, list[Document], EmeansModel]:
    docs = make_docs(60, n_clusters=3)
    path = str(tmp_path / "m.npz")
    EmeansFitter(k=3, model_path=path).run(docs)
    model = load_model(path)
    return path, docs, model


def _nearest(model: EmeansModel, doc: Document) -> int:
    x = np.asarray(doc.embedding, dtype=np.float64)
    if model.spherical:
        x = x / np.linalg.norm(x)
    return int(np.argmax(model.centroids @ x))


def test_assigns_missing_cluster_with_model(fitted: tuple[str, list[Document], EmeansModel]) -> None:
    path, docs, model = fitted
    clustered = EmeansDocumentClusterer(model_path=path).run(copy.deepcopy(docs))["documents"]
    for i, doc in enumerate(clustered):
        doc.score = 0.1 + i * 0.001
    stripped = [d for d in clustered if _nearest(model, d) == _nearest(model, clustered[0])][:2]
    for d in stripped:
        d.meta.pop("cluster_id")
        d.score = 5.0 + (d is stripped[0])
    result = EmeansClusterRanker(top_k=None, model_path=path).run(clustered)["documents"]
    assert any(r is stripped[0] for r in result)
    assert all(r is not stripped[1] for r in result)
    for d in stripped:
        assert "cluster_id" not in d.meta
    assert len({id(r) for r in result}) == len(result) == 3


def test_assignment_matches_brute_force(fitted: tuple[str, list[Document], EmeansModel]) -> None:
    path, docs, model = fitted
    for i, doc in enumerate(docs):
        doc.score = float(i)
    result = EmeansClusterRanker(top_k=None, model_path=path).run(docs)["documents"]
    expected: dict[int, Document] = {}
    for doc in docs:
        expected[_nearest(model, doc)] = doc  # increasing score: last wins
    assert {id(d) for d in result} == {id(d) for d in expected.values()}


def test_no_embedding_falls_back_to_missing_cluster(fitted: tuple[str, list[Document], EmeansModel]) -> None:
    path, docs, _ = fitted
    bare = [_doc("p", None, 0.5), _doc("q", None, 0.4), docs[0]]
    docs[0].score = 0.1
    assert _ids(EmeansClusterRanker(model_path=path).run(bare)) == ["p", "q", docs[0].id]
    assert _ids(EmeansClusterRanker(model_path=path, missing_cluster="drop").run(bare)) == [docs[0].id]


def test_single_warning_covers_assigned_and_kept(
    fitted: tuple[str, list[Document], EmeansModel], caplog: pytest.LogCaptureFixture
) -> None:
    path, docs, _ = fitted
    docs = [*docs[:3], _doc("bare", None, 0.2)]
    for d in docs[:3]:
        d.score = 0.3
    with caplog.at_level(logging.WARNING):
        EmeansClusterRanker(model_path=path).run(docs)
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "assigned 3" in message
    assert "kept 1" in message


def test_run_model_path_overrides_init(fitted: tuple[str, list[Document], EmeansModel]) -> None:
    path, docs, _ = fitted
    docs[0].score = 0.5
    result = EmeansClusterRanker(model_path="/nonexistent/m.npz").run(docs[:1], model_path=path)
    assert result["documents"][0] is docs[0]


def test_model_not_loaded_when_all_clustered(basic_docs: list[Document]) -> None:
    result = EmeansClusterRanker(model_path="/nonexistent/m.npz").run(basic_docs)
    assert _ids(result) == ["B", "D", "C"]


def test_model_not_loaded_when_nothing_has_embedding() -> None:
    docs = [_doc("a", None, 0.2)]
    result = EmeansClusterRanker(model_path="/nonexistent/m.npz").run(docs)
    assert _ids(result) == ["a"]


def test_missing_model_file_raises(make_docs: Callable[..., list[Document]], tmp_path: Path) -> None:
    docs = make_docs(3)
    with pytest.raises(FileNotFoundError):
        EmeansClusterRanker(model_path=str(tmp_path / "absent.npz")).run(docs)


def test_dimension_mismatch_names_both_dimensions(
    fitted: tuple[str, list[Document], EmeansModel], make_docs: Callable[..., list[Document]]
) -> None:
    path, _, model = fitted
    docs = make_docs(2, d=8)
    with pytest.raises(ValueError, match=rf"(?=.*\b8\b)(?=.*\b{model.d}\b)"):
        EmeansClusterRanker(model_path=path).run(docs)


def test_nan_embedding_names_document(fitted: tuple[str, list[Document], EmeansModel]) -> None:
    path, _, model = fitted
    emb = [0.1] * model.d
    emb[0] = float("nan")
    doc = Document(id="nandoc", content="x", embedding=emb)
    with pytest.raises(ValueError, match="nandoc"):
        EmeansClusterRanker(model_path=path).run([doc])


def test_zero_embedding_with_spherical_model(fitted: tuple[str, list[Document], EmeansModel]) -> None:
    path, _, model = fitted
    assert model.spherical
    doc = Document(id="zerodoc", content="x", embedding=[0.0] * model.d)
    with pytest.raises(ValueError, match="zerodoc"):
        EmeansClusterRanker(model_path=path).run([doc])


def test_unnormalised_embedding_same_cluster(fitted: tuple[str, list[Document], EmeansModel]) -> None:
    path, docs, _ = fitted
    base = docs[5]
    scaled = Document(id="scaled", content="s", embedding=[v * 5 for v in base.embedding or []], score=0.9)
    original = list(scaled.embedding or [])
    base.score = 0.1
    result = EmeansClusterRanker(model_path=path, top_k=None).run([base, scaled])["documents"]
    assert [d.id for d in result] == ["scaled"]
    assert scaled.embedding == original


def test_model_cached_between_runs(
    fitted: tuple[str, list[Document], EmeansModel], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path, docs, _ = fitted
    calls: list[str] = []
    real = ranker_module.load_model

    def spy(p: str) -> EmeansModel:
        calls.append(p)
        return real(p)

    monkeypatch.setattr(ranker_module, "load_model", spy)
    ranker = EmeansClusterRanker(model_path=path)
    ranker.run(docs[:2])
    ranker.run(docs[:2])
    assert len(calls) == 1

    other = str(tmp_path / "other.npz")
    shutil.copy(path, other)
    ranker.run(docs[:2], model_path=other)
    assert len(calls) == 2

    shutil.copy(path, other)
    stat = os.stat(other)
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    ranker.run(docs[:2], model_path=other)
    assert len(calls) == 3
