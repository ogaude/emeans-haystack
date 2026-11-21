# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import logging

import numpy as np
import pytest
from haystack import Document

from haystack_integrations.components.rankers.emeans import EmeansClusterRanker


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
