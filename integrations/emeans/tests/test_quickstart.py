# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Offline run of the Quick Start wiring, with a mock embedder in place of the real one."""

from dataclasses import replace
from pathlib import Path

import pytest
from haystack import Document, Pipeline
from haystack.components.embedders import MockDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.core.errors import PipelineRuntimeError
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter
from haystack_integrations.components.rankers.emeans import EmeansClusterRanker

N_DOCS = 200
K = 4


def _build_pipeline(store: InMemoryDocumentStore, model_path: str, *, overwrite: bool = False) -> Pipeline:
    indexing = Pipeline()
    indexing.add_component("embedder", MockDocumentEmbedder(dimension=16))
    indexing.add_component("fitter", EmeansFitter(target_cluster_size=50, model_path=model_path, overwrite=overwrite))
    indexing.add_component("clusterer", EmeansDocumentClusterer())
    indexing.add_component("writer", DocumentWriter(document_store=store))
    indexing.connect("embedder.documents", "fitter.documents")
    indexing.connect("embedder.documents", "clusterer.documents")
    indexing.connect("fitter.model_path", "clusterer.model_path")
    indexing.connect("clusterer.documents", "writer.documents")
    return indexing


def _inputs() -> dict:
    return {"embedder": {"documents": [Document(content=f"text number {i}") for i in range(N_DOCS)]}}


def test_quickstart_pipeline(tmp_path: Path) -> None:
    model_path = str(tmp_path / "clusters.npz")
    store = InMemoryDocumentStore()
    result = _build_pipeline(store, model_path).run(_inputs(), include_outputs_from={"fitter"})

    stats = result["fitter"]["stats"]
    assert stats["k"] == K
    assert stats["backend"] == "cpu"
    assert result["fitter"]["model_path"] == model_path
    assert Path(model_path).exists()

    stored = store.filter_documents()
    assert len(stored) == N_DOCS
    for doc in stored:
        cluster_id = doc.meta["cluster_id"]
        assert type(cluster_id) is int
        assert cluster_id in range(K)


def test_quickstart_rerun_without_overwrite(tmp_path: Path) -> None:
    model_path = tmp_path / "clusters.npz"
    pipeline = _build_pipeline(InMemoryDocumentStore(), str(model_path))
    pipeline.run(_inputs())
    before = model_path.read_bytes()

    with pytest.raises(PipelineRuntimeError) as excinfo:
        pipeline.run(_inputs())

    assert isinstance(excinfo.value.__cause__, FileExistsError)
    assert model_path.read_bytes() == before


def test_quickstart_rerun_with_overwrite(tmp_path: Path) -> None:
    model_path = str(tmp_path / "clusters.npz")
    _build_pipeline(InMemoryDocumentStore(), model_path).run(_inputs())

    store = InMemoryDocumentStore()
    _build_pipeline(store, model_path, overwrite=True).run(_inputs())

    assert len(store.filter_documents()) == N_DOCS


def test_ranker_one_per_cluster(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    _build_pipeline(store, str(tmp_path / "clusters.npz")).run(_inputs())
    scored = [replace(doc, score=float(i)) for i, doc in enumerate(store.filter_documents())]

    ranked = EmeansClusterRanker(top_k=3).run(documents=scored)["documents"]

    assert 1 <= len(ranked) <= 3
    cluster_ids = [doc.meta["cluster_id"] for doc in ranked]
    assert len(set(cluster_ids)) == len(cluster_ids)
    assert ranked[0].score == max(doc.score for doc in scored)
