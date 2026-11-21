# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Quick Start run with the real SentenceTransformers embedder (downloads a model; run locally)."""

from pathlib import Path

import pytest
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.core.errors import PipelineRuntimeError
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter

pytestmark = pytest.mark.integration

embedders = pytest.importorskip("haystack_integrations.components.embedders.sentence_transformers")

_TOPICS = {
    "cooking": ["Simmer the {} sauce slowly.", "Season the {} soup with salt.", "Bake the {} bread until golden."],
    "football": ["The {} striker scored a goal.", "The {} defender blocked the shot.", "The {} coach changed tactics."],
    "astronomy": ["The {} galaxy is far away.", "A {} star exploded long ago.", "The {} telescope saw a new moon."],
}
_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
          "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
          "eighteenth", "nineteenth", "twentieth"]  # fmt: skip


def _docs() -> list[Document]:
    return [Document(content=t.format(w)) for ts in _TOPICS.values() for t in ts for w in _WORDS[:7]][:63]


def _build_pipeline(store: InMemoryDocumentStore, model_path: str, *, overwrite: bool = False) -> Pipeline:
    indexing = Pipeline()
    indexing.add_component("embedder", embedders.SentenceTransformersDocumentEmbedder())
    indexing.add_component("fitter", EmeansFitter(target_cluster_size=20, model_path=model_path, overwrite=overwrite))
    indexing.add_component("clusterer", EmeansDocumentClusterer())
    indexing.add_component("writer", DocumentWriter(document_store=store))
    indexing.connect("embedder.documents", "fitter.documents")
    indexing.connect("embedder.documents", "clusterer.documents")
    indexing.connect("fitter.model_path", "clusterer.model_path")
    indexing.connect("clusterer.documents", "writer.documents")
    return indexing


def test_quickstart_real_embedder(tmp_path: Path) -> None:
    model_path = str(tmp_path / "clusters.npz")
    store = InMemoryDocumentStore()
    pipeline = _build_pipeline(store, model_path)

    result = pipeline.run({"embedder": {"documents": _docs()}}, include_outputs_from={"fitter"})

    assert result["fitter"]["stats"]["backend"] == "cpu"
    stored = store.filter_documents()
    assert len(stored) == len(_docs())
    assert all(type(doc.meta["cluster_id"]) is int for doc in stored)

    with pytest.raises(PipelineRuntimeError) as excinfo:
        pipeline.run({"embedder": {"documents": _docs()}})
    assert isinstance(excinfo.value.__cause__, FileExistsError)
