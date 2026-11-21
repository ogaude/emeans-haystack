# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Example 2: retrieve a diverse top 5, the best document from each cluster, with EmeansClusterRanker."""

# ruff: noqa: T201 (examples print their results)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _shared import index_corpus, output_dir
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.rankers.emeans import EmeansClusterRanker


def main() -> None:
    """Run the example."""
    store = InMemoryDocumentStore()
    index_corpus(store, str(output_dir() / "clusters.npz"))

    query = Pipeline()
    query.add_component("embedder", SentenceTransformersTextEmbedder())
    query.add_component("retriever", InMemoryEmbeddingRetriever(document_store=store, top_k=50))
    query.add_component("ranker", EmeansClusterRanker(top_k=5))
    query.connect("embedder.embedding", "retriever.query_embedding")
    query.connect("retriever.documents", "ranker.documents")

    text = "how do solar panels work?"
    out = query.run({"embedder": {"text": text}}, include_outputs_from={"retriever"})
    docs = out["ranker"]["documents"]  # up to 5 documents, each the best match from a different cluster

    print(f"query: {text}\n\ndiverse top {len(docs)} (one per cluster):")
    for doc in docs:
        print(f"  {doc.score:.3f} cluster={doc.meta['cluster_id']} {doc.content}")
    print("\nplain retriever top 5 (for contrast):")
    for doc in out["retriever"]["documents"][:5]:
        print(f"  {doc.score:.3f} cluster={doc.meta['cluster_id']} {doc.content}")


if __name__ == "__main__":
    main()
