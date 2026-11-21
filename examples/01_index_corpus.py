# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Example 1: embed a corpus, fit e-means, assign cluster ids and write to an InMemoryDocumentStore."""

# ruff: noqa: T201 (examples print their results)

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _shared import TOPICS, corpus_documents, output_dir
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
)
from haystack_integrations.components.preprocessors.emeans import (
    EmeansDocumentClusterer,
    EmeansFitter,
)


def main() -> None:
    """Run the example."""
    docs = corpus_documents()  # any list of texts
    store = InMemoryDocumentStore()
    print(f"corpus: {len(docs)} documents, {len(TOPICS)} topics")

    indexing = Pipeline()
    indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
    # overwrite=True: without it a re-run raises FileExistsError, because refitting changes every cluster_id.
    indexing.add_component(
        "fitter",
        EmeansFitter(target_cluster_size=25, model_path=str(output_dir() / "clusters.npz"), overwrite=True),
    )
    indexing.add_component("clusterer", EmeansDocumentClusterer())
    indexing.add_component("writer", DocumentWriter(document_store=store))
    indexing.connect("embedder.documents", "fitter.documents")
    indexing.connect("embedder.documents", "clusterer.documents")
    indexing.connect("fitter.model_path", "clusterer.model_path")  # also makes the clusterer run after the fitter
    indexing.connect("clusterer.documents", "writer.documents")

    result = indexing.run({"embedder": {"documents": docs}}, include_outputs_from={"fitter"})
    print(result["fitter"]["stats"]["k"], result["fitter"]["stats"]["backend"])
    stored = store.filter_documents()
    print(stored[0].meta["cluster_id"])

    topics_per_cluster: dict[int, Counter[str]] = defaultdict(Counter)
    for doc in stored:
        topics_per_cluster[doc.meta["cluster_id"]][doc.meta["topic"]] += 1
    for cluster_id in sorted(topics_per_cluster):
        print(f"cluster {cluster_id}: {dict(topics_per_cluster[cluster_id].most_common())}")


if __name__ == "__main__":
    main()
