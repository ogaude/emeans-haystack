# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Example 3: assign new documents to the clusters of an existing model with EmeansDocumentClusterer."""

# ruff: noqa: T201 (examples print their results)

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _shared import index_corpus, output_dir
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer

NEW_TEXTS = [
    "Wind turbines and solar batteries are reshaping the electricity market.",
    "She simmered the onions slowly before folding in the fresh basil.",
    "The winger scored twice in the cup final on Saturday.",
    "Astronomers detected a faint signal from a pulsar near the galactic centre.",
    "Refactoring the parser made the unit tests much faster.",
    "Water the young lettuce every morning and add some compost.",
    "The pianist performed a sonata at the evening recital.",
    "Central banks held interest rates steady to curb inflation.",
]


def main() -> None:
    """Run the example."""
    store = InMemoryDocumentStore()
    model_path = str(output_dir() / "clusters.npz")
    index_corpus(store, model_path)  # creates the model and indexes the corpus

    embedder = SentenceTransformersDocumentEmbedder()
    embedder.warm_up()
    new_docs = embedder.run(documents=[Document(content=text) for text in NEW_TEXTS])["documents"]

    # Assign new documents with the saved model: the cluster ids are the same ones the indexed corpus uses.
    clusterer = EmeansDocumentClusterer(model_path=model_path, return_distance=True)
    out = clusterer.run(documents=new_docs)["documents"]

    topics_by_cluster: dict[int, Counter[str]] = {}
    for doc in store.filter_documents():
        topics_by_cluster.setdefault(doc.meta["cluster_id"], Counter())[doc.meta["topic"]] += 1

    for doc in out:
        cluster_id = doc.meta["cluster_id"]
        topic = topics_by_cluster[cluster_id].most_common(1)[0][0] if cluster_id in topics_by_cluster else "?"
        print(f"cluster={cluster_id} similarity={doc.meta['cluster_similarity']:.3f} topic={topic}: {doc.content}")


if __name__ == "__main__":
    main()
