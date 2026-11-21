# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Example 1: embed a corpus, fit e-means, assign cluster ids and write to an InMemoryDocumentStore."""

# ruff: noqa: T201 (examples print their results)

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _shared import TOPICS, corpus_documents, index_corpus, output_dir
from haystack.document_stores.in_memory import InMemoryDocumentStore


def main() -> None:
    """Run the example."""
    docs = corpus_documents()  # any list of texts
    store = InMemoryDocumentStore()
    print(f"corpus: {len(docs)} documents, {len(TOPICS)} topics")

    # The Quick Start indexing pipeline (embedder -> fitter + clusterer -> writer) lives in _shared.index_corpus,
    # so that examples 01 and 02 index the corpus the same way (target_cluster_size=25, overwrite=True).
    result = index_corpus(store, str(output_dir() / "clusters.npz"))
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
