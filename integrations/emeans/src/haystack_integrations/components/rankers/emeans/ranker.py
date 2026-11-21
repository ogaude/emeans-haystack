# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import numpy as np
from haystack import Document, component, default_from_dict, default_to_dict, logging

from haystack_integrations.components.preprocessors.emeans._backend import cpu_assigner
from haystack_integrations.components.preprocessors.emeans._model import ModelCache, load_model
from haystack_integrations.components.preprocessors.emeans._utils import (
    check_positive_int,
    extract_embeddings,
    normalize_rows,
)

logger = logging.getLogger(__name__)


def _cluster_id_of(doc: Document) -> int | None:
    """Return the document's usable integer `cluster_id`, or `None` when absent or not an integer."""
    value = doc.meta.get("cluster_id")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


@component
class EmeansClusterRanker:
    """
    Keeps the best-scored document of each cluster among retrieved documents.

    The ranker never clusters: it reads the `meta["cluster_id"]` written at indexing time by
    `EmeansDocumentClusterer`.

    ### Usage example

    ```python
    from haystack_integrations.components.rankers.emeans import EmeansClusterRanker

    ranker = EmeansClusterRanker(top_k=5)
    diverse = ranker.run(documents=retrieved_documents)["documents"]
    ```
    """

    def __init__(
        self,
        top_k: int | None = 10,
        model_path: str | None = None,
        missing_cluster: Literal["keep", "drop"] = "keep",
    ) -> None:
        """
        Create an EmeansClusterRanker.

        :param top_k: Maximum documents returned. `None` returns one document per cluster present.
        :param model_path: Optional model used to assign documents that have no `cluster_id` but have an embedding.
        :param missing_cluster: What to do with documents that have no `cluster_id` and cannot be assigned:
            `"keep"` treats each as its own cluster, `"drop"` removes them.
        """
        check_positive_int("top_k", top_k, allow_none=True)
        if missing_cluster not in ("keep", "drop"):
            msg = f"missing_cluster must be 'keep' or 'drop', got {missing_cluster!r}"
            raise ValueError(msg)
        self.top_k = top_k
        self.model_path = model_path
        self.missing_cluster = missing_cluster
        self._models = ModelCache()

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with all init parameters.
        """
        return default_to_dict(self, top_k=self.top_k, model_path=self.model_path, missing_cluster=self.missing_cluster)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmeansClusterRanker":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary produced by `to_dict`.
        :returns: The deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], top_k: int | None = None, model_path: str | None = None) -> dict[str, Any]:
        """
        Return the best-scored document of each cluster, best first.

        :param documents: Retrieved documents with `meta["cluster_id"]`.
        :param top_k: Maximum documents returned; replaces the init value.
        :param model_path: Model used to assign documents without `cluster_id`; replaces the init value.
        :returns: A dictionary with the kept input `documents`, unchanged.
        """
        check_positive_int("top_k", top_k, allow_none=True)
        limit = top_k if top_k is not None else self.top_k
        if not documents:
            return {"documents": []}

        def key(i: int) -> tuple[bool, float, int]:
            score = documents[i].score
            return (score is None, -(score if score is not None else 0.0), i)

        n_unscored = sum(1 for doc in documents if doc.score is None)
        if n_unscored:
            logger.warning(
                "{n} documents have no score; they rank after scored documents.",
                n=n_unscored,
            )

        cluster_ids = [_cluster_id_of(doc) for doc in documents]
        n_assigned = 0
        path = model_path if model_path is not None else self.model_path
        if path is not None:
            todo = [i for i, c in enumerate(cluster_ids) if c is None and documents[i].embedding is not None]
            if todo:
                model = self._models.get(path, load_model)
                subset = [documents[i] for i in todo]
                positions, x = extract_embeddings(subset, expected_dim=model.d)
                if model.spherical:
                    x = normalize_rows(x, [subset[p].id for p in positions])
                _, labels = cpu_assigner(model.centroids, spherical=model.spherical).assign(x)
                for p, label in zip(positions, labels, strict=True):
                    cluster_ids[todo[p]] = int(label)
                n_assigned = len(positions)

        groups: dict[tuple[str, int], int] = {}
        n_missing = 0
        for i in range(len(documents)):
            cluster = cluster_ids[i]
            if cluster is None:
                n_missing += 1
                if self.missing_cluster == "drop":
                    continue
                group = ("m", i)
            else:
                group = ("c", cluster)
            if group not in groups or key(i) < key(groups[group]):
                groups[group] = i
        if n_missing or n_assigned:
            kept = n_missing if self.missing_cluster == "keep" else 0
            logger.warning(
                "{n} documents have no usable cluster_id (assigned {assigned}, kept {kept}, dropped {dropped}).",
                n=n_missing + n_assigned,
                assigned=n_assigned,
                kept=kept,
                dropped=n_missing - kept,
            )

        ordered = sorted(groups.values(), key=key)
        if limit is not None:
            ordered = ordered[:limit]
        return {"documents": [documents[i] for i in ordered]}
