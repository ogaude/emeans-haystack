# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any, Literal

import numpy as np
from haystack import Document, component, default_from_dict, default_to_dict, logging

from haystack_integrations.components.preprocessors.emeans._backend import cpu_assigner
from haystack_integrations.components.preprocessors.emeans._model import EmeansModel, load_model
from haystack_integrations.components.preprocessors.emeans._utils import (
    check_positive_int,
    extract_embeddings,
    normalize_rows,
)

logger = logging.getLogger(__name__)


@component
class EmeansDocumentClusterer:
    """
    Assigns each document to the nearest centroid of a saved e-means model.

    Every output document with an embedding gets `meta["cluster_id"]`, an `int` in `[0, k)`.

    ### Usage example

    ```python
    from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer

    clusterer = EmeansDocumentClusterer(model_path="clusters.npz")
    clustered = clusterer.run(documents=documents_with_embeddings)["documents"]
    print(clustered[0].meta["cluster_id"])
    ```
    """

    def __init__(
        self,
        model_path: str | None = None,
        return_distance: bool = False,
        batch_size: int = 65536,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        gpu_threshold: float = 1e10,
        fit_if_missing: bool = False,
        fit_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Create an EmeansDocumentClusterer.

        :param model_path: Model to load. May instead (or also) come from the `model_path` run input, which wins.
        :param return_distance: Also write the distance (Euclidean model) or cosine similarity (spherical model)
            to the assigned centroid.
        :param batch_size: Documents assigned per chunk. Affects memory only, not results.
        :param device: `"cpu"`, `"cuda"` or `"auto"`.
        :param gpu_threshold: With `device="auto"`, a GPU is used only when `n * k` is at least this value.
        :param fit_if_missing: Fit a model on this run's documents when no file exists at the model path.
        :param fit_params: Fitter parameters used when `fit_if_missing` is set, except `model_path`.
        """
        check_positive_int("batch_size", batch_size)
        self.model_path = model_path
        self.return_distance = return_distance
        self.batch_size = batch_size
        self.device = device
        self.gpu_threshold = gpu_threshold
        self.fit_if_missing = fit_if_missing
        self.fit_params = fit_params
        self._model: EmeansModel | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with all init parameters.
        """
        return default_to_dict(
            self,
            model_path=self.model_path,
            return_distance=self.return_distance,
            batch_size=self.batch_size,
            device=self.device,
            gpu_threshold=self.gpu_threshold,
            fit_if_missing=self.fit_if_missing,
            fit_params=self.fit_params,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmeansDocumentClusterer":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary produced by `to_dict`.
        :returns: The deserialized component.
        """
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """Load the model for the init `model_path`, if one is set."""
        if self.model_path is not None:
            self._load(self.model_path)

    def _load(self, path: str) -> EmeansModel:
        if self._model is None or self._model.path != path:
            self._model = load_model(path)
        return self._model

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], model_path: str | None = None) -> dict[str, Any]:
        """
        Write `meta["cluster_id"]` on every document that has an embedding.

        :param documents: Documents to cluster. They are not modified; copies are returned in the same order.
        :param model_path: Model to load; replaces the init value.
        :returns: A dictionary with the `documents`, in input order.
        :raises ValueError: If no model path is available, or on invalid embeddings.
        :raises FileNotFoundError: If the model file does not exist.
        """
        path = model_path if model_path is not None else self.model_path
        if path is None:
            msg = "No model path: set model_path at init or pass it to run()"
            raise ValueError(msg)
        model = self._load(path)

        positions, x = extract_embeddings(documents, expected_dim=model.d)
        n_skipped = len(documents) - len(positions)
        if n_skipped:
            logger.warning("{n} documents without embedding were returned unchanged", n=n_skipped)
        if model.spherical and len(positions):
            x = normalize_rows(x, [documents[p].id for p in positions])

        assigner = cpu_assigner(model.centroids, spherical=model.spherical)
        metas: dict[int, dict[str, Any]] = {}
        for lo in range(0, len(positions), self.batch_size):
            scores, labels = assigner.assign(x[lo : lo + self.batch_size])
            for offset, (score, label) in enumerate(zip(scores, labels, strict=True)):
                doc = documents[positions[lo + offset]]
                meta = {**doc.meta, "cluster_id": int(label)}
                if self.return_distance:
                    if model.spherical:
                        meta["cluster_similarity"] = float(np.clip(score, -1.0, 1.0))
                    else:
                        meta["cluster_distance"] = float(score)
                metas[positions[lo + offset]] = meta

        out = [replace(doc, meta=metas[i]) if i in metas else doc for i, doc in enumerate(documents)]
        return {"documents": out}
