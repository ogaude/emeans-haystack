# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import replace
from typing import Any, Literal

import numpy as np
from haystack import Document, component, default_from_dict, default_to_dict, logging

from haystack_integrations.components.preprocessors.emeans._backend import DEVICES, assigner_for, select_backend
from haystack_integrations.components.preprocessors.emeans._model import EmeansModel, ModelCache, load_model
from haystack_integrations.components.preprocessors.emeans._utils import (
    check_bool,
    check_positive_int,
    check_threshold,
    extract_embeddings,
    normalize_rows,
)
from haystack_integrations.components.preprocessors.emeans.fitter import EmeansFitter

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
        check_bool("return_distance", return_distance)
        check_bool("fit_if_missing", fit_if_missing)
        check_threshold("gpu_threshold", gpu_threshold)
        if device not in DEVICES:
            msg = f"device must be 'auto', 'cpu' or 'cuda', got {device!r}"
            raise ValueError(msg)
        if fit_if_missing:
            self._validate_fit_params(fit_params)
        self.model_path = model_path
        self.return_distance = return_distance
        self.batch_size = int(batch_size)
        self.device = device
        self.gpu_threshold = gpu_threshold
        self.fit_if_missing = fit_if_missing
        self.fit_params = fit_params
        self._models = ModelCache()

    @staticmethod
    def _validate_fit_params(fit_params: dict[str, Any] | None) -> None:
        if not isinstance(fit_params, dict) or not fit_params:
            msg = f"fit_params must be a non-empty dict when fit_if_missing=True, got {fit_params!r}"
            raise ValueError(msg)
        params = fit_params
        forbidden = sorted({"model_path", "overwrite"} & params.keys())
        if forbidden:
            msg = f"fit_params must not contain {forbidden}: the model path is the clusterer's and it never overwrites"
            raise ValueError(msg)
        try:
            # The fitter constructor only validates; it does no file access.
            EmeansFitter(model_path="<fit_if_missing>", **params)
        except TypeError as e:
            msg = f"fit_params is not a valid fitter configuration: {e}"
            raise ValueError(msg) from e

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
        """Load the model for the init `model_path`, if one is set and available."""
        if self.model_path is None:
            return
        if os.path.exists(self.model_path) or not self.fit_if_missing:
            self._ensure_model(self.model_path)

    def _ensure_model(self, path: str, documents: list[Document] | None = None) -> tuple[EmeansModel, bool]:
        """Load the model, auto-fitting it first when allowed; also return whether the fit chose the backend."""
        warned = False
        if documents is not None and self.fit_if_missing and not os.path.exists(path):
            warned = self._auto_fit(path, documents)
        return self._models.get(path, load_model), warned

    def _auto_fit(self, path: str, documents: list[Document]) -> bool:
        """
        Fit and save a model on `documents`.

        :returns: True when the fit ran with this component's device settings, so any backend fallback warning
            for this run has already been logged.
        """
        logger.info(
            "No model at {path}: fitting one on this run's documents; the clusters depend on the first batch seen",
            path=path,
        )
        params = {"device": self.device, "gpu_threshold": self.gpu_threshold, **(self.fit_params or {})}
        fitter = EmeansFitter(model_path=path, overwrite=False, **params)
        try:
            fitter.run(documents=documents)
            return params["device"] == self.device and params["gpu_threshold"] == self.gpu_threshold
        except FileExistsError:
            # Another writer created the file first; use theirs.
            logger.info("Model {path} was created concurrently; using it", path=path)
            return False

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], model_path: str | None = None) -> dict[str, Any]:
        """
        Write `meta["cluster_id"]` on every document that has an embedding.

        :param documents: Documents to cluster. They are not modified; copies are returned in the same order.
            Documents without embedding are returned unchanged.
        :param model_path: Model to load; replaces the init value.
        :returns: A dictionary with the `documents`, in input order.
        :raises ValueError: If no model path is available, or on invalid embeddings.
        :raises FileNotFoundError: If the model file does not exist and `fit_if_missing` is not set.
        :raises RuntimeError: If `device="cuda"` and no GPU backend is usable.
        """
        path = model_path if model_path is not None else self.model_path
        if path is None:
            msg = "No model path: set model_path at init or pass it to run()"
            raise ValueError(msg)
        if not documents:
            return {"documents": []}

        positions, x = extract_embeddings(documents)
        n_skipped = len(documents) - len(positions)
        if n_skipped:
            logger.warning("{n} documents without embedding were returned unchanged", n=n_skipped)
        if not positions:
            return {"documents": list(documents)}

        if self.device == "cuda":
            # Fail before _ensure_model, which may auto-fit and write a model file; k does not matter for "cuda".
            select_backend(device=self.device, n=len(positions), k=1, gpu_threshold=self.gpu_threshold)
        model, fit_warned = self._ensure_model(path, documents)
        if x.shape[1] != model.d:
            msg = f"Embedding dimension mismatch: documents have dimension {x.shape[1]}, but the model has {model.d}"
            raise ValueError(msg)
        # An auto-fit in this run already logged any fallback warning; log it once per run (D-03).
        backend = select_backend(
            device=self.device,
            n=len(positions),
            k=model.k,
            gpu_threshold=self.gpu_threshold,
            warn=not fit_warned,
        )
        if model.spherical:
            x = normalize_rows(x, [documents[p].id for p in positions])

        assigner = assigner_for(backend, model.centroids, spherical=model.spherical)
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
                        meta["cluster_distance"] = float(max(score, 0.0))
                metas[positions[lo + offset]] = meta

        out = [replace(doc, meta=metas[i]) if i in metas else doc for i, doc in enumerate(documents)]
        return {"documents": out}
