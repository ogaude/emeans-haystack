# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

from haystack import Document, component, default_from_dict, default_to_dict, logging

from haystack_integrations.components.preprocessors.emeans._backend import fit_model, select_backend
from haystack_integrations.components.preprocessors.emeans._model import save_model
from haystack_integrations.components.preprocessors.emeans._utils import (
    check_positive_int,
    extract_embeddings,
    normalize_rows,
    resolve_k,
)

logger = logging.getLogger(__name__)


def _package_version() -> str:
    try:
        return version("emeans-haystack")
    except PackageNotFoundError:
        return "unknown"


@component
class EmeansFitter:
    """
    Trains e-means on the embeddings of the input documents and writes a model file.

    The model file is the only link between fitting and assignment: connect `model_path` to an
    `EmeansDocumentClusterer` to give documents stable `cluster_id`s.

    ### Usage example

    ```python
    from haystack_integrations.components.preprocessors.emeans import EmeansFitter

    fitter = EmeansFitter(model_path="clusters.npz", target_cluster_size=50)
    result = fitter.run(documents=documents_with_embeddings)
    print(result["stats"]["k"])
    ```
    """

    def __init__(
        self,
        model_path: str,
        k: int | None = None,
        target_cluster_size: int | None = None,
        spherical: bool = True,
        nepochs: float = 5.0,
        precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto",
        device: Literal["auto", "cpu", "cuda"] = "auto",
        gpu_threshold: float = 1e10,
        seed: int = 1234,
        deterministic: bool = False,
        max_points_per_centroid: int | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Create an EmeansFitter.

        :param model_path: Where the `.npz` model file is written. Parent directories must exist.
        :param k: Number of clusters. Exactly one of `k` and `target_cluster_size` must be set.
        :param target_cluster_size: Average documents per cluster;
            `k = max(1, min(round(n / target_cluster_size), n))` with `n` the number of usable documents.
        :param spherical: If `True`, cluster by cosine similarity: embeddings are L2-normalised (on a copy).
            If `False`, cluster by Euclidean distance.
        :param nepochs: Passes over the data; fractional values are allowed.
        :param precision: Number format of the nearest-centroid search on GPU. On CPU everything is FP32.
        :param device: `"cpu"`, `"cuda"` or `"auto"`.
        :param gpu_threshold: With `device="auto"`, a GPU is used only when `n * k` is at least this value.
        :param seed: Seeds starting centroids, shuffling and cluster revival.
        :param deterministic: On GPU, request deterministic algorithms during training. No effect on CPU.
        :param max_points_per_centroid: If set and `n > max_points_per_centroid * k`, train on a random subset.
        :param overwrite: If `False`, an existing file at `model_path` raises `FileExistsError`.
        """
        check_positive_int("k", k, allow_none=True)
        check_positive_int("target_cluster_size", target_cluster_size, allow_none=True)
        self._check_exactly_one(k, target_cluster_size)
        self.model_path = model_path
        self.k = k
        self.target_cluster_size = target_cluster_size
        self.spherical = spherical
        self.nepochs = nepochs
        self.precision = precision
        self.device = device
        self.gpu_threshold = gpu_threshold
        self.seed = seed
        self.deterministic = deterministic
        self.max_points_per_centroid = max_points_per_centroid
        self.overwrite = overwrite

    @staticmethod
    def _check_exactly_one(k: int | None, target_cluster_size: int | None) -> None:
        if (k is None) == (target_cluster_size is None):
            msg = "Exactly one of k and target_cluster_size must be set"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with all init parameters.
        """
        return default_to_dict(
            self,
            model_path=self.model_path,
            k=self.k,
            target_cluster_size=self.target_cluster_size,
            spherical=self.spherical,
            nepochs=self.nepochs,
            precision=self.precision,
            device=self.device,
            gpu_threshold=self.gpu_threshold,
            seed=self.seed,
            deterministic=self.deterministic,
            max_points_per_centroid=self.max_points_per_centroid,
            overwrite=self.overwrite,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmeansFitter":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary produced by `to_dict`.
        :returns: The deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(model_path=str, stats=dict)
    def run(
        self,
        documents: list[Document],
        k: int | None = None,
        target_cluster_size: int | None = None,
        model_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Fit e-means on the embeddings of `documents` and write the model file.

        :param documents: Documents with embeddings. Documents without an embedding are skipped.
        :param k: Number of clusters. Passing `k` or `target_cluster_size` replaces both init values.
        :param target_cluster_size: Average documents per cluster.
        :param model_path: Where to write the model; replaces the init value.
        :returns: A dictionary with `model_path` (the file written) and `stats` (a report of the fit).
        :raises FileExistsError: If the file exists and `overwrite` is `False`.
        :raises ValueError: On invalid parameters or embeddings.
        """
        if k is not None or target_cluster_size is not None:
            check_positive_int("k", k, allow_none=True)
            check_positive_int("target_cluster_size", target_cluster_size, allow_none=True)
            use_k, use_tcs = k, target_cluster_size
        else:
            use_k, use_tcs = self.k, self.target_cluster_size
        self._check_exactly_one(use_k, use_tcs)
        path = model_path if model_path is not None else self.model_path

        if os.path.exists(path) and not self.overwrite:
            msg = f"Model file already exists: {path}. Pass overwrite=True to replace it."
            raise FileExistsError(msg)

        positions, x = extract_embeddings(documents)
        n_usable = len(positions)
        n_skipped = len(documents) - n_usable
        if n_skipped:
            logger.warning("Skipped {n} documents without embedding", n=n_skipped)
        if n_usable < 2:  # noqa: PLR2004
            msg = f"At least 2 documents with an embedding are required, got {n_usable}"
            raise ValueError(msg)
        if use_k is not None and use_k > n_usable:
            msg = f"k={use_k} is larger than the number of usable documents ({n_usable})"
            raise ValueError(msg)

        if self.spherical:
            x = normalize_rows(x, [documents[p].id for p in positions])
        n_clusters = resolve_k(n_usable, use_k, use_tcs)
        backend = select_backend(device=self.device, n=n_usable, k=n_clusters, gpu_threshold=self.gpu_threshold)
        fit = fit_model(
            x,
            k=n_clusters,
            spherical=self.spherical,
            nepochs=self.nepochs,
            seed=self.seed,
            max_points_per_centroid=self.max_points_per_centroid,
            backend=backend,
        )
        info = fit.info
        metadata = {
            "spherical": self.spherical,
            "n_train": info["n_train"],
            "seed": self.seed,
            "nepochs": self.nepochs,
            "backend": info["backend"],
            "assignment_backend": info["assignment_backend"],
            "precision_used": info["precision_used"],
            "emeans_source": info["emeans_source"],
            "faiss_version": info["faiss_version"],
            "emeans_haystack_version": _package_version(),
            "recipe": info["recipe"],
        }
        save_model(path, fit.centroids, metadata, overwrite=self.overwrite)

        stats: dict[str, Any] = {
            "k": n_clusters,
            "d": int(x.shape[1]),
            "n_documents": len(documents),
            "n_skipped": n_skipped,
            "n_train": info["n_train"],
            "spherical": self.spherical,
            "backend": info["backend"],
            "assignment_backend": info["assignment_backend"],
            "device_name": info["device_name"],
            "precision_used": info["precision_used"],
            "fallback_reason": backend.fallback_reason,
            "emeans_source": info["emeans_source"],
            "faiss_version": info["faiss_version"],
            "train_seconds": info["train_seconds"],
        }
        return {"model_path": path, "stats": stats}
