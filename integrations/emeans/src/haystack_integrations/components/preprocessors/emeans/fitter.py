# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Literal

import numpy as np
from haystack import Document, component, default_from_dict, default_to_dict, logging

from haystack_integrations.components.preprocessors.emeans._backend import DEVICES, fit_model, select_backend
from haystack_integrations.components.preprocessors.emeans._model import UNIT_NORM_TOLERANCE, save_model
from haystack_integrations.components.preprocessors.emeans._utils import (
    check_bool,
    check_integer,
    check_nepochs,
    check_positive_int,
    check_threshold,
    extract_embeddings,
    normalize_rows,
    optional_int,
    resolve_k,
)

logger = logging.getLogger(__name__)

_PRECISIONS = ("auto", "fp32", "bf16", "fp16")
_MIN_USABLE_DOCUMENTS = 2
_EMPTY_CENTROID_NORM = 0.5


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
        :param deterministic: On GPU, request deterministic algorithms during training. No effect on CPU. On
            NVIDIA, export `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting the process for guaranteed repeats.
            PyTorch's setting is process-wide: while the fit trains, GPU work in other threads (other fits,
            cluster assignment, backend probes) also runs with deterministic algorithms on.
        :param max_points_per_centroid: If set and `n > max_points_per_centroid * k`, train on a random subset.
        :param overwrite: If `False`, an existing file at `model_path` raises `FileExistsError`.
        """
        check_positive_int("k", k, allow_none=True)
        check_positive_int("target_cluster_size", target_cluster_size, allow_none=True)
        check_positive_int("max_points_per_centroid", max_points_per_centroid, allow_none=True)
        self._check_exactly_one(k, target_cluster_size)
        check_nepochs(nepochs)
        check_bool("spherical", spherical)
        check_bool("deterministic", deterministic)
        check_bool("overwrite", overwrite)
        check_integer("seed", seed)
        check_threshold("gpu_threshold", gpu_threshold)
        if device not in DEVICES:
            msg = f"device must be one of {DEVICES}, got {device!r}"
            raise ValueError(msg)
        if precision not in _PRECISIONS:
            msg = f"precision must be one of {_PRECISIONS}, got {precision!r}"
            raise ValueError(msg)
        self.model_path = model_path
        self.k = optional_int(k)
        self.target_cluster_size = optional_int(target_cluster_size)
        self.spherical = spherical
        self.nepochs = nepochs
        self.precision = precision
        self.device = device
        self.gpu_threshold = gpu_threshold
        self.seed = int(seed)
        self.deterministic = deterministic
        self.max_points_per_centroid = optional_int(max_points_per_centroid)
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
        :raises FileNotFoundError: If the parent directory of the model path does not exist.
        :raises IsADirectoryError: If the model path is a directory and `overwrite` is `True`.
        :raises RuntimeError: If `device="cuda"` and no GPU backend is usable.
        :raises ValueError: On invalid parameters or embeddings, or if the fit produced a non-finite centroid or an
            empty spherical centroid.
        """
        if k is not None or target_cluster_size is not None:
            check_positive_int("k", k, allow_none=True)
            check_positive_int("target_cluster_size", target_cluster_size, allow_none=True)
            use_k, use_tcs = optional_int(k), optional_int(target_cluster_size)
        else:
            use_k, use_tcs = self.k, self.target_cluster_size
        self._check_exactly_one(use_k, use_tcs)
        path = model_path if model_path is not None else self.model_path

        parent = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(parent):
            msg = f"parent directory of model_path does not exist: {parent}"
            raise FileNotFoundError(msg)
        if os.path.exists(path) and not self.overwrite:
            msg = f"Model file already exists: {path}. Pass overwrite=True to replace it."
            raise FileExistsError(msg)
        if os.path.isdir(path):
            msg = f"model_path is a directory, not a file: {path}"
            raise IsADirectoryError(msg)

        positions, x = extract_embeddings(documents)
        n_usable = len(positions)
        n_skipped = len(documents) - n_usable
        if n_skipped:
            logger.warning("Skipped {n} documents without embedding", n=n_skipped)
        if n_usable < _MIN_USABLE_DOCUMENTS:
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
            precision=self.precision,
            deterministic=self.deterministic,
        )
        info = fit.info
        # The GPU engine does not check its data for NaN or infinity, and NaN passes every norm comparison below.
        # Check finiteness first, on every backend, so the fitter never writes a model its own loader refuses.
        finite_rows = np.isfinite(fit.centroids).all(axis=1)
        if not finite_rows.all():
            n_bad = int(np.count_nonzero(~finite_rows))
            msg = (
                f"The fit produced {n_bad} non-finite centroids out of {n_clusters}; "
                "use precision='fp32' or check the embeddings"
            )
            raise ValueError(msg)
        if self.spherical:
            norms = np.linalg.norm(fit.centroids.astype(np.float64), axis=1)
            n_empty = int(np.count_nonzero(norms < _EMPTY_CENTROID_NORM))
            if n_empty:
                msg = (
                    f"The fit produced {n_empty} empty centroids out of {n_clusters}; "
                    "use a smaller k or a larger nepochs"
                )
                raise ValueError(msg)
            # Same tolerance as load_model, so the fitter never writes a model its own loader refuses.
            n_off = int(np.count_nonzero(np.abs(norms - 1.0) > UNIT_NORM_TOLERANCE))
            if n_off:
                msg = f"The fit produced {n_off} spherical centroids out of {n_clusters} that do not have unit norm"
                raise ValueError(msg)
        metadata = {
            "spherical": self.spherical,
            "n_train": info["n_train"],
            "seed": self.seed,
            "nepochs": float(self.nepochs),
            "backend": info["backend"],
            "assignment_backend": info["assignment_backend"],
            "precision_used": info["precision_used"],
            "emeans_source": info["emeans_source"],
            "faiss_version": info["faiss_version"],
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
            "nepochs": info["nepochs"],
            "batch_size": info["batch_size"],
            "alpha0": info["alpha0"],
            "n_splits": info["n_splits"],
            "imbalance": info["imbalance"],
            "train_seconds": info["train_seconds"],
        }
        logger.info(
            "Fitted e-means on the {backend} backend (device {device_name}, precision {precision}): k={k} on {n} "
            "documents in {seconds:.2f}s{hint}",
            backend=info["backend"],
            device_name=info["device_name"],
            precision=info["precision_used"],
            k=n_clusters,
            n=n_usable,
            seconds=info["train_seconds"],
            hint=f" {info['backend_hint']}" if info.get("backend_hint") else "",
        )
        return {"model_path": path, "stats": stats}
