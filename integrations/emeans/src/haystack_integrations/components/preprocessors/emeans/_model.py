# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import hashlib
import json
import os
import secrets
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

FORMAT_VERSION = 1

METADATA_KEYS: tuple[str, ...] = (
    "format_version",
    "k",
    "d",
    "spherical",
    "centroids_sha256",
    "created_at",
    "n_train",
    "seed",
    "nepochs",
    "backend",
    "assignment_backend",
    "precision_used",
    "emeans_source",
    "faiss_version",
    "emeans_haystack_version",
    "recipe",
)
_COMPUTED_KEYS = ("format_version", "k", "d", "centroids_sha256", "created_at", "emeans_haystack_version")
_REQUIRED_KEYS = ("format_version", "k", "d", "spherical", "centroids_sha256")
UNIT_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class EmeansModel:
    """A loaded model file: centroids plus the metadata stored with them."""

    path: str
    centroids: np.ndarray
    metadata: dict[str, Any]

    @property
    def k(self) -> int:
        """Number of clusters."""
        return int(self.centroids.shape[0])

    @property
    def d(self) -> int:
        """Embedding dimension."""
        return int(self.centroids.shape[1])

    @property
    def spherical(self) -> bool:
        """Whether the model clusters by cosine similarity."""
        return self.metadata["spherical"] is True


class ModelCache:
    """
    Remembers the last model loaded and reloads it only when the path or the file changes.

    A file counts as changed when its device, inode, size or modification time differ. `save_model` always writes
    a new inode, so every rewrite by this package is detected even when the modification time is unchanged
    (coarse filesystem timestamps, `cp -p`, restored backups).
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._key: tuple[str, int, int, int, int] | None = None
        self._model: EmeansModel | None = None

    def get(self, path: str, loader: Callable[[str], EmeansModel]) -> EmeansModel:
        """
        Return the model at `path`, calling `loader(path)` unless the cached model comes from the same file.

        :param path: Path of the model file.
        :param loader: Function that loads a model file, normally `load_model`.
        """
        try:
            st = os.stat(path)
            key: tuple[str, int, int, int, int] | None = (
                os.path.abspath(path),
                st.st_dev,
                st.st_ino,
                st.st_size,
                st.st_mtime_ns,
            )
        except OSError:
            key = None  # the loader reports the missing or unreadable file
        if self._model is None or key is None or key != self._key:
            self._model = loader(path)
            self._key = key
        return self._model


def model_checksum(centroids: np.ndarray, *, spherical: bool) -> str:
    """
    Return the checksum stored as `centroids_sha256`.

    Definition (model format version 1): the lowercase hex SHA-256 of the ASCII header
    `spherical=<1|0>;k=<k>;d=<d>;` immediately followed by the centroid values as float32 in C (row-major)
    order. The header binds the clustering mode and the shape to the centroid bytes, so editing `spherical`,
    `k` or `d` in the metadata is detected like editing a centroid value.

    :param centroids: float32 centroids of shape `(k, d)`.
    :param spherical: Whether the model clusters by cosine similarity.
    """
    k, d = centroids.shape
    header = f"spherical={int(spherical)};k={int(k)};d={int(d)};".encode("ascii")
    body = np.ascontiguousarray(centroids, dtype=np.float32).tobytes(order="C")
    return hashlib.sha256(header + body).hexdigest()


def save_model(path: str, centroids: np.ndarray, metadata: dict[str, Any], *, overwrite: bool) -> None:
    """
    Write a model file atomically.

    The data goes to a temporary file in the target directory first, so a crash never leaves a partial model.

    :param path: Target `.npz` path.
    :param centroids: Centroids of shape `(k, d)`; stored as float32.
    :param metadata: The caller-supplied metadata keys; format version, `k`, `d`, checksum, creation time and package
        version are added here.
    :param overwrite: If `False`, an existing file at `path` raises `FileExistsError`.
    :raises ValueError: If `metadata` does not have exactly the expected keys, or `spherical` is not a bool.
    """
    expected_keys = set(METADATA_KEYS) - set(_COMPUTED_KEYS)
    if set(metadata) != expected_keys:
        msg = (
            f"metadata keys must be exactly {sorted(expected_keys)}; "
            f"missing {sorted(expected_keys - set(metadata))}, unexpected {sorted(set(metadata) - expected_keys)}"
        )
        raise ValueError(msg)
    if not isinstance(metadata["spherical"], bool):
        msg = f"metadata spherical must be True or False, got {metadata['spherical']!r}"
        raise ValueError(msg)
    c = np.ascontiguousarray(centroids, dtype=np.float32)
    meta: dict[str, Any] = {
        **metadata,
        "format_version": FORMAT_VERSION,
        "k": int(c.shape[0]),
        "d": int(c.shape[1]),
        "centroids_sha256": model_checksum(c, spherical=metadata["spherical"]),
        "created_at": datetime.now(UTC).isoformat(),
        "emeans_haystack_version": _package_version(),
    }
    parent = os.path.dirname(os.path.abspath(path))
    tmp_path = ""
    try:
        fd, tmp_path = _create_temp(parent, os.path.basename(path))
        with os.fdopen(fd, "wb") as tmp:
            np.savez(tmp, centroids=c, metadata=np.array(json.dumps(meta, sort_keys=True, default=_json_default)))
            tmp.flush()
            os.fsync(tmp.fileno())
        if overwrite:
            os.replace(tmp_path, path)
        else:
            _create_exclusive(tmp_path, path)
    finally:
        if tmp_path:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)


def _package_version() -> str:
    try:
        return version("emeans-haystack")
    except PackageNotFoundError:
        return "unknown"


def _create_temp(parent: str, name: str) -> tuple[int, str]:
    """
    Create a new temporary file next to the target and return its descriptor and path.

    The file is opened with mode `0o666` so that the kernel applies the process umask, as for any normal file.
    Reading the umask with `os.umask` would change it for every thread of the process for a moment.
    """
    for _ in range(100):
        tmp_path = os.path.join(parent, f".{name}.{secrets.token_hex(8)}.tmp")
        try:
            return os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666), tmp_path
        except FileExistsError:
            continue
    msg = f"could not create a temporary file in {parent}"
    raise FileExistsError(msg)


def _create_exclusive(tmp_path: str, path: str) -> None:
    try:
        os.link(tmp_path, path)
    except FileExistsError:
        raise
    except OSError:
        # Hard links are not supported on this filesystem: fall back to a plain existence check.
        if os.path.exists(path):
            raise FileExistsError(path) from None
        os.replace(tmp_path, path)


def _json_default(value: object) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    msg = f"Object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def load_model(path: str) -> EmeansModel:
    """
    Read a model file written by `save_model` and validate it.

    Loading never unpickles anything, so a model file cannot run code. The checks run in this order: `file`,
    `arrays`, `format_version`, `metadata`, `shape`, `finite`, `unit_norm`, `checksum`.

    :param path: Path of the `.npz` model file.
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file is not a valid, supported model; the message names the file and the failed check.
    """
    if not os.path.exists(path):
        msg = f"Model file not found: {path}"
        raise FileNotFoundError(msg)
    if not os.path.isfile(path):
        raise _invalid(path, "file", "not a regular file")
    centroids, metadata = _read_arrays(path)
    _check_format_version(path, metadata)
    _check_metadata_types(path, metadata)
    _check_shape(path, centroids, metadata)
    if not np.isfinite(centroids).all():
        raise _invalid(path, "finite", "centroids contain NaN or infinite values")
    if metadata["spherical"]:
        norms = np.linalg.norm(centroids.astype(np.float64), axis=1)
        bad = int(np.count_nonzero(np.abs(norms - 1.0) > UNIT_NORM_TOLERANCE))
        if bad:
            raise _invalid(path, "unit_norm", f"{bad} centroid rows of a spherical model do not have unit norm")
    if model_checksum(centroids, spherical=metadata["spherical"]) != metadata["centroids_sha256"]:
        detail = "centroids, spherical, k or d do not match centroids_sha256; the file is corrupt or modified"
        raise _invalid(path, "checksum", detail)
    return EmeansModel(path=str(path), centroids=centroids, metadata=metadata)


def _invalid(path: str, check: str, detail: str) -> ValueError:
    return ValueError(f"{path}: {check} check failed: {detail}")


def _read_arrays(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        loaded = np.load(path, allow_pickle=False)
    except (ValueError, EOFError, zipfile.BadZipFile, KeyError, OSError) as e:
        raise _invalid(path, "arrays", f"cannot read as an .npz archive ({e})") from e
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise _invalid(path, "arrays", "not an .npz archive")
    with loaded as z:
        if set(z.files) != {"centroids", "metadata"}:
            raise _invalid(path, "arrays", f"expected arrays centroids and metadata, found {sorted(z.files)}")
        try:
            centroids = np.array(z["centroids"])
            raw = z["metadata"]
        except (ValueError, EOFError, zipfile.BadZipFile, KeyError, OSError) as e:
            raise _invalid(path, "arrays", f"cannot read arrays ({e})") from e
    if raw.ndim != 0 or raw.dtype.kind != "U":
        raise _invalid(path, "arrays", "metadata is not a 0-d string array")
    try:
        metadata = json.loads(str(raw))
    except ValueError as e:
        raise _invalid(path, "arrays", f"metadata is not valid JSON ({e})") from e
    if not isinstance(metadata, dict):
        raise _invalid(path, "arrays", "metadata is not a JSON object")
    missing = [key for key in _REQUIRED_KEYS if key not in metadata]
    if missing:
        raise _invalid(path, "arrays", f"metadata lacks keys {missing}")
    return centroids, metadata


def _check_format_version(path: str, metadata: dict[str, Any]) -> None:
    version = metadata["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise _invalid(path, "format_version", f"format_version must be an integer, got {version!r}")
    if version > FORMAT_VERSION:
        raise _invalid(
            path,
            "format_version",
            f"model format_version {version} is newer than supported version {FORMAT_VERSION}; upgrade emeans-haystack",
        )
    if version < 1:
        raise _invalid(path, "format_version", f"unsupported format_version {version}")


def _check_metadata_types(path: str, metadata: dict[str, Any]) -> None:
    # A truthy string such as "false" must never select the spherical geometry, nor a falsy one the Euclidean.
    spherical = metadata["spherical"]
    if not isinstance(spherical, bool):
        raise _invalid(path, "metadata", f"metadata spherical must be true or false, got {spherical!r}")
    checksum = metadata["centroids_sha256"]
    if not isinstance(checksum, str):
        raise _invalid(path, "metadata", f"metadata centroids_sha256 must be a string, got {checksum!r}")


def _check_shape(path: str, centroids: np.ndarray, metadata: dict[str, Any]) -> None:
    k, d = metadata["k"], metadata["d"]
    for name, value in (("k", k), ("d", d)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise _invalid(path, "shape", f"metadata {name} must be a positive integer, got {value!r}")
    if centroids.dtype != np.float32:
        raise _invalid(path, "shape", f"centroids must be float32, got {centroids.dtype}")
    if centroids.ndim != 2 or centroids.shape != (k, d):  # noqa: PLR2004
        raise _invalid(path, "shape", f"centroids have shape {centroids.shape}, metadata says ({k}, {d})")
