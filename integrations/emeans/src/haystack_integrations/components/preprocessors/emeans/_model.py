# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

FORMAT_VERSION = 1


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
        return bool(self.metadata["spherical"])


def save_model(path: str, centroids: np.ndarray, metadata: dict[str, Any], *, overwrite: bool) -> None:
    """
    Write a model file atomically.

    The data goes to a temporary file in the target directory first, so a crash never leaves a partial model.

    :param path: Target `.npz` path.
    :param centroids: Centroids of shape `(k, d)`; stored as float32.
    :param metadata: Extra metadata; format version, `k`, `d`, checksum and creation time are added here.
    :param overwrite: If `False`, an existing file at `path` raises `FileExistsError`.
    """
    c = np.ascontiguousarray(centroids, dtype=np.float32)
    meta: dict[str, Any] = {
        **metadata,
        "format_version": FORMAT_VERSION,
        "k": int(c.shape[0]),
        "d": int(c.shape[1]),
        "centroids_sha256": hashlib.sha256(c.tobytes()).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    parent = os.path.dirname(os.path.abspath(path))
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(dir=parent, suffix=".tmp", delete=False) as tmp:
            tmp_path = tmp.name
            np.savez(tmp, centroids=c, metadata=np.array(json.dumps(meta, sort_keys=True, default=_json_default)))
            tmp.flush()
            os.fsync(tmp.fileno())
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)
        if overwrite:
            os.replace(tmp_path, path)
        else:
            _create_exclusive(tmp_path, path)
    finally:
        if tmp_path:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)


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
    Read a model file written by `save_model`.

    Loading never unpickles anything, so a model file cannot run code.

    :param path: Path of the `.npz` model file.
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file cannot be read as a model.
    """
    if not os.path.exists(path):
        msg = f"Model file not found: {path}"
        raise FileNotFoundError(msg)
    try:
        with np.load(path, allow_pickle=False) as z:
            if not isinstance(z, np.lib.npyio.NpzFile):
                msg = "not an .npz archive"
                raise ValueError(msg)
            centroids = np.array(z["centroids"])
            metadata = json.loads(str(z["metadata"]))
    except (ValueError, EOFError, zipfile.BadZipFile, KeyError, OSError) as e:
        msg = f"{path}: cannot read model file ({e})"
        raise ValueError(msg) from e
    if not isinstance(metadata, dict):
        msg = f"{path}: metadata is not a JSON object"
        raise ValueError(msg)
    return EmeansModel(path=path, centroids=centroids, metadata=metadata)
