# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""
Download the pinned Cohere Wikipedia embeddings into the gitignored cache and verify them.

The file is a headerless float32 row-major matrix of 1,000,000 x 768 L2-normalised rows. The download is pinned to one
dataset revision, resumes a partial `.part` file with an HTTP Range request, and is renamed into place only after its
size and sha256 match. Run it again at any time with `--verify-only` to re-check the file on disk.
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

DATASET = "YoKONCy/Cohere-1M-wikipedia-768d"
REVISION = "25ad59fd88a30aec3370cbc4c46dcbfc1e19591f"
FILENAME = "cohere_train.f32"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILENAME}"
SHA256 = "d4bd7224f525ff1722ef4a77d0437a01c5f2c624857ead009f0a9c0e8e167961"
SIZE = 3_072_000_000
CACHE_DIR = Path(os.environ.get("BENCH_CACHE_DIR", Path(__file__).resolve().parent / "cache"))

_CHUNK = 1024 * 1024
_PROGRESS_EVERY = 256 * 1024 * 1024
_TIMEOUT_S = 60
_PARTIAL_CONTENT = 206


def sha256_of(path: Path) -> str:
    """
    Stream a file through sha256.

    :param path: File to hash.
    :returns: Hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, sha256: str, size: int) -> str | None:
    """
    Check a file's size and sha256.

    :param path: File to check.
    :param sha256: Expected hex digest.
    :param size: Expected size in bytes.
    :returns: `None` when the file matches, otherwise the reason it does not.
    """
    if not path.is_file():
        return f"{path} does not exist"
    actual_size = path.stat().st_size
    if actual_size != size:
        return f"{path} has {actual_size} bytes, expected {size}"
    digest = sha256_of(path)
    if digest != sha256:
        return f"{path} has sha256 {digest}, expected {sha256}"
    return None


def fetch(
    url: str,
    dest: Path,
    sha256: str,
    size: int,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> int:
    """
    Download `url` to `dest` with resume and verification.

    :param url: HTTPS URL of the pinned file.
    :param dest: Final path; written only after verification passes.
    :param sha256: Expected hex digest.
    :param size: Expected size in bytes.
    :param urlopen: Opener, replaceable in tests.
    :returns: 0 on success, 1 on a size or digest mismatch, 2 for a non-https URL.
    """
    if not url.startswith("https://"):
        sys.stderr.write(f"Refusing non-https URL: {url}\n")
        return 2
    if verify(dest, sha256, size) is None:
        sys.stderr.write(f"{dest} is present and verified\n")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url)  # noqa: S310 (https checked above)
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urlopen(request, timeout=_TIMEOUT_S) as response:
        if have and response.getcode() != _PARTIAL_CONTENT:
            sys.stderr.write("Server ignored the Range request; restarting from zero\n")
            have = 0
        written = have
        next_report = (written // _PROGRESS_EVERY + 1) * _PROGRESS_EVERY
        with part.open("ab" if have else "wb") as out:
            while chunk := response.read(_CHUNK):
                out.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    sys.stderr.write(f"{written / 1e9:.2f} / {size / 1e9:.2f} GB\n")
                    next_report += _PROGRESS_EVERY

    reason = verify(part, sha256, size)
    if reason is not None:
        sys.stderr.write(f"Verification failed, not renaming: {reason}\n")
        return 1
    os.replace(part, dest)
    sys.stderr.write(f"Verified and stored {dest}\n")
    return 0


def main() -> int:
    """
    Run the command line.

    :returns: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--verify-only", action="store_true", help="only check the file already on disk")
    args = parser.parse_args()
    dest = args.dest_dir / FILENAME
    if args.verify_only:
        reason = verify(dest, SHA256, SIZE)
        if reason is not None:
            sys.stderr.write(f"{reason}\n")
            return 1
        sys.stderr.write(f"{dest} verified\n")
        return 0
    return fetch(URL, dest, SHA256, SIZE)


if __name__ == "__main__":
    sys.exit(main())
