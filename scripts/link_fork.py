# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""
Install a copy of the e-means-rocm fork over the faiss-cpu e-means copy in the current environment.

This is for local GPU testing only. The fork is copied, not symlinked, so pip/uv operations on faiss-cpu can never
write to or delete files in the fork clone. Re-run it after editing the fork. Run `--restore` before upgrading,
reinstalling or uninstalling faiss-cpu, and run it again afterwards.
"""

import os
import shutil
import sys
from pathlib import Path

# Written into the copied directory so the script can tell its fork copy apart from the faiss-cpu copy.
_MARKER = ".emeans-rocm-fork-copy"
_BACKUP_NAME = "e_means.pypi-orig"


def _emeans_dir() -> Path | None:
    """Return the environment's `faiss/contrib/e_means` path, or `None` when it is not inside an active venv."""
    if sys.prefix == sys.base_prefix:
        sys.stderr.write("Refusing to run outside a virtual environment; run it through hatch\n")
        return None

    import faiss.contrib  # noqa: PLC0415

    dst = Path(faiss.contrib.__file__).parent / "e_means"
    if not dst.parent.resolve().is_relative_to(Path(sys.prefix).resolve()):
        sys.stderr.write(f"Refusing to modify {dst}: it is not inside the active environment {sys.prefix}\n")
        return None
    return dst


def _is_fork_copy(dst: Path) -> bool:
    """Return whether `dst` is a fork copy made by this script (or a symlink left by an older version of it)."""
    return dst.is_symlink() or (dst / _MARKER).is_file()


def _remove_fork_copy(dst: Path) -> None:
    """Remove a fork copy, or only the link itself for a legacy symlink, never the fork clone it points at."""
    if dst.is_symlink():
        dst.unlink()
    else:
        shutil.rmtree(dst)


def _restore(dst: Path) -> int:
    backup = dst.with_name(_BACKUP_NAME)
    fork_copy = _is_fork_copy(dst)
    if not fork_copy and dst.is_dir():
        sys.stdout.write(f"{dst} is already the faiss-cpu copy\n")
        return 0
    # Check for the backup before removing anything, so a failed restore never leaves the env without e_means.
    if not backup.is_dir():
        kept = f"; left {dst} in place" if fork_copy else ""
        sys.stderr.write(f"No backup at {backup}{kept}. Reinstall faiss-cpu to restore {dst}\n")
        return 2
    if fork_copy:
        _remove_fork_copy(dst)
    backup.rename(dst)
    sys.stdout.write(f"restored {dst} from {backup}\n")
    return 0


def main() -> int:
    """
    Replace the environment's `faiss/contrib/e_means` with a copy of the fork named by `EMEANS_ROCM_FORK`.

    The faiss-cpu copy is kept as `e_means.pypi-orig`. With `--restore`, remove the fork copy and put the faiss-cpu
    copy back from that backup; nothing is removed when there is no backup.

    :returns: 0 on success, 2 when `EMEANS_ROCM_FORK` is unset or does not point at a fork clone, when not run
        inside a virtual environment, or when `--restore` finds no backup.
    """
    if "--restore" in sys.argv[1:]:
        dst = _emeans_dir()
        return 2 if dst is None else _restore(dst)

    fork = os.environ.get("EMEANS_ROCM_FORK")
    if not fork or not (Path(fork) / "contrib" / "e_means").is_dir():
        sys.stderr.write("Set EMEANS_ROCM_FORK to a clone of the e-means-rocm fork\n")
        return 2

    dst = _emeans_dir()
    if dst is None:
        return 2
    source = (Path(fork) / "contrib" / "e_means").resolve()
    # Build the copy beside dst first, so a failed copy never leaves the env without an e_means package.
    staging = dst.with_name("e_means.fork-staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (staging / _MARKER).write_text(f"{source}\n", encoding="utf-8")

    if _is_fork_copy(dst):
        _remove_fork_copy(dst)
    elif dst.exists():
        # A real directory is the current faiss-cpu copy (for example after an upgrade): it replaces any older
        # backup, so a later restore matches the installed faiss version.
        backup = dst.with_name(_BACKUP_NAME)
        if backup.exists():
            sys.stdout.write(f"replacing the older backup {backup} with the current {dst}\n")
            shutil.rmtree(backup)
        dst.rename(backup)
    staging.rename(dst)
    sys.stdout.write(f"copied {source} to {dst}\n")
    sys.stdout.write("Re-run link after editing the fork. Run unlink before upgrading or reinstalling faiss-cpu.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
