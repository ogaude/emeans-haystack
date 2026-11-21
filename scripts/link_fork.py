# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Link the e-means-rocm fork over the faiss-cpu e-means copy in the current environment (local GPU testing only)."""

import os
import shutil
import sys
from pathlib import Path


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


def _restore(dst: Path) -> int:
    backup = dst.with_name("e_means.pypi-orig")
    if dst.is_symlink():
        dst.unlink()
    if backup.is_dir() and not dst.exists():
        backup.rename(dst)
        sys.stdout.write(f"restored {dst} from {backup}\n")
    elif dst.is_dir():
        sys.stdout.write(f"{dst} is already the faiss-cpu copy\n")
    else:
        sys.stderr.write(f"No backup at {backup}; reinstall faiss-cpu to restore {dst}\n")
        return 2
    return 0


def main() -> int:
    """
    Replace the environment's `faiss/contrib/e_means` with a symlink to the fork named by `EMEANS_ROCM_FORK`.

    With `--restore`, remove the symlink and put the faiss-cpu copy back from its backup.

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
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        # A real directory is the current faiss-cpu copy (for example after an upgrade): it replaces any older
        # backup, so a later restore matches the installed faiss version.
        backup = dst.with_name("e_means.pypi-orig")
        if backup.exists():
            sys.stdout.write(f"replacing the older backup {backup} with the current {dst}\n")
            shutil.rmtree(backup)
        dst.rename(backup)
    target = (Path(fork) / "contrib" / "e_means").resolve()
    dst.symlink_to(target)
    sys.stdout.write(f"{dst} -> {target}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
