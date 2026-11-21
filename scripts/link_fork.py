# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Link the e-means-rocm fork over the faiss-cpu e-means copy in the current environment (local GPU testing only)."""

import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    """
    Replace the environment's `faiss/contrib/e_means` with a symlink to the fork named by `EMEANS_ROCM_FORK`.

    :returns: 0 on success, 2 when `EMEANS_ROCM_FORK` is unset or does not point at a fork clone.
    """
    fork = os.environ.get("EMEANS_ROCM_FORK")
    if not fork or not (Path(fork) / "contrib" / "e_means").is_dir():
        sys.stderr.write("Set EMEANS_ROCM_FORK to a clone of the e-means-rocm fork\n")
        return 2

    import faiss.contrib  # noqa: PLC0415

    dst = Path(faiss.contrib.__file__).parent / "e_means"
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        backup = dst.with_name("e_means.pypi-orig")
        if backup.exists():
            sys.stdout.write(f"removing {dst} (backup {backup} already exists)\n")
            shutil.rmtree(dst)
        else:
            dst.rename(backup)
    target = (Path(fork) / "contrib" / "e_means").resolve()
    dst.symlink_to(target)
    sys.stdout.write(f"{dst} -> {target}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
