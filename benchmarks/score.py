# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""One shared exact FP32 nearest-centroid scorer, used for every method and mode."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import faiss
import numpy as np

from data import load_dataset


def score(x: np.ndarray, centroids: np.ndarray, chunk: int = 65536) -> tuple[float, float, np.ndarray]:
    """
    Score centroids by mean squared distance to the nearest centroid (exact, FP32).

    :param x: Data matrix.
    :param centroids: Centroid matrix of shape `(k, d)`.
    :param chunk: Rows searched per call.
    :returns: The mean objective, the summed objective and the int64 labels.
    """
    n, d = x.shape
    index = faiss.IndexFlatL2(d)
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    labels = np.empty(n, dtype=np.int64)
    total = 0.0
    for lo in range(0, n, chunk):
        part = np.ascontiguousarray(x[lo : lo + chunk], dtype=np.float32)
        dist, idx = index.search(part, 1)
        total += float(dist.astype(np.float64).sum())
        labels[lo : lo + len(part)] = idx[:, 0]
    return total / n, total, labels


def main() -> int:
    """Score a saved centroid file against a dataset and print the result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, default=768)
    parser.add_argument("--centroids", required=True)
    args = parser.parse_args()
    x, _ = load_dataset(args.data, args.n, args.d)
    centroids = np.load(args.centroids, allow_pickle=False)
    objective, objective_sum, _ = score(x, centroids)
    sys.stdout.write(
        json.dumps({"objective": objective, "objective_sum": objective_sum, "n": args.n, "k": len(centroids)}) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
