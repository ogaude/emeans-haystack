# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""
Example 4: fit e-means on a large corpus on a GPU (synthetic 768-d embeddings).

Run it from integrations/emeans in an env with a GPU build of PyTorch, for example
`hatch run gpu:run ../../examples/04_large_corpus_gpu.py` (or `gpu-fork:run` with emeans-rocm installed).
It uses device="cuda", so it raises where no GPU backend is usable.
"""

# ruff: noqa: T201 (examples print their results)

import argparse
from pathlib import Path

import numpy as np
from haystack import Document

from haystack_integrations.components.preprocessors.emeans import EmeansFitter

DIM = 768
N_CENTRES = 1024
BLOCK = 10_000


def make_documents(n: int, seed: int) -> list[Document]:
    """Build `n` synthetic unit-norm 768-d embeddings scattered around 1024 random centres."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((N_CENTRES, DIM)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    documents: list[Document] = []
    for start in range(0, n, BLOCK):
        size = min(BLOCK, n - start)
        rows = centres[rng.integers(0, N_CENTRES, size)] + 0.3 * rng.standard_normal((size, DIM)).astype(np.float32)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        documents.extend(
            Document(content=f"synthetic document {start + i}", embedding=row.tolist()) for i, row in enumerate(rows)
        )
    return documents


def main() -> None:
    """Run the example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200_000, help="number of documents")
    parser.add_argument("--k", type=int, default=16_384, help="number of clusters")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Same folder as _shared.output_dir(); _shared is not imported because it needs sentence-transformers-haystack,
    # which the gpu envs do not install.
    out = Path(__file__).parent / ".output"
    out.mkdir(exist_ok=True)

    # Each Document holds its embedding as a Python list (about 25 KB per 768-d document), so host RAM,
    # not GPU memory, bounds --n.
    documents = make_documents(args.n, args.seed)
    print(f"documents: {len(documents)}")

    # On upstream e-means on AMD, bf16 runs in FP32 with one warning and reports gpu-pytorch;
    # with emeans-rocm or on NVIDIA it runs BF16.
    fitter = EmeansFitter(
        k=args.k,
        model_path=str(out / "large.npz"),
        device="cuda",
        precision="bf16",
        deterministic=True,
        overwrite=True,
    )
    stats = fitter.run(documents=documents)["stats"]
    print(f"backend: {stats['backend']}")
    print(f"precision_used: {stats['precision_used']}")
    print(f"train_seconds: {stats['train_seconds']:.2f}")
    print(f"device_name: {stats['device_name']}")
    print(f"emeans_source: {stats['emeans_source']}")
    print(f"k: {stats['k']}")


if __name__ == "__main__":
    main()
