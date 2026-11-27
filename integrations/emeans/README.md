# emeans-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/emeans-haystack.svg)](https://pypi.org/project/emeans-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/emeans-haystack.svg)](https://pypi.org/project/emeans-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/emeans)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/emeans/CHANGELOG.md)

---

## Installation

```bash
pip install emeans-haystack            # CPU: brings haystack-ai, faiss-cpu and numpy
```

Until the package is on PyPI, install it from a clone: `pip install ./integrations/emeans`.

Requirements: Python 3.11 or newer, `haystack-ai>=3.5`, `faiss-cpu>=1.16.0` (the first release that contains
`faiss.contrib.e_means`), `numpy>=2.0`.

**GPU, NVIDIA:**

```bash
pip install "emeans-haystack[gpu]"     # adds PyTorch from PyPI (CUDA build)
```

**GPU, AMD (ROCm):** install the ROCm build of PyTorch first, from AMD's or PyTorch's ROCm index, then install
`emeans-haystack` **without** `[gpu]`. The `[gpu]` extra pulls the PyPI build of PyTorch, which would replace your
ROCm build and leave you on CPU.

On AMD, upstream e-means runs only in native FP32 through plain PyTorch. For the fast Triton kernels and BF16/FP16 on
RX 9000 (gfx120x) GPUs, run `pip install emeans-rocm` after installing ROCm PyTorch; there is no extra for it.
[emeans-rocm](https://pypi.org/project/emeans-rocm/) (import name `emeans_rocm`, source at
https://github.com/ogaude/e-means-rocm) is the author's package of Meta's e-means code under its MIT license. This package supports emeans-rocm releases `>=0.2.5,<0.3`; the install line stays unpinned. A release outside that range is not used: GPU fits run on the stock faiss-cpu e-means engine (on AMD in native FP32, `stats["backend"]` `"gpu-pytorch"`), a warning (once per process) names the installed version and the range, and `stats["emeans_source"]` is `"faiss-cpu <version>"`. A source checkout of emeans-rocm on the path (not the installed wheel) has no checkable version and is used as is. When it
is installed, the GPU backends use it on every GPU, AMD and NVIDIA. Its NVIDIA path has not been re-run on hardware,
so NVIDIA users who want the verified path leave it uninstalled. On AMD, if its Triton backend fails the start-up
probe (for example, no Triton in your ROCm PyTorch), fits run in native FP32, and a warning (once per process) and
`stats["fallback_reason"]` give the probe error. If every one of its GPU backends fails the probe on your GPU, the
stock faiss-cpu e-means engine is tried on the GPU in native FP32 before falling back to the CPU; a warning and
`stats["fallback_reason"]` name the backends that failed. CPU fits always use the faiss-cpu e-means. The fitter's
`stats["emeans_source"]` names the package and version that ran, for example `"emeans-rocm 0.2.6"` or
`"faiss-cpu 1.16.0"`, and `stats["backend"]` which backend ran.

The package passes `unused_threshold=0.1` (`0.0` when d <= 2) to every e-means engine, so it keeps mass-based
revival of unused clusters, the rule of stock faiss and emeans-rocm 0.2.5, also on emeans-rocm 0.2.6, whose default is
cost-based revival; `stats["unused_threshold"]` reports the value used. emeans-rocm 0.2.6 places revived centroids
differently (along the donor cluster's long axis), so GPU fits at d > 2 are not the same as on 0.2.5; CPU fits and the
stock faiss-cpu engine are unchanged. Models saved with an earlier version load and assign unchanged, because
assignment from saved centroids does not depend on the engine version. A refit on emeans-rocm 0.2.6 gives assignments
(`cluster_id`s) that differ from those of a 0.2.5 fit about as much as a different seed does, while the objectives stay
close (figures under Benchmarks). The benchmark figures below are for emeans-rocm 0.2.6.

Importing the package never imports PyTorch. PyTorch is loaded only when a GPU backend is considered.

## Quick Start

A CPU run of fit, cluster and rank on random embeddings; it downloads no model.

```python
import numpy as np
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter
from haystack_integrations.components.rankers.emeans import EmeansClusterRanker

rng = np.random.default_rng(0)
docs = [Document(content=f"document {i}", embedding=rng.normal(size=64).tolist()) for i in range(500)]

# Fit once and save the model.
fitted = EmeansFitter(target_cluster_size=50, model_path="clusters.npz", overwrite=True).run(documents=docs)
stats = fitted["stats"]

# Assign a stable cluster_id to every document.
clustered = EmeansDocumentClusterer().run(documents=docs, model_path=fitted["model_path"])["documents"]

# Retrieve, then keep the best document of each cluster.
store = InMemoryDocumentStore()
store.write_documents(clustered)
query_embedding = rng.normal(size=64).tolist()
retriever = InMemoryEmbeddingRetriever(document_store=store, top_k=50)
retrieved = retriever.run(query_embedding=query_embedding)["documents"]
diverse = EmeansClusterRanker(top_k=5).run(documents=retrieved)["documents"]

print(stats["k"], stats["backend"])
for doc in diverse:
    print(doc.score, doc.meta["cluster_id"])
```

In real use a Haystack document embedder, for example `SentenceTransformersDocumentEmbedder` from
`sentence-transformers-haystack`, sits in front. The full pipelines are in the repository's
[examples](https://github.com/ogaude/emeans-haystack/tree/main/examples): 01 indexing, 02 diverse retrieval, 03 new
documents, 04 GPU.

## Benchmarks

- e-means needs fewer passes than Lloyd's k-means (`faiss.Kmeans`, scikit-learn `KMeans`) to reach the same objective in all 5 sweep comparisons (k = 4096 and 16384); its passes are not faster, and at its default 5 passes Lloyd's k-means keeps the better objective in 3 settings (below). On 1M Cohere embeddings (768-d) with k = 16384 (pass-sweep medians), CPU e-means after 5 passes (objective 0.18569) beats `faiss.Kmeans` after 25 passes (0.18746); at the 25-pass points a pass costs about the same (114 vs 111 s/pass), so 575 s vs 2765 s.
- On an AMD RX 9060 XT with `emeans-rocm` 0.2.6, GPU e-means was faster than CPU e-means cold in all 16 settings and warm in all 12 that have a CPU warm time. FP32 against FP32 at 1M x 65536 on Cohere (cold, end-to-end, CPU 2232 s): native FP32 31.1x, Triton FP32 27.5x; Triton BF16 on the GPU against FP32 on the CPU: 187.0x (11.9 s). The smallest gain is 4.1x at random 100k x 1024. Against emeans-rocm 0.2.5 on the same cells, the median cold fit time ratio is 0.99 (warm 0.85). A refit on 0.2.6 gives assignments that differ from a 0.2.5 fit about as much as a different seed does: on Cohere 100k the same-seed ARI between versions is 0.27-0.40, against 0.25-0.39 between 0.2.5 seeds and 1.0 for a BF16 0.2.5 rerun. The objectives stay close: within 1.342 % of 0.2.5 and lower on 0.2.6 in 71 of 96 main-cell records, the largest difference at Cohere 100k x 65536 (1.5 rows per centroid). Models saved before load and assign unchanged ([comparison](../../benchmarks/RESULTS.md)).
- Without a GPU, on real data (Cohere), CPU e-means finishes sooner than Lloyd's k-means with an objective within 0.5 % in 6 of 7 comparable settings (11 of 14 over both datasets). Lloyd's k-means has the better objective in 3 settings: 100k x 65536 on both datasets (degenerate: 1.5 rows per centroid; e-means +22.22 % on Cohere, +36.81 % on random) and random 100k x 16384 (e-means +7.90 %).
- On real data, FP16/BF16 objectives are between -0.070 % and +0.055 % of FP32, but at about 100k rows they change document assignments almost as much as reseeding does: use FP32 when stable `cluster_id`s matter.
- Full tables, protocol, every cut made for time or memory and the settings not run: [benchmarks/RESULTS.md](../../benchmarks/RESULTS.md). Random data (Gaussian noise, no cluster structure) is used for speed; its objectives appear here only as the exceptions noted above.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

GPU tests are marked `integration` and `gpu`, so `hatch run test:unit` and CI never run them. They are local
only: run them with `hatch run gpu:gpu-tests` (upstream e-means) or `hatch run gpu-fork:gpu-tests` (installs
`emeans-rocm` 0.2.6 from PyPI). They skip when PyTorch sees no GPU.

## License

`emeans-haystack` is distributed under the Apache-2.0 license (see `LICENSE.txt`).

Third-party notices:

- e-means is developed by Meta FAIR and ships in faiss (`faiss.contrib.e_means`) under the MIT license
  (Copyright (c) Facebook, Inc. and its affiliates). This package depends on faiss-cpu and does not include its code.
- emeans-rocm (optional, not a dependency) packages Meta's e-means code and is distributed under the MIT license.

## Citation

If you use e-means, cite the paper:

```bibtex
@misc{bessa2026emeans,
  title  = {{e-means}: outperforming {k-means} (in one-tenth the epochs)},
  author = {Bessa, Swann and Mazar{\'e}, Pierre-Emmanuel and Szilvasy, Gergely and
            Touvron, Hugo and Picard, David and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  year   = {2026},
  note   = {Preprint},
  url    = {https://arxiv.org/abs/2611.09501}
}
```
