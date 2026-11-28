# emeans-haystack

A [Haystack](https://haystack.deepset.ai) 3.x integration that exposes e-means as three components: a fitter that
trains it on document embeddings and saves a model, a clusterer that assigns a `cluster_id` to documents, and a
query-time ranker that keeps the best document of each cluster. e-means is a clustering algorithm from Meta FAIR
([paper](https://arxiv.org/abs/2611.09501)) that ships in faiss as `faiss.contrib.e_means`.

e-means is a different algorithm from k-means. Lloyd's k-means appears in this repository only as the baseline in the
benchmarks.

The package lives in [`integrations/emeans/`](integrations/emeans/README.md), and its README is the per-package
reference. It is kept identical to what would be contributed to haystack-core-integrations (see
[`UPSTREAM.md`](UPSTREAM.md)).

## Components

| Component | Haystack type | Import | Role |
|---|---|---|---|
| `EmeansFitter` | preprocessors | `haystack_integrations.components.preprocessors.emeans` | Fits e-means on document embeddings, saves a `.npz` model, returns `model_path` and `stats` |
| `EmeansDocumentClusterer` | preprocessors | `haystack_integrations.components.preprocessors.emeans` | Assigns `meta["cluster_id"]` to each document from a saved model |
| `EmeansClusterRanker` | rankers | `haystack_integrations.components.rankers.emeans` | Keeps the best-scored document of each cluster among retrieved documents (`top_k`) |

The ranker never clusters: it reads the `cluster_id` written at indexing time by the clusterer.

## Installation

```bash
pip install emeans-haystack                  # CPU: brings haystack-ai, faiss-cpu and numpy
```

For development, install from a clone instead: `pip install ./integrations/emeans`.

Requirements: Python 3.11 or newer, `haystack-ai>=3.5`, `faiss-cpu>=1.16.0` (the first release that contains
`faiss.contrib.e_means`), `numpy>=2.0`.

**GPU, NVIDIA:**

```bash
pip install "emeans-haystack[gpu]"           # adds PyTorch from PyPI (CUDA build)
```

**GPU, AMD (ROCm):** install the ROCm build of PyTorch first, from AMD's or PyTorch's ROCm index, then install the
package **without** `[gpu]`. The `[gpu]` extra pulls the PyPI build of PyTorch, which would replace your ROCm build and
leave you on CPU.

On AMD, upstream e-means runs only in native FP32 through plain PyTorch. For the fast Triton kernels and BF16/FP16 on
RX 9000 (gfx120x) GPUs, run `pip install emeans-rocm` after installing ROCm PyTorch; there is no extra for it.
`emeans-rocm` is optional and not a dependency. This package supports releases `>=0.2.5,<0.3`; the install line stays
unpinned. A release outside that range is not used: GPU fits run on the stock faiss-cpu e-means engine, a warning
(once per process) names the installed version and the range, and `stats["emeans_source"]` is `"faiss-cpu <version>"`.

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
`sentence-transformers-haystack`, sits in front. The full pipelines are in [`examples/`](examples/): 01 indexing,
02 diverse retrieval, 03 new documents, 04 GPU.

## Backend selection

`EmeansFitter` chooses where to run with two parameters:

- `device`: `"auto"` (default), `"cpu"` or `"cuda"`. With `"auto"`, a GPU is used only when `n * k` is at least
  `gpu_threshold` (default `1e10`). With `"cuda"` the fitter raises `RuntimeError` when no GPU backend is usable.
- `precision`: `"auto"` (default), `"fp32"`, `"bf16"` or `"fp16"`. It sets the number format of the nearest-centroid
  search on the GPU. On CPU everything is FP32.

CPU fits always use the faiss-cpu e-means. On a GPU, `emeans-rocm` is used when it is installed and in the supported
range (on any GPU, AMD or NVIDIA); otherwise the stock faiss e-means runs on PyTorch. If the Triton backend fails its
start-up probe, fits run in native FP32 and a warning plus `stats["fallback_reason"]` give the probe error. If every
GPU backend fails the probe, the stock engine is tried on the GPU in native FP32 before falling back to the CPU.

The fit statistics tell you what ran: `stats["backend"]`, `stats["emeans_source"]` (package and version, for example
`"faiss-cpu 1.16.0"`), `stats["fallback_reason"]` and `stats["unused_threshold"]`.

Use FP32 when stable `cluster_id`s matter: FP16/BF16 objectives stay close to FP32, but at about 100k rows they change
document assignments almost as much as reseeding does.

**Reproducible GPU fits:** `deterministic=False` is the default. With it, two GPU fits with the same `seed` can end in
different clusterings. emeans-rocm 0.2.6 sums FP32/BF16 batch statistics without atomic additions only at d ≥ 128; every
other path, including every d < 128, uses atomic additions. `deterministic=True` turns on
`torch.use_deterministic_algorithms(True)` for the duration of training and then restores the previous setting. The
setting is process-wide, so other GPU work in the process runs deterministically meanwhile. On NVIDIA, also export
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting the process. On CPU the parameter has no effect: CPU training is
already deterministic. Measured on an RX 9060 XT with emeans-rocm 0.2.6 (Triton, FP32, d = 768): an FP32 rerun with
`deterministic=True` gave identical centroids, assignments and objective in all 48 cells of the precision study
([benchmarks/RESULTS.md](benchmarks/RESULTS.md)). Runs without `deterministic=True`, and d < 128, were not measured.

## Benchmarks

**e-means needs fewer passes to reach the same objective. It is not faster per pass.**

- Fewer passes than Lloyd's k-means (`faiss.Kmeans`, scikit-learn `KMeans`) to reach the same objective in all 5 sweep
  comparisons (k = 4096 and 16384). On 1M Cohere embeddings (768-d) with k = 16384, CPU e-means after 5 passes
  (objective 0.18569) beats `faiss.Kmeans` after 25 passes (0.18746). A pass costs about the same (114 vs 111 s/pass),
  so the wall time is 575 s vs 2765 s.
- On an AMD RX 9060 XT with `emeans-rocm` 0.2.6, GPU e-means was faster than CPU e-means cold in all 16 settings. At
  1M x 65536 on Cohere, FP32 against FP32: native FP32 31.1x, Triton FP32 27.5x; Triton BF16 on the GPU against FP32 on
  the CPU: 187.0x. The smallest gain is 4.1x at random 100k x 1024.
- Lloyd's k-means wins some cells. At its default 5 passes, CPU e-means has the lower objective in 11 of 14 comparable
  main cells. Lloyd's k-means is better at Cohere 100k x 65536 (e-means +22.22 %), random 100k x 16384 (+7.90 %) and
  random 100k x 65536 (+36.81 %). The two 65536 cells are degenerate, with 1.5 rows per centroid.
- Per pass, CPU e-means is not faster: scikit-learn `KMeans` is cheaper per pass at 100k.
- Random data (Gaussian noise, no cluster structure) is used for speed only.

Every cut made for time or memory and every setting not run is listed in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) (see [Cuts and choices](benchmarks/RESULTS.md#cuts-and-choices) and
[Not run](benchmarks/RESULTS.md#not-run)). To reproduce, see [`benchmarks/README.md`](benchmarks/README.md).

## Examples

| File | What it shows |
|---|---|
| [`examples/01_index_corpus.py`](examples/01_index_corpus.py) | Embed a corpus, fit e-means, assign cluster ids and write to an `InMemoryDocumentStore` |
| [`examples/02_diverse_retrieval.py`](examples/02_diverse_retrieval.py) | Retrieve a diverse top 5, the best document from each cluster, with `EmeansClusterRanker` |
| [`examples/03_cluster_new_documents.py`](examples/03_cluster_new_documents.py) | Assign new documents to the clusters of an existing model with `EmeansDocumentClusterer` |
| [`examples/04_large_corpus_gpu.py`](examples/04_large_corpus_gpu.py) | Fit e-means on a large corpus on a GPU (synthetic 768-d embeddings) |
| [`examples/_shared.py`](examples/_shared.py) | Shared helpers: a small multi-topic corpus and the Quick Start indexing pipeline |

Examples 01 to 03 use `SentenceTransformersDocumentEmbedder`, which downloads a model on first use. Run them from
`integrations/emeans`:

```bash
cd integrations/emeans
hatch run examples:run ../../examples/01_index_corpus.py
```

Example 04 needs an environment with a GPU build of PyTorch:

```bash
hatch run gpu:run ../../examples/04_large_corpus_gpu.py        # or gpu-fork:run with emeans-rocm installed
```

## Repository layout

| Path | Content |
|---|---|
| [`integrations/emeans/`](integrations/emeans/) | The package, laid out to move unchanged into deepset-ai/haystack-core-integrations: `src/`, `tests/`, `pydoc/`, `pyproject.toml`, `CHANGELOG.md` |
| [`benchmarks/`](benchmarks/) | Benchmark scripts with their own `pyproject.toml`, the grid runner, committed results and [`RESULTS.md`](benchmarks/RESULTS.md) |
| [`examples/`](examples/) | Runnable pipelines built on the components |
| [`hayhooks-smoke/`](hayhooks-smoke/) | Local smoke test that serves the clusterer and ranker with Hayhooks on CPU, with its own `pyproject.toml` |
| [`UPSTREAM.md`](UPSTREAM.md) | Hand-off checklist and draft proposal for the upstream move |
| [`.github/workflows/emeans.yml`](.github/workflows/emeans.yml) | The CI workflow |
| [`LICENSE`](LICENSE) | Apache-2.0 |

## Development

All tooling runs through hatch, never `python` or `pip` directly. Run from `integrations/emeans`:

```bash
cd integrations/emeans
hatch run fmt-check        # ruff lint and format check
hatch run fmt              # ruff fix and format
hatch run test:types       # mypy on both component namespaces
hatch run test:unit        # unit tests
hatch run docs             # build the API docs
```

GPU tests are marked `integration` and `gpu`, so `hatch run test:unit` and CI never run them. They are local only:
`hatch run gpu:gpu-tests` (upstream e-means) or `hatch run gpu-fork:gpu-tests` (installs `emeans-rocm` 0.2.9 from
PyPI). They skip when PyTorch sees no GPU. The `gpu` env installs an exact ROCm build of torch from AMD's index; the
comment above `[tool.hatch.envs.gpu]` in `integrations/emeans/pyproject.toml` records the accepted risk of that setup.

### Hayhooks smoke test

`hayhooks-smoke/` checks that `EmeansDocumentClusterer` and `EmeansClusterRanker` work when served by Hayhooks 2.5.0
on CPU. It fits a small model on synthetic embeddings, starts a local Hayhooks server on a free `127.0.0.1` port,
deploys both components as pipeline wrappers that rebuild the pipeline with `Pipeline.loads()` (so `to_dict` and
`from_dict` are exercised), calls the run endpoints over HTTP and asserts that `cluster_id` comes back as a JSON
integer and that the ranker returns one best document per cluster. The wrappers are used because the YAML endpoint of
Hayhooks 2.5.0 passes `list[Document]` inputs to components as plain dicts. It is local only, not part of CI, and
downloads no model. It lives outside `integrations/emeans/` so the integration stays a clean copy for the upstream move.

```bash
cd hayhooks-smoke
hatch run smoke
hatch run lint
```

## CI

The workflow [`.github/workflows/emeans.yml`](.github/workflows/emeans.yml) is named `Test / emeans`.

- Triggers: pull requests, pushes to `main`, and a nightly schedule (`0 0 * * *`).
- Matrix: a push runs Python 3.11 only; pull requests and the schedule run `["3.11", "3.15"]`.
- Lint (`hatch run fmt-check && hatch run test:types`) runs on Python 3.11 only.
- Unit tests run with `hatch run test:unit-cov-retry`, plus a run against the lowest direct dependency versions.
- The nightly run also tests against Haystack `main`.
- A Build job runs `hatch build` and checks the distributions.

GPU tests cannot run on GitHub Actions. Actions are pinned to commit SHAs.

## License

`emeans-haystack` is distributed under the Apache-2.0 license (see [`LICENSE`](LICENSE)).

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
