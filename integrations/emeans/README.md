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
`faiss.contrib.e_means`), `numpy>=1.25`.

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
https://github.com/ogaude/e-means-rocm) is the author's package of Meta's e-means code under its MIT license. When it
is installed, the GPU backends use it on every GPU, AMD and NVIDIA. Its NVIDIA path has not been re-run on hardware,
so NVIDIA users who want the verified path leave it uninstalled. On AMD, if its Triton backend fails the start-up
probe (for example, no Triton in your ROCm PyTorch), fits run in native FP32, and a warning (once per process) and
`stats["fallback_reason"]` give the probe error. If every one of its GPU backends fails the probe on your GPU, the
stock faiss-cpu e-means engine is tried on the GPU in native FP32 before falling back to the CPU; a warning and
`stats["fallback_reason"]` name the backends that failed. CPU fits always use the faiss-cpu e-means. The fitter's
`stats["emeans_source"]` names the package and version that ran, for example `"emeans-rocm 0.2.5"` or
`"faiss-cpu 1.16.0"`, and `stats["backend"]` which backend ran.

Importing the package never imports PyTorch. PyTorch is loaded only when a GPU backend is considered.

## Benchmarks

- e-means needs fewer passes than Lloyd's k-means (`faiss.Kmeans`, scikit-learn `KMeans`) to reach the same objective in all 5 sweep comparisons (k = 4096 and 16384); its passes are not faster, and at its default 5 passes Lloyd's k-means keeps the better objective in 3 settings (below). On 1M Cohere embeddings (768-d) with k = 16384 (pass-sweep medians), CPU e-means after 5 passes (objective 0.18569) beats `faiss.Kmeans` after 25 passes (0.18746); at the 25-pass points a pass costs about the same (114 vs 111 s/pass), so 575 s vs 2765 s.
- On an AMD RX 9060 XT with `emeans-rocm` 0.2.5, GPU e-means was faster than CPU e-means cold in all 16 settings and warm in all 12 that have a CPU warm time. FP32 against FP32 at 1M x 65536 on Cohere (cold, end-to-end, CPU 2232 s): native FP32 31.1x, Triton FP32 27.6x; Triton BF16 on the GPU against FP32 on the CPU: 207.7x (10.7 s). The smallest gain is 3.9x at random 100k x 1024.
- On real data (Cohere) without a GPU, CPU e-means finishes sooner than Lloyd's k-means with an objective within 0.5 % in 6 of 7 comparable settings; the exception is the degenerate 100k x 65536 (1.5 rows per centroid), where Lloyd's k-means has the better objective (e-means +22.22 %). Over both datasets it is 11 of 14 comparable settings (2 have no Lloyd run); on random data Lloyd's k-means is better at 100k x 16384 (e-means +7.90 %, reported as found; Cohere at the same size: -0.69 %) and at the degenerate 100k x 65536 (+36.81 %).
- On real data, FP16/BF16 objectives are between -0.121 % and +0.108 % of FP32, but at about 100k rows they change document assignments almost as much as reseeding does: use FP32 when stable `cluster_id`s matter.
- Full tables, protocol, every cut made for time or memory and the settings not run: [benchmarks/RESULTS.md](../../benchmarks/RESULTS.md). Random data (Gaussian noise, no cluster structure) is used for speed; its objectives appear here only as the exceptions noted above.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

GPU tests are marked `integration` and `gpu`, so `hatch run test:unit` and CI never run them. They are local
only: run them with `hatch run gpu:gpu-tests` (upstream e-means) or `hatch run gpu-fork:gpu-tests` (installs
`emeans-rocm` 0.2.5 from PyPI). They skip when PyTorch sees no GPU.
