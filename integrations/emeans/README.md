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
so NVIDIA users who want the verified path leave it uninstalled. CPU fits always use the faiss-cpu e-means. The
fitter's `stats["emeans_source"]` names the package and version that ran, for example `"emeans-rocm 0.2.5"` or
`"faiss-cpu 1.16.0"`, and `stats["backend"]` which backend ran.

Importing the package never imports PyTorch. PyTorch is loaded only when a GPU backend is considered.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

GPU tests are marked `integration` and `gpu`, so `hatch run test:unit` and CI never run them. They are local
only: run them with `hatch run gpu:gpu-tests` (upstream e-means) or `hatch run gpu-fork:gpu-tests` (installs
`emeans-rocm` 0.2.5 from PyPI). They skip when PyTorch sees no GPU.
