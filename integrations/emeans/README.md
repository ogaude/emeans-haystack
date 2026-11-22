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
RX 9000 (gfx120x) GPUs, install the [e-means-rocm](https://github.com/ogaude/e-means-rocm) fork as described in its
README. It replaces the `faiss/contrib/e_means` folder from the wheel; upgrading `faiss-cpu` undoes that replacement.
The fitter's `stats["emeans_source"]` shows which copy was used, and `stats["backend"]` which backend ran.

Importing the package never imports PyTorch. PyTorch is loaded only when a GPU backend is considered.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

<!-- If integration tests need credentials or a running service, document it here. Otherwise, delete this comment. -->
