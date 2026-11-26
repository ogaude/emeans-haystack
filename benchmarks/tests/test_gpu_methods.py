# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from data import starts
from methods import METHODS

HERE = Path(__file__).resolve().parent.parent
GPU_METHODS = ("emeans-gpu-native-fp32", "emeans-gpu-triton-fp32", "emeans-gpu-triton-bf16")
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _gpu_visible() -> bool:
    if not HAS_TORCH:
        return False
    torch = importlib.import_module("torch")
    return bool(torch.cuda.is_available())


def _emeans_rocm() -> bool:
    from haystack_integrations.components.preprocessors.emeans._backend import emeans_rocm_importable  # noqa: PLC0415

    return emeans_rocm_importable()


requires_gpu = pytest.mark.skipif(not _gpu_visible(), reason="needs torch with a visible GPU")
requires_stock = pytest.mark.skipif(
    not _gpu_visible() or _emeans_rocm(), reason="needs the bench-gpu env (stock e-means)"
)
requires_emeans_rocm = pytest.mark.skipif(
    not _gpu_visible() or not _emeans_rocm(), reason="needs the bench-gpu-fork env (emeans-rocm installed)"
)


def _data() -> tuple[np.ndarray, np.ndarray]:
    x = np.random.default_rng(0).standard_normal((20_000, 64)).astype(np.float32)
    return x, starts(x, 64, 0)


@pytest.mark.gpu
@requires_stock
def test_native_fp32_cell() -> None:
    x, init = _data()
    res = METHODS["emeans-gpu-native-fp32"].train(x, init, k=64, spherical=False, seed=0, passes=None)
    assert res.centroids.dtype == np.float32
    assert res.centroids.shape == (64, 64)
    assert res.info["backend"] == "gpu-pytorch"
    assert res.info["precision_used"] == "fp32"
    assert res.info["n_train"] == 20_000
    assert res.passes_used == 5.0


@pytest.mark.gpu
@requires_emeans_rocm
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_triton_cells(precision: str) -> None:
    x, init = _data()
    res = METHODS[f"emeans-gpu-triton-{precision}"].train(x, init, k=64, spherical=False, seed=0, passes=None)
    assert res.centroids.shape == (64, 64)
    assert res.info["backend"] == "gpu-triton"
    assert res.info["precision_used"] == precision


@pytest.mark.gpu
@requires_stock
def test_wrong_env_fails_loudly() -> None:
    x, init = _data()
    with pytest.raises(RuntimeError, match="gpu-triton"):
        METHODS["emeans-gpu-triton-fp32"].train(x, init, k=64, spherical=False, seed=0, passes=None)


def test_available_reasons() -> None:
    for name in GPU_METHODS:
        spec = METHODS[name]
        assert spec.device == "gpu"
        assert spec.engine == "e-means"
        reason = spec.available()
        if not HAS_TORCH:
            assert reason is not None
            assert "torch" in reason
        elif not _gpu_visible():
            assert reason is not None
        elif name.startswith("emeans-gpu-triton-"):
            if _emeans_rocm():
                assert reason is None
            else:
                assert reason is not None
                assert "bench-gpu-fork" in reason
        elif _emeans_rocm():
            assert reason is not None
            assert "bench-gpu env" in reason
        else:
            assert reason is None


def test_versions_record_emeans_source() -> None:
    from haystack_integrations.components.preprocessors.emeans._backend import detect_emeans_source  # noqa: PLC0415

    import worker  # noqa: PLC0415

    source = worker._versions()["emeans_source"]
    assert source == detect_emeans_source()
    if _emeans_rocm():
        assert source is not None
        assert source.startswith("emeans-rocm ")
    else:
        assert source is not None
        assert source.startswith("faiss-cpu ")


def test_versions_record_engine_per_method() -> None:
    from haystack_integrations.components.preprocessors.emeans._backend import (  # noqa: PLC0415
        _cpu_source,
        detect_emeans_source,
    )

    import worker  # noqa: PLC0415

    # CPU e-means always runs the stock faiss-cpu engine, whatever env the cell runs in.
    assert worker._versions("emeans-cpu")["emeans_source"] == _cpu_source()
    # Lloyd baselines run no e-means engine.
    assert worker._versions("faiss-lloyd")["emeans_source"] is None
    assert worker._versions("sklearn-lloyd")["emeans_source"] is None
    for method in GPU_METHODS:
        assert worker._versions(method)["emeans_source"] == detect_emeans_source()


@pytest.mark.gpu
@requires_gpu
def test_worker_gpu_fields(tmp_path: Path) -> None:
    method = "emeans-gpu-triton-bf16" if _emeans_rocm() else "emeans-gpu-native-fp32"
    cmd = [
        sys.executable, "-I", str(HERE / "worker.py"), "--method", method, "--data", "random", "--n", "20000",
        "--d", "64", "--k", "64", "--seed", "0", "--warm", "yes", "--scratch", str(tmp_path),
    ]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=HERE)
    rec = json.loads(out.stdout.strip().splitlines()[-1])
    assert rec["transfer_s"] > 0
    assert rec["cold_s"] > 0
    assert rec["warm_s"] > 0
    assert rec["info"]["train_seconds"] > 0
    assert rec["host"]["gpu"]
    assert rec["versions"]["torch"]
    assert rec["versions"]["hip"]
    assert rec["info"]["triton_cache"] in ("persistent", "cleared")
