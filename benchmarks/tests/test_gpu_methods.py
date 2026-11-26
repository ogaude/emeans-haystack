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
    from haystack_integrations.components.preprocessors.emeans._backend import emeans_rocm_enabled  # noqa: PLC0415

    return emeans_rocm_enabled()


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


def _run_worker_in_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, name: str, info: dict
) -> dict:
    import worker  # noqa: PLC0415
    from methods import MethodSpec, TrainResult  # noqa: PLC0415

    real = METHODS[name]

    def train(x: np.ndarray, init: np.ndarray, **kwargs: object) -> TrainResult:  # noqa: ARG001
        return TrainResult(init.astype(np.float32), 1.0, dict(info))

    fake = MethodSpec(name, real.engine, real.device, real.default_passes, train, lambda: None)
    monkeypatch.setattr(worker, "METHODS", {**METHODS, name: fake})
    monkeypatch.setattr(worker, "_transfer_seconds", lambda x: 0.001)  # noqa: ARG005
    monkeypatch.setattr(worker, "_gpu_versions", lambda: {"torch": None, "hip": None})
    monkeypatch.setattr(worker, "_gpu_name", lambda: "fake gpu")
    argv = ["worker.py", "--method", name, "--data", "random", "--n", "500", "--d", "8", "--k", "4", "--seed", "0"]
    monkeypatch.setattr(sys, "argv", [*argv, "--scratch", str(tmp_path)])
    assert worker.main() == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_worker_record_uses_fit_engine_label(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    info = {"emeans_source": "faiss-cpu 0.0.1-fit", "train_seconds": 0.1}
    rec = _run_worker_in_process(monkeypatch, capsys, tmp_path, "emeans-gpu-triton-bf16", info)
    assert rec["versions"]["emeans_source"] == "faiss-cpu 0.0.1-fit"
    assert rec["info"]["emeans_source"] == "faiss-cpu 0.0.1-fit"


def test_worker_record_lloyd_keeps_no_engine(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rec = _run_worker_in_process(monkeypatch, capsys, tmp_path, "faiss-lloyd", {"train_seconds": 0.1})
    assert rec["versions"]["emeans_source"] is None


def test_worker_record_falls_back_to_env_label(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import worker  # noqa: PLC0415

    rec = _run_worker_in_process(monkeypatch, capsys, tmp_path, "emeans-cpu", {"train_seconds": 0.1})
    assert rec["versions"]["emeans_source"] == worker._versions("emeans-cpu")["emeans_source"]
    assert rec["versions"]["emeans_source"] is not None


def test_gpu_reason_refuses_out_of_range_emeans_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    from haystack_integrations.components.preprocessors.emeans import _backend  # noqa: PLC0415

    import methods  # noqa: PLC0415
    import precision  # noqa: PLC0415

    class FakeTorch:
        class cuda:  # noqa: N801
            @staticmethod
            def is_available() -> bool:
                return True

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    # Patch the inputs of the range gate, not the gate itself, so the version check really runs.
    monkeypatch.setattr(_backend, "emeans_rocm_importable", lambda: True)
    monkeypatch.setattr(_backend, "_emeans_rocm_is_packaged", lambda: True)
    monkeypatch.setattr(_backend, "_dist_version", lambda _dist: "0.3.0")
    monkeypatch.setattr(_backend, "_UNSUPPORTED_WARNED", set())
    assert _backend.emeans_rocm_enabled() is False
    assert methods._gpu_reason(want_emeans_rocm=True) == "needs the bench-gpu-fork env (emeans-rocm installed)"
    assert methods._gpu_reason(want_emeans_rocm=False) is None
    refusal = precision._gpu_refusal()
    assert refusal is not None
    assert "emeans-rocm 0.3.0 is outside the supported range" in refusal
    # The study refuses to run, so the refusal must not also claim a fallback to the stock engine.
    assert "stock" not in refusal


def test_gpu_reason_accepts_in_range_emeans_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    from haystack_integrations.components.preprocessors.emeans import _backend  # noqa: PLC0415

    import methods  # noqa: PLC0415

    class FakeTorch:
        class cuda:  # noqa: N801
            @staticmethod
            def is_available() -> bool:
                return True

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(_backend, "emeans_rocm_importable", lambda: True)
    monkeypatch.setattr(_backend, "_emeans_rocm_is_packaged", lambda: True)
    monkeypatch.setattr(_backend, "_dist_version", lambda _dist: "0.2.5")
    assert methods._gpu_reason(want_emeans_rocm=True) is None
    assert methods._gpu_reason(want_emeans_rocm=False) == "needs the bench-gpu env (stock e-means)"
