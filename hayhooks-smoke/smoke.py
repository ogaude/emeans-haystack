# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""
Serve the e-means clusterer and ranker with Hayhooks on CPU and call them over HTTP.

The script fits a small model on synthetic embeddings, serializes two single-component pipelines with
``Pipeline.dumps()``, deploys them to a local Hayhooks server as pipeline wrappers (the server rebuilds them with
``Pipeline.loads()``, which exercises ``to_dict``/``from_dict``), and checks the JSON responses. Nothing is
downloaded and the server is always stopped on exit.

The wrappers are used instead of ``hayhooks pipeline deploy-yaml`` because the YAML endpoint of Hayhooks 2.5.0 hands
``list[Document]`` inputs to the components as plain dicts.
"""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from haystack import Document, Pipeline
from haystack_integrations.components.preprocessors.emeans import EmeansDocumentClusterer, EmeansFitter
from haystack_integrations.components.rankers.emeans import EmeansClusterRanker

N_DOCS = 600
DIM = 32
N_CENTERS = 8
TOP_K = 5
STARTUP_BUDGET_S = 60.0
HTTP_TIMEOUT_S = 60.0
HTTP_OK = 200
HTTP_MULTIPLE_CHOICES = 300
MIN_CLUSTERS = 2


def make_documents() -> list[Document]:
    """Build deterministic documents with embeddings drawn around well separated centers."""
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=10.0, size=(N_CENTERS, DIM))
    labels = rng.integers(0, N_CENTERS, size=N_DOCS)
    points = centers[labels] + rng.normal(scale=0.5, size=(N_DOCS, DIM))
    return [Document(content=f"doc {i}", embedding=points[i].astype(float).tolist()) for i in range(N_DOCS)]


WRAPPER_TEMPLATE = """\
from typing import Any

from haystack import Document, Pipeline
from hayhooks import BasePipelineWrapper

PIPELINE_YAML = {yaml_source!r}


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline.loads(PIPELINE_YAML)

    def run_api(self, documents: list[dict[str, Any]]{extra_param}) -> dict[str, Any]:
        docs = [Document.from_dict(doc) for doc in documents]
        result = self.pipeline.run({{"{component}": {{"documents": docs{extra_arg}}}}})
        return {{"documents": [doc.to_dict(flatten=False) for doc in result["{component}"]["documents"]]}}
"""


def wrapper_files(pipeline: Pipeline, component: str, *, with_top_k: bool) -> dict[str, str]:
    """
    Return the deploy_files payload: a wrapper that rebuilds the pipeline with ``Pipeline.loads()``.

    The YAML deployment path of Hayhooks 2.5.0 passes ``list[Document]`` inputs to the pipeline as plain dicts, so the
    components receive dicts instead of documents. The wrapper converts them with ``Document.from_dict`` and still
    loads the pipeline from its serialized form, which exercises ``to_dict``/``from_dict`` of our components.
    """
    source = WRAPPER_TEMPLATE.format(
        yaml_source=pipeline.dumps(),
        component=component,
        extra_param=", top_k: int | None = None" if with_top_k else "",
        extra_arg=', "top_k": top_k' if with_top_k else "",
    )
    return {"pipeline_wrapper.py": source}


def free_port() -> int:
    """Return a free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    """Send a JSON request and return the status code and the parsed JSON body (or raw text)."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:  # noqa: S310
            status, body = response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read().decode()
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def wait_ready(process: subprocess.Popen[bytes], base: str, log_path: Path) -> None:
    """Poll /status until the server answers; fail when it exits early or the budget runs out."""
    deadline = time.monotonic() + STARTUP_BUDGET_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            msg = f"hayhooks exited early with code {process.returncode}"
            raise RuntimeError(msg)
        with contextlib.suppress(OSError):
            status, _ = http("GET", f"{base}/status")
            if status == HTTP_OK:
                return
        time.sleep(0.25)
    msg = f"hayhooks not ready after {STARTUP_BUDGET_S:.0f} s (log: {log_path})"
    raise TimeoutError(msg)


def stop(process: subprocess.Popen[bytes]) -> None:
    """Terminate the server process group, escalating to SIGKILL, and reap it."""
    if process.poll() is not None:
        return
    pgid = os.getpgid(process.pid)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        process.wait(timeout=15)


def deploy(base: str, name: str, files: dict[str, str]) -> None:
    """Deploy a wrapper pipeline over HTTP and require a 2xx answer."""
    status, body = http("POST", f"{base}/deploy_files", {"name": name, "files": files, "overwrite": True})
    if not HTTP_OK <= status < HTTP_MULTIPLE_CHOICES:
        msg = f"deploy of {name} failed: HTTP {status} {body}"
        raise RuntimeError(msg)


def check_cluster(base: str, docs: list[Document], k: int) -> list[dict[str, Any]]:
    """Call the clusterer endpoint and assert integer cluster ids; return the clustered documents."""
    payload = {"documents": [doc.to_dict(flatten=False) for doc in docs]}
    status, body = http("POST", f"{base}/emeans_cluster/run", payload)
    assert status == HTTP_OK, f"emeans_cluster/run: HTTP {status} {body}"  # noqa: S101
    out = body["result"]["documents"]
    assert len(out) == len(docs), f"expected {len(docs)} documents, got {len(out)}"  # noqa: S101
    ids = [doc["meta"]["cluster_id"] for doc in out]
    bad = [v for v in ids if type(v) is not int]
    assert not bad, f"cluster_id is not a JSON integer: {bad[:5]!r}"  # noqa: S101
    assert all(0 <= v < k for v in ids), f"cluster_id outside [0, {k})"  # noqa: S101
    assert len(set(ids)) >= MIN_CLUSTERS, "expected at least two distinct clusters"  # noqa: S101
    return out  # type: ignore[no-any-return]


def check_rank(base: str, clustered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Call the ranker endpoint with scored documents and assert one best document per cluster."""
    rng = np.random.default_rng(1)
    scored = [{**doc, "score": float(rng.random())} for doc in clustered]
    status, body = http("POST", f"{base}/emeans_rank/run", {"documents": scored, "top_k": TOP_K})
    assert status == HTTP_OK, f"emeans_rank/run: HTTP {status} {body}"  # noqa: S101
    out = body["result"]["documents"]
    distinct = {doc["meta"]["cluster_id"] for doc in scored}
    assert len(out) == min(TOP_K, len(distinct)), f"expected {min(TOP_K, len(distinct))} documents, got {len(out)}"  # noqa: S101
    out_ids = [doc["meta"]["cluster_id"] for doc in out]
    assert all(type(v) is int for v in out_ids), f"cluster_id is not a JSON integer: {out_ids!r}"  # noqa: S101
    assert len(set(out_ids)) == len(out_ids), f"more than one document for a cluster: {out_ids!r}"  # noqa: S101
    best: dict[int, dict[str, Any]] = {}
    for doc in scored:
        cid = doc["meta"]["cluster_id"]
        if cid not in best or doc["score"] > best[cid]["score"]:
            best[cid] = doc
    for doc in out:
        expected = best[doc["meta"]["cluster_id"]]["id"]
        assert doc["id"] == expected, f"cluster {doc['meta']['cluster_id']}: not the best-scored document"  # noqa: S101
    scores = [doc["score"] for doc in out]
    assert scores == sorted(scores, reverse=True), f"scores not in descending order: {scores!r}"  # noqa: S101
    return out  # type: ignore[no-any-return]


def run(tmp: Path) -> None:
    """Fit, serve, deploy and exercise both pipelines."""
    docs = make_documents()
    model_path = tmp / "model.npz"
    fit = EmeansFitter(k=N_CENTERS, device="cpu", model_path=str(model_path), overwrite=True).run(documents=docs)
    print(f"fitted model: {fit['model_path']} stats={fit['stats']}")  # noqa: T201

    cluster = Pipeline()
    cluster.add_component("clusterer", EmeansDocumentClusterer(model_path=str(model_path), device="cpu"))
    cluster_files = wrapper_files(cluster, "clusterer", with_top_k=False)
    rank = Pipeline()
    rank.add_component("ranker", EmeansClusterRanker(top_k=TOP_K))
    rank_files = wrapper_files(rank, "ranker", with_top_k=True)

    pipelines_dir = tmp / "pipelines"
    pipelines_dir.mkdir()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = tmp / "hayhooks.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(  # noqa: S603
            [
                str(Path(sys.executable).parent / "hayhooks"),
                "run",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--pipelines-dir",
                str(pipelines_dir),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ),
        )
        try:
            wait_ready(process, base, log_path)
            deploy(base, "emeans_cluster", cluster_files)
            deploy(base, "emeans_rank", rank_files)
            clustered = check_cluster(base, docs, int(fit["stats"].get("k", N_CENTERS)))
            print(f"emeans_cluster/run: OK, {len(clustered)} documents, integer cluster_ids")  # noqa: T201
            ranked = check_rank(base, clustered)
            print(  # noqa: T201
                f"emeans_rank/run: OK, {len(ranked)} documents, cluster_ids={[d['meta']['cluster_id'] for d in ranked]}"
            )
        except BaseException:
            log.flush()
            tail = log_path.read_text(errors="replace").splitlines()[-30:]
            print("--- hayhooks log tail ---\n" + "\n".join(tail), file=sys.stderr)  # noqa: T201
            raise
        finally:
            stop(process)


def main() -> int:
    """Run the smoke test and return the process exit code."""
    with tempfile.TemporaryDirectory(prefix="hayhooks-smoke-") as tmp:
        try:
            run(Path(tmp))
        except Exception as error:
            print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)  # noqa: T201
            return 1
    print(f"PASS (haystack-ai {version('haystack-ai')}, hayhooks {version('hayhooks')})")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
