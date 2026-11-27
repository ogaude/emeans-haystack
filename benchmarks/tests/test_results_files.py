# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""Data invariants of the committed emeans-rocm 0.2.6 follow-up results (phase 5)."""

import json
import re
import tomllib
from collections import Counter
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BENCH = Path(__file__).resolve().parents[1]
RESULTS = BENCH / "results"
OLD = "emeans-rocm 0.2.5"
NEW = "emeans-rocm 0.2.6"

# Timestamps are ISO 8601 with an explicit "+00:00" offset (UTC). The owner's machine runs on CET (UTC+1) in November
# 2026, so local time is UTC+1. A fixed offset avoids depending on a tz database; the quiet-hours rule is checked in
# both UTC and local time so that either reading passes only when the run was outside 23:00-07:00.
CET = timezone(timedelta(hours=1))


def _load(name: str) -> list[dict[str, Any]]:
    lines = (RESULTS / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _triton(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["method"].startswith("emeans-gpu-triton-")]


def _source(row: dict[str, Any]) -> str | None:
    versions = row.get("versions") or {}  # skipped records can carry versions: null
    return versions.get("emeans_source")


def test_bench_gpu_fork_env_pins_emeans_rocm_0_2_6_exactly() -> None:
    config = tomllib.loads((BENCH / "pyproject.toml").read_text(encoding="utf-8"))
    deps = config["tool"]["hatch"]["envs"]["bench-gpu-fork"]["extra-dependencies"]
    assert [d for d in deps if re.match(r"emeans-rocm\b", d)] == ["emeans-rocm==0.2.6"]


def test_speed_jsonl_holds_372_ok_triton_records_labelled_0_2_6() -> None:
    triton = _triton(_load("speed.jsonl"))
    assert len(triton) == 372
    assert {r["method"] for r in triton} == {"emeans-gpu-triton-fp32", "emeans-gpu-triton-bf16"}
    assert {_source(r) for r in triton} == {NEW}
    assert {r["status"] for r in triton} == {"ok"}
    assert {r["env"] for r in triton} == {"bench-gpu-fork"}


def test_speed_jsonl_triton_has_186_unique_cells_each_cold_and_warm() -> None:
    triton = _triton(_load("speed.jsonl"))
    keys = [r["key"] for r in triton]
    assert len(set(keys)) == 372  # no duplicate record
    timings: dict[str, set[str]] = {}
    for r in triton:
        cell = r["key"].rsplit("|", 1)[0]
        assert r["key"].endswith("|" + r["timing"])
        timings.setdefault(cell, set()).add(r["timing"])
    assert len(timings) == 186
    assert all(t == {"cold", "warm"} for t in timings.values())


def test_speed_jsonl_non_triton_records_keep_the_faiss_cpu_label() -> None:
    rows = [r for r in _load("speed.jsonl") if not r["method"].startswith("emeans-gpu-triton-")]
    native = [r for r in rows if r["method"] == "emeans-gpu-native-fp32"]
    cpu = [r for r in rows if r["method"] == "emeans-cpu"]
    assert len(native) == 186
    assert len(cpu) == 186
    assert {_source(r) for r in native + cpu} == {"faiss-cpu"}
    # no non-Triton record may claim an emeans-rocm version it was not run with
    assert not [r for r in rows if (_source(r) or "").startswith("emeans-rocm")]


def test_precision_jsonl_has_48_records_all_labelled_0_2_6() -> None:
    rows = _load("precision.jsonl")
    assert len(rows) == 48
    assert {_source(r) for r in rows} == {NEW}
    assert len({r["key"] for r in rows}) == 48
    gpu_variants = [v for r in rows for v in r["variants"].values() if v["backend"] == "gpu-triton"]
    assert gpu_variants
    assert {v["emeans_source"] for v in gpu_variants} == {NEW}


def test_baseline_0_2_5_files_stay_labelled_0_2_5() -> None:
    speed = _load("speed-emeans-rocm-0.2.5.jsonl")
    precision = _load("precision-emeans-rocm-0.2.5.jsonl")
    assert len(speed) == 372
    assert len(precision) == 48
    assert {_source(r) for r in speed + precision} == {OLD}
    assert {r["status"] for r in speed} == {"ok"}
    assert {r["method"] for r in speed} == {"emeans-gpu-triton-fp32", "emeans-gpu-triton-bf16"}


def test_version_ari_file_compares_0_2_5_against_0_2_6() -> None:
    rows = _load("version-ari-emeans-rocm-0.2.5-0.2.6.jsonl")
    assert len(rows) == 8
    assert Counter((r["old_source"], r["new_source"]) for r in rows) == {(OLD, NEW): 8}


def _stamps(rows: list[dict[str, Any]]) -> list[datetime]:
    out = []
    for r in rows:
        stamp = datetime.fromisoformat(r["timestamp"])
        assert stamp.tzinfo is not None, f"timestamp without offset: {r['timestamp']}"
        out.append(stamp)
    return out


def test_0_2_6_runs_stayed_outside_the_23_to_07_quiet_hours() -> None:
    stamps = _stamps(_triton(_load("speed.jsonl"))) + _stamps(_load("precision.jsonl"))
    assert len(stamps) == 372 + 48
    for stamp in stamps:
        for zone in (UTC, CET):
            hour = stamp.astimezone(zone).hour
            assert 7 <= hour < 23, f"{stamp.isoformat()} falls in quiet hours ({zone})"
