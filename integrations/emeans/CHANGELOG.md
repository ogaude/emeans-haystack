# Changelog

## 0.1.2

- The README now states the supported emeans-rocm range (`>=0.2.5,<0.3`) and what happens outside it: a release outside the range is not used, GPU fits run on the stock faiss-cpu e-means engine (on AMD in native FP32), a warning (once per process) names the installed version and the range, and `stats["emeans_source"]` is `"faiss-cpu <version>"`.
- Benchmark records now carry `unused_threshold` (benchmarks only, not the package).
- The README benchmark links are now absolute GitHub URLs, so they work on the PyPI page.
- First release on PyPI: `pip install emeans-haystack`. The README no longer asks to install from a clone.
- No code or API change: the public API and the `.npz` model format are unchanged.

## 0.1.1

- The package now passes an explicit `unused_threshold` (0.1, or 0.0 when d <= 2) to every e-means engine: stock faiss-cpu, emeans-rocm, the GPU probe and the assigners. emeans-rocm 0.2.6 changed its default to a cost-based threshold; the explicit value keeps the mass-based revival of 0.2.5 and stock faiss.
- `stats["unused_threshold"]` reports the value used.
- emeans-rocm 0.2.6 is tested; the supported range stays `>=0.2.5,<0.3`. The local GPU envs pin `emeans-rocm==0.2.6`.
- Benchmarks were rerun on emeans-rocm 0.2.6 (see `benchmarks/RESULTS.md`). Models saved with an earlier version load and assign unchanged, because assignment from saved centroids does not depend on the engine version. A refit on emeans-rocm 0.2.6 gives assignments (`cluster_id`s) that differ from a 0.2.5 fit about as much as a different seed does (on Cohere 100k, same-seed ARI between versions 0.27-0.40, between 0.2.5 seeds 0.25-0.39), because 0.2.6 places revived centroids differently at d > 2. Objectives stay close: within 1.342 %, lower on 0.2.6 in 71 of 96 main-cell records, largest at Cohere 100k x 65536 (1.5 rows per centroid).
- The public API and the `.npz` model format are unchanged.

## 0.1.0

First release.
