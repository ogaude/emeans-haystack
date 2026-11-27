# Changelog

## 0.1.1

- The fitter now passes an explicit `unused_threshold` (0.1, or 0.0 when d <= 2) to every e-means engine: stock faiss-cpu, emeans-rocm, the GPU probe and the assigners. emeans-rocm 0.2.6 changed its default to a cost-based threshold; the explicit value keeps the mass-based revival of 0.2.5 and stock faiss.
- `stats["unused_threshold"]` reports the value used.
- emeans-rocm 0.2.6 is tested; the supported range stays `>=0.2.5,<0.3`. The local GPU envs pin `emeans-rocm==0.2.6`.
- Benchmarks were rerun on emeans-rocm 0.2.6 (see `benchmarks/RESULTS.md`). GPU centroids at d > 2 differ slightly from those on 0.2.5, because 0.2.6 places revived centroids differently.
- The public API and the `.npz` model format are unchanged.

## 0.1.0

First release.
