# Benchmark results

Generated 2026-11-27T14:46:09+00:00.
- host: rx9060xt
- CPU: Intel(R) Core(TM) Ultra 5 225
- cores: 10
- GPU: AMD Radeon RX 9060 XT
- torch: 2.14.0+rocm10.1.0
- HIP: 7.16.26385
- faiss: 1.16.0
- scikit-learn: 1.9.1
- e-means engine per method:
  - emeans-cpu: faiss-cpu
  - emeans-gpu-native-fp32: faiss-cpu
  - emeans-gpu-triton-bf16: emeans-rocm 0.2.6
  - emeans-gpu-triton-fp32: emeans-rocm 0.2.6

## Headline

- e-means needs fewer passes (epochs) than Lloyd's k-means (`faiss.Kmeans`, scikit-learn `KMeans`) to reach the same objective in all 5 sweep comparisons (k = 4096 and 16384). It is not faster per pass on CPU; the wall-time gain comes from the fewer passes. Exceptions at 5 passes in the main cells: see the line below.
- cohere 100k x 4096: CPU e-means reaches the objective `faiss.Kmeans` has after 25 passes (0.21433) within 5 passes (0.20978), pass ratio 5.0-12.5x; at the 25-pass points an e-means pass costs +16 % (3.42 vs 2.95 s/pass), at the 5 passes quoted e-means takes 3.71 s/pass (+26 % vs `faiss.Kmeans`); 18.5 s vs 73.7 s on the same CPU (pass-sweep medians; the main-cell runs give 17.1 s vs 73.2 s).
- cohere 1M x 16384: CPU e-means reaches the objective `faiss.Kmeans` has after 25 passes (0.18746) within 5 passes (0.18569), pass ratio 5.0-12.5x; at the 25-pass points a pass costs about the same (114 vs 111 s/pass), at the 5 passes quoted e-means takes 115 s/pass (+4 % vs `faiss.Kmeans`); 575 s vs 2765 s on the same CPU (pass-sweep medians; the main-cell runs give 578 s vs 2797 s).
- random 100k x 4096: CPU e-means reaches the objective `faiss.Kmeans` has after 25 passes (719.94) within 2 passes (716.73), pass ratio 12.5-25.0x; at the 25-pass points an e-means pass costs +13 % (3.08 vs 2.73 s/pass), at the 2 passes quoted e-means takes 3.18 s/pass (+17 % vs `faiss.Kmeans`); 6.36 s vs 68.1 s on the same CPU (pass-sweep medians).
- Per pass, CPU e-means is not faster than Lloyd: scikit-learn `KMeans` is cheaper per pass at 100k (cohere 100k x 4096: 3.42 vs 1.07 s/pass; random 100k x 4096: 3.08 vs 1.03 s/pass, 25-pass points), so the wall-time gain there is smaller than the pass gain.
- At cohere 1M x 16384, `faiss.Kmeans` used all 25 passes and was still improving (10 -> 25 passes: 0.18856 -> 0.18746), so 1M comparisons are against Lloyd at 25 passes, not at convergence.
- random 100k x 4096: after 5 passes the CPU e-means objective is flat and not monotone (5/10/25 passes: 713.15 / 713.67 / 713.45); reported as found.
- At its default 5 passes CPU e-means has a lower objective than the best Lloyd's k-means run in 11 of 14 comparable main cells; Lloyd is better at cohere 100k x 65536 (+22.22 %, degenerate: 1.5 rows per centroid), random 100k x 16384 (+7.90 %, reported as found; cohere at the same size: -0.69 %), random 100k x 65536 (+36.81 %, degenerate: 1.5 rows per centroid).
- GPU (AMD Radeon RX 9060 XT): the best GPU e-means run is faster than CPU e-means (cold in all 16 cells, warm in all 12 cells that have a CPU warm time); at cohere 1M x 65536 (cold, end-to-end, CPU e-means 2232 s), FP32 against FP32: `emeans-gpu-native-fp32` 31.1x (71.9 s), `emeans-gpu-triton-fp32` 27.5x (81.1 s); `emeans-gpu-triton-bf16` 187.0x (11.9 s), BF16 on the GPU against FP32 on the CPU; smallest gain 4.1x at random 100k x 1024 (cold: `emeans-gpu-triton-bf16` 1.28 s end-to-end vs 5.21 s).

## Protocol

- Every method trains the same k from the same starting centroids: a seeded sample of k distinct data rows.
- Passes per method: e-means `nepochs` 5 (the `EmeansFitter` default), `faiss.Kmeans` `niter` 25, scikit-learn `KMeans` `max_iter` 300, `tol` 1e-4, `n_init` 1. One e-means pass costs about the same distance evaluations as one Lloyd iteration, so the pass sweep compares equal work.
- No subsampling: `faiss.Kmeans` `max_points_per_centroid` 2**30, e-means `None`.
- scikit-learn runs at 100k rows only, to save time: `faiss-lloyd` is the Lloyd baseline at every size, so the 1M rows have no scikit-learn entries.
- Final objective: mean squared distance to the nearest centroid, from one exact FP32 CPU scorer (`faiss.IndexFlatL2`), never a library's own objective. The `vs faiss` column is the relative difference to the Lloyd `faiss-lloyd` median objective in the same cell.
- Real data is spherical. `faiss-lloyd` (`faiss.Kmeans(spherical=True)`) normalises its centroids to unit length after every iteration: it is the true spherical Lloyd baseline, and spherical comparisons should be read against it. scikit-learn has no spherical mode: `sklearn-lloyd` runs Euclidean Lloyd on the unit-norm rows and projects its final centroids to unit length once, after training, so it is scored on the sphere but is not spherical Lloyd (no projection per iteration). The Cohere file's rows are not unit-norm (raw L2 norms about 12 to 17.5): the harness normalises every row to unit length when it loads them. Random data is Euclidean and is for speed only: Gaussian noise has no cluster structure.
- The real 100k set is a seeded, sorted random subset of the 1M Cohere rows: seed 20261105 (`COHERE_SUBSET_SEED` in `data.py`), row-index sha256 `19fd6db7de0e1f3c90a6bf0a4ad6e370a4a92fe07bbaa17b590018286ca6ac72`. Per record, 216 of the 594 real-data speed records carry the subset seed, row-index sha256 and raw norm range (`emeans-gpu-triton-bf16`, `emeans-gpu-triton-fp32`), and so do all 48 precision records. The `emeans-cpu`, `emeans-gpu-native-fp32`, `faiss-lloyd`, `sklearn-lloyd` records predate these fields.
- Cold: the first run in a fresh process, including the backend probe and Triton JIT with the persistent Triton cache (recorded per row). Warm: a repeat in the same process. Host-to-device transfer is timed separately and subtracted; GPU end-to-end below adds it back.
- Time cap: a cell whose probe estimate exceeds the cap is listed as `not run: est. X h`, never dropped; a worker past its hard timeout is listed as `timed out after X h`.
- 100k x 65536 is degenerate: about 1.5 points per centroid.
- Lloyd's k-means is the baseline here; e-means is not k-means.
- Marks are computed from the data. `GPU does not help`: the best GPU e-means end-to-end median is not below the CPU e-means median. `CPU e-means wins`: CPU e-means is strictly faster than every other method (GPU end-to-end) and its objective is within OBJECTIVE_TOLERANCE = 0.005 (0.5 %) of the best Lloyd median. A `(cold)` suffix means the cold timings were compared.

## Passes to the same objective

The measured advantage of e-means is fewer passes (epochs) to reach the same objective, not faster passes (per-pass cost below). Target: the Lloyd's k-means median objective at its largest swept pass count (25 requested; passes actually used shown). `e-means passes`: the fewest swept CPU e-means passes whose median objective is at or below the target. Swept points are 1, 2, 5, 10, 25, nothing is interpolated, so the pass ratio is a range: e-means got there after the previous point and by this one. `one step earlier` is the previous point's distance to the target. `Lloyd to match e-means at 5`: the fewest swept Lloyd passes reaching CPU e-means' objective at its default 5 passes. Seconds are cold medians on the same CPU; s/pass is taken from each method's 25-pass point.

Per-pass cost of CPU e-means relative to Lloyd's k-means (same CPU, cold medians):

- sweep, 25-pass points, cohere 100k x 4096 vs `faiss-lloyd`: +15.9 % (3.42 vs 2.95 s/pass)
- sweep, 25-pass points, cohere 100k x 4096 vs `sklearn-lloyd`: 3.2x (3.42 vs 1.07 s/pass)
- sweep, 25-pass points, cohere 1M x 16384 vs `faiss-lloyd`: +3.5 % (114 vs 111 s/pass)
- sweep, 25-pass points, random 100k x 4096 vs `faiss-lloyd`: +12.9 % (3.08 vs 2.73 s/pass)
- sweep, 25-pass points, random 100k x 4096 vs `sklearn-lloyd`: 3.0x (3.08 vs 1.03 s/pass)
- main cells, k = 1024, vs `faiss-lloyd`: +47 % to +56 % (4 cells)
- main cells, k = 4096, vs `faiss-lloyd`: +13 % to +17 % (4 cells)
- main cells, k = 16384, vs `faiss-lloyd`: +3 % to +7 % (4 cells)
- main cells, k = 65536, vs `faiss-lloyd`: -4 % to +3 % (2 cells)
- main cells, k = 1024, vs `sklearn-lloyd`: 4.1x to 4.6x (2 cells)
- main cells, k = 4096, vs `sklearn-lloyd`: 3.3x (2 cells)
- main cells, k = 16384, vs `sklearn-lloyd`: 3.0x to 3.1x (2 cells)
- main cells, k = 65536, vs `sklearn-lloyd`: 2.6x to 3.0x (2 cells)

| data | n | k | Lloyd method | Lloyd passes used | Lloyd objective | Lloyd s/pass | Lloyd s | e-means passes | e-means objective | e-means s/pass | e-means s | pass ratio | time ratio | one step earlier | Lloyd to match e-means at 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cohere | 100k | 4096 | faiss-lloyd | 25 | 0.21433 | 2.95 | 73.7 | 5 | 0.20978 | 3.42 | 18.5 | 5.0-12.5x | 4.0x | 2 passes: +0.07 % | not within 25 |
| cohere | 100k | 4096 | sklearn-lloyd | 25 | 0.21448 | 1.07 | 26.8 | 2 | 0.21447 | 3.42 | 7.07 | 12.5-25.0x | 3.8x | 1 pass: +5.28 % | not within 25 |
| cohere | 1M | 16384 | faiss-lloyd | 25 | 0.18746 | 111 | 2765 | 5 | 0.18569 | 114 | 575 | 5.0-12.5x | 4.8x | 2 passes: +0.79 % | not within 25 |
| random | 100k | 4096 | faiss-lloyd | 25 | 719.94 | 2.73 | 68.1 | 2 | 716.73 | 3.08 | 6.36 | 12.5-25.0x | 10.7x | 1 pass: +0.78 % | not within 25 |
| random | 100k | 4096 | sklearn-lloyd | 25 | 719.94 | 1.03 | 25.8 | 2 | 716.73 | 3.08 | 6.36 | 12.5-25.0x | 4.1x | 1 pass: +0.78 % | not within 25 |

## e-means at 5 passes vs Lloyd's k-means, per main cell

CPU e-means at its default 5 passes against the Lloyd's k-means run (`faiss.Kmeans`, scikit-learn `KMeans`) with the lowest median objective in the cell, at the passes it used (medians over seeds). scikit-learn stops on `tol` 1e-4 (converged); `faiss.Kmeans` runs at most 25 passes and may stop earlier. Cold medians on the same CPU. `CPU e-means wins (CPU only)`: without a GPU, CPU e-means is faster than every CPU Lloyd run and its objective is within 0.5 % of the best Lloyd median.

| data | mode | n | k | rows per centroid | e-means objective | e-means s | e-means s/pass | best Lloyd | Lloyd passes | Lloyd objective | Lloyd s | Lloyd s/pass | e-means vs Lloyd | mark |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cohere | spherical | 100k | 1024 | 97.7 | 0.23717 | 5.63 | 1.13 | sklearn-lloyd | converged (tol 1e-4) after 71 | 0.23913 | 17.4 | 0.245 | -0.82 % | CPU e-means wins (CPU only) |
| cohere | spherical | 100k | 4096 | 24.4 | 0.20978 | 17.1 | 3.42 | faiss-lloyd | ran all 25 | 0.21433 | 73.2 | 2.93 | -2.12 % | CPU e-means wins (CPU only) |
| cohere | spherical | 100k | 16384 | 6.1 | 0.16549 | 61.5 | 12.3 | faiss-lloyd | stopped after 20 of 25 | 0.16664 | 230 | 11.5 | -0.69 % | CPU e-means wins (CPU only) |
| cohere | spherical | 100k | 65536 | 1.5 | 0.066174 | 233 | 46.6 | faiss-lloyd | stopped after 6 of 25 | 0.054142 | 272 | 45.4 | +22.22 % |  |
| cohere | spherical | 1M | 1024 | 976.6 | 0.23943 | 51.6 | 10.3 | faiss-lloyd | ran all 25 | 0.23995 | 175 | 7 | -0.22 % | CPU e-means wins (CPU only) |
| cohere | spherical | 1M | 4096 | 244.1 | 0.21664 | 156 | 31.3 | faiss-lloyd | ran all 25 | 0.21764 | 691 | 27.6 | -0.46 % | CPU e-means wins (CPU only) |
| cohere | spherical | 1M | 16384 | 61.0 | 0.18569 | 578 | 116 | faiss-lloyd | ran all 25 | 0.18746 | 2797 | 112 | -0.94 % | CPU e-means wins (CPU only) |
| cohere | spherical | 1M | 65536 | 15.3 | 0.15387 | 2232 | 446 | not run | - | - | - | - | - |  |
| random | euclidean | 100k | 1024 | 97.7 | 744 | 5.21 | 1.04 | sklearn-lloyd | converged (tol 1e-4) after 38 | 747.36 | 9.55 | 0.251 | -0.45 % | CPU e-means wins (CPU only) |
| random | euclidean | 100k | 4096 | 24.4 | 713.15 | 15.8 | 3.15 | sklearn-lloyd | converged (tol 1e-4) after 28 | 719.94 | 26.8 | 0.957 | -0.94 % | CPU e-means wins (CPU only) |
| random | euclidean | 100k | 16384 | 6.1 | 669.01 | 57.3 | 11.5 | sklearn-lloyd | converged (tol 1e-4) after 19 | 620 | 71.1 | 3.74 | +7.90 % |  |
| random | euclidean | 100k | 65536 | 1.5 | 336.13 | 217 | 43.4 | faiss-lloyd | stopped after 8 of 25 | 245.68 | 363 | 45.4 | +36.81 % |  |
| random | euclidean | 1M | 1024 | 976.6 | 754.51 | 54.9 | 11 | faiss-lloyd | ran all 25 | 755.16 | 180 | 7.2 | -0.09 % | CPU e-means wins (CPU only) |
| random | euclidean | 1M | 4096 | 244.1 | 747.51 | 158 | 31.6 | faiss-lloyd | ran all 25 | 749.74 | 701 | 28 | -0.30 % | CPU e-means wins (CPU only) |
| random | euclidean | 1M | 16384 | 61.0 | 732.03 | 557 | 111 | faiss-lloyd | ran all 25 | 737.29 | 2643 | 106 | -0.71 % | CPU e-means wins (CPU only) |
| random | euclidean | 1M | 65536 | 15.3 | 688.6 | 2140 | 428 | not run | - | - | - | - | - |  |

## Upstream claims, checked

Quoted from `faiss/contrib/e_means/README.md` in the faiss-cpu 1.16.0 wheel (square brackets mark words added here); the measured lines are computed from the tables on this page.

1. > On most [Lloyd's] k-means applications, e-means can achieve the same loss in about 10 times fewer epochs.

   - cohere 100k x 4096 vs `faiss-lloyd` at 25 passes: e-means within 5 passes, pass ratio 5.0-12.5x.
   - cohere 100k x 4096 vs `sklearn-lloyd` at 25 passes: e-means within 2 passes, pass ratio 12.5-25.0x.
   - cohere 1M x 16384 vs `faiss-lloyd` at 25 passes: e-means within 5 passes, pass ratio 5.0-12.5x.
   - random 100k x 4096 vs `faiss-lloyd` at 25 passes: e-means within 2 passes, pass ratio 12.5-25.0x.
   - random 100k x 4096 vs `sklearn-lloyd` at 25 passes: e-means within 2 passes, pass ratio 12.5-25.0x.

2. > A value of 5 is typically sufficient to beat Lloyd's loss at convergence.

   - against scikit-learn `KMeans` run to convergence (tol 1e-4; 100k rows): e-means at 5 passes is lower in 5 of 8 cells; higher at cohere 100k x 65536 +22.17 %, random 100k x 16384 +7.90 %, random 100k x 65536 +36.81 %.
   - against `faiss.Kmeans` at up to 25 passes: e-means at 5 passes is lower in 11 of 14 cells; higher at cohere 100k x 65536 +22.22 %, random 100k x 16384 +7.90 %, random 100k x 65536 +36.81 %.
   - Not checked (no Lloyd run, see Not run): cohere 1M x 65536, random 1M x 65536.

3. > e-means delivers a **~10x speedup in epochs and wall-clock** while achieving **lower loss** compared to Lloyd's k-means

   - CPU, cohere 100k x 4096: e-means reaches `faiss-lloyd`'s 25-pass objective 4.0x faster in wall time.
   - CPU, cohere 100k x 4096: e-means reaches `sklearn-lloyd`'s 25-pass objective 3.8x faster in wall time.
   - CPU, cohere 1M x 16384: e-means reaches `faiss-lloyd`'s 25-pass objective 4.8x faster in wall time.
   - CPU, random 100k x 4096: e-means reaches `faiss-lloyd`'s 25-pass objective 10.7x faster in wall time.
   - CPU, random 100k x 4096: e-means reaches `sklearn-lloyd`'s 25-pass objective 4.1x faster in wall time.
   - Per pass, CPU e-means is not faster than Lloyd (see the s/pass columns); the wall-time gain comes from fewer passes.

4. > GPUs are much faster, so we recommend using them for configurations where n x k > 10^10

   - On this host the GPU already helps below 1e10: smallest gain below 1e10 is 4.1x at random 100k x 1024 (cold: `emeans-gpu-triton-bf16` 1.28 s end-to-end vs 5.21 s).

## Main results

Cold and warm are `median [min-max]` seconds over seeds.

| data | mode | n | k | n x k | method | backend | precision | passes used | cold s | warm s | transfer s | objective | vs faiss | marks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cohere | spherical | 100000 | 1024 | 1.0e8 | emeans-cpu | cpu | fp32 | 5 | 5.63 [5.61-5.66] | 6.4 [5.8-6.49] | - | 0.23717 (n=3) | -0.83 % |  |
| cohere | spherical | 100000 | 1024 | 1.0e8 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 2.19 [2.19-2.2] | 0.269 [0.269-0.27] | 0.0128 | 0.23719 (n=3) | -0.82 % |  |
| cohere | spherical | 100000 | 1024 | 1.0e8 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.3 [1.29-1.31] | 0.0686 [0.0683-0.069] | 0.0124 | 0.23713 (n=3) | -0.84 % |  |
| cohere | spherical | 100000 | 1024 | 1.0e8 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 1.35 [1.34-2.36] | 0.181 [0.181-0.184] | 0.0153 | 0.23717 (n=3) | -0.83 % |  |
| cohere | spherical | 100000 | 1024 | 1.0e8 | faiss-lloyd | lloyd | fp32 | 25 | 18.3 [18.2-19.2] | 19.7 [18.3-20.7] | - | 0.23915 (n=3) | +0.00 % |  |
| cohere | spherical | 100000 | 1024 | 1.0e8 | sklearn-lloyd | lloyd | fp32 | 71 | 17.4 [15.7-18.7] | 17.9 [16.4-18.1] | - | 0.23913 (n=3) | -0.01 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | emeans-cpu | cpu | fp32 | 5 | 17.1 [17.1-17.5] | 17 [17-18.6] | - | 0.20978 (n=3) | -2.12 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 2.59 [2.51-2.6] | 0.628 [0.56-0.628] | 0.0157 | 0.20985 (n=3) | -2.09 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.34 [1.33-1.35] | 0.148 [0.147-0.148] | 0.0156 | 0.20971 (n=3) | -2.15 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 1.79 [1.76-1.79] | 0.633 [0.632-0.636] | 0.0157 | 0.20975 (n=3) | -2.14 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | faiss-lloyd | lloyd | fp32 | 25 | 73.2 [72.7-74] | 74.3 [74.2-75.4] | - | 0.21433 (n=3) | +0.00 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | sklearn-lloyd | lloyd | fp32 | 35 | 35.8 [32.6-37.8] | 35.7 [32.5-37.7] | - | 0.21448 (n=3) | +0.07 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | emeans-cpu | cpu | fp32 | 5 | 61.5 [61.3-62.2] | 62.5 [62.2-62.8] | - | 0.16549 (n=3) | -0.69 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 4.09 [4.06-4.12] | 2.05 [2.05-2.05] | 0.0192 | 0.16541 (n=3) | -0.74 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.56 [1.56-1.58] | 0.368 [0.368-0.37] | 0.0155 | 0.16535 (n=3) | -0.77 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 3.42 [3.41-3.43] | 2.35 [2.35-2.35] | 0.0176 | 0.16526 (n=3) | -0.83 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | faiss-lloyd | lloyd | fp32 | 20 | 230 [225-239] | 228 [221-238] | - | 0.16664 (n=3) | +0.00 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | sklearn-lloyd | lloyd | fp32 | 20 | 81.9 [78-90.4] | - | - | 0.16692 (n=3) | +0.17 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | emeans-cpu | cpu | fp32 | 5 | 233 [233-233] | 233 [232-235] | - | 0.066174 (n=3) | +22.22 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 9.7 [8.73-9.72] | 7.72 [6.75-7.73] | 0.0218 | 0.066463 (n=3) | +22.76 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 2.56 [2.55-3.25] | 1.4 [1.37-1.56] | 0.0274 | 0.066163 (n=3) | +22.20 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 12.2 [10.9-12.2] | 11.2 [9.84-11.4] | 0.0259 | 0.065477 (n=3) | +20.94 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | faiss-lloyd | lloyd | fp32 | 6 | 272 [227-321] | - | - | 0.054142 (n=3) | +0.00 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | sklearn-lloyd | lloyd | fp32 | 6 | 94.5 [63.9-95.6] | - | - | 0.054165 (n=3) | +0.04 % |  |
| cohere | spherical | 1000000 | 1024 | 1.0e9 | emeans-cpu | cpu | fp32 | 5 | 51.6 [51.6-52.2] | 51.7 [51.1-52.1] | - | 0.23943 (n=3) | -0.22 % |  |
| cohere | spherical | 1000000 | 1024 | 1.0e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 4.21 [4.21-4.26] | 2.42 [2.34-2.45] | 0.162 | 0.23947 (n=3) | -0.20 % |  |
| cohere | spherical | 1000000 | 1024 | 1.0e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 2.84 [2.81-2.88] | 0.383 [0.382-0.393] | 0.154 | 0.23949 (n=3) | -0.19 % |  |
| cohere | spherical | 1000000 | 1024 | 1.0e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 3.91 [3.31-4.16] | 1.36 [1.36-1.38] | 0.154 | 0.23947 (n=3) | -0.20 % |  |
| cohere | spherical | 1000000 | 1024 | 1.0e9 | faiss-lloyd | lloyd | fp32 | 25 | 175 [175-175] | 176 [176-176] | - | 0.23995 (n=3) | +0.00 % |  |
| cohere | spherical | 1000000 | 4096 | 4.1e9 | emeans-cpu | cpu | fp32 | 5 | 156 [156-157] | 156 [156-157] | - | 0.21664 (n=3) | -0.46 % |  |
| cohere | spherical | 1000000 | 4096 | 4.1e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 7.55 [6.82-7.61] | 5.73 [5.04-5.79] | 0.156 | 0.21668 (n=3) | -0.44 % |  |
| cohere | spherical | 1000000 | 4096 | 4.1e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 3.52 [3.39-3.56] | 0.796 [0.772-0.798] | 0.154 | 0.21663 (n=3) | -0.46 % |  |
| cohere | spherical | 1000000 | 4096 | 4.1e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 7.43 [7.39-7.71] | 4.81 [4.8-5.09] | 0.124 | 0.21663 (n=3) | -0.46 % |  |
| cohere | spherical | 1000000 | 4096 | 4.1e9 | faiss-lloyd | lloyd | fp32 | 25 | 691 [690-691] | - | - | 0.21764 (n=3) | +0.00 % |  |
| cohere | spherical | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-cpu | cpu | fp32 | 5 | 578 [571-580] | - | - | 0.18569 (n=3) | -0.94 % |  |
| cohere | spherical | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 20.7 [20.6-20.8] | 18.9 [18.8-18.9] | 0.129 | 0.18573 (n=3) | -0.93 % |  |
| cohere | spherical | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 5.44 [5.13-5.79] | 2.48 [2.48-2.53] | 0.161 | 0.18551 (n=3) | -1.04 % |  |
| cohere | spherical | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 22.3 [22.1-22.4] | 20 [19.8-20.3] | 0.157 | 0.18553 (n=3) | -1.03 % |  |
| cohere | spherical | 1000000 | 16384 | 1.6e10 (>= 1e10) | faiss-lloyd | lloyd | fp32 | 25 | 2797 [2773-2878] | - | - | 0.18746 (n=3) | +0.00 % |  |
| cohere | spherical | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-cpu | cpu | fp32 | 5 | 2232 [2230-2256] | - | - | 0.15387 (n=3) | - |  |
| cohere | spherical | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 71.8 [71.7-71.9] | 70.1 [70.1-70.1] | 0.133 | 0.15389 (n=3) | - |  |
| cohere | spherical | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 11.8 [11.6-12] | 9.16 [9.03-9.67] | 0.163 | 0.15346 (n=3) | - |  |
| cohere | spherical | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 81 [79.5-81.2] | 79 [77.4-79] | 0.168 | 0.15345 (n=3) | - |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | emeans-cpu | cpu | fp32 | 5 | 5.21 [5.18-5.22] | 5.1 [5.06-5.22] | - | 744 (n=3) | -0.45 % |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 2.17 [2.17-2.21] | 0.279 [0.263-0.28] | 0.0129 | 744.03 (n=3) | -0.45 % |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.27 [1.26-1.28] | 0.0648 [0.0648-0.065] | 0.0126 | 743.83 (n=3) | -0.47 % |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 1.35 [1.32-1.45] | 0.189 [0.186-0.189] | 0.0126 | 743.81 (n=3) | -0.48 % |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | faiss-lloyd | lloyd | fp32 | 25 | 16.7 [16-16.7] | 17 [17-17.1] | - | 747.37 (n=3) | +0.00 % |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | sklearn-lloyd | lloyd | fp32 | 38 | 9.55 [7.25-11.8] | 9.43 [7.3-11.7] | - | 747.36 (n=3) | -0.00 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | emeans-cpu | cpu | fp32 | 5 | 15.8 [15.5-16.4] | 15.8 [15.8-15.8] | - | 713.15 (n=3) | -0.94 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 2.55 [2.48-2.58] | 0.641 [0.583-0.643] | 0.0142 | 713.17 (n=3) | -0.94 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.35 [1.35-1.38] | 0.141 [0.141-0.142] | 0.0157 | 713.47 (n=3) | -0.90 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 1.81 [1.8-1.82] | 0.649 [0.645-0.653] | 0.0129 | 713.5 (n=3) | -0.89 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | faiss-lloyd | lloyd | fp32 | 25 | 67.5 [66.6-67.9] | 68.5 [67.9-69.1] | - | 719.94 (n=3) | +0.00 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | sklearn-lloyd | lloyd | fp32 | 28 | 26.8 [25.2-33.4] | 26.6 [25-33.1] | - | 719.94 (n=3) | -0.00 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | emeans-cpu | cpu | fp32 | 5 | 57.3 [55.6-57.3] | 57.4 [56.5-58] | - | 669.01 (n=3) | +7.90 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 4.13 [4.07-4.14] | 2.06 [2.06-2.06] | 0.0169 | 666.57 (n=3) | +7.51 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.57 [1.55-1.59] | 0.35 [0.349-0.362] | 0.0178 | 664.89 (n=3) | +7.24 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 3.52 [3.52-3.56] | 2.43 [2.42-2.45] | 0.0151 | 666.72 (n=3) | +7.54 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | faiss-lloyd | lloyd | fp32 | 20 | 215 [183-242] | 215 [183-238] | - | 620 (n=3) | +0.00 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | sklearn-lloyd | lloyd | fp32 | 19 | 71.1 [63.2-74.2] | - | - | 620 (n=3) | -0.00 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | emeans-cpu | cpu | fp32 | 5 | 217 [217-218] | 217 [217-218] | - | 336.13 (n=3) | +36.81 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 9.68 [9.68-9.7] | 7.67 [7.67-7.67] | 0.0218 | 335.72 (n=3) | +36.65 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 2.54 [2.49-3.14] | 1.32 [1.32-1.32] | 0.0266 | 335.22 (n=3) | +36.45 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 11.5 [11.4-11.5] | 10.3 [10.3-10.5] | 0.0276 | 334.9 (n=3) | +36.32 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | faiss-lloyd | lloyd | fp32 | 8 | 363 [300-381] | - | - | 245.68 (n=3) | +0.00 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | sklearn-lloyd | lloyd | fp32 | 7 | 118 [104-118] | - | - | 245.68 (n=3) | +0.00 % |  |
| random | euclidean | 1000000 | 1024 | 1.0e9 | emeans-cpu | cpu | fp32 | 5 | 54.9 [54.1-55.1] | 54.9 [54.9-56.2] | - | 754.51 (n=3) | -0.09 % |  |
| random | euclidean | 1000000 | 1024 | 1.0e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 4.38 [4.37-4.39] | 2.61 [2.59-2.61] | 0.158 | 754.51 (n=3) | -0.09 % |  |
| random | euclidean | 1000000 | 1024 | 1.0e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 3.26 [2.93-3.5] | 0.384 [0.381-0.467] | 0.157 | 754.5 (n=3) | -0.09 % |  |
| random | euclidean | 1000000 | 1024 | 1.0e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 3.53 [3.43-4.2] | 1.42 [1.31-1.46] | 0.153 | 754.5 (n=3) | -0.09 % |  |
| random | euclidean | 1000000 | 1024 | 1.0e9 | faiss-lloyd | lloyd | fp32 | 25 | 180 [177-181] | 179 [177-181] | - | 755.16 (n=3) | +0.00 % |  |
| random | euclidean | 1000000 | 4096 | 4.1e9 | emeans-cpu | cpu | fp32 | 5 | 158 [149-159] | 161 [152-162] | - | 747.51 (n=3) | -0.30 % |  |
| random | euclidean | 1000000 | 4096 | 4.1e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 7.69 [7.65-7.76] | 5.92 [5.92-5.93] | 0.156 | 747.49 (n=3) | -0.30 % |  |
| random | euclidean | 1000000 | 4096 | 4.1e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 3.24 [2.84-3.83] | 0.776 [0.77-0.806] | 0.154 | 747.41 (n=3) | -0.31 % |  |
| random | euclidean | 1000000 | 4096 | 4.1e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 7.75 [7.59-7.88] | 5.05 [5.04-5.15] | 0.124 | 747.42 (n=3) | -0.31 % |  |
| random | euclidean | 1000000 | 4096 | 4.1e9 | faiss-lloyd | lloyd | fp32 | 25 | 701 [689-715] | - | - | 749.74 (n=3) | +0.00 % |  |
| random | euclidean | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-cpu | cpu | fp32 | 5 | 557 [556-558] | - | - | 732.03 (n=3) | -0.71 % |  |
| random | euclidean | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 20.9 [20.8-21] | 19.1 [19.1-19.1] | 0.129 | 732.01 (n=3) | -0.72 % |  |
| random | euclidean | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 5.07 [4.95-5.2] | 2.45 [2.45-2.45] | 0.126 | 731.92 (n=3) | -0.73 % |  |
| random | euclidean | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 23.6 [23-23.9] | 21 [20.9-21.1] | 0.156 | 731.89 (n=3) | -0.73 % |  |
| random | euclidean | 1000000 | 16384 | 1.6e10 (>= 1e10) | faiss-lloyd | lloyd | fp32 | 25 | 2643 [2641-2652] | - | - | 737.29 (n=3) | +0.00 % |  |
| random | euclidean | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-cpu | cpu | fp32 | 5 | 2140 [2136-2165] | - | - | 688.6 (n=3) | - |  |
| random | euclidean | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 63.4 [63.4-72.2] | 61.8 [61.8-70.6] | 0.133 | 688.64 (n=3) | - |  |
| random | euclidean | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 11.6 [11.5-12.1] | 9.09 [8.94-9.1] | 0.137 | 689.56 (n=3) | - |  |
| random | euclidean | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 85.4 [83.7-85.9] | 83.5 [82-84.6] | 0.167 | 689.59 (n=3) | - |  |

## Where the GPU does not help / where CPU e-means wins

No cell carries `GPU does not help` or `CPU e-means wins`.

- The GPU helps in every main cell measured: the best GPU e-means run (end-to-end, transfer included) is faster than CPU e-means (cold in all 16 cells, warm in all 12 cells that have a CPU warm time), including the smallest n x k (1.0e8). Cells without a CPU warm time have no warm comparison.
- Smallest GPU speedup over CPU e-means: 4.1x at random 100k x 1024 (cold: `emeans-gpu-triton-bf16` 1.28 s end-to-end vs 5.21 s).
- Largest: 187.0x at cohere 1M x 65536 (cold: `emeans-gpu-triton-bf16` 11.9 s end-to-end vs 2232 s), BF16 on the GPU against FP32 on the CPU.
- Per GPU method at cohere 1M x 65536 (cold, end-to-end, CPU e-means 2232 s), FP32 against FP32: `emeans-gpu-native-fp32` 31.1x (71.9 s), `emeans-gpu-triton-fp32` 27.5x (81.1 s); `emeans-gpu-triton-bf16` 187.0x (11.9 s), BF16 on the GPU against FP32 on the CPU.
- CPU e-means is never the fastest method here, because a GPU e-means row is in every cell; `CPU e-means wins` therefore never applies on this host.
- Without a GPU, `CPU e-means wins (CPU only)` in 11 of 14 comparable main cells (2 have no Lloyd run): CPU e-means finishes sooner than every CPU Lloyd run with an objective within 0.5 % of the best one: cohere 100k x 1024, cohere 100k x 4096, cohere 100k x 16384, cohere 1M x 1024, cohere 1M x 4096, cohere 1M x 16384, random 100k x 1024, random 100k x 4096, random 1M x 1024, random 1M x 4096, random 1M x 16384.
  - cohere: 6 of 7 comparable main cells.
  - random: 5 of 7 comparable main cells.
- Lloyd's k-means has the better objective at: cohere 100k x 65536 (+22.22 %, degenerate: 1.5 rows per centroid); random 100k x 16384 (+7.90 %, reported as found; cohere at the same size: -0.69 %); random 100k x 65536 (+36.81 %, degenerate: 1.5 rows per centroid).
- No Lloyd run, so no comparison: cohere 1M x 65536, random 1M x 65536 (see Not run).

## GPU engines

The Triton rows (`emeans-gpu-triton-fp32`, `emeans-gpu-triton-bf16`) and the precision study ran on `emeans-rocm` 0.2.6 from PyPI on 2026-11-27, with `unused_threshold=0.1` passed explicitly (mass-based revival, the default of 0.2.5 and of stock faiss; the own default of 0.2.6 is cost-based revival and is not used here). The native rows (`emeans-gpu-native-fp32`) ran on 2026-11-22 on the stock e-means engine of faiss-cpu 1.16.0 through plain PyTorch (recorded as `faiss-cpu`; the records' faiss version is 1.16.0) and were not rerun. The 0.2.5 records are kept in `results/speed-emeans-rocm-0.2.5.jsonl` and `results/precision-emeans-rocm-0.2.5.jsonl` and are compared with the 0.2.6 records below.

Largest cell, cohere (spherical) 1M x 65536, cold medians over seeds:

| method | backend | precision | passes | cold s | end-to-end s | objective [min-max] |
|---|---|---|---|---|---|---|
| emeans-cpu | cpu | fp32 | 5 | 2232 | 2232 | 0.15387 [0.15384-0.15388] |
| emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 71.8 | 71.9 | 0.15389 [0.15387-0.1539] |
| emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 11.8 | 11.9 | 0.15346 [0.15343-0.15347] |
| emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 81 | 81.1 | 0.15345 [0.15344-0.15349] |

Native FP32 is faster than Triton FP32 (warm, end-to-end) in 10 of 16 main cells: cohere 100k x 4096, cohere 100k x 16384, cohere 100k x 65536, cohere 1M x 16384, cohere 1M x 65536, random 100k x 4096, random 100k x 16384, random 100k x 65536, random 1M x 16384, random 1M x 65536.

### emeans-rocm 0.2.5 vs 0.2.6

Both versions ran the same Triton cells with the same seeds and starting centroids. emeans-haystack passes `unused_threshold=0.1` (mass-based revival), which equals the default of 0.2.5, so the 0.2.5 records (made with no explicit threshold) and the 0.2.6 records use the same revival rule. `emeans-rocm` 0.2.6 places revived centroids along the donor's long axis (upstream faiss change #5975), so centroids can still differ. The 0.2.5 records are kept in `results/speed-emeans-rocm-0.2.5.jsonl` and `results/precision-emeans-rocm-0.2.5.jsonl`; the ARI figures come from `compare_versions.py`.

- Speed: 372 matching records. Warm time ratio 0.2.6 / 0.2.5, median 0.85 [0.26-1.27]. Cold time ratio, median 0.99 [0.79-1.78].
- bf16 main-cell objectives: largest difference 0.378 %, 7 of 48 records within 0.01 %.
- fp32 main-cell objectives: largest difference 1.342 %, 11 of 48 records within 0.01 %.
- Main-cell objectives, all precisions: 0.2.6 lower than 0.2.5 in 71 of 96 records; largest difference -1.342 % (emeans-gpu-triton-fp32 cohere 100k x 65536, 1.5 rows per centroid).

| method | cell | kind | passes | objective 0.2.5 | objective 0.2.6 | difference | inside 0.2.5 seed range | warm ratio | cold ratio | ARI 0.2.6 vs 0.2.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| emeans-gpu-triton-bf16 | cohere 100k x 1024 | main | default | 0.23717 | 0.23713 | -0.016 % | yes | 0.43 | 1.01 | 0.3937 |
| emeans-gpu-triton-bf16 | cohere 100k x 16384 | main | default | 0.16554 | 0.16535 | -0.109 % | no | 0.93 | 0.97 | - |
| emeans-gpu-triton-bf16 | cohere 100k x 4096 | main | default | 0.20997 | 0.20971 | -0.123 % | no | 0.69 | 0.99 | 0.2701 |
| emeans-gpu-triton-bf16 | cohere 100k x 4096 | sweep | 1 | 0.22642 | 0.22668 | +0.114 % | no | 1.11 | 0.98 | - |
| emeans-gpu-triton-bf16 | cohere 100k x 4096 | sweep | 10 | 0.20829 | 0.20803 | -0.127 % | no | 0.56 | 0.92 | - |
| emeans-gpu-triton-bf16 | cohere 100k x 4096 | sweep | 2 | 0.21449 | 0.21434 | -0.072 % | no | 0.91 | 1.00 | - |
| emeans-gpu-triton-bf16 | cohere 100k x 4096 | sweep | 25 | 0.20727 | 0.20699 | -0.135 % | no | 0.49 | 0.80 | - |
| emeans-gpu-triton-bf16 | cohere 100k x 4096 | sweep | 5 | 0.20997 | 0.20971 | -0.123 % | no | 0.69 | 1.01 | - |
| emeans-gpu-triton-bf16 | cohere 100k x 65536 | main | default | 0.066313 | 0.066163 | -0.226 % | no | 1.14 | 1.03 | - |
| emeans-gpu-triton-bf16 | cohere 1M x 1024 | main | default | 0.23947 | 0.23949 | +0.007 % | yes | 0.27 | 1.14 | - |
| emeans-gpu-triton-bf16 | cohere 1M x 16384 | main | default | 0.18567 | 0.18551 | -0.088 % | no | 0.73 | 1.20 | - |
| emeans-gpu-triton-bf16 | cohere 1M x 16384 | sweep | 1 | 0.19689 | 0.19628 | -0.309 % | no | 0.80 | 1.67 | - |
| emeans-gpu-triton-bf16 | cohere 1M x 16384 | sweep | 10 | 0.18451 | 0.18449 | -0.009 % | yes | 0.71 | 0.98 | - |
| emeans-gpu-triton-bf16 | cohere 1M x 16384 | sweep | 2 | 0.18882 | 0.18851 | -0.163 % | no | 0.81 | 1.52 | - |
| emeans-gpu-triton-bf16 | cohere 1M x 16384 | sweep | 25 | 0.18378 | 0.18379 | +0.005 % | yes | 0.68 | 0.79 | - |
| emeans-gpu-triton-bf16 | cohere 1M x 16384 | sweep | 5 | 0.18567 | 0.18551 | -0.088 % | no | 0.73 | 1.17 | - |
| emeans-gpu-triton-bf16 | cohere 1M x 4096 | main | default | 0.21667 | 0.21663 | -0.019 % | no | 0.44 | 1.22 | - |
| emeans-gpu-triton-bf16 | cohere 1M x 65536 | main | default | 0.15397 | 0.15346 | -0.335 % | no | 0.96 | 1.10 | - |
| emeans-gpu-triton-bf16 | random 100k x 1024 | main | default | 743.91 | 743.83 | -0.010 % | no | 0.40 | 0.95 | 0.0006 |
| emeans-gpu-triton-bf16 | random 100k x 16384 | main | default | 666.72 | 664.89 | -0.274 % | no | 0.89 | 0.96 | - |
| emeans-gpu-triton-bf16 | random 100k x 4096 | main | default | 713.2 | 713.47 | +0.039 % | no | 0.68 | 0.99 | 0.0005 |
| emeans-gpu-triton-bf16 | random 100k x 4096 | sweep | 1 | 725.26 | 726.21 | +0.131 % | no | 1.17 | 1.01 | - |
| emeans-gpu-triton-bf16 | random 100k x 4096 | sweep | 10 | 713.71 | 713.41 | -0.043 % | no | 0.55 | 0.92 | - |
| emeans-gpu-triton-bf16 | random 100k x 4096 | sweep | 2 | 716.85 | 718.17 | +0.184 % | no | 0.85 | 0.94 | - |
| emeans-gpu-triton-bf16 | random 100k x 4096 | sweep | 25 | 713.47 | 712.9 | -0.080 % | no | 0.49 | 0.81 | - |
| emeans-gpu-triton-bf16 | random 100k x 4096 | sweep | 5 | 713.2 | 713.47 | +0.039 % | no | 0.68 | 0.98 | - |
| emeans-gpu-triton-bf16 | random 100k x 65536 | main | default | 334.98 | 335.22 | +0.071 % | yes | 1.11 | 1.03 | - |
| emeans-gpu-triton-bf16 | random 1M x 1024 | main | default | 754.5 | 754.5 | -0.001 % | no | 0.26 | 1.28 | - |
| emeans-gpu-triton-bf16 | random 1M x 16384 | main | default | 731.98 | 731.92 | -0.009 % | no | 0.72 | 1.12 | - |
| emeans-gpu-triton-bf16 | random 1M x 4096 | main | default | 747.51 | 747.41 | -0.013 % | no | 0.43 | 1.13 | - |
| emeans-gpu-triton-bf16 | random 1M x 65536 | main | default | 688.66 | 689.56 | +0.130 % | no | 0.95 | 1.09 | - |
| emeans-gpu-triton-fp32 | cohere 100k x 1024 | main | default | 0.23711 | 0.23717 | +0.027 % | no | 0.65 | 1.00 | 0.3950 |
| emeans-gpu-triton-fp32 | cohere 100k x 16384 | main | default | 0.16542 | 0.16526 | -0.094 % | no | 0.98 | 0.95 | - |
| emeans-gpu-triton-fp32 | cohere 100k x 4096 | main | default | 0.20978 | 0.20975 | -0.013 % | no | 0.91 | 0.99 | 0.2703 |
| emeans-gpu-triton-fp32 | cohere 100k x 4096 | sweep | 1 | 0.22619 | 0.22681 | +0.276 % | no | 1.07 | 0.98 | - |
| emeans-gpu-triton-fp32 | cohere 100k x 4096 | sweep | 10 | 0.20826 | 0.208 | -0.124 % | no | 0.85 | 0.93 | - |
| emeans-gpu-triton-fp32 | cohere 100k x 4096 | sweep | 2 | 0.21436 | 0.21444 | +0.039 % | yes | 0.96 | 0.97 | - |
| emeans-gpu-triton-fp32 | cohere 100k x 4096 | sweep | 25 | 0.2071 | 0.20701 | -0.046 % | no | 0.81 | 0.87 | - |
| emeans-gpu-triton-fp32 | cohere 100k x 4096 | sweep | 5 | 0.20985 | 0.20975 | -0.050 % | no | 0.91 | 0.98 | - |
| emeans-gpu-triton-fp32 | cohere 100k x 65536 | main | default | 0.06626 | 0.065477 | -1.182 % | no | 1.02 | 1.12 | - |
| emeans-gpu-triton-fp32 | cohere 1M x 1024 | main | default | 0.23951 | 0.23947 | -0.017 % | yes | 0.54 | 1.10 | - |
| emeans-gpu-triton-fp32 | cohere 1M x 16384 | main | default | 0.1857 | 0.18553 | -0.093 % | no | 0.95 | 1.02 | - |
| emeans-gpu-triton-fp32 | cohere 1M x 16384 | sweep | 1 | 0.19686 | 0.19627 | -0.299 % | no | 1.00 | 1.24 | - |
| emeans-gpu-triton-fp32 | cohere 1M x 16384 | sweep | 10 | 0.18456 | 0.18451 | -0.025 % | no | 0.94 | 0.97 | - |
| emeans-gpu-triton-fp32 | cohere 1M x 16384 | sweep | 2 | 0.18885 | 0.18842 | -0.230 % | no | 0.96 | 1.14 | - |
| emeans-gpu-triton-fp32 | cohere 1M x 16384 | sweep | 25 | 0.18377 | 0.18383 | +0.032 % | no | 0.92 | 0.94 | - |
| emeans-gpu-triton-fp32 | cohere 1M x 16384 | sweep | 5 | 0.1857 | 0.18553 | -0.090 % | no | 0.95 | 1.03 | - |
| emeans-gpu-triton-fp32 | cohere 1M x 4096 | main | default | 0.21667 | 0.21663 | -0.020 % | no | 0.78 | 1.03 | - |
| emeans-gpu-triton-fp32 | cohere 1M x 65536 | main | default | 0.15387 | 0.15345 | -0.278 % | no | 0.99 | 1.00 | - |
| emeans-gpu-triton-fp32 | random 100k x 1024 | main | default | 744.05 | 743.81 | -0.032 % | no | 0.66 | 0.91 | 0.0005 |
| emeans-gpu-triton-fp32 | random 100k x 16384 | main | default | 666.68 | 666.72 | +0.006 % | yes | 0.96 | 0.94 | - |
| emeans-gpu-triton-fp32 | random 100k x 4096 | main | default | 713.21 | 713.5 | +0.041 % | yes | 0.90 | 0.97 | 0.0005 |
| emeans-gpu-triton-fp32 | random 100k x 4096 | sweep | 1 | 725.23 | 726.19 | +0.132 % | no | 1.01 | 0.97 | - |
| emeans-gpu-triton-fp32 | random 100k x 4096 | sweep | 10 | 713.69 | 713.44 | -0.035 % | no | 0.84 | 0.94 | - |
| emeans-gpu-triton-fp32 | random 100k x 4096 | sweep | 2 | 716.79 | 718.23 | +0.201 % | no | 0.96 | 0.95 | - |
| emeans-gpu-triton-fp32 | random 100k x 4096 | sweep | 25 | 713.58 | 712.86 | -0.101 % | no | 0.80 | 0.87 | - |
| emeans-gpu-triton-fp32 | random 100k x 4096 | sweep | 5 | 713.25 | 713.5 | +0.036 % | yes | 0.90 | 0.96 | - |
| emeans-gpu-triton-fp32 | random 100k x 65536 | main | default | 335.34 | 334.9 | -0.129 % | no | 1.00 | 0.99 | - |
| emeans-gpu-triton-fp32 | random 1M x 1024 | main | default | 754.5 | 754.5 | -0.001 % | no | 0.54 | 0.96 | - |
| emeans-gpu-triton-fp32 | random 1M x 16384 | main | default | 732.01 | 731.89 | -0.015 % | no | 0.94 | 1.01 | - |
| emeans-gpu-triton-fp32 | random 1M x 4096 | main | default | 747.51 | 747.42 | -0.012 % | no | 0.78 | 1.04 | - |
| emeans-gpu-triton-fp32 | random 1M x 65536 | main | default | 688.67 | 689.59 | +0.133 % | no | 0.99 | 1.00 | - |

Assignment agreement (100k subset), median over seeds:

| method | cell | ARI 0.2.6 vs 0.2.5 | ARI between 0.2.5 seeds | ARI 0.2.5 rerun | objective in 0.2.5 range | ARI at least seed-to-seed |
|---|---|---|---|---|---|---|
| emeans-gpu-triton-fp32 | random 100k x 1024 | 0.0005 | 0.0005 | 0.0005 | yes | yes |
| emeans-gpu-triton-fp32 | random 100k x 4096 | 0.0005 | 0.0004 | 0.0005 | yes | yes |
| emeans-gpu-triton-bf16 | random 100k x 1024 | 0.0006 | 0.0005 | 1.0000 | no | yes |
| emeans-gpu-triton-bf16 | random 100k x 4096 | 0.0005 | 0.0004 | 1.0000 | no | yes |
| emeans-gpu-triton-fp32 | cohere 100k x 1024 | 0.3950 | 0.3824 | 0.4512 | yes | yes |
| emeans-gpu-triton-fp32 | cohere 100k x 4096 | 0.2703 | 0.2519 | 0.3407 | yes | yes |
| emeans-gpu-triton-bf16 | cohere 100k x 1024 | 0.3937 | 0.3861 | 1.0000 | yes | yes |
| emeans-gpu-triton-bf16 | cohere 100k x 4096 | 0.2701 | 0.2535 | 1.0000 | no | yes |

Baseline of the ARI part: the gate compared the ARI between versions at the same seed with the ARI between different 0.2.5 seeds. That baseline is weak: fits from one seed share their starting centroids, so the comparison leans towards passing, and a version change that moves assignments as much as a reseed still passes. The fair baseline is the same-seed 0.2.5 rerun (`ARI 0.2.5 rerun`). bf16 0.2.5 reruns are deterministic (ARI 1.0). fp32 0.2.5 reruns are not reproducible (same-seed ARI 0.34-0.45 on cohere). On cohere, 0.2.6 against 0.2.5 at the same seed gives an ARI of 0.27-0.40, about the agreement of a reseed (0.25-0.39 between 0.2.5 seeds). On random data (no cluster structure) the ARI between versions and between seeds is 0.0004-0.0006, chance level, so the ARI part compares noise there. The gate rule and its committed result (R-02) are kept as run.

- Precision: 48 matching records; 89 of 192 variant drift_gap medians differ by more than 0.005 between the versions.
- fp16: median share of documents in the same cluster as FP32, 0.2.5 0.209, 0.2.6 0.233.
- bf16: median share of documents in the same cluster as FP32, 0.2.5 0.207, 0.2.6 0.223.
- Precision reference objective: median relative difference 0.014 %.

emeans-rocm 0.2.6 has a median objective outside the min-max of the 3 seeds of 0.2.5 of the same cell in 52 cells (emeans-gpu-triton-bf16 cohere 100k x 16384 (main, default passes); emeans-gpu-triton-bf16 cohere 100k x 4096 (main, default passes); emeans-gpu-triton-bf16 cohere 100k x 4096 (sweep, 1.0 passes); emeans-gpu-triton-bf16 cohere 100k x 4096 (sweep, 10.0 passes); emeans-gpu-triton-bf16 cohere 100k x 4096 (sweep, 2.0 passes); emeans-gpu-triton-bf16 cohere 100k x 4096 (sweep, 25.0 passes); emeans-gpu-triton-bf16 cohere 100k x 4096 (sweep, 5.0 passes); emeans-gpu-triton-bf16 cohere 100k x 65536 (main, default passes); emeans-gpu-triton-bf16 cohere 1M x 16384 (main, default passes); emeans-gpu-triton-bf16 cohere 1M x 16384 (sweep, 1.0 passes) and more); the largest main-cell objective difference is bf16 0.378 %, fp32 1.342 %. The smoke gate passed its objective part in 5 of 8 settings and its ARI part in 8 of 8.

Gate and override: before the full 0.2.6 rerun, a smoke gate on the 100k subset (8 settings x 3 seeds) passed its assignment part in 8 of 8 settings and its objective part in 5 of 8; the 3 failures were all bf16 (two with a better objective than the 0.2.5 range, random k = 4096 worse by 0.020 against a 0.95 seed spread), so the gate result is a failure (exit 3). The owner then checked bf16 on 6 seeds: 0.2.6 is never worse than 0.2.5 there, and at random k = 1024 and Cohere k = 4096 it is better than every 0.2.5 value. On that evidence the owner overrode the gate on 2026-11-27 and the full rerun went ahead.

What this means for users: models saved with an earlier version load and assign unchanged, because assignment from saved centroids does not depend on the engine version. A refit on emeans-rocm 0.2.6 gives GPU centroids at d > 2 whose assignments (`cluster_id`s) differ from those of a 0.2.5 fit about as much as a different seed does (ARI paragraph above), because 0.2.6 places revived centroids differently (same revival rule, different split placement). The objectives stay close (main-cell objective lines above).

Timings, as measured: the Triton fit times of 0.2.6 are about those of 0.2.5 when cold (median ratio in the speed line above) and shorter when warm. The whole runs took longer, by about 15-20 %: the grid 4 h 35 min against 3 h 58 min, the precision study 4964 s against about 4006 s. These wall times include everything around the fits and are not interpreted here; speed is not the claim of this page.

## Pass sweep

### cohere (spherical), n=100000, k=4096, n x k = 4.1e8

| method | passes | passes used | objective | cold s | warm s | seeds |
|---|---|---|---|---|---|---|
| emeans-cpu | 1 | 1 | 0.22581 | 3.86 | 3.68 | 3 |
| emeans-cpu | 2 | 2 | 0.21447 | 7.07 | 7.08 | 3 |
| emeans-cpu | 5 | 5 | 0.20978 | 18.5 | 17.4 | 3 |
| emeans-cpu | 10 | 10 | 0.20813 | 35 | 34.7 | 3 |
| emeans-cpu | 25 | 25 | 0.20711 | 85.5 | 84.8 | 3 |
| emeans-gpu-native-fp32 | 1 | 1 | 0.22637 | 2.1 | 0.166 | 3 |
| emeans-gpu-native-fp32 | 2 | 2 | 0.21435 | 2.2 | 0.287 | 3 |
| emeans-gpu-native-fp32 | 5 | 5 | 0.20981 | 2.55 | 0.633 | 3 |
| emeans-gpu-native-fp32 | 10 | 10 | 0.20821 | 3.1 | 1.2 | 3 |
| emeans-gpu-native-fp32 | 25 | 25 | 0.20713 | 4.86 | 2.92 | 3 |
| emeans-gpu-triton-bf16 | 1 | 1 | 0.22668 | 1.18 | 0.0741 | 3 |
| emeans-gpu-triton-bf16 | 2 | 2 | 0.21434 | 1.2 | 0.0889 | 3 |
| emeans-gpu-triton-bf16 | 5 | 5 | 0.20971 | 1.35 | 0.147 | 3 |
| emeans-gpu-triton-bf16 | 10 | 10 | 0.20803 | 1.43 | 0.224 | 3 |
| emeans-gpu-triton-bf16 | 25 | 25 | 0.20699 | 1.67 | 0.465 | 3 |
| emeans-gpu-triton-fp32 | 1 | 1 | 0.22681 | 1.29 | 0.228 | 3 |
| emeans-gpu-triton-fp32 | 2 | 2 | 0.21444 | 1.37 | 0.309 | 3 |
| emeans-gpu-triton-fp32 | 5 | 5 | 0.20975 | 1.77 | 0.633 | 3 |
| emeans-gpu-triton-fp32 | 10 | 10 | 0.208 | 2.27 | 1.1 | 3 |
| emeans-gpu-triton-fp32 | 25 | 25 | 0.20701 | 3.7 | 2.57 | 3 |
| faiss-lloyd | 1 | 1 | 0.22799 | 2.93 | 3.71 | 3 |
| faiss-lloyd | 2 | 2 | 0.22069 | 5.72 | 5.74 | 3 |
| faiss-lloyd | 5 | 5 | 0.21594 | 15.2 | 14.4 | 3 |
| faiss-lloyd | 10 | 10 | 0.21471 | 29.5 | 29.6 | 3 |
| faiss-lloyd | 25 | 25 | 0.21433 | 73.7 | 74 | 3 |
| sklearn-lloyd | 1 | 1 | 0.22799 | 2.41 | 2.07 | 3 |
| sklearn-lloyd | 2 | 2 | 0.22082 | 3.42 | 3.13 | 3 |
| sklearn-lloyd | 5 | 5 | 0.21608 | 6.43 | 6.14 | 3 |
| sklearn-lloyd | 10 | 10 | 0.21483 | 11.5 | 11.3 | 3 |
| sklearn-lloyd | 25 | 25 | 0.21448 | 26.8 | 26.6 | 3 |

### cohere (spherical), n=1000000, k=16384, n x k = 1.6e10 (>= 1e10)

| method | passes | passes used | objective | cold s | warm s | seeds |
|---|---|---|---|---|---|---|
| emeans-cpu | 1 | 1 | 0.19692 | 116 | 117 | 3 |
| emeans-cpu | 2 | 2 | 0.18894 | 231 | 231 | 3 |
| emeans-cpu | 5 | 5 | 0.18569 | 575 | - | 3 |
| emeans-cpu | 10 | 10 | 0.18461 | 1144 | - | 3 |
| emeans-cpu | 25 | 25 | 0.18377 | 2861 | - | 3 |
| emeans-gpu-native-fp32 | 1 | 1 | 0.19685 | 5.76 | 3.87 | 3 |
| emeans-gpu-native-fp32 | 2 | 2 | 0.18885 | 9.56 | 7.71 | 3 |
| emeans-gpu-native-fp32 | 5 | 5 | 0.18569 | 20.6 | 18.8 | 3 |
| emeans-gpu-native-fp32 | 10 | 10 | 0.18453 | 38.9 | 37.2 | 3 |
| emeans-gpu-native-fp32 | 25 | 25 | 0.18373 | 94 | 92.4 | 3 |
| emeans-gpu-triton-bf16 | 1 | 1 | 0.19628 | 3.14 | 0.576 | 3 |
| emeans-gpu-triton-bf16 | 2 | 2 | 0.18851 | 3.89 | 1.14 | 3 |
| emeans-gpu-triton-bf16 | 5 | 5 | 0.18551 | 5.32 | 2.47 | 3 |
| emeans-gpu-triton-bf16 | 10 | 10 | 0.18449 | 7.54 | 4.64 | 3 |
| emeans-gpu-triton-bf16 | 25 | 25 | 0.18379 | 13.9 | 11.3 | 3 |
| emeans-gpu-triton-fp32 | 1 | 1 | 0.19627 | 6.78 | 4.38 | 3 |
| emeans-gpu-triton-fp32 | 2 | 2 | 0.18842 | 11.1 | 8.37 | 3 |
| emeans-gpu-triton-fp32 | 5 | 5 | 0.18553 | 22.8 | 19.9 | 3 |
| emeans-gpu-triton-fp32 | 10 | 10 | 0.18451 | 41 | 38.7 | 3 |
| emeans-gpu-triton-fp32 | 25 | 25 | 0.18383 | 99 | 95.3 | 3 |
| faiss-lloyd | 1 | 1 | 0.20697 | 112 | 114 | 3 |
| faiss-lloyd | 2 | 2 | 0.19781 | 226 | 228 | 3 |
| faiss-lloyd | 5 | 5 | 0.19096 | 570 | - | 3 |
| faiss-lloyd | 10 | 10 | 0.18856 | 1154 | - | 3 |
| faiss-lloyd | 25 | 25 | 0.18746 | 2765 | - | 3 |

### random (euclidean), n=100000, k=4096, n x k = 4.1e8

| method | passes | passes used | objective | cold s | warm s | seeds |
|---|---|---|---|---|---|---|
| emeans-cpu | 1 | 1 | 725.55 | 3.24 | 3.33 | 3 |
| emeans-cpu | 2 | 2 | 716.73 | 6.36 | 6.4 | 3 |
| emeans-cpu | 5 | 5 | 713.15 | 15.7 | 15.8 | 3 |
| emeans-cpu | 10 | 10 | 713.67 | 31.3 | 31.2 | 3 |
| emeans-cpu | 25 | 25 | 713.45 | 76.9 | 76.7 | 3 |
| emeans-gpu-native-fp32 | 1 | 1 | 725.3 | 2.1 | 0.168 | 3 |
| emeans-gpu-native-fp32 | 2 | 2 | 716.7 | 2.2 | 0.283 | 3 |
| emeans-gpu-native-fp32 | 5 | 5 | 713.13 | 2.58 | 0.643 | 3 |
| emeans-gpu-native-fp32 | 10 | 10 | 713.66 | 3.01 | 1.1 | 3 |
| emeans-gpu-native-fp32 | 25 | 25 | 713.5 | 4.95 | 3.01 | 3 |
| emeans-gpu-triton-bf16 | 1 | 1 | 726.21 | 1.2 | 0.0732 | 3 |
| emeans-gpu-triton-bf16 | 2 | 2 | 718.17 | 1.18 | 0.0817 | 3 |
| emeans-gpu-triton-bf16 | 5 | 5 | 713.47 | 1.36 | 0.142 | 3 |
| emeans-gpu-triton-bf16 | 10 | 10 | 713.41 | 1.43 | 0.22 | 3 |
| emeans-gpu-triton-bf16 | 25 | 25 | 712.9 | 1.7 | 0.463 | 3 |
| emeans-gpu-triton-fp32 | 1 | 1 | 726.19 | 1.3 | 0.218 | 3 |
| emeans-gpu-triton-fp32 | 2 | 2 | 718.23 | 1.37 | 0.31 | 3 |
| emeans-gpu-triton-fp32 | 5 | 5 | 713.5 | 1.82 | 0.647 | 3 |
| emeans-gpu-triton-fp32 | 10 | 10 | 713.44 | 2.35 | 1.15 | 3 |
| emeans-gpu-triton-fp32 | 25 | 25 | 712.86 | 3.86 | 2.66 | 3 |
| faiss-lloyd | 1 | 1 | 722.2 | 2.62 | 2.71 | 3 |
| faiss-lloyd | 2 | 2 | 721.03 | 5.37 | 5.54 | 3 |
| faiss-lloyd | 5 | 5 | 720.22 | 13.4 | 13.6 | 3 |
| faiss-lloyd | 10 | 10 | 720 | 26.8 | 27.6 | 3 |
| faiss-lloyd | 25 | 25 | 719.94 | 68.1 | 68.2 | 3 |
| sklearn-lloyd | 1 | 1 | 722.2 | 2.2 | 1.94 | 3 |
| sklearn-lloyd | 2 | 2 | 721.03 | 3.45 | 2.97 | 3 |
| sklearn-lloyd | 5 | 5 | 720.22 | 5.99 | 5.7 | 3 |
| sklearn-lloyd | 10 | 10 | 720.01 | 10.6 | 10.6 | 3 |
| sklearn-lloyd | 25 | 25 | 719.94 | 25.8 | 24.8 | 3 |

## Cuts and choices

Every cut made to save time or memory, with its reason and what it hides. A setting is one (data, n, k); a cell is a setting run by one method; a cell-seed is one seed of a cell, with a cold and a warm record; a sweep point is one pass count of a sweep cell-seed.

| cut or choice | reason | what it hides |
|---|---|---|
| scikit-learn runs at 100k rows only (`METHOD_NS` in `run_grid.py`) | at 1M it would cost most of a 30-40 h CPU grid for a reference point `faiss-lloyd` already gives | scikit-learn speed and objective at 1M; there `faiss-lloyd` is the only Lloyd baseline |
| 60-minute cap per cell-seed (`DEFAULT_CAP_MIN`); a cell-seed whose one-pass probe estimate exceeds it is not run | one Lloyd cell-seed at 1M x 65536 would take about 3 h | not run: `faiss-lloyd` cohere 1M x 65536 (est. 3.0-3.1 h); CPU e-means ran it in 2232 s (37 min), GPU e-means in 11.9-81.1 s; `faiss-lloyd` random 1M x 65536 (est. 2.9-3.2 h); CPU e-means ran it in 2140 s (36 min), GPU e-means in 11.8-85.5 s. No Lloyd objective there, so no e-means vs Lloyd comparison |
| Warm repeat skipped for CPU cell-seeds estimated over 300 s (`WARM_MAX_S`) | a warm repeat would double the cost of the largest CPU cell-seeds | 60 warm records (42 main cell-seeds, 18 sweep points; emeans-cpu, faiss-lloyd, sklearn-lloyd); their cold times are shown |
| Pass sweep (1, 2, 5, 10, 25 passes) on 3 settings only (`SWEEP_CELLS`): random 100k x 4096, cohere 100k x 4096, cohere 1M x 16384 | each sweep point is a full run per method and seed | passes-to-objective at other settings; the main cells run fixed passes |
| `faiss.Kmeans` capped at `niter` 25; e-means at its default 5 passes | fixed, stated passes per method; a Lloyd pass at 1M x 16384 costs about 2 min on CPU | Lloyd at convergence at 1M: it was still improving (cohere 1M x 16384 10 -> 25 passes: 0.18856 -> 0.18746) |
| 3 seeds per cell (`SEEDS`) | cost | the spread is a min-max over 3 seeds, not a confidence interval |
| Precision study at k = 1024, 4096 only | cost (6 fits per precision cell-seed) | precision effects at larger k |
| Cell-seeds 4-7 % slower than the clean 2026-11-26 reruns were kept, not rerun (owner, 2026-11-26): emeans-cpu random 1M x 4096 seed 0 (vs seed 2) 31.9 vs 29.8 s/pass (+6.8 %); emeans-cpu random 1M x 4096 seed 1 (vs seed 2) 31.6 vs 29.8 s/pass (+5.8 %); faiss-lloyd cohere 1M x 16384 seed 1 main (vs its 25-pass sweep point) 115.1 vs 110.6 s/pass (+4.1 %) | the gap matches day-to-day variation; a rerun would cost hours | per-pass timings vary by 4-7 % between days and machine states; objectives and pass counts are not affected |
| CPU cells ran with 8 of the 10 cores (`BENCH_THREADS=8`), `nice 19`, a systemd memory cap of 12 GiB and swap off | keep the machine usable and stop one cell from taking all RAM during multi-day runs | CPU timings at 10 threads; CPU s/pass would likely be somewhat lower with all cores |
| GPU cells ran with the default 10 host threads | the GPU does the work; host threads only feed it | nothing measured here; CPU and GPU thread counts differ, so GPU-vs-CPU ratios are host-specific |
| No heavy runs 23:00-07:00 (owner rule); a watcher stopped each grid at 22:00 after the running cell | quiet hours on a shared personal machine | the CPU grid ran over 2026-11-23 to 2026-11-26, so CPU rows come from different days |
| Load-contaminated cell-seeds were rerun: 6 cell-seeds (cold and warm, 12 records) across 4 cells, dropped and rerun on 2026-11-26 on a quiet machine, the drop logged with the dropped lines kept | concurrent jobs on the machine (agent sessions, an image-model probe, a container that pushed the machine into swap) had inflated their per-pass times | the first, slower timings of those cell-seeds; they are kept locally, not published |
| scikit-learn cohere records (54) were dropped and rerun on 2026-11-24 after a scoring fix | its centroids were off the unit sphere and scored on a different geometry than the other methods | nothing: the replaced records were wrong, not slow |
| The precision study's memory cap was raised from 12 to 16 GiB mid-run (2026-11-26) after 1563 cgroup `memory.max` events; 0 OOM kills, peak 12.07 GiB | the study was being throttled at the cap | precision results are unaffected (deterministic fits); its train times are not used anywhere |
| Precision guidance: option B (owner, 2026-11-23); the random-data spike contradictions of the 0.2.6 records were accepted by the owner on 2026-11-27 (those of the 0.2.5 records on 2026-11-23) | on random data FP16/BF16 drift equals or slightly exceeds the FP32 other-seed drift; the guidance adds a caution instead of a claim | nothing; the contradictions are listed in the Precision section |
| `sklearn-lloyd` on spherical data is Euclidean Lloyd on unit-norm rows with one projection of the final centroids to unit length (scikit-learn has no spherical mode) | scikit-learn is a reference point; `faiss-lloyd` (`spherical=True`, normalised every iteration) is the true spherical Lloyd baseline | how true spherical Lloyd would behave in scikit-learn; read spherical comparisons against `faiss-lloyd` |
| ARI between 0.2.5 and 0.2.6 on 8 settings x 3 seeds only (n = 100k, k = 1024 and 4096, random and Cohere, Triton FP32 and BF16) | the 0.2.5 grid kept no centroids or assignments; a full 0.2.5 rerun with saved centroids would cost about 5 more GPU hours and 6-7 GB | assignment agreement at 1M rows, at k = 16384 and 65536 and on sweep cells; objectives and times are compared on all 62 Triton cells x 3 seeds (186 cell-seed runs per timing) |

## Not run

| data | n | k | kind | passes | method | timing | status | reason | seeds |
|---|---|---|---|---|---|---|---|---|---|
| cohere | 100000 | 16384 | main | - | sklearn-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 100000 | 65536 | main | - | faiss-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 100000 | 65536 | main | - | sklearn-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 4096 | main | - | faiss-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 16384 | main | - | emeans-cpu | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 16384 | main | - | faiss-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 16384 | sweep | 5 | emeans-cpu | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 16384 | sweep | 5 | faiss-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 16384 | sweep | 10 | emeans-cpu | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 16384 | sweep | 10 | faiss-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 16384 | sweep | 25 | emeans-cpu | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 16384 | sweep | 25 | faiss-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 65536 | main | - | emeans-cpu | warm | skipped | skipped (cost) | 0,1,2 |
| cohere | 1000000 | 65536 | main | - | faiss-lloyd | cold | skipped | not run: est. 3.1 h | 0 |
| cohere | 1000000 | 65536 | main | - | faiss-lloyd | cold | skipped | not run: est. 3.0 h | 1,2 |
| cohere | 1000000 | 65536 | main | - | faiss-lloyd | warm | skipped | not run: est. 3.1 h | 0 |
| cohere | 1000000 | 65536 | main | - | faiss-lloyd | warm | skipped | not run: est. 3.0 h | 1,2 |
| random | 100000 | 16384 | main | - | sklearn-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| random | 100000 | 65536 | main | - | faiss-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| random | 100000 | 65536 | main | - | sklearn-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| random | 1000000 | 4096 | main | - | faiss-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| random | 1000000 | 16384 | main | - | emeans-cpu | warm | skipped | skipped (cost) | 0,1,2 |
| random | 1000000 | 16384 | main | - | faiss-lloyd | warm | skipped | skipped (cost) | 0,1,2 |
| random | 1000000 | 65536 | main | - | emeans-cpu | warm | skipped | skipped (cost) | 0,1,2 |
| random | 1000000 | 65536 | main | - | faiss-lloyd | cold | skipped | not run: est. 2.9 h | 0 |
| random | 1000000 | 65536 | main | - | faiss-lloyd | cold | skipped | not run: est. 3.0 h | 1 |
| random | 1000000 | 65536 | main | - | faiss-lloyd | cold | skipped | not run: est. 3.2 h | 2 |
| random | 1000000 | 65536 | main | - | faiss-lloyd | warm | skipped | not run: est. 2.9 h | 0 |
| random | 1000000 | 65536 | main | - | faiss-lloyd | warm | skipped | not run: est. 3.0 h | 1 |
| random | 1000000 | 65536 | main | - | faiss-lloyd | warm | skipped | not run: est. 3.2 h | 2 |

## Precision

All fits are deterministic from the same shared start; the reference is FP32. The baselines are an FP32 deterministic rerun and an FP32 run with another seed and another start. FP16 and BF16 run on the emeans-rocm package's Triton rung. Centroids are matched one to one (Hungarian) before measuring. `drift_gap` 1.0 means a centroid moved as far as its nearest other reference centroid; the table shows the median over seeds of the per-run median and the maximum over seeds of the per-run maximum. Labels and objective come from the shared exact FP32 scorer. Runs that fell back to another precision or produced non-finite centroids are left out of the statistics and counted under `excluded`. Cells show `median [min-max]` over seeds.

| data | mode | n | k | variant | seeds | drift_gap median | drift_gap max | share same cluster | objective vs FP32 (%) | excluded |
|---|---|---|---|---|---|---|---|---|---|---|
| cohere | euclidean | 100000 | 1024 | fp16 | 3 | 0.436 | 2.51 | 0.56 [0.54-0.592] | -0.0244 [-0.0524--0.0111] | 0 |
| cohere | euclidean | 100000 | 1024 | bf16 | 3 | 0.484 | 2.44 | 0.536 [0.516-0.538] | -0.0181 [-0.0401--0.00366] | 0 |
| cohere | euclidean | 100000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | euclidean | 100000 | 1024 | fp32_other_seed | 3 | 0.587 | 2.49 | 0.473 [0.454-0.476] | -0.000242 [-0.0238-0.0279] | 0 |
| cohere | euclidean | 100000 | 4096 | fp16 | 3 | 0.643 | 4.14 | 0.457 [0.452-0.458] | -0.0162 [-0.0404-0.02] | 0 |
| cohere | euclidean | 100000 | 4096 | bf16 | 3 | 0.693 | 3.99 | 0.424 [0.422-0.426] | -0.0104 [-0.0347-0.0552] | 0 |
| cohere | euclidean | 100000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | euclidean | 100000 | 4096 | fp32_other_seed | 3 | 0.798 | 4.05 | 0.359 [0.357-0.361] | -0.0572 [-0.104-0.0646] | 0 |
| cohere | euclidean | 1000000 | 1024 | fp16 | 3 | 0.0848 | 2.97 | 0.799 [0.759-0.808] | -0.00655 [-0.00977-0.0174] | 0 |
| cohere | euclidean | 1000000 | 1024 | bf16 | 3 | 0.117 | 3.05 | 0.752 [0.749-0.756] | 0.0118 [0.0109-0.0205] | 0 |
| cohere | euclidean | 1000000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | euclidean | 1000000 | 1024 | fp32_other_seed | 3 | 0.315 | 2.37 | 0.581 [0.565-0.592] | 0.00128 [-0.037-0.0129] | 0 |
| cohere | euclidean | 1000000 | 4096 | fp16 | 3 | 0.109 | 2.57 | 0.725 [0.722-0.735] | 0.00632 [-0.0152-0.0109] | 0 |
| cohere | euclidean | 1000000 | 4096 | bf16 | 3 | 0.134 | 2.55 | 0.692 [0.689-0.693] | 0.00253 [-0.0138-0.014] | 0 |
| cohere | euclidean | 1000000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | euclidean | 1000000 | 4096 | fp32_other_seed | 3 | 0.424 | 2.7 | 0.532 [0.531-0.535] | 0.00874 [-0.0353-0.0334] | 0 |
| cohere | spherical | 100000 | 1024 | fp16 | 3 | 0.495 | 2.34 | 0.522 [0.519-0.524] | -0.0661 [-0.0696--0.0513] | 0 |
| cohere | spherical | 100000 | 1024 | bf16 | 3 | 0.538 | 2.4 | 0.5 [0.491-0.505] | -0.0178 [-0.0302--0.00855] | 0 |
| cohere | spherical | 100000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | spherical | 100000 | 1024 | fp32_other_seed | 3 | 0.604 | 2.85 | 0.465 [0.447-0.466] | -0.00653 [-0.026-0.00459] | 0 |
| cohere | spherical | 100000 | 4096 | fp16 | 3 | 0.683 | 5.62 | 0.431 [0.427-0.435] | -0.00462 [-0.0387-0.0482] | 0 |
| cohere | spherical | 100000 | 4096 | bf16 | 3 | 0.722 | 7.88 | 0.409 [0.408-0.412] | -0.000267 [-0.0333-0.0306] | 0 |
| cohere | spherical | 100000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | spherical | 100000 | 4096 | fp32_other_seed | 3 | 0.803 | 7.35 | 0.359 [0.358-0.36] | -0.0157 [-0.128-0.0773] | 0 |
| cohere | spherical | 1000000 | 1024 | fp16 | 3 | 0.109 | 2.13 | 0.758 [0.733-0.769] | 0.00801 [0.00792-0.0122] | 0 |
| cohere | spherical | 1000000 | 1024 | bf16 | 3 | 0.145 | 2.08 | 0.717 [0.709-0.734] | 0.0138 [-0.0273-0.0457] | 0 |
| cohere | spherical | 1000000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | spherical | 1000000 | 1024 | fp32_other_seed | 3 | 0.341 | 2.4 | 0.579 [0.575-0.584] | -0.00642 [-0.013-0.00491] | 0 |
| cohere | spherical | 1000000 | 4096 | fp16 | 3 | 0.138 | 2.33 | 0.689 [0.683-0.691] | 0.00139 [-0.000839-0.0206] | 0 |
| cohere | spherical | 1000000 | 4096 | bf16 | 3 | 0.169 | 2.59 | 0.664 [0.653-0.665] | 0.00188 [-0.0214-0.0034] | 0 |
| cohere | spherical | 1000000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | spherical | 1000000 | 4096 | fp32_other_seed | 3 | 0.411 | 2.58 | 0.538 [0.538-0.539] | -0.0148 [-0.0302-0.0235] | 0 |
| random | euclidean | 100000 | 1024 | fp16 | 3 | 1.17 | 2.65 | 0.0104 [0.01-0.0107] | 0.0186 [0.00491-0.0352] | 0 |
| random | euclidean | 100000 | 1024 | bf16 | 3 | 1.17 | 2.7 | 0.0106 [0.0102-0.0106] | 0.00736 [-0.0228-0.00996] | 0 |
| random | euclidean | 100000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | euclidean | 100000 | 1024 | fp32_other_seed | 3 | 1.18 | 2.6 | 0.0105 [0.0104-0.0106] | -0.00726 [-0.0179-0.016] | 0 |
| random | euclidean | 100000 | 4096 | fp16 | 3 | 1.21 | 4.53 | 0.0341 [0.034-0.0352] | -0.0168 [-0.034--0.00513] | 0 |
| random | euclidean | 100000 | 4096 | bf16 | 3 | 1.22 | 3.67 | 0.0348 [0.0341-0.0356] | -0.0162 [-0.0245--0.00423] | 0 |
| random | euclidean | 100000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | euclidean | 100000 | 4096 | fp32_other_seed | 3 | 1.21 | 4.15 | 0.0323 [0.0321-0.0328] | -0.0873 [-0.131-0.07] | 0 |
| random | euclidean | 1000000 | 1024 | fp16 | 3 | 1 | 1.67 | 0.00377 [0.00376-0.00382] | 0.000456 [0.000301-0.00131] | 0 |
| random | euclidean | 1000000 | 1024 | bf16 | 3 | 1 | 1.64 | 0.00385 [0.00378-0.00389] | 7.82e-05 [-9.49e-05-0.00177] | 0 |
| random | euclidean | 1000000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | euclidean | 1000000 | 1024 | fp32_other_seed | 3 | 1 | 1.63 | 0.00382 [0.0038-0.0039] | 0.000203 [-0.00114-0.00118] | 0 |
| random | euclidean | 1000000 | 4096 | fp16 | 3 | 1.13 | 2.36 | 0.00236 [0.00235-0.00244] | 0.00428 [0.000128-0.00622] | 0 |
| random | euclidean | 1000000 | 4096 | bf16 | 3 | 1.13 | 2.52 | 0.00236 [0.00233-0.00241] | 0.00114 [-0.00436-0.0012] | 0 |
| random | euclidean | 1000000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | euclidean | 1000000 | 4096 | fp32_other_seed | 3 | 1.13 | 2.45 | 0.00232 [0.00229-0.00236] | -0.00189 [-0.00226-0.000384] | 0 |
| random | spherical | 100000 | 1024 | fp16 | 3 | 0.996 | 1.04 | 0.0114 [0.0109-0.0117] | 0.00105 [-0.000542-0.00417] | 0 |
| random | spherical | 100000 | 1024 | bf16 | 3 | 0.996 | 1.05 | 0.0115 [0.0112-0.0119] | 0.00348 [0.00308-0.0076] | 0 |
| random | spherical | 100000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | spherical | 100000 | 1024 | fp32_other_seed | 3 | 0.997 | 1.05 | 0.0114 [0.0111-0.0117] | -0.00363 [-0.00471-0.00333] | 0 |
| random | spherical | 100000 | 4096 | fp16 | 3 | 0.99 | 1.04 | 0.0394 [0.0391-0.0394] | -0.00717 [-0.00936-0.00512] | 0 |
| random | spherical | 100000 | 4096 | bf16 | 3 | 0.99 | 1.04 | 0.0382 [0.0381-0.0387] | 0.00219 [-0.00481-0.00227] | 0 |
| random | spherical | 100000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | spherical | 100000 | 4096 | fp32_other_seed | 3 | 0.99 | 1.04 | 0.0376 [0.0369-0.0381] | -0.000335 [-0.00215-0.00447] | 0 |
| random | spherical | 1000000 | 1024 | fp16 | 3 | 0.997 | 1.04 | 0.00383 [0.00371-0.00383] | 0.00318 [-0.000483-0.00528] | 0 |
| random | spherical | 1000000 | 1024 | bf16 | 3 | 0.998 | 1.04 | 0.00382 [0.0038-0.00383] | 0.00184 [-0.000211-0.00344] | 0 |
| random | spherical | 1000000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | spherical | 1000000 | 1024 | fp32_other_seed | 3 | 0.997 | 1.03 | 0.00378 [0.0037-0.00384] | 0.000129 [-0.00232-0.0054] | 0 |
| random | spherical | 1000000 | 4096 | fp16 | 3 | 1 | 1.04 | 0.00244 [0.00242-0.00257] | 0.00191 [0.00127-0.00253] | 0 |
| random | spherical | 1000000 | 4096 | bf16 | 3 | 1 | 1.04 | 0.00249 [0.00249-0.00258] | -0.000397 [-0.00101-0.00347] | 0 |
| random | spherical | 1000000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | spherical | 1000000 | 4096 | fp32_other_seed | 3 | 1 | 1.04 | 0.00244 [0.0024-0.00251] | -0.00107 [-0.00144-0.00168] | 0 |

### Storage floor

FP32 reference centroids rounded to FP16 or BF16 and back: the least drift possible.

| data | mode | n | k | variant | seeds | drift_gap median | drift_gap max | share same cluster | objective vs FP32 (%) | excluded |
|---|---|---|---|---|---|---|---|---|---|---|
| cohere | euclidean | 100000 | 1024 | round16 | 3 | 0.000863 | 0.00159 | 1 [1-1] | 1.83e-05 [1.78e-05-2.1e-05] | 0 |
| cohere | euclidean | 100000 | 1024 | roundbf | 3 | 0.00691 | 0.0134 | 1 [1-1] | 0.000991 [0.000956-0.000994] | 0 |
| cohere | euclidean | 100000 | 4096 | round16 | 3 | 0.000724 | 0.0027 | 1 [1-1] | 2.04e-05 [2.02e-05-2.21e-05] | 0 |
| cohere | euclidean | 100000 | 4096 | roundbf | 3 | 0.00578 | 0.0185 | 1 [1-1] | 0.00114 [0.00113-0.00115] | 0 |
| cohere | euclidean | 1000000 | 1024 | round16 | 3 | 0.00089 | 0.0021 | 1 [1-1] | 1.85e-05 [1.63e-05-1.9e-05] | 0 |
| cohere | euclidean | 1000000 | 1024 | roundbf | 3 | 0.00717 | 0.0141 | 0.999 [0.999-0.999] | 0.000962 [0.000935-0.000963] | 0 |
| cohere | euclidean | 1000000 | 4096 | round16 | 3 | 0.000784 | 0.0019 | 1 [1-1] | 1.99e-05 [1.89e-05-2.03e-05] | 0 |
| cohere | euclidean | 1000000 | 4096 | roundbf | 3 | 0.00629 | 0.0145 | 1 [1-1] | 0.00108 [0.00108-0.00109] | 0 |
| cohere | spherical | 100000 | 1024 | round16 | 3 | 0.000805 | 0.00158 | 1 [1-1] | 0.000143 [-0.000294-0.000248] | 0 |
| cohere | spherical | 100000 | 1024 | roundbf | 3 | 0.00648 | 0.013 | 1 [0.999-1] | 0.000931 [0.000769-0.00152] | 0 |
| cohere | spherical | 100000 | 4096 | round16 | 3 | 0.000681 | 0.00458 | 1 [1-1] | -0.000151 [-0.000186-8.22e-05] | 0 |
| cohere | spherical | 100000 | 4096 | roundbf | 3 | 0.00545 | 0.0352 | 1 [1-1] | 0.00106 [0.001-0.00113] | 0 |
| cohere | spherical | 1000000 | 1024 | round16 | 3 | 0.000831 | 0.00157 | 1 [1-1] | -0.000243 [-0.000298--1.44e-05] | 0 |
| cohere | spherical | 1000000 | 1024 | roundbf | 3 | 0.00664 | 0.0119 | 0.998 [0.998-0.998] | 0.000891 [6.49e-05-0.0036] | 0 |
| cohere | spherical | 1000000 | 4096 | round16 | 3 | 0.000736 | 0.00188 | 1 [1-1] | 5.52e-05 [3.01e-05-0.000139] | 0 |
| cohere | spherical | 1000000 | 4096 | roundbf | 3 | 0.00586 | 0.0138 | 0.999 [0.999-0.999] | 0.0011 [0.000396-0.00154] | 0 |
| random | euclidean | 100000 | 1024 | round16 | 3 | 0.000185 | 0.000245 | 1 [1-1] | -2.3e-09 [-1.06e-07-9.04e-08] | 0 |
| random | euclidean | 100000 | 1024 | roundbf | 3 | 0.00149 | 0.00198 | 1 [1-1] | 8.79e-06 [8.11e-06-9.46e-06] | 0 |
| random | euclidean | 100000 | 4096 | round16 | 3 | 0.000197 | 0.000534 | 1 [1-1] | -1.12e-07 [-1.52e-07--8.02e-08] | 0 |
| random | euclidean | 100000 | 4096 | roundbf | 3 | 0.00158 | 0.00405 | 1 [1-1] | 2.08e-05 [2.04e-05-2.09e-05] | 0 |
| random | euclidean | 1000000 | 1024 | round16 | 3 | 0.000157 | 0.000259 | 1 [1-1] | -1.18e-07 [-1.92e-07--5.13e-08] | 0 |
| random | euclidean | 1000000 | 1024 | roundbf | 3 | 0.00125 | 0.00198 | 0.999 [0.999-0.999] | 3.19e-06 [2.51e-06-4.1e-06] | 0 |
| random | euclidean | 1000000 | 4096 | round16 | 3 | 0.000179 | 0.000281 | 1 [1-1] | -1.07e-07 [-1.16e-07--6.67e-09] | 0 |
| random | euclidean | 1000000 | 4096 | roundbf | 3 | 0.00143 | 0.00231 | 1 [1-1] | 6.94e-06 [6.59e-06-7.16e-06] | 0 |
| random | spherical | 100000 | 1024 | round16 | 3 | 0.000156 | 0.000174 | 1 [1-1] | 4.4e-05 [1.49e-05-9.67e-05] | 0 |
| random | spherical | 100000 | 1024 | roundbf | 3 | 0.00125 | 0.00138 | 1 [1-1] | 8.42e-05 [-0.000182-0.000266] | 0 |
| random | spherical | 100000 | 4096 | round16 | 3 | 0.000157 | 0.000177 | 1 [1-1] | -3.55e-05 [-4.49e-05--6.08e-06] | 0 |
| random | spherical | 100000 | 4096 | roundbf | 3 | 0.00126 | 0.00142 | 1 [1-1] | 2.34e-06 [-6.65e-05-0.000181] | 0 |
| random | spherical | 1000000 | 1024 | round16 | 3 | 0.000156 | 0.000174 | 1 [1-1] | -7.52e-06 [-5.06e-05-0.000121] | 0 |
| random | spherical | 1000000 | 1024 | roundbf | 3 | 0.00125 | 0.00138 | 0.999 [0.999-0.999] | -0.00025 [-0.000396--0.000158] | 0 |
| random | spherical | 1000000 | 4096 | round16 | 3 | 0.000157 | 0.000176 | 1 [1-1] | -1.68e-05 [-1.81e-05-2.01e-05] | 0 |
| random | spherical | 1000000 | 4096 | roundbf | 3 | 0.00126 | 0.00144 | 1 [1-1] | 5.66e-06 [-0.000117-0.00019] | 0 |

### Reduced precision against reseeding

Ratio: FP16 or BF16 drift_gap median over the FP32 other-seed drift_gap median (1.00: reduced precision moves the centroids as much as a new seed does). Share: median share of documents in the same cluster as under the FP32 reference.

| data | mode | n | k | FP16 / reseed | BF16 / reseed | share FP16 | share BF16 | share reseed |
|---|---|---|---|---|---|---|---|---|
| cohere | euclidean | 100k | 1024 | 0.74 | 0.82 | 0.56 | 0.536 | 0.473 |
| cohere | euclidean | 100k | 4096 | 0.81 | 0.87 | 0.457 | 0.424 | 0.359 |
| cohere | euclidean | 1M | 1024 | 0.27 | 0.37 | 0.799 | 0.752 | 0.581 |
| cohere | euclidean | 1M | 4096 | 0.26 | 0.32 | 0.725 | 0.692 | 0.532 |
| cohere | spherical | 100k | 1024 | 0.82 | 0.89 | 0.522 | 0.5 | 0.465 |
| cohere | spherical | 100k | 4096 | 0.85 | 0.90 | 0.431 | 0.409 | 0.359 |
| cohere | spherical | 1M | 1024 | 0.32 | 0.43 | 0.758 | 0.717 | 0.579 |
| cohere | spherical | 1M | 4096 | 0.34 | 0.41 | 0.689 | 0.664 | 0.538 |
| random | euclidean | 100k | 1024 | 1.00 | 1.00 | 0.0104 | 0.0106 | 0.0105 |
| random | euclidean | 100k | 4096 | 1.01 | 1.01 | 0.0341 | 0.0348 | 0.0323 |
| random | euclidean | 1M | 1024 | 1.00 | 1.00 | 0.00377 | 0.00385 | 0.00382 |
| random | euclidean | 1M | 4096 | 1.00 | 1.00 | 0.00236 | 0.00236 | 0.00232 |
| random | spherical | 100k | 1024 | 1.00 | 1.00 | 0.0114 | 0.0115 | 0.0114 |
| random | spherical | 100k | 4096 | 1.00 | 1.00 | 0.0394 | 0.0382 | 0.0376 |
| random | spherical | 1M | 1024 | 1.00 | 1.00 | 0.00383 | 0.00382 | 0.00378 |
| random | spherical | 1M | 4096 | 1.00 | 1.00 | 0.00244 | 0.00249 | 0.00244 |

- cohere: ratio 0.26-0.90.
- random: ratio 1.00-1.01.
- Real data: FP16/BF16 objective vs FP32 between -0.070 % and +0.055 % over seeds.

Use FP32 when stable `cluster_id`s matter: at about 100k rows, FP16 and BF16 change document assignments almost as much as reseeding does (owner decision, option B, 2026-11-23).

Spike check: results contradict the Phase 2 spike; reported as measured, guidance not changed:

- random euclidean n=100000 k=4096: fp16 drift_gap median 1.21 is not below the FP32 other-seed baseline 1.21
- random euclidean n=100000 k=4096: bf16 drift_gap median 1.22 is not below the FP32 other-seed baseline 1.21
- random euclidean n=1000000 k=1024: bf16 drift_gap median 1 is not below the FP32 other-seed baseline 1
- random euclidean n=1000000 k=4096: bf16 drift_gap median 1.13 is not below the FP32 other-seed baseline 1.13
- random spherical n=1000000 k=1024: fp16 drift_gap median 0.997 is not below the FP32 other-seed baseline 0.997
- random spherical n=1000000 k=1024: bf16 drift_gap median 0.998 is not below the FP32 other-seed baseline 0.997

Each contradiction is a reduced-precision drift equal to or slightly above the FP32 other-seed drift (random euclidean 100k x 4096: fp16 1.214, bf16 1.216 vs 1.206; random euclidean 1M x 1024: bf16 1.001 vs 1.000; random euclidean 1M x 4096: bf16 1.128 vs 1.127; random spherical 1M x 1024: fp16 0.997, bf16 0.998 vs 0.997). These 6 contradictions of the 0.2.6 records were accepted by the owner on 2026-11-27 (those of the 0.2.5 records on 2026-11-23), who kept option B: the caution above, no other change to the precision guidance.
