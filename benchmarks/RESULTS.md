# Benchmark results

Generated 2026-11-26T16:57:18+00:00.
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
  - emeans-gpu-triton-bf16: emeans-rocm 0.2.5
  - emeans-gpu-triton-fp32: emeans-rocm 0.2.5

## Headline

- e-means needs fewer passes (epochs) than Lloyd's k-means (`faiss.Kmeans`, scikit-learn `KMeans`) to reach the same objective in all 5 sweep comparisons (k = 4096 and 16384). It is not faster per pass on CPU; the wall-time gain comes from the fewer passes. Exceptions at 5 passes in the main cells: see the line below.
- cohere 100k x 4096: CPU e-means reaches the objective `faiss.Kmeans` has after 25 passes (0.21433) within 5 passes (0.20978), pass ratio 5.0-12.5x; at the 25-pass points an e-means pass costs +16 % (3.42 vs 2.95 s/pass), at the 5 passes quoted e-means takes 3.71 s/pass (+26 % vs `faiss.Kmeans`); 18.5 s vs 73.7 s on the same CPU (pass-sweep medians; the main-cell runs give 17.1 s vs 73.2 s).
- cohere 1M x 16384: CPU e-means reaches the objective `faiss.Kmeans` has after 25 passes (0.18746) within 5 passes (0.18569), pass ratio 5.0-12.5x; at the 25-pass points a pass costs about the same (114 vs 111 s/pass), at the 5 passes quoted e-means takes 115 s/pass (+4 % vs `faiss.Kmeans`); 575 s vs 2765 s on the same CPU (pass-sweep medians; the main-cell runs give 578 s vs 2797 s).
- random 100k x 4096: CPU e-means reaches the objective `faiss.Kmeans` has after 25 passes (719.94) within 2 passes (716.73), pass ratio 12.5-25.0x; at the 25-pass points an e-means pass costs +13 % (3.08 vs 2.73 s/pass), at the 2 passes quoted e-means takes 3.18 s/pass (+17 % vs `faiss.Kmeans`); 6.36 s vs 68.1 s on the same CPU (pass-sweep medians).
- Per pass, CPU e-means is not faster than Lloyd: scikit-learn `KMeans` is cheaper per pass at 100k (cohere 100k x 4096: 3.42 vs 1.07 s/pass; random 100k x 4096: 3.08 vs 1.03 s/pass, 25-pass points), so the wall-time gain there is smaller than the pass gain.
- At cohere 1M x 16384, `faiss.Kmeans` used all 25 passes and was still improving (10 -> 25 passes: 0.18856 -> 0.18746), so 1M comparisons are against Lloyd at 25 passes, not at convergence.
- random 100k x 4096: after 5 passes the CPU e-means objective is flat and not monotone (5/10/25 passes: 713.15 / 713.67 / 713.45); reported as found.
- At its default 5 passes CPU e-means has a lower objective than the best Lloyd's k-means run in 11 of 14 comparable main cells; Lloyd is better at cohere 100k x 65536 (+22.22 %, degenerate: 1.5 rows per centroid), random 100k x 16384 (+7.90 %, reported as found; cohere at the same size: -0.69 %), random 100k x 65536 (+36.81 %, degenerate: 1.5 rows per centroid).
- GPU (AMD Radeon RX 9060 XT): the best GPU e-means run is faster than CPU e-means (cold in all 16 cells, warm in all 12 cells that have a CPU warm time); at cohere 1M x 65536 (cold, end-to-end, CPU e-means 2232 s), FP32 against FP32: `emeans-gpu-native-fp32` 31.1x (71.9 s), `emeans-gpu-triton-fp32` 27.6x (80.8 s); `emeans-gpu-triton-bf16` 207.7x (10.7 s), BF16 on the GPU against FP32 on the CPU; smallest gain 3.9x at random 100k x 1024 (cold: `emeans-gpu-triton-bf16` 1.35 s end-to-end vs 5.21 s).

## Protocol

- Every method trains the same k from the same starting centroids: a seeded sample of k distinct data rows.
- Passes per method: e-means `nepochs` 5 (the `EmeansFitter` default), `faiss.Kmeans` `niter` 25, scikit-learn `KMeans` `max_iter` 300, `tol` 1e-4, `n_init` 1. One e-means pass costs about the same distance evaluations as one Lloyd iteration, so the pass sweep compares equal work.
- No subsampling: `faiss.Kmeans` `max_points_per_centroid` 2**30, e-means `None`.
- scikit-learn runs at 100k rows only, to save time: `faiss-lloyd` is the Lloyd baseline at every size, so the 1M rows have no scikit-learn entries.
- Final objective: mean squared distance to the nearest centroid, from one exact FP32 CPU scorer (`faiss.IndexFlatL2`), never a library's own objective. The `vs faiss` column is the relative difference to the Lloyd `faiss-lloyd` median objective in the same cell.
- Real data is spherical. `faiss-lloyd` (`faiss.Kmeans(spherical=True)`) normalises its centroids to unit length after every iteration: it is the true spherical Lloyd baseline, and spherical comparisons should be read against it. scikit-learn has no spherical mode: `sklearn-lloyd` runs Euclidean Lloyd on the unit-norm rows and projects its final centroids to unit length once, after training, so it is scored on the sphere but is not spherical Lloyd (no projection per iteration). The Cohere file's rows are not unit-norm (raw L2 norms about 12 to 17.5): the harness normalises every row to unit length when it loads them. Random data is Euclidean and is for speed only: Gaussian noise has no cluster structure.
- The real 100k set is a seeded, sorted random subset of the 1M Cohere rows: seed 20261105 (`COHERE_SUBSET_SEED` in `data.py`), row-index sha256 `19fd6db7de0e1f3c90a6bf0a4ad6e370a4a92fe07bbaa17b590018286ca6ac72`. The records published here do not carry the subset seed, row-index sha256 or raw norm range per record; records written by later runs do.
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

   - On this host the GPU already helps below 1e10: smallest gain below 1e10 is 3.9x at random 100k x 1024 (cold: `emeans-gpu-triton-bf16` 1.35 s end-to-end vs 5.21 s).

## Main results

Cold and warm are `median [min-max]` seconds over seeds.

| data | mode | n | k | n x k | method | backend | precision | passes used | cold s | warm s | transfer s | objective | vs faiss | marks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cohere | spherical | 100000 | 1024 | 1.0e8 | emeans-cpu | cpu | fp32 | 5 | 5.63 [5.61-5.66] | 6.4 [5.8-6.49] | - | 0.23717 (n=3) | -0.83 % |  |
| cohere | spherical | 100000 | 1024 | 1.0e8 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 2.19 [2.19-2.2] | 0.269 [0.269-0.27] | 0.0128 | 0.23719 (n=3) | -0.82 % |  |
| cohere | spherical | 100000 | 1024 | 1.0e8 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.29 [1.29-1.29] | 0.16 [0.159-0.16] | 0.0125 | 0.23717 (n=3) | -0.83 % |  |
| cohere | spherical | 100000 | 1024 | 1.0e8 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 1.36 [1.36-1.37] | 0.279 [0.276-0.279] | 0.0126 | 0.23711 (n=3) | -0.85 % |  |
| cohere | spherical | 100000 | 1024 | 1.0e8 | faiss-lloyd | lloyd | fp32 | 25 | 18.3 [18.2-19.2] | 19.7 [18.3-20.7] | - | 0.23915 (n=3) | +0.00 % |  |
| cohere | spherical | 100000 | 1024 | 1.0e8 | sklearn-lloyd | lloyd | fp32 | 71 | 17.4 [15.7-18.7] | 17.9 [16.4-18.1] | - | 0.23913 (n=3) | -0.01 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | emeans-cpu | cpu | fp32 | 5 | 17.1 [17.1-17.5] | 17 [17-18.6] | - | 0.20978 (n=3) | -2.12 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 2.59 [2.51-2.6] | 0.628 [0.56-0.628] | 0.0157 | 0.20985 (n=3) | -2.09 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.35 [1.34-1.37] | 0.214 [0.213-0.214] | 0.0127 | 0.20997 (n=3) | -2.03 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 1.8 [1.8-1.82] | 0.697 [0.695-0.698] | 0.0156 | 0.20978 (n=3) | -2.12 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | faiss-lloyd | lloyd | fp32 | 25 | 73.2 [72.7-74] | 74.3 [74.2-75.4] | - | 0.21433 (n=3) | +0.00 % |  |
| cohere | spherical | 100000 | 4096 | 4.1e8 | sklearn-lloyd | lloyd | fp32 | 35 | 35.8 [32.6-37.8] | 35.7 [32.5-37.7] | - | 0.21448 (n=3) | +0.07 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | emeans-cpu | cpu | fp32 | 5 | 61.5 [61.3-62.2] | 62.5 [62.2-62.8] | - | 0.16549 (n=3) | -0.69 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 4.09 [4.06-4.12] | 2.05 [2.05-2.05] | 0.0192 | 0.16541 (n=3) | -0.74 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.61 [1.6-1.61] | 0.396 [0.395-0.396] | 0.0168 | 0.16554 (n=3) | -0.66 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 3.57 [3.56-3.58] | 2.39 [2.39-2.39] | 0.0171 | 0.16542 (n=3) | -0.74 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | faiss-lloyd | lloyd | fp32 | 20 | 230 [225-239] | 228 [221-238] | - | 0.16664 (n=3) | +0.00 % |  |
| cohere | spherical | 100000 | 16384 | 1.6e9 | sklearn-lloyd | lloyd | fp32 | 20 | 81.9 [78-90.4] | - | - | 0.16692 (n=3) | +0.17 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | emeans-cpu | cpu | fp32 | 5 | 233 [233-233] | 233 [232-235] | - | 0.066174 (n=3) | +22.22 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 9.7 [8.73-9.72] | 7.72 [6.75-7.73] | 0.0218 | 0.066463 (n=3) | +22.76 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 2.47 [2.46-2.52] | 1.22 [1.22-1.23] | 0.02 | 0.066313 (n=3) | +22.48 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 10.9 [10.9-10.9] | 9.72 [9.71-11.1] | 0.0221 | 0.06626 (n=3) | +22.38 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | faiss-lloyd | lloyd | fp32 | 6 | 272 [227-321] | - | - | 0.054142 (n=3) | +0.00 % |  |
| cohere | spherical | 100000 | 65536 | 6.6e9 | sklearn-lloyd | lloyd | fp32 | 6 | 94.5 [63.9-95.6] | - | - | 0.054165 (n=3) | +0.04 % |  |
| cohere | spherical | 1000000 | 1024 | 1.0e9 | emeans-cpu | cpu | fp32 | 5 | 51.6 [51.6-52.2] | 51.7 [51.1-52.1] | - | 0.23943 (n=3) | -0.22 % |  |
| cohere | spherical | 1000000 | 1024 | 1.0e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 4.21 [4.21-4.26] | 2.42 [2.34-2.45] | 0.162 | 0.23947 (n=3) | -0.20 % |  |
| cohere | spherical | 1000000 | 1024 | 1.0e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 2.5 [2.48-2.5] | 1.44 [1.44-1.45] | 0.154 | 0.23947 (n=3) | -0.20 % |  |
| cohere | spherical | 1000000 | 1024 | 1.0e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 3.54 [3.51-3.54] | 2.54 [2.51-2.54] | 0.124 | 0.23951 (n=3) | -0.19 % |  |
| cohere | spherical | 1000000 | 1024 | 1.0e9 | faiss-lloyd | lloyd | fp32 | 25 | 175 [175-175] | 176 [176-176] | - | 0.23995 (n=3) | +0.00 % |  |
| cohere | spherical | 1000000 | 4096 | 4.1e9 | emeans-cpu | cpu | fp32 | 5 | 156 [156-157] | 156 [156-157] | - | 0.21664 (n=3) | -0.46 % |  |
| cohere | spherical | 1000000 | 4096 | 4.1e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 7.55 [6.82-7.61] | 5.73 [5.04-5.79] | 0.156 | 0.21668 (n=3) | -0.44 % |  |
| cohere | spherical | 1000000 | 4096 | 4.1e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 2.87 [2.85-2.88] | 1.81 [1.81-1.81] | 0.155 | 0.21667 (n=3) | -0.44 % |  |
| cohere | spherical | 1000000 | 4096 | 4.1e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 7.19 [7.18-7.25] | 6.18 [6.17-6.2] | 0.155 | 0.21667 (n=3) | -0.44 % |  |
| cohere | spherical | 1000000 | 4096 | 4.1e9 | faiss-lloyd | lloyd | fp32 | 25 | 691 [690-691] | - | - | 0.21764 (n=3) | +0.00 % |  |
| cohere | spherical | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-cpu | cpu | fp32 | 5 | 578 [571-580] | - | - | 0.18569 (n=3) | -0.94 % |  |
| cohere | spherical | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 20.7 [20.6-20.8] | 18.9 [18.8-18.9] | 0.129 | 0.18573 (n=3) | -0.93 % |  |
| cohere | spherical | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 4.55 [4.54-4.58] | 3.41 [3.41-3.41] | 0.126 | 0.18567 (n=3) | -0.96 % |  |
| cohere | spherical | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 21.9 [21.9-22] | 20.9 [20.9-21] | 0.156 | 0.1857 (n=3) | -0.94 % |  |
| cohere | spherical | 1000000 | 16384 | 1.6e10 (>= 1e10) | faiss-lloyd | lloyd | fp32 | 25 | 2797 [2773-2878] | - | - | 0.18746 (n=3) | +0.00 % |  |
| cohere | spherical | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-cpu | cpu | fp32 | 5 | 2232 [2230-2256] | - | - | 0.15387 (n=3) | - |  |
| cohere | spherical | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 71.8 [71.7-71.9] | 70.1 [70.1-70.1] | 0.133 | 0.15389 (n=3) | - |  |
| cohere | spherical | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 10.6 [10.6-10.9] | 9.51 [9.5-9.53] | 0.133 | 0.15397 (n=3) | - |  |
| cohere | spherical | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 80.6 [80.4-81.1] | 79.7 [78.2-79.8] | 0.137 | 0.15387 (n=3) | - |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | emeans-cpu | cpu | fp32 | 5 | 5.21 [5.18-5.22] | 5.1 [5.06-5.22] | - | 744 (n=3) | -0.45 % |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 2.17 [2.17-2.21] | 0.279 [0.263-0.28] | 0.0129 | 744.03 (n=3) | -0.45 % |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.33 [1.33-1.34] | 0.16 [0.159-0.161] | 0.0124 | 743.91 (n=3) | -0.46 % |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 1.48 [1.45-1.5] | 0.287 [0.282-0.289] | 0.0125 | 744.05 (n=3) | -0.44 % |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | faiss-lloyd | lloyd | fp32 | 25 | 16.7 [16-16.7] | 17 [17-17.1] | - | 747.37 (n=3) | +0.00 % |  |
| random | euclidean | 100000 | 1024 | 1.0e8 | sklearn-lloyd | lloyd | fp32 | 38 | 9.55 [7.25-11.8] | 9.43 [7.3-11.7] | - | 747.36 (n=3) | -0.00 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | emeans-cpu | cpu | fp32 | 5 | 15.8 [15.5-16.4] | 15.8 [15.8-15.8] | - | 713.15 (n=3) | -0.94 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 2.55 [2.48-2.58] | 0.641 [0.583-0.643] | 0.0142 | 713.17 (n=3) | -0.94 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.36 [1.36-1.38] | 0.206 [0.206-0.209] | 0.0156 | 713.2 (n=3) | -0.94 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 1.86 [1.83-1.88] | 0.718 [0.715-0.718] | 0.0126 | 713.21 (n=3) | -0.94 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | faiss-lloyd | lloyd | fp32 | 25 | 67.5 [66.6-67.9] | 68.5 [67.9-69.1] | - | 719.94 (n=3) | +0.00 % |  |
| random | euclidean | 100000 | 4096 | 4.1e8 | sklearn-lloyd | lloyd | fp32 | 28 | 26.8 [25.2-33.4] | 26.6 [25-33.1] | - | 719.94 (n=3) | -0.00 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | emeans-cpu | cpu | fp32 | 5 | 57.3 [55.6-57.3] | 57.4 [56.5-58] | - | 669.01 (n=3) | +7.90 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 4.13 [4.07-4.14] | 2.06 [2.06-2.06] | 0.0169 | 666.57 (n=3) | +7.51 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 1.65 [1.61-1.7] | 0.392 [0.391-0.392] | 0.0141 | 666.72 (n=3) | +7.54 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 3.75 [3.73-3.77] | 2.52 [2.52-2.53] | 0.014 | 666.68 (n=3) | +7.53 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | faiss-lloyd | lloyd | fp32 | 20 | 215 [183-242] | 215 [183-238] | - | 620 (n=3) | +0.00 % |  |
| random | euclidean | 100000 | 16384 | 1.6e9 | sklearn-lloyd | lloyd | fp32 | 19 | 71.1 [63.2-74.2] | - | - | 620 (n=3) | -0.00 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | emeans-cpu | cpu | fp32 | 5 | 217 [217-218] | 217 [217-218] | - | 336.13 (n=3) | +36.81 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 9.68 [9.68-9.7] | 7.67 [7.67-7.67] | 0.0218 | 335.72 (n=3) | +36.65 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 2.45 [2.44-2.47] | 1.19 [1.18-1.19] | 0.0233 | 334.98 (n=3) | +36.35 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 11.5 [11.4-11.5] | 10.3 [10.3-10.3] | 0.0228 | 335.34 (n=3) | +36.49 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | faiss-lloyd | lloyd | fp32 | 8 | 363 [300-381] | - | - | 245.68 (n=3) | +0.00 % |  |
| random | euclidean | 100000 | 65536 | 6.6e9 | sklearn-lloyd | lloyd | fp32 | 7 | 118 [104-118] | - | - | 245.68 (n=3) | +0.00 % |  |
| random | euclidean | 1000000 | 1024 | 1.0e9 | emeans-cpu | cpu | fp32 | 5 | 54.9 [54.1-55.1] | 54.9 [54.9-56.2] | - | 754.51 (n=3) | -0.09 % |  |
| random | euclidean | 1000000 | 1024 | 1.0e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 4.38 [4.37-4.39] | 2.61 [2.59-2.61] | 0.158 | 754.51 (n=3) | -0.09 % |  |
| random | euclidean | 1000000 | 1024 | 1.0e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 2.54 [2.53-2.54] | 1.45 [1.45-1.45] | 0.155 | 754.5 (n=3) | -0.09 % |  |
| random | euclidean | 1000000 | 1024 | 1.0e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 3.66 [3.65-3.75] | 2.6 [2.57-2.63] | 0.155 | 754.5 (n=3) | -0.09 % |  |
| random | euclidean | 1000000 | 1024 | 1.0e9 | faiss-lloyd | lloyd | fp32 | 25 | 180 [177-181] | 179 [177-181] | - | 755.16 (n=3) | +0.00 % |  |
| random | euclidean | 1000000 | 4096 | 4.1e9 | emeans-cpu | cpu | fp32 | 5 | 158 [149-159] | 161 [152-162] | - | 747.51 (n=3) | -0.30 % |  |
| random | euclidean | 1000000 | 4096 | 4.1e9 | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 7.69 [7.65-7.76] | 5.92 [5.92-5.93] | 0.156 | 747.49 (n=3) | -0.30 % |  |
| random | euclidean | 1000000 | 4096 | 4.1e9 | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 2.86 [2.85-2.88] | 1.82 [1.82-1.82] | 0.155 | 747.51 (n=3) | -0.30 % |  |
| random | euclidean | 1000000 | 4096 | 4.1e9 | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 7.49 [7.48-7.51] | 6.48 [6.48-6.5] | 0.155 | 747.51 (n=3) | -0.30 % |  |
| random | euclidean | 1000000 | 4096 | 4.1e9 | faiss-lloyd | lloyd | fp32 | 25 | 701 [689-715] | - | - | 749.74 (n=3) | +0.00 % |  |
| random | euclidean | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-cpu | cpu | fp32 | 5 | 557 [556-558] | - | - | 732.03 (n=3) | -0.71 % |  |
| random | euclidean | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 20.9 [20.8-21] | 19.1 [19.1-19.1] | 0.129 | 732.01 (n=3) | -0.72 % |  |
| random | euclidean | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 4.54 [4.5-4.54] | 3.4 [3.37-3.41] | 0.126 | 731.98 (n=3) | -0.72 % |  |
| random | euclidean | 1000000 | 16384 | 1.6e10 (>= 1e10) | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 23.5 [23.4-23.6] | 22.3 [22.3-22.5] | 0.156 | 732.01 (n=3) | -0.72 % |  |
| random | euclidean | 1000000 | 16384 | 1.6e10 (>= 1e10) | faiss-lloyd | lloyd | fp32 | 25 | 2643 [2641-2652] | - | - | 737.29 (n=3) | +0.00 % |  |
| random | euclidean | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-cpu | cpu | fp32 | 5 | 2140 [2136-2165] | - | - | 688.6 (n=3) | - |  |
| random | euclidean | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 63.4 [63.4-72.2] | 61.8 [61.8-70.6] | 0.133 | 688.64 (n=3) | - |  |
| random | euclidean | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 10.6 [10.6-10.6] | 9.6 [9.53-9.61] | 0.161 | 688.66 (n=3) | - |  |
| random | euclidean | 1000000 | 65536 | 6.6e10 (>= 1e10) | emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 85.2 [83.7-85.4] | 83.1 [83-84.5] | 0.16 | 688.67 (n=3) | - |  |

## Where the GPU does not help / where CPU e-means wins

No cell carries `GPU does not help` or `CPU e-means wins`.

- The GPU helps in every main cell measured: the best GPU e-means run (end-to-end, transfer included) is faster than CPU e-means (cold in all 16 cells, warm in all 12 cells that have a CPU warm time), including the smallest n x k (1.0e8). Cells without a CPU warm time have no warm comparison.
- Smallest GPU speedup over CPU e-means: 3.9x at random 100k x 1024 (cold: `emeans-gpu-triton-bf16` 1.35 s end-to-end vs 5.21 s).
- Largest: 207.7x at cohere 1M x 65536 (cold: `emeans-gpu-triton-bf16` 10.7 s end-to-end vs 2232 s), BF16 on the GPU against FP32 on the CPU.
- Per GPU method at cohere 1M x 65536 (cold, end-to-end, CPU e-means 2232 s), FP32 against FP32: `emeans-gpu-native-fp32` 31.1x (71.9 s), `emeans-gpu-triton-fp32` 27.6x (80.8 s); `emeans-gpu-triton-bf16` 207.7x (10.7 s), BF16 on the GPU against FP32 on the CPU.
- CPU e-means is never the fastest method here, because a GPU e-means row is in every cell; `CPU e-means wins` therefore never applies on this host.
- Without a GPU, `CPU e-means wins (CPU only)` in 11 of 14 comparable main cells (2 have no Lloyd run): CPU e-means finishes sooner than every CPU Lloyd run with an objective within 0.5 % of the best one: cohere 100k x 1024, cohere 100k x 4096, cohere 100k x 16384, cohere 1M x 1024, cohere 1M x 4096, cohere 1M x 16384, random 100k x 1024, random 100k x 4096, random 1M x 1024, random 1M x 4096, random 1M x 16384.
  - cohere: 6 of 7 comparable main cells.
  - random: 5 of 7 comparable main cells.
- Lloyd's k-means has the better objective at: cohere 100k x 65536 (+22.22 %, degenerate: 1.5 rows per centroid); random 100k x 16384 (+7.90 %, reported as found; cohere at the same size: -0.69 %); random 100k x 65536 (+36.81 %, degenerate: 1.5 rows per centroid).
- No Lloyd run, so no comparison: cohere 1M x 65536, random 1M x 65536 (see Not run).

## GPU engines

The Triton rows (`emeans-gpu-triton-fp32`, `emeans-gpu-triton-bf16`) and the precision study ran on `emeans-rocm` 0.2.5 from PyPI on 2026-11-26. The native rows (`emeans-gpu-native-fp32`) ran on 2026-11-22 on the stock e-means engine of faiss-cpu 1.16.0 through plain PyTorch (recorded as `faiss-cpu`; the records' faiss version is 1.16.0). `emeans-rocm` 0.2.6 (upcoming) revives clusters by cost by default, which changes training results, so these figures describe 0.2.5 and do not describe 0.2.6.

Largest cell, cohere (spherical) 1M x 65536, cold medians over seeds:

| method | backend | precision | passes | cold s | end-to-end s | objective [min-max] |
|---|---|---|---|---|---|---|
| emeans-cpu | cpu | fp32 | 5 | 2232 | 2232 | 0.15387 [0.15384-0.15388] |
| emeans-gpu-native-fp32 | gpu-pytorch | fp32 | 5 | 71.8 | 71.9 | 0.15389 [0.15387-0.1539] |
| emeans-gpu-triton-bf16 | gpu-triton | bf16 | 5 | 10.6 | 10.7 | 0.15397 [0.15396-0.154] |
| emeans-gpu-triton-fp32 | gpu-triton | fp32 | 5 | 80.6 | 80.8 | 0.15387 [0.15384-0.15388] |

Native FP32 is faster than Triton FP32 (warm, end-to-end) in 15 of 16 main cells: cohere 100k x 1024, cohere 100k x 4096, cohere 100k x 16384, cohere 100k x 65536, cohere 1M x 1024, cohere 1M x 4096, cohere 1M x 16384, cohere 1M x 65536, random 100k x 1024, random 100k x 4096, random 100k x 16384, random 100k x 65536, random 1M x 4096, random 1M x 16384, random 1M x 65536.

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
| emeans-gpu-triton-bf16 | 1 | 1 | 0.22642 | 1.19 | 0.0667 | 3 |
| emeans-gpu-triton-bf16 | 2 | 2 | 0.21449 | 1.22 | 0.097 | 3 |
| emeans-gpu-triton-bf16 | 5 | 5 | 0.20997 | 1.34 | 0.213 | 3 |
| emeans-gpu-triton-bf16 | 10 | 10 | 0.20829 | 1.53 | 0.399 | 3 |
| emeans-gpu-triton-bf16 | 25 | 25 | 0.20727 | 2.1 | 0.948 | 3 |
| emeans-gpu-triton-fp32 | 1 | 1 | 0.22619 | 1.31 | 0.215 | 3 |
| emeans-gpu-triton-fp32 | 2 | 2 | 0.21436 | 1.42 | 0.32 | 3 |
| emeans-gpu-triton-fp32 | 5 | 5 | 0.20985 | 1.79 | 0.698 | 3 |
| emeans-gpu-triton-fp32 | 10 | 10 | 0.20826 | 2.43 | 1.3 | 3 |
| emeans-gpu-triton-fp32 | 25 | 25 | 0.2071 | 4.28 | 3.18 | 3 |
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
| emeans-gpu-triton-bf16 | 1 | 1 | 0.19689 | 1.88 | 0.72 | 3 |
| emeans-gpu-triton-bf16 | 2 | 2 | 0.18882 | 2.56 | 1.41 | 3 |
| emeans-gpu-triton-bf16 | 5 | 5 | 0.18567 | 4.53 | 3.4 | 3 |
| emeans-gpu-triton-bf16 | 10 | 10 | 0.18451 | 7.71 | 6.57 | 3 |
| emeans-gpu-triton-bf16 | 25 | 25 | 0.18378 | 17.6 | 16.5 | 3 |
| emeans-gpu-triton-fp32 | 1 | 1 | 0.19686 | 5.48 | 4.38 | 3 |
| emeans-gpu-triton-fp32 | 2 | 2 | 0.18885 | 9.78 | 8.69 | 3 |
| emeans-gpu-triton-fp32 | 5 | 5 | 0.1857 | 22.1 | 21 | 3 |
| emeans-gpu-triton-fp32 | 10 | 10 | 0.18456 | 42.3 | 41.3 | 3 |
| emeans-gpu-triton-fp32 | 25 | 25 | 0.18377 | 105 | 103 | 3 |
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
| emeans-gpu-triton-bf16 | 1 | 1 | 725.26 | 1.23 | 0.0626 | 3 |
| emeans-gpu-triton-bf16 | 2 | 2 | 716.85 | 1.27 | 0.0967 | 3 |
| emeans-gpu-triton-bf16 | 5 | 5 | 713.2 | 1.38 | 0.207 | 3 |
| emeans-gpu-triton-bf16 | 10 | 10 | 713.71 | 1.58 | 0.4 | 3 |
| emeans-gpu-triton-bf16 | 25 | 25 | 713.47 | 2.09 | 0.948 | 3 |
| emeans-gpu-triton-fp32 | 1 | 1 | 725.23 | 1.35 | 0.216 | 3 |
| emeans-gpu-triton-fp32 | 2 | 2 | 716.79 | 1.49 | 0.324 | 3 |
| emeans-gpu-triton-fp32 | 5 | 5 | 713.25 | 1.87 | 0.718 | 3 |
| emeans-gpu-triton-fp32 | 10 | 10 | 713.69 | 2.47 | 1.36 | 3 |
| emeans-gpu-triton-fp32 | 25 | 25 | 713.58 | 4.46 | 3.32 | 3 |
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
| 60-minute cap per cell-seed (`DEFAULT_CAP_MIN`); a cell-seed whose one-pass probe estimate exceeds it is not run | one Lloyd cell-seed at 1M x 65536 would take about 3 h | not run: `faiss-lloyd` cohere 1M x 65536 (est. 3.0-3.1 h); CPU e-means ran it in 2232 s (37 min), GPU e-means in 10.7-80.8 s; `faiss-lloyd` random 1M x 65536 (est. 2.9-3.2 h); CPU e-means ran it in 2140 s (36 min), GPU e-means in 10.8-85.4 s. No Lloyd objective there, so no e-means vs Lloyd comparison |
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
| Precision guidance: option B (owner, 2026-11-23); the 4 random-data spike contradictions are accepted as known | on random data FP16/BF16 drift equals or slightly exceeds the FP32 other-seed drift; the guidance adds a caution instead of a claim | nothing; the contradictions are listed in the Precision section |
| `sklearn-lloyd` on spherical data is Euclidean Lloyd on unit-norm rows with one projection of the final centroids to unit length (scikit-learn has no spherical mode) | scikit-learn is a reference point; `faiss-lloyd` (`spherical=True`, normalised every iteration) is the true spherical Lloyd baseline | how true spherical Lloyd would behave in scikit-learn; read spherical comparisons against `faiss-lloyd` |

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
| cohere | euclidean | 100000 | 1024 | fp16 | 3 | 0.554 | 2.55 | 0.481 [0.477-0.492] | -0.0155 [-0.0279-0.0602] | 0 |
| cohere | euclidean | 100000 | 1024 | bf16 | 3 | 0.58 | 3.26 | 0.476 [0.467-0.478] | -0.0119 [-0.0213-0.042] | 0 |
| cohere | euclidean | 100000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | euclidean | 100000 | 1024 | fp32_other_seed | 3 | 0.614 | 2.51 | 0.453 [0.447-0.453] | 0.0275 [-0.0202-0.0289] | 0 |
| cohere | euclidean | 100000 | 4096 | fp16 | 3 | 0.764 | 2.91 | 0.38 [0.378-0.384] | -0.111 [-0.121-0.033] | 0 |
| cohere | euclidean | 100000 | 4096 | bf16 | 3 | 0.766 | 2.81 | 0.381 [0.376-0.382] | 0.0136 [-0.0215-0.0525] | 0 |
| cohere | euclidean | 100000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | euclidean | 100000 | 4096 | fp32_other_seed | 3 | 0.818 | 2.41 | 0.35 [0.342-0.35] | -0.0651 [-0.0736--0.0368] | 0 |
| cohere | euclidean | 1000000 | 1024 | fp16 | 3 | 0.152 | 2.23 | 0.711 [0.683-0.721] | -0.0208 [-0.0331--0.0157] | 0 |
| cohere | euclidean | 1000000 | 1024 | bf16 | 3 | 0.166 | 2.55 | 0.693 [0.667-0.701] | -0.0301 [-0.0364-0.000361] | 0 |
| cohere | euclidean | 1000000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | euclidean | 1000000 | 1024 | fp32_other_seed | 3 | 0.343 | 3.31 | 0.582 [0.57-0.588] | -0.0429 [-0.0728-0.0376] | 0 |
| cohere | euclidean | 1000000 | 4096 | fp16 | 3 | 0.185 | 2.65 | 0.651 [0.646-0.653] | -0.0158 [-0.0224--0.00465] | 0 |
| cohere | euclidean | 1000000 | 4096 | bf16 | 3 | 0.205 | 3.02 | 0.636 [0.635-0.643] | -0.0298 [-0.0374--0.00393] | 0 |
| cohere | euclidean | 1000000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | euclidean | 1000000 | 4096 | fp32_other_seed | 3 | 0.433 | 2.68 | 0.53 [0.526-0.535] | -0.0159 [-0.0223-0.0227] | 0 |
| cohere | spherical | 100000 | 1024 | fp16 | 3 | 0.574 | 2.56 | 0.481 [0.477-0.482] | -0.0253 [-0.0276-0.00875] | 0 |
| cohere | spherical | 100000 | 1024 | bf16 | 3 | 0.561 | 2.77 | 0.481 [0.481-0.484] | -0.01 [-0.0197-0.00569] | 0 |
| cohere | spherical | 100000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | spherical | 100000 | 1024 | fp32_other_seed | 3 | 0.615 | 2.72 | 0.457 [0.447-0.467] | 0.0166 [-0.0789-0.12] | 0 |
| cohere | spherical | 100000 | 4096 | fp16 | 3 | 0.755 | 2.55 | 0.388 [0.385-0.388] | 0.0652 [0.0346-0.094] | 0 |
| cohere | spherical | 100000 | 4096 | bf16 | 3 | 0.763 | 3.1 | 0.384 [0.378-0.387] | 0.0775 [0.0249-0.108] | 0 |
| cohere | spherical | 100000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | spherical | 100000 | 4096 | fp32_other_seed | 3 | 0.812 | 2.54 | 0.356 [0.354-0.357] | -0.0285 [-0.119-0.0578] | 0 |
| cohere | spherical | 1000000 | 1024 | fp16 | 3 | 0.146 | 2.84 | 0.706 [0.679-0.712] | 0.0132 [-0.00955-0.0182] | 0 |
| cohere | spherical | 1000000 | 1024 | bf16 | 3 | 0.192 | 2.3 | 0.674 [0.653-0.677] | -0.00467 [-0.00755-0.032] | 0 |
| cohere | spherical | 1000000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | spherical | 1000000 | 1024 | fp32_other_seed | 3 | 0.329 | 3.08 | 0.586 [0.576-0.589] | -0.0317 [-0.0509-0.0318] | 0 |
| cohere | spherical | 1000000 | 4096 | fp16 | 3 | 0.201 | 2.99 | 0.638 [0.636-0.644] | -0.038 [-0.041-0.00743] | 0 |
| cohere | spherical | 1000000 | 4096 | bf16 | 3 | 0.218 | 2.69 | 0.634 [0.621-0.635] | -0.0328 [-0.0331--0.00105] | 0 |
| cohere | spherical | 1000000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| cohere | spherical | 1000000 | 4096 | fp32_other_seed | 3 | 0.417 | 2.46 | 0.539 [0.534-0.547] | -0.0173 [-0.0524--0.0147] | 0 |
| random | euclidean | 100000 | 1024 | fp16 | 3 | 1.19 | 2.75 | 0.0101 [0.0101-0.0107] | -0.00609 [-0.00617-0.00358] | 0 |
| random | euclidean | 100000 | 1024 | bf16 | 3 | 1.18 | 2.59 | 0.0106 [0.0102-0.0107] | -0.00223 [-0.0204-0.000176] | 0 |
| random | euclidean | 100000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | euclidean | 100000 | 1024 | fp32_other_seed | 3 | 1.19 | 2.58 | 0.0102 [0.0101-0.0103] | -0.00353 [-0.0109-0.0169] | 0 |
| random | euclidean | 100000 | 4096 | fp16 | 3 | 1.2 | 6.05 | 0.0343 [0.0342-0.0353] | -0.0261 [-0.0448-0.00859] | 0 |
| random | euclidean | 100000 | 4096 | bf16 | 3 | 1.21 | 4.56 | 0.0346 [0.0342-0.0357] | -0.0212 [-0.0266-0.00554] | 0 |
| random | euclidean | 100000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | euclidean | 100000 | 4096 | fp32_other_seed | 3 | 1.19 | 4.25 | 0.0329 [0.0319-0.0332] | -0.121 [-0.165-0.124] | 0 |
| random | euclidean | 1000000 | 1024 | fp16 | 3 | 1 | 1.78 | 0.0039 [0.00384-0.00398] | 0.000668 [-0.000374-0.00132] | 0 |
| random | euclidean | 1000000 | 1024 | bf16 | 3 | 1 | 1.8 | 0.00384 [0.00374-0.00393] | -2.05e-05 [-0.000523-0.0014] | 0 |
| random | euclidean | 1000000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | euclidean | 1000000 | 1024 | fp32_other_seed | 3 | 1 | 1.8 | 0.00382 [0.00375-0.00395] | -0.000599 [-0.000641-0.00111] | 0 |
| random | euclidean | 1000000 | 4096 | fp16 | 3 | 1.14 | 2.55 | 0.00241 [0.0024-0.00242] | -0.00158 [-0.00316--0.00103] | 0 |
| random | euclidean | 1000000 | 4096 | bf16 | 3 | 1.14 | 2.6 | 0.00241 [0.00238-0.00244] | -0.000656 [-0.000904-0.000525] | 0 |
| random | euclidean | 1000000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | euclidean | 1000000 | 4096 | fp32_other_seed | 3 | 1.14 | 2.37 | 0.00237 [0.00234-0.00238] | 0.000134 [-2.47e-05-0.000275] | 0 |
| random | spherical | 100000 | 1024 | fp16 | 3 | 0.996 | 1.04 | 0.0109 [0.0106-0.0114] | 0.000856 [-0.00115-0.00417] | 0 |
| random | spherical | 100000 | 1024 | bf16 | 3 | 0.996 | 1.05 | 0.011 [0.0106-0.0119] | 0.00287 [-0.00539-0.0076] | 0 |
| random | spherical | 100000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | spherical | 100000 | 1024 | fp32_other_seed | 3 | 0.996 | 1.04 | 0.0111 [0.0111-0.0115] | -0.00363 [-0.00531-0.00659] | 0 |
| random | spherical | 100000 | 4096 | fp16 | 3 | 0.99 | 1.05 | 0.0395 [0.0393-0.0395] | 0.00151 [-0.0148-0.0126] | 0 |
| random | spherical | 100000 | 4096 | bf16 | 3 | 0.99 | 1.05 | 0.0381 [0.038-0.0382] | -0.00614 [-0.00687-0.00486] | 0 |
| random | spherical | 100000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | spherical | 100000 | 4096 | fp32_other_seed | 3 | 0.99 | 1.05 | 0.0376 [0.0376-0.0378] | -0.00623 [-0.0164-0.0158] | 0 |
| random | spherical | 1000000 | 1024 | fp16 | 3 | 0.997 | 1.04 | 0.00394 [0.00386-0.00395] | 0.000318 [-0.00528-0.00397] | 0 |
| random | spherical | 1000000 | 1024 | bf16 | 3 | 0.997 | 1.04 | 0.00388 [0.0038-0.00392] | 0.000572 [-0.00311-0.0028] | 0 |
| random | spherical | 1000000 | 1024 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | spherical | 1000000 | 1024 | fp32_other_seed | 3 | 0.997 | 1.05 | 0.00387 [0.00383-0.00391] | -0.0019 [-0.00348-0.00343] | 0 |
| random | spherical | 1000000 | 4096 | fp16 | 3 | 1 | 1.04 | 0.00251 [0.00248-0.00258] | -0.000535 [-0.00103-3.85e-05] | 0 |
| random | spherical | 1000000 | 4096 | bf16 | 3 | 1 | 1.04 | 0.00248 [0.00245-0.00253] | -0.00109 [-0.00282-0.00159] | 0 |
| random | spherical | 1000000 | 4096 | fp32_rerun | 3 | 0 | 0 | 1 [1-1] | 0 [0-0] | 0 |
| random | spherical | 1000000 | 4096 | fp32_other_seed | 3 | 1 | 1.05 | 0.00245 [0.00243-0.00254] | -0.00166 [-0.00199--0.00137] | 0 |

### Storage floor

FP32 reference centroids rounded to FP16 or BF16 and back: the least drift possible.

| data | mode | n | k | variant | seeds | drift_gap median | drift_gap max | share same cluster | objective vs FP32 (%) | excluded |
|---|---|---|---|---|---|---|---|---|---|---|
| cohere | euclidean | 100000 | 1024 | round16 | 3 | 0.00087 | 0.00184 | 1 [1-1] | 1.81e-05 [1.63e-05-1.97e-05] | 0 |
| cohere | euclidean | 100000 | 1024 | roundbf | 3 | 0.00699 | 0.0143 | 1 [1-1] | 0.000994 [0.000969-0.00101] | 0 |
| cohere | euclidean | 100000 | 4096 | round16 | 3 | 0.000735 | 0.00186 | 1 [1-1] | 2.14e-05 [2.12e-05-2.16e-05] | 0 |
| cohere | euclidean | 100000 | 4096 | roundbf | 3 | 0.00582 | 0.0153 | 1 [1-1] | 0.00111 [0.0011-0.00113] | 0 |
| cohere | euclidean | 1000000 | 1024 | round16 | 3 | 0.000889 | 0.00173 | 1 [1-1] | 1.69e-05 [1.55e-05-1.79e-05] | 0 |
| cohere | euclidean | 1000000 | 1024 | roundbf | 3 | 0.0071 | 0.0162 | 0.999 [0.999-0.999] | 0.000971 [0.000939-0.000971] | 0 |
| cohere | euclidean | 1000000 | 4096 | round16 | 3 | 0.000784 | 0.00197 | 1 [1-1] | 2.02e-05 [1.97e-05-2.17e-05] | 0 |
| cohere | euclidean | 1000000 | 4096 | roundbf | 3 | 0.00626 | 0.0148 | 0.999 [0.999-1] | 0.00109 [0.00108-0.00109] | 0 |
| cohere | spherical | 100000 | 1024 | round16 | 3 | 0.000815 | 0.00163 | 1 [1-1] | -5.84e-05 [-0.000242-0.000198] | 0 |
| cohere | spherical | 100000 | 1024 | roundbf | 3 | 0.00657 | 0.0132 | 1 [1-1] | 0.00168 [0.000214-0.00343] | 0 |
| cohere | spherical | 100000 | 4096 | round16 | 3 | 0.00069 | 0.00164 | 1 [1-1] | -2.7e-05 [-0.000201-9.73e-05] | 0 |
| cohere | spherical | 100000 | 4096 | roundbf | 3 | 0.0055 | 0.0127 | 1 [1-1] | 0.00116 [0.001-0.00148] | 0 |
| cohere | spherical | 1000000 | 1024 | round16 | 3 | 0.000831 | 0.0017 | 1 [1-1] | 7.6e-05 [2.97e-05-0.000213] | 0 |
| cohere | spherical | 1000000 | 1024 | roundbf | 3 | 0.00667 | 0.0138 | 0.998 [0.998-0.998] | 0.0015 [-0.00059-0.00167] | 0 |
| cohere | spherical | 1000000 | 4096 | round16 | 3 | 0.000737 | 0.00184 | 1 [1-1] | 6.32e-05 [2.42e-05-0.000161] | 0 |
| cohere | spherical | 1000000 | 4096 | roundbf | 3 | 0.00592 | 0.0132 | 0.999 [0.999-0.999] | 0.000903 [0.000192-0.0019] | 0 |
| random | euclidean | 100000 | 1024 | round16 | 3 | 0.000188 | 0.000253 | 1 [1-1] | 5.01e-08 [1.93e-08-5.36e-08] | 0 |
| random | euclidean | 100000 | 1024 | roundbf | 3 | 0.0015 | 0.00199 | 1 [1-1] | 7.99e-06 [6.91e-06-9.38e-06] | 0 |
| random | euclidean | 100000 | 4096 | round16 | 3 | 0.000196 | 0.000813 | 1 [1-1] | -1.36e-07 [-1.86e-07--9.52e-08] | 0 |
| random | euclidean | 100000 | 4096 | roundbf | 3 | 0.00156 | 0.00558 | 1 [1-1] | 2.09e-05 [1.98e-05-2.1e-05] | 0 |
| random | euclidean | 1000000 | 1024 | round16 | 3 | 0.000157 | 0.000287 | 1 [1-1] | -1.53e-07 [-1.6e-07--1.16e-07] | 0 |
| random | euclidean | 1000000 | 1024 | roundbf | 3 | 0.00126 | 0.00225 | 0.999 [0.999-0.999] | 4.42e-06 [3.61e-06-4.52e-06] | 0 |
| random | euclidean | 1000000 | 4096 | round16 | 3 | 0.000181 | 0.000298 | 1 [1-1] | -4.64e-08 [-1.62e-07--2.81e-08] | 0 |
| random | euclidean | 1000000 | 4096 | roundbf | 3 | 0.00145 | 0.00238 | 1 [1-1] | 6.95e-06 [6.26e-06-7.8e-06] | 0 |
| random | spherical | 100000 | 1024 | round16 | 3 | 0.000156 | 0.000174 | 1 [1-1] | 1.46e-05 [-4.51e-05-6.87e-05] | 0 |
| random | spherical | 100000 | 1024 | roundbf | 3 | 0.00125 | 0.0014 | 1 [1-1] | 8.34e-05 [4.2e-06-9.06e-05] | 0 |
| random | spherical | 100000 | 4096 | round16 | 3 | 0.000157 | 0.000178 | 1 [1-1] | -1.77e-05 [-2.94e-05-2.43e-05] | 0 |
| random | spherical | 100000 | 4096 | roundbf | 3 | 0.00126 | 0.00143 | 1 [1-1] | 5.17e-05 [-0.000112-6.06e-05] | 0 |
| random | spherical | 1000000 | 1024 | round16 | 3 | 0.000156 | 0.000173 | 1 [1-1] | -5.59e-06 [-1.84e-05-5.03e-05] | 0 |
| random | spherical | 1000000 | 1024 | roundbf | 3 | 0.00124 | 0.00139 | 0.999 [0.999-0.999] | 0.000362 [-0.000304-0.000618] | 0 |
| random | spherical | 1000000 | 4096 | round16 | 3 | 0.000157 | 0.000178 | 1 [1-1] | 5.34e-06 [-1.92e-05-1.39e-05] | 0 |
| random | spherical | 1000000 | 4096 | roundbf | 3 | 0.00126 | 0.00142 | 1 [1-1] | -0.000275 [-0.000306--0.000173] | 0 |

### Reduced precision against reseeding

Ratio: FP16 or BF16 drift_gap median over the FP32 other-seed drift_gap median (1.00: reduced precision moves the centroids as much as a new seed does). Share: median share of documents in the same cluster as under the FP32 reference.

| data | mode | n | k | FP16 / reseed | BF16 / reseed | share FP16 | share BF16 | share reseed |
|---|---|---|---|---|---|---|---|---|
| cohere | euclidean | 100k | 1024 | 0.90 | 0.94 | 0.481 | 0.476 | 0.453 |
| cohere | euclidean | 100k | 4096 | 0.93 | 0.94 | 0.38 | 0.381 | 0.35 |
| cohere | euclidean | 1M | 1024 | 0.44 | 0.48 | 0.711 | 0.693 | 0.582 |
| cohere | euclidean | 1M | 4096 | 0.43 | 0.47 | 0.651 | 0.636 | 0.53 |
| cohere | spherical | 100k | 1024 | 0.93 | 0.91 | 0.481 | 0.481 | 0.457 |
| cohere | spherical | 100k | 4096 | 0.93 | 0.94 | 0.388 | 0.384 | 0.356 |
| cohere | spherical | 1M | 1024 | 0.44 | 0.58 | 0.706 | 0.674 | 0.586 |
| cohere | spherical | 1M | 4096 | 0.48 | 0.52 | 0.638 | 0.634 | 0.539 |
| random | euclidean | 100k | 1024 | 0.99 | 0.99 | 0.0101 | 0.0106 | 0.0102 |
| random | euclidean | 100k | 4096 | 1.01 | 1.01 | 0.0343 | 0.0346 | 0.0329 |
| random | euclidean | 1M | 1024 | 1.00 | 1.00 | 0.0039 | 0.00384 | 0.00382 |
| random | euclidean | 1M | 4096 | 1.00 | 1.00 | 0.00241 | 0.00241 | 0.00237 |
| random | spherical | 100k | 1024 | 1.00 | 1.00 | 0.0109 | 0.011 | 0.0111 |
| random | spherical | 100k | 4096 | 1.00 | 1.00 | 0.0395 | 0.0381 | 0.0376 |
| random | spherical | 1M | 1024 | 1.00 | 1.00 | 0.00394 | 0.00388 | 0.00387 |
| random | spherical | 1M | 4096 | 1.00 | 1.00 | 0.00251 | 0.00248 | 0.00245 |

- cohere: ratio 0.43-0.94.
- random: ratio 0.99-1.01.
- Real data: FP16/BF16 objective vs FP32 between -0.121 % and +0.108 % over seeds.

Use FP32 when stable `cluster_id`s matter: at about 100k rows, FP16 and BF16 change document assignments almost as much as reseeding does (owner decision, option B, 2026-11-23).

Spike check: results contradict the Phase 2 spike; reported as measured, guidance not changed:

- random euclidean n=100000 k=4096: fp16 drift_gap median 1.2 is not below the FP32 other-seed baseline 1.19
- random euclidean n=100000 k=4096: bf16 drift_gap median 1.21 is not below the FP32 other-seed baseline 1.19
- random euclidean n=1000000 k=1024: bf16 drift_gap median 1 is not below the FP32 other-seed baseline 1
- random euclidean n=1000000 k=4096: fp16 drift_gap median 1.14 is not below the FP32 other-seed baseline 1.14

Each contradiction is a reduced-precision drift equal to or slightly above the FP32 other-seed drift (random euclidean 100k x 4096: fp16 1.205, bf16 1.208 vs 1.190; random euclidean 1M x 1024: bf16 1.001 vs 1.001; random euclidean 1M x 4096: fp16 1.144 vs 1.144). The owner accepted them as known and chose option B on 2026-11-23: the caution above, no other change to the precision guidance.
