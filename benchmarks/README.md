# Benchmarks

Scripts that compare e-means with Lloyd's k-means on random and real 768-d vectors. Everything runs
through `hatch run`, from this `benchmarks/` folder. Nothing here is published or installed as a package.

## What this compares

| Method | What it is | Env |
|---|---|---|
| `emeans-gpu-triton-fp32`, `emeans-gpu-triton-bf16` | e-means on GPU through emeans-rocm 0.2.6 (PyPI) (Triton kernels, FP32 and BF16) | `bench-gpu-fork` |
| `emeans-gpu-native-fp32` | e-means on GPU through stock upstream e-means (native FP32 rung) | `bench-gpu` |
| `emeans-cpu` | e-means on CPU (`NumpyEmeans`) | `bench-cpu` |
| `faiss-lloyd` | `faiss.Kmeans`, Lloyd's k-means baseline | `bench-cpu` |
| `sklearn-lloyd` | scikit-learn `KMeans(algorithm="lloyd")`, Lloyd's k-means baseline | `bench-cpu` |

e-means is not k-means. Lloyd's k-means appears only as the baseline.

## Protocol

The same protocol is printed at the top of `RESULTS.md`.

- Every method in a cell starts from the same seeded sample of k data rows (for e-means this goes through a
  private `init_centroids` keyword of the package's `fit_model`).
- Passes are stated per method: e-means 5 passes, `faiss.Kmeans` `niter=25`, scikit-learn `max_iter=300`,
  `tol=1e-4`, `n_init=1`. `faiss.Kmeans` can stop early, so the pass check accepts `0 < used <= niter` and the
  record keeps the passes actually used. A pass sweep (1, 2, 5, 10, 25) runs on three cells.
- No silent subsampling (faiss `max_points_per_centroid` is raised so it never subsamples).
- The final objective is computed by one shared exact scorer (CPU, FP32 distances, float64 accumulation) for
  every method.
- Cold time is the first run in a fresh process. Warm time is a repeat in the same process (skipped for CPU
  cells estimated over 5 minutes). Host-to-GPU transfer is timed separately.
- At least 3 seeds (0, 1, 2).
- Grid: k = 1024, 4096, 16384, 65536 at n = 100k and 1M, on random Euclidean data and on real spherical data.
- Real data is the Cohere Wikipedia embeddings. The file is not unit-norm, so the loader normalises each row.
- Spherical baselines: `faiss-lloyd` (`faiss.Kmeans(spherical=True)`) normalises its centroids to unit length
  after every iteration, so it is the true spherical Lloyd baseline; read spherical comparisons against it.
  scikit-learn has no spherical mode: `sklearn-lloyd` runs Euclidean Lloyd on the unit-norm rows, then projects
  its final centroids to unit length once, after training. It is scored on the sphere like the others, but it is
  not spherical Lloyd (no projection per iteration).
  The real 100k set is a seeded, sorted random subset of the 1M rows: seed 20261105 (`COHERE_SUBSET_SEED` in
  `data.py`), row-index sha256 `19fd6db7de0e1f3c90a6bf0a4ad6e370a4a92fe07bbaa17b590018286ca6ac72`. The published records predate the per-record fields; records
  written since carry `subset_seed`, `subset_rows_sha256` and the raw norm range.
- Each cell has a per-seed time cap (default 60 minutes). A cell whose estimate is over the cap is reported as
  `not run: est. X h`, never dropped.

## Setup

Three hatch envs, all local. The two GPU envs are AMD ROCm only and are never built in CI.

| Env | Use |
|---|---|
| `bench-cpu` | CPU methods, data fetch, report, lint, types |
| `bench-gpu` | stock upstream e-means on GPU (`torch` ROCm build pinned in `pyproject.toml`, installed from the AMD index) |
| `bench-gpu-fork` | emeans-rocm 0.2.6 (PyPI) on GPU, and the precision study |

The first run of each env builds it. After editing the package, run `hatch env prune` so the envs pick it up.

`bench-gpu-fork` installs `emeans-rocm==0.2.6` from PyPI on top of the ROCm torch of `bench-gpu`, and its records
store `versions.emeans_source` as `"emeans-rocm 0.2.6"` (the stock envs store `"faiss-cpu 1.16.0"`; records from
before this change say `"e-means-rocm"` / `"faiss-cpu"`).

```bash
cd benchmarks
export BENCH_HOST_LABEL="my-host"                          # free-text label stored in the records
export BENCH_THREADS=10                                    # optional, CPU threads (default 10)
```

Set `EMEANS_ROCM_GFX` if your card is not `gfx1200`.

## Data

```bash
hatch run bench-cpu:run fetch_data.py
hatch run bench-cpu:run fetch_data.py --verify-only
```

The first command downloads the Cohere file (about 3 GB, pinned dataset revision and sha256) into `cache/`.
`--verify-only` checks the file already on disk. `cache/` is gitignored.

## Smoke

A tiny grid (10k x 256, one seed, a couple of passes) that proves every method runs in its env on both data
types. Smoke output goes to `scratch/smoke/` and never to `results/`.

```bash
rm -rf scratch/smoke
hatch run bench-cpu:run run_grid.py --smoke --data random cohere
hatch run bench-gpu:run run_grid.py --smoke --data random cohere
hatch run bench-gpu-fork:run run_grid.py --smoke --data random cohere
hatch run bench-gpu-fork:run precision.py --smoke
hatch run bench-cpu:run report.py --smoke
```

Each env skips the methods it cannot run and says so. `scratch/smoke/RESULTS.md` should list all six methods.
Run the tests too: `hatch run bench-cpu:test`, `hatch run bench-gpu:test`, `hatch run bench-gpu-fork:test`.

## Full run

Run one env at a time, never concurrently: CPU and GPU runs would disturb each other's timings and can
exhaust host RAM. Every run appends to `results/speed.jsonl`. Order:

```bash
hatch run bench-gpu-fork:run run_grid.py --methods emeans-gpu-triton-fp32 emeans-gpu-triton-bf16
hatch run bench-gpu:run run_grid.py --methods emeans-gpu-native-fp32
hatch run bench-cpu:run run_grid.py --methods emeans-cpu faiss-lloyd sklearn-lloyd   # the long one
```

To keep the desktop usable during the long CPU run, give it fewer threads, the lowest priority and a memory
ceiling. A cell that hits the ceiling is killed on its own and recorded as failed, instead of the whole system
swapping:

```bash
BENCH_THREADS=8 systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \
  nice -n 19 hatch run bench-cpu:run run_grid.py --methods emeans-cpu faiss-lloyd sklearn-lloyd
```

The published CPU runs used this 12 GiB cap (peak 7.5 GiB, at k = 65536). The precision study ran under the same
12 GiB cap and was raised to 16 GiB mid-run after it was throttled at the cap (peak 12.07 GiB, no OOM kill); give
the precision study 16 GiB (`-p MemoryMax=16G`).

Every CPU method gets the same thread count, and each record stores it (`host.omp_threads`). Keep other heavy work
off the machine while it runs, or the timings suffer. A failed or timed-out cell counts as done on resume: delete
its lines from `results/speed.jsonl` to run it again.

Always pass `--methods`: without it each env runs every method it can, so the GPU envs would also run the CPU
methods, and those rows would come from the wrong env.

The runs are resumable. Interrupt one (Ctrl-C) and run the same command again; finished cells are skipped.
Each cell runs in its own fresh process with a hard timeout.

Options:

- `--cap-min 60` per-seed time cap in minutes (default 60); `--no-cap` measures the expensive Lloyd cells in full.
- `--timeout-min M` hard limit per worker process.
- `--methods ...`, `--data random cohere`, `--seeds 0 1 2`, `--kind main|sweep|all`, `--results PATH`.

Cells in the full grid: `bench-gpu-fork` 186 (96 main, 90 sweep), `bench-gpu` 93 (48 main, 45 sweep),
`bench-cpu` 240 (120 main, 120 sweep). scikit-learn runs at 100k only: faiss is the Lloyd baseline at every
size, and scikit-learn at 1M would take most of the CPU time for little extra information. The 1M-row faiss cells
dominate the time.

Estimates from one probe on a 10-core CPU and an RX 9060 XT (they will vary; the run measures them):

| n x k | n | k | faiss per iteration | faiss 25 iterations | e-means CPU 5 passes |
|---|---|---|---|---|---|
| 1.0e8 | 100k | 1024 | 0.6 s | 16 s | 3 s |
| 4.1e8 | 100k | 4096 | 2.6 s | 1.1 min | 13 s |
| 1.6e9 | 100k | 16384 | 10 s | 4.3 min | 52 s |
| 6.5e9 | 100k | 65536 | 42 s | 18 min | 3.5 min |
| 1.0e9 | 1M | 1024 | 6.4 s | 2.7 min | 32 s |
| 4.1e9 | 1M | 4096 | 26 s | 11 min | 2.2 min |
| 1.6e10 | 1M | 16384 | 105 s | 44 min | 8.8 min |
| 6.5e10 | 1M | 65536 | 7 min | 2.9 h | 35 min |

The whole speed grid is on the order of 30 to 40 hours of serial CPU time; GPU envs take minutes to about an
hour each.

## Precision study

FP16 and BF16 against an FP32 reference, plus an FP32 rerun and an FP32 other-seed baseline, at k = 1024 and
4096, n = 100k and 1M, both modes, random and real data, 3 seeds (48 cells). It needs
bench-gpu-fork (emeans-rocm installed) and is resumable:

```bash
hatch run bench-gpu-fork:run precision.py
```

Records go to `results/precision.jsonl`.

## Report

```bash
hatch run bench-cpu:run report.py --check
hatch run bench-cpu:run report.py
```

`--check` lists problems (missing cells, fewer than 3 seeds, failed or timed-out cells, results that
contradict the earlier precision spike). `report.py` writes `RESULTS.md` from `results/speed.jsonl` and
`results/precision.jsonl`.

Local only: `report.py` can add a comparison of the Triton records with those of the earlier fork build, which were
replaced by the `emeans-rocm` 0.2.5 rerun (itself replaced by the 0.2.6 rerun) and are not committed. The published `RESULTS.md` leaves it out, because
its source records are not published. On the owner's machine all 192 precision drift_gap medians of the two builds
were bit-identical. To render it into a local page, pass the dropped records and a scratch output:

```bash
hatch run bench-cpu:run report.py --out scratch/RESULTS-with-fork.md \
  --fork-speed scratch/dropped-fork-gpu-speed-20261126T112236.jsonl \
  --fork-precision scratch/dropped-fork-gpu-precision-20261126T112236.jsonl
```

### Version comparison

`RESULTS.md` compares emeans-rocm 0.2.5 with 0.2.6 on the Triton cells.

- The 0.2.5 Triton and precision records were moved out of `results/speed.jsonl` and `results/precision.jsonl` into
  `results/speed-emeans-rocm-0.2.5.jsonl` and `results/precision-emeans-rocm-0.2.5.jsonl` by
  `scratch/drop_025_triton.py` (logged) before the 0.2.6 run wrote the main files.
- The 0.2.5 grid kept no centroids. Centroids of both versions were captured on the 100k subset (8 settings, seeds
  0-2) with `scratch/capture_versions.sh <dir> <seeds>`: run it in `bench-gpu-fork` with the version installed,
  outside quiet hours.
- `hatch run bench-cpu:run compare_versions.py` scores them: a setting passes when the 0.2.6 median objective lies
  inside the min-max of the 0.2.5 objectives and the median ARI between versions is at least the median ARI between
  0.2.5 seeds; any failing setting exits 3 and needs an owner decision. The result is in
  `results/version-ari-emeans-rocm-0.2.5-0.2.6.jsonl`.
- `report.py` renders the comparison when given the three files:

```bash
hatch run bench-cpu:run report.py \
  --prev-speed results/speed-emeans-rocm-0.2.5.jsonl \
  --prev-precision results/precision-emeans-rocm-0.2.5.jsonl \
  --version-ari results/version-ari-emeans-rocm-0.2.5-0.2.6.jsonl
```

## What is committed

Committed: the scripts, tests, `RESULTS.md` and `results/*.jsonl` (records carry no hostname or username; the
host label comes from `BENCH_HOST_LABEL`). Not committed: `cache/` (the 3 GB data file) and `scratch/`
(smoke output, temporary centroids).
