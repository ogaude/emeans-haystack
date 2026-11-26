# SPDX-FileCopyrightText: 2026-present ogaude <ogaude.code@proton.me>
#
# SPDX-License-Identifier: Apache-2.0

"""
Fixed text of RESULTS.md: decisions, run conditions and quoted claims that the records cannot hold.

Everything measured is computed by `report.py` from the records; this module only holds what was decided or
observed outside them (how the runs were scheduled, which records were replaced and why).
"""

# The claims are quoted from `faiss/contrib/e_means/README.md` as shipped in the faiss-cpu 1.16.0 wheel.
UPSTREAM_SOURCE = "`faiss/contrib/e_means/README.md` in the faiss-cpu 1.16.0 wheel"
CLAIM_EPOCHS = (
    "On most [Lloyd's] k-means applications, e-means can achieve the same loss in about 10 times fewer epochs."
)
CLAIM_FIVE = "A value of 5 is typically sufficient to beat Lloyd's loss at convergence."
CLAIM_WALL = (
    "e-means delivers a **~10x speedup in epochs and wall-clock** while achieving **lower loss** compared to "
    "Lloyd's k-means"
)
CLAIM_GPU = "GPUs are much faster, so we recommend using them for configurations where n x k > 10^10"

# Cuts and choices made for time or memory, with their reason and what they hide. Rows whose numbers live in the
# records or in run_grid.py are built in report.py; these rows describe run conditions the records do not hold.
STATIC_CUTS: tuple[tuple[str, str, str], ...] = (
    (
        (
            "CPU cells ran with 8 of the 10 cores (`BENCH_THREADS=8`), `nice 19`, a systemd memory cap of 12 GiB and "
            "swap off"
        ),
        "keep the machine usable and stop one cell from taking all RAM during multi-day runs",
        "CPU timings at 10 threads; CPU s/pass would likely be somewhat lower with all cores",
    ),
    (
        "GPU cells ran with the default 10 host threads",
        "the GPU does the work; host threads only feed it",
        "nothing measured here; CPU and GPU thread counts differ, so GPU-vs-CPU ratios are host-specific",
    ),
    (
        "No heavy runs 23:00-07:00 (owner rule); a watcher stopped each grid at 22:00 after the running cell",
        "quiet hours on a shared personal machine",
        "the CPU grid ran over 2026-11-23 to 2026-11-26, so CPU rows come from different days",
    ),
    (
        (
            "Load-contaminated cells were rerun: 6 cells (12 records) dropped and rerun on 2026-11-26 on a quiet "
            "machine, the drop logged with the dropped lines kept"
        ),
        (
            "concurrent jobs on the machine (agent sessions, an image-model probe, a container that pushed the machine "
            "into swap) had inflated their per-pass times"
        ),
        "the first, slower timings of those cells; they are kept locally, not published",
    ),
    (
        (
            "Cells 4-6 % slower than the clean 2026-11-26 reruns were kept, not rerun (owner, 2026-11-26): emeans-cpu "
            "random 1M x 4096 seeds 0-1 (31.6-31.9 vs 29.8 s/pass), faiss-lloyd cohere 1M x 16384 seed 1 main "
            "(115.1 vs about 110.6 s/pass)"
        ),
        "the gap matches day-to-day variation; a rerun would cost hours",
        ("per-pass timings vary by 4-6 % between days and machine states; objectives and pass counts are not affected"),
    ),
    (
        "scikit-learn cohere records (54) were dropped and rerun on 2026-11-24 after a scoring fix",
        "its centroids were off the unit sphere and scored on a different geometry than the other methods",
        "nothing: the replaced records were wrong, not slow",
    ),
    (
        (
            "The precision study's memory cap was raised from 12 to 16 GiB mid-run (2026-11-26) after 1563 cgroup "
            "`memory.max` events; 0 OOM kills, peak 12.07 GiB"
        ),
        "the study was being throttled at the cap",
        "precision results are unaffected (deterministic fits); its train times are not used anywhere",
    ),
    (
        (
            "Precision guidance: option B (owner, 2026-11-23); the 4 random-data spike contradictions are accepted as "
            "known"
        ),
        (
            "on random data FP16/BF16 drift equals the FP32 other-seed drift; the guidance adds a caution instead of "
            "a claim"
        ),
        "nothing; the contradictions are listed in the Precision section",
    ),
    (
        (
            "`sklearn-lloyd` on spherical data is Euclidean Lloyd on unit-norm rows with one projection of the final "
            "centroids to unit length (scikit-learn has no spherical mode)"
        ),
        (
            "scikit-learn is a reference point; `faiss-lloyd` (`spherical=True`, normalised every iteration) is the "
            "true spherical Lloyd baseline"
        ),
        "how true spherical Lloyd would behave in scikit-learn; read spherical comparisons against `faiss-lloyd`",
    ),
)

GPU_ENGINE_NOTE = (
    "The Triton rows (`emeans-gpu-triton-fp32`, `emeans-gpu-triton-bf16`) and the precision study ran on "
    "`emeans-rocm` 0.2.5 from PyPI on 2026-11-26. The native rows (`emeans-gpu-native-fp32`) ran on 2026-11-22 on "
    "the stock e-means engine of faiss-cpu 1.16.0 through plain PyTorch (recorded as `faiss-cpu`; the records' "
    "faiss version is 1.16.0). `emeans-rocm` 0.2.6 (upcoming) revives clusters by cost by default, which changes "
    "training results, so these figures describe 0.2.5 and do not describe 0.2.6."
)

FORK_NOTE = (
    "Before 0.2.5 was published, the Triton cells and the precision study ran on an earlier build of the "
    "author's e-means-rocm fork (commit not recorded). Those records were dropped from `results/` on 2026-11-26 "
    "and replaced by the 0.2.5 rerun; they are kept locally and are not published. This comparison is computed "
    "from them only when `report.py` is given `--fork-speed` and `--fork-precision`."
)

OPTION_B_CAUTION = (
    "Use FP32 when stable `cluster_id`s matter: at about 100k rows, FP16 and BF16 change document assignments "
    "almost as much as reseeding does (owner decision, option B, 2026-11-23)."
)
