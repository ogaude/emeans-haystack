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
            "Load-contaminated cell-seeds were rerun: 6 cell-seeds (cold and warm, 12 records) across 4 cells, "
            "dropped and rerun on 2026-11-26 on a quiet machine, the drop logged with the dropped lines kept"
        ),
        (
            "concurrent jobs on the machine (agent sessions, an image-model probe, a container that pushed the machine "
            "into swap) had inflated their per-pass times"
        ),
        "the first, slower timings of those cell-seeds; they are kept locally, not published",
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
            "Precision guidance: option B (owner, 2026-11-23); the random-data spike contradictions of the 0.2.6 "
            "records were accepted by the owner on 2026-11-27 (those of the 0.2.5 records on 2026-11-23)"
        ),
        (
            "on random data FP16/BF16 drift equals or slightly exceeds the FP32 other-seed drift; the guidance adds "
            "a caution instead of a claim"
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

# Cell-seeds kept although slower than a clean rerun of the same work: (kept key, reference key, label).
KEPT_SLOW: tuple[tuple[str, str, str], ...] = (
    (
        "emeans-cpu|random|1000000|4096|euclidean|0|main|default|cold",
        "emeans-cpu|random|1000000|4096|euclidean|2|main|default|cold",
        "emeans-cpu random 1M x 4096 seed 0 (vs seed 2)",
    ),
    (
        "emeans-cpu|random|1000000|4096|euclidean|1|main|default|cold",
        "emeans-cpu|random|1000000|4096|euclidean|2|main|default|cold",
        "emeans-cpu random 1M x 4096 seed 1 (vs seed 2)",
    ),
    (
        "faiss-lloyd|cohere|1000000|16384|spherical|1|main|default|cold",
        "faiss-lloyd|cohere|1000000|16384|spherical|1|sweep|25|cold",
        "faiss-lloyd cohere 1M x 16384 seed 1 main (vs its 25-pass sweep point)",
    ),
)

GPU_ENGINE_NOTE = (
    "The Triton rows (`emeans-gpu-triton-fp32`, `emeans-gpu-triton-bf16`) and the precision study ran on "
    "`emeans-rocm` 0.2.6 from PyPI on 2026-11-27, with `unused_threshold=0.1` passed explicitly (mass-based "
    "revival, the default of 0.2.5 and of stock faiss; the own default of 0.2.6 is cost-based revival and is not "
    "used here). The native rows (`emeans-gpu-native-fp32`) ran on 2026-11-22 on the stock e-means engine of "
    "faiss-cpu 1.16.0 through plain PyTorch (recorded as `faiss-cpu`; the records' faiss version is 1.16.0) and "
    "were not rerun. The 0.2.5 records are kept in `results/speed-emeans-rocm-0.2.5.jsonl` and "
    "`results/precision-emeans-rocm-0.2.5.jsonl` and are compared with the 0.2.6 records below."
)

FORK_NOTE = (
    "Before 0.2.5 was published, the Triton cells and the precision study ran on an earlier build of the "
    "author's e-means-rocm fork (commit not recorded). Those records were replaced on 2026-11-26 by the 0.2.5 "
    "rerun, which the 0.2.6 rerun of 2026-11-27 replaced in turn (the 0.2.5 records are kept in labelled files). "
    "The fork records are kept locally and are not published. This comparison is computed from them only when "
    "`report.py` is given `--fork-speed` and `--fork-precision`."
)

OPTION_B_CAUTION = (
    "Use FP32 when stable `cluster_id`s matter: at about 100k rows, FP16 and BF16 change document assignments "
    "almost as much as reseeding does (owner decision, option B, 2026-11-23)."
)

VERSION_NOTE = (
    "Both versions ran the same Triton cells with the same seeds and starting centroids. emeans-haystack passes "
    "`unused_threshold=0.1` (mass-based revival), which equals the default of 0.2.5, so the 0.2.5 records (made with "
    "no explicit threshold) and the 0.2.6 records use the same revival rule. `emeans-rocm` 0.2.6 places revived "
    "centroids along the donor's long axis (upstream faiss change #5975), so centroids can still differ. The 0.2.5 "
    "records are kept in `results/speed-emeans-rocm-0.2.5.jsonl` and `results/precision-emeans-rocm-0.2.5.jsonl`; "
    "the ARI figures come from `compare_versions.py`."
)

VERSION_ARI_CUT: tuple[str, str, str] = (
    (
        "ARI between 0.2.5 and 0.2.6 on 8 settings x 3 seeds only (n = 100k, k = 1024 and 4096, random and Cohere, "
        "Triton FP32 and BF16)"
    ),
    (
        "the 0.2.5 grid kept no centroids or assignments; a full 0.2.5 rerun with saved centroids would cost about 5 "
        "more GPU hours and 6-7 GB"
    ),
    (
        "assignment agreement at 1M rows, at k = 16384 and 65536 and on sweep cells; objectives and times are "
        "compared on all 186 Triton cells"
    ),
)

# Owner history, shown only while the records match it (report.py checks the counts): the number of spike
# contradictions in the 0.2.6 precision records the owner accepted on 2026-11-27, and the smoke-gate outcome
# (settings, settings passing the objective part, settings passing the ARI part) the owner overrode on 2026-11-27.
ACCEPTED_CONTRADICTIONS_026 = 6
VERSION_GATE_RECORDED = (8, 5, 8)

VERSION_GATE_UNMATCHED = (
    "Gate history: the smoke gate run before the full 0.2.6 rerun failed (exit 3) and was overridden by the owner on "
    "2026-11-27. The ARI records given here do not match that run's outcome, so its details are not repeated."
)

VERSION_GATE_NOTE = (
    "Gate and override: before the full 0.2.6 rerun, a smoke gate on the 100k subset (8 settings x 3 seeds) "
    "passed its assignment part in 8 of 8 settings and its objective part in 5 of 8; the 3 failures were all "
    "bf16 (two with a better objective than the 0.2.5 range, random k = 4096 worse by 0.020 against a 0.95 seed "
    "spread), so the gate result is a failure (exit 3). The owner then checked bf16 on 6 seeds: 0.2.6 is never "
    "worse than 0.2.5 there, and at random k = 1024 and Cohere k = 4096 it is better than every 0.2.5 value. On that "
    "evidence the owner overrode the gate on 2026-11-27 and the full rerun went ahead."
)

VERSION_CENTROIDS_NOTE = (
    "GPU centroids at d > 2 differ slightly from 0.2.5, because 0.2.6 places revived centroids differently "
    "(same revival rule, different split placement). The agreement figures above measure how much that moves "
    "assignments."
)

VERSION_TIME_NOTE = (
    "Timings, as measured: the Triton fit times of 0.2.6 are about those of 0.2.5 when cold (median ratio in the "
    "speed line above) and shorter when warm. The whole runs took longer, by about 15-20 %: "
    "the grid 4 h 35 min against 3 h 58 min, the precision study 4964 s against about 4006 s. "
    "These wall times include everything around the fits and are not interpreted "
    "here; speed is not the claim of this page."
)
