# Upstream hand-off: emeans-haystack -> haystack-core-integrations

This file records what must change when `integrations/emeans/` is contributed to
[deepset-ai/haystack-core-integrations](https://github.com/deepset-ai/haystack-core-integrations), plus a draft
new-integration proposal. The goal is that the move needs no change inside `integrations/emeans/` other than the items
under "Differences to resolve on the move".

**Nothing has been opened upstream: no issue, no pull request. This repository has no remote.** The proposal below is a
draft only.

Checked against the live upstream files on **2026-11-26** (read-only, unauthenticated fetches of public URLs):

- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/CONTRIBUTING.md
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/AGENTS.md
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/.github/ISSUE_TEMPLATE/new-integration-proposal.md
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/.github/labeler.yml
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/.github/workflows/CI_coverage_comment.yml
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/README.md (Inventory table)
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/scripts/create_new_integration.py
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/scripts/utils/templates/workflow.yml
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/scripts/utils/templates/pyproject.toml
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/scripts/utils/templates/readme_table_row.txt
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/integrations/pyversity/README.md
- https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations/main/LICENSE

Re-fetch before any real use: the upstream rules change.

## Checklist

Status: Done / N/A / Todo-on-move. "Reviewer to confirm" means the rule is probably met but was not verified in a
dedicated check.

| Rule | Source | Status | Evidence |
|---|---|---|---|
| Name is `PREFIX-haystack` | CONTRIBUTING "Create a new integration" | Done | package `emeans-haystack`, folder `integrations/emeans/` |
| Scaffold creates the integration folder, workflow, `labeler.yml` entry, `CI_coverage_comment.yml` entry and root README row | CONTRIBUTING, `scripts/create_new_integration.py` | Done for folder and workflow (by hand); the three monorepo edits are Todo-on-move | `.github/workflows/emeans.yml`; see "Monorepo edits on the move" |
| Use Hatch only; never run `python` or `pip` directly | AGENTS "Environment" | Done | every command in the CI workflow and the docs goes through `hatch run` |
| Versioning by git tag `integrations/<name>-v<version>`; only maintainers release | AGENTS "Versioning" | Done | `tag-pattern` and `git_describe_command` in `pyproject.toml`; upstream maintainers release, the 0.1.0 tag lives only in this repository |
| Changelog is auto-generated, not edited by hand | AGENTS "Changelogs" | N/A | `integrations/emeans/CHANGELOG.md` is a stub; do not hand-edit it |
| Target Haystack 3.x; create clients and models in `warm_up()`; keep `__init__` light | AGENTS "API Design" | Reviewer to confirm | `EmeansDocumentClusterer.warm_up()` loads the saved model; the fitter and the ranker hold no model, but a reviewer should confirm the split |
| Keep `to_dict()`/`from_dict()` symmetric with `__init__`; build on `default_to_dict`/`default_from_dict` | AGENTS "API Design" | Done | `to_dict`/`from_dict` on all three components; round-trip tests in `integrations/emeans/tests/` |
| Add async APIs only for native async I/O | AGENTS "API Design" | Done | no `run_async` anywhere; the components are CPU/GPU compute, not async I/O |
| `haystack_integrations` roots and intermediate directories are namespace-only (no `__init__.py`); re-export the public API in the concrete `__init__.py`; `py.typed` at the exposed boundary | AGENTS "Code Style", "Type System" | Done | `py.typed` in `components/preprocessors/` and `components/rankers/`; namespace layout is asserted by the conformance tests |
| `pydoc/config_docusaurus.yml` lists the public modules | AGENTS "Documentation" | Done | three modules, see below |
| Every source and test file starts with the SPDX/Apache-2.0 header; `LICENSE.txt` carries real year and holder | AGENTS "Documentation" | Done here, Todo-on-move for the holder | SPDX header on every tracked `.py`; `LICENSE.txt` holder is `ogaude`, upstream copies `Copyright 2023-present deepset GmbH` (see differences) |
| `pyproject.toml` aligned with `scripts/utils/templates/pyproject.toml`: real `authors`, `description`, `keywords`, Python metadata matching tested versions, docs env, shared Ruff config with test-only ignores, markers with `--strict-markers`, every importable package in the `types` env | AGENTS "Config" | Done, with the deltas listed below | `integrations/emeans/pyproject.toml`; `keywords = []` is empty: Todo-on-move, add keywords |
| Dependencies: only directly used, explicit minimums, no upper bounds unless required | AGENTS "Dependencies" | Done | `haystack-ai>=3.5`, `faiss-cpu>=1.16.0`, `numpy>=2.0`, `packaging>=22`. `numpy>=2.0` is not the faiss integration's `numpy<2` pin: faiss-cpu 1.16.0 imports `numpy._core`, which exists only in numpy 2.x, and the lowest-direct run proves numpy 2.0.0 works |
| Workflows pin `uses:` to full commit SHAs | AGENTS "Config" | Done | `checkout` and `setup-python` pinned to the same SHAs as the live scaffold template (v7.0.1, v7.1.0) |
| Workflows test the min and max supported Python on every supported OS | AGENTS "Config" | Todo-on-move (reviewer to confirm) | Linux only, Python 3.11 and 3.15. e-means in faiss-cpu is only verified on Linux. The scaffold template has Windows/macOS paths; maintainers decide whether a Linux-only matrix is accepted, as the faiss integration does |
| Keep `integrations/*/README.md` minimal and template-aligned; link to canonical docs | AGENTS "Documentation" | Todo-on-move | our README has install, GPU, quick start and benchmark sections; the template/pyversity README is badges, two links and Contributing. Move the content to the integration docs page |
| Haystack-style docstrings: summary line, `:param`, `:returns:`, `:raises:`, `### Usage example` | AGENTS "Documentation" | Reviewer to confirm | public components have docstrings; not audited line by line against this rule |
| Update `Document`s with `dataclasses.replace(...)`, never in place | AGENTS "Code Style" | Todo | the clusterer uses `replace`; `tests/test_ranker.py` and others set `doc.score` in place when building fixtures. Switch the fixtures to `replace(...)` before contributing |
| Log with `from haystack import logging` and `{placeholder}` templates | AGENTS "Code Style" | Reviewer to confirm | not audited |
| Import required dependencies at module top; lazy or `try` imports only for optional dependencies | AGENTS "General" | Done | only `torch`, `emeans_rocm` and `faiss.contrib.e_means` internals are imported lazily, all in `_backend.py`; importing the package never imports torch |
| Raise `TypeError` for wrong types, `ValueError` for invalid values | AGENTS "General" | Reviewer to confirm | not audited |
| Tests organised per component, parametrized duplicates, no one-off files | AGENTS "Testing" | Reviewer to confirm | tests are one file per component plus backend/model/conformance files |
| Sort the README integrations table alphabetically | AGENTS "Documentation" | Todo-on-move | `emeans-haystack` goes between `elasticsearch-haystack` and `faiss-haystack` |
| Community contributors may have only one open PR at a time; check `gh pr list --author @me` first | CONTRIBUTING, AGENTS "Pull Requests" | Todo-on-move | planning note; check before opening anything |
| Contributor authored 100 % of the content, has the rights, and agrees the content may be provided under the project license | CONTRIBUTING | Todo-on-move | state it in the proposal (draft below); emeans-rocm is a separate MIT package by the same author and is only an optional runtime dependency |

## Monorepo edits on the move

The scaffold (`scripts/create_new_integration.py`) makes these edits outside `integrations/emeans/`. A contribution PR
must include them. Formats below are from the 2026-11-26 fetch.

1. `.github/labeler.yml`: a block in alphabetical position, between `integration:elasticsearch` and
   `integration:faiss`:

   ```yaml
   integration:emeans:
     - changed-files:
         - any-glob-to-any-file: "integrations/emeans/**/*"
         - any-glob-to-any-file: ".github/workflows/emeans.yml"
   ```

2. `.github/workflows/CI_coverage_comment.yml`: add `- "Test / emeans"` to the `workflows:` list. The list is
   alphabetical: after `- "Test / elasticsearch"` and before `- "Test / faiss"`.

3. Root `README.md`, Inventory table: one new row, alphabetical, between the `elasticsearch-haystack` and
   `faiss-haystack` rows. The scaffold generates it from `scripts/utils/templates/readme_table_row.txt`. The pattern
   has the package link, a type label, and four badges: PyPI version, the `Test / emeans` workflow badge, and two
   coverage badges (`python-coverage-comment-action-data-emeans` and `...-emeans-combined`). The type label is
   `Preprocessor, Ranker` (the scaffold takes one component type; the table's other multi-type rows, for example
   `Embedder, Generator`, show the comma form).

4. Maintainers create the GitHub label `integration:emeans` (listed in the proposal checklist).

5. `.github/workflows/emeans.yml`: take the scaffold's `workflow.yml` and keep only the emeans-specific differences
   (see below), rather than moving our standalone file as is.

## Differences to resolve on the move

| Item | Where | Recommended resolution |
|---|---|---|
| README link `[benchmarks/RESULTS.md](../../benchmarks/RESULTS.md)` points outside `integrations/emeans/` | `integrations/emeans/README.md`, `## Benchmarks` | `benchmarks/` does not move. Replace with an absolute URL to wherever RESULTS.md is hosted, or drop the link. The README also has an absolute `examples/` link to `github.com/ogaude/emeans-haystack/tree/main/examples`; the same applies (`examples/` does not move). Editing `## Benchmarks` fails `tests/test_readme.py` (see its row) |
| `benchmarks/` and `examples/` stay in this repository | repo root | not part of the contribution; the integration must not import from them |
| `[project.urls]` and `authors` point to `github.com/ogaude/emeans-haystack` and `ogaude` | `pyproject.toml` | use the template URLs (`.../deepset-ai/haystack-core-integrations/tree/main/integrations/emeans#readme`, `.../issues`, `.../tree/main/integrations/emeans`); keep a real author or take maintainers' preference |
| `LICENSE.txt` copyright holder: ours is `Copyright 2026-present ogaude <ogaude.code@proton.me>`; the scaffold copies the monorepo LICENSE with `Copyright 2023-present deepset GmbH` | `integrations/emeans/LICENSE.txt`, SPDX headers | ask the maintainers which holder they want for contributed code; the SPDX header holder in source and test files is `ogaude` today. Either holder passes the tests: `test_license_copyright` checks only `LICENSE.txt` itself (Apache-2.0 text, no placeholders, a named holder), and `test_license_matches_repo_root` runs only in this repository (it skips when the root has no `UPSTREAM.md`) |
| `[gpu] = ["torch"]` extra is novel upstream (no core integration declares a torch extra) | `pyproject.toml` | keep if reviewers accept; the fallback is to drop it and keep README-only install instructions. Note the ROCm trap: `[gpu]` pulls the PyPI torch build and replaces an installed ROCm torch, so AMD users must not use it |
| Personal hatch envs `gpu`, `gpu-fork` (ROCm torch from the AMD extra index with `unsafe-best-match`; `emeans-rocm==0.2.5` pin) and `examples` (adds `sentence-transformers-haystack`) | `pyproject.toml` | droppable; the upstream template has none of them, and they are not used by CI. The `gpu` marker registration stays (`--strict-markers` is on) |
| `types` script targets two modules (`-p ...preprocessors.emeans -p ...rankers.emeans`); the template's targets one | `pyproject.toml` | keep: AGENTS requires every importable package in the `types` env |
| `pydoc/config_docusaurus.yml` module paths | `integrations/emeans/pydoc/` | the scaffold writes a single placeholder module; keep ours: `haystack_integrations.components.preprocessors.emeans.fitter`, `...preprocessors.emeans.clusterer`, `...rankers.emeans.ranker` |
| Workflow steps the scaffold template has and ours dropped: coverage comment (`python-coverage-comment-action`), upload of the coverage-comment artifact, "Run integration tests" (the GPU tests would be collected), combined coverage on push, Slack notify on a failed nightly | `.github/workflows/emeans.yml` | on the move take the template and re-enable the coverage steps; the integration-test step passes only if the GPU tests skip without a GPU (they use `importorskip`/CUDA checks; re-verify). Ours adds a `build` job (wheel and sdist metadata checks, `fetch-depth: 0`); keep it only if maintainers want it |
| Runner label `ubuntu-latest` (template uses its own matrix helper) | workflow | follow the template |
| README length against the minimal-README expectation | `integrations/emeans/README.md` | reduce to badges, integration page, changelog and Contributing (pyversity pattern); move installation, GPU and benchmark text to the integration docs page; `tests/test_readme.py` must change with it (see its row) |
| `tests/test_readme.py` checks README content: it runs the `## Quick Start` block, sha256-locks the `## Benchmarks` section (`BENCHMARKS_SHA256`), and requires `## Citation`, `## License` and the `numpy>=2.0` text. The two README rows above (the `../../benchmarks/RESULTS.md` link, the minimal README) break every test in it | `integrations/emeans/tests/test_readme.py` | change it in the same commit as the README. If only the RESULTS.md link changes, update `BENCHMARKS_SHA256` to the new section's hash. When the README is reduced, delete the file, or move the Quick Start exec and the citation checks to wherever the moved text lives (the integration docs page) |
| `CHANGELOG.md` stub | `integrations/emeans/CHANGELOG.md` | generated upstream; do not carry hand edits |
| Version `0.1.0` tag `integrations/emeans-v0.1.0` exists only in this repository | git | upstream maintainers release; nothing to move |
| `keywords = []` | `pyproject.toml` | add relevant keywords (scaffold next step 6) |
| In-place `.score` mutation in test fixtures | `tests/` | switch to `dataclasses.replace` (see checklist) |

## Draft new-integration proposal

**Draft only. Not opened.** Title suggestion: `New integration: emeans (e-means clustering components)`. Label:
`new integration` (the template front matter: `name: New Integration Proposal`, `labels: new integration`). Headings
below follow the template as fetched on 2026-11-26.

### Summary and motivation

e-means is a clustering algorithm from Meta FAIR ([arXiv 2611.09501](https://arxiv.org/abs/2611.09501)) that ships in
faiss as `faiss.contrib.e_means` since faiss v1.16.0 (`faiss-cpu` 1.16.0 on PyPI, 2026-11-10). It is a different
algorithm from Lloyd's k-means; Lloyd's k-means appears only as the benchmark baseline.

The proposed integration, `emeans-haystack`, has three components:

- `EmeansFitter` (type `preprocessors`): trains e-means on the embeddings of a document collection and saves a model
  file.
- `EmeansDocumentClusterer` (type `preprocessors`): loads the saved model and writes a stable `cluster_id` into each
  document's `meta`, so new documents can be assigned to existing clusters at indexing time.
- `EmeansClusterRanker` (type `rankers`): at query time keeps the best document of each cluster, giving
  one-per-cluster diversity without clustering at query time.

Overlap, stated honestly: `pyversity-haystack` already offers diversity ranking (MMR/DPP) and `faiss-haystack` is a
document store on faiss. This integration does neither: it assigns persistent cluster ids and ranks one document per
cluster from those ids.

### Adoption signals

All numbers fetched 2026-11-26.

- **GitHub stars** of the main repository: `facebookresearch/faiss` has 41,936 stars (GitHub API, last push
  2026-11-25). e-means itself has no separate repository; it lives in faiss `contrib/e_means`.
- **PyPI downloads in the last 30 days** of the Python package: `faiss-cpu` 12,975,129 (pypistats API, last month).
  e-means has no separate package, so this is the user base it ships to, not demand for e-means.
- **Release activity**: e-means first shipped in faiss v1.16.0, released to PyPI on 2026-11-10. faiss releases
  regularly.
- **Maintenance**: faiss is maintained by Meta FAIR (commits within the last day at fetch time).
- **Haystack community demand**: none yet. No issue or pull request in `haystack-core-integrations` mentions e-means
  (GitHub issue search for `emeans` returned 0 results; the quoted phrase `"e-means"` returned 2 unrelated results).
- **Anything else**: the technology is very new (the algorithm and its faiss release are about two weeks old), so
  the numbers above are for faiss as a whole. Interest is expected to grow as the paper circulates because e-means
  reaches a given clustering objective in fewer passes than Lloyd's k-means, which matters for large embedding
  collections that Haystack users index.

### Detailed design

- **Components and I/O.** `EmeansFitter.run(documents)` reads `Document.embedding`, fits e-means with `k` clusters,
  writes the model to `model_path` and returns `stats` (backend, precision, timings, cluster balance).
  `EmeansDocumentClusterer.run(documents)` returns the documents with `meta["cluster_id"]`.
  `EmeansClusterRanker.run(documents, top_k)` returns the best document per cluster by score, at most `top_k`.
- **Model file.** A `.npz` file loaded with `allow_pickle=False`; loading never executes code.
- **Devices.** CPU always works through `faiss-cpu`. A GPU is used only when PyTorch is installed (`[gpu]` extra);
  importing the package never imports PyTorch. On AMD, the optional, separate `emeans-rocm` package (MIT, same author
  as this proposal) provides Triton kernels; it is never a dependency.
- **Dependencies.** `haystack-ai>=3.5`, `faiss-cpu>=1.16.0`, `numpy>=2.0`, `packaging>=22`.
- **Non-goals.** No query-time clustering, no `run_async`, no CLI, no own configuration file or environment
  variables.
- **Benchmarks.** The claim is fewer passes, not speed: e-means needs fewer passes than Lloyd's k-means to reach the
  same objective in all 5 sweep comparisons (k = 4096 and 16384), while its passes are not faster, and at its default
  5 passes Lloyd's k-means keeps the better objective in 3 settings. Full tables, protocol and every cut made for time
  or memory are in `benchmarks/RESULTS.md` of the author's repository. That repository has no public URL yet, so the
  proposal would attach the file or link it once it exists.

### Checklist

Copied from the template, all unchecked:

- [ ] The code is documented with docstrings and was merged in the `main` branch
- [ ] Docs are published at https://docs.haystack.deepset.ai/
- [ ] There is a Github workflow running the tests for the integration nightly and at every PR
- [ ] A new label named like `integration:<your integration name>` has been added to the list of labels for this repository
- [ ] The `labeler.yml` file has been updated
- [ ] The package has been released on PyPI
- [ ] An integration tile with a usage example has been added to https://github.com/deepset-ai/haystack-integrations
- [ ] The integration has been listed in the Inventory section of the repository README
- [ ] The feature was announced through social media

### Contributor notes

- One open pull request at a time; nothing is open today (and nothing has been opened by this repository's work).
- The contributor authored 100 % of the content, has the rights to it, and agrees it may be provided under the project
  license. `emeans-rocm` is a separate MIT package by the same author and is only an optional runtime dependency.
