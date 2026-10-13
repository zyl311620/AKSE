# Local verification record

Date: 2026-10-13. Host: Windows. Runtime: Python 3.12.14.

## Passed

- `python -m unittest discover -s tests -v`: **32 tests passed**.
- Offline CLI demonstration: detection, five hypothesis trajectories, recurrence filtering, temporal/semantic refinement, assimilation, retrieval, and final answering completed.
- CLI integration: synthetic fixture AKSE and baseline runs, manifest/prediction generation, annotation template, evaluation, and paired bootstrap comparison completed.
- HTTP adapter protocol check: request URL/body, output budget, seed, response extraction, and usage recording tested with a mocked transport. No actual external request or paid inference was used.
- BNeCR schema audit: 3,205 rows; supplied snapshot copied without byte changes; SHA-256 matches the source.
- Setuptools wheel construction: `akse_research-0.1.0-py3-none-any.whl` successfully built with the project's build backend.
- Clean upload ZIP: the packaging tool verifies CRC integrity and excludes outputs, weights, caches, `.env` secrets, build directories, and Git internals.

The ZIP is intended for uploading this repository's source. The wheel contains the Python package; datasets and research input files remain separate repository assets. The packaging commands do not push to GitHub.

## Not verified

- Real Baichuan2, BGE-M3, GPU memory/throughput, or the opt-in legacy chat adapter.
- Authenticated external model endpoints and backend-specific seed determinism.
- BERTScore checkpoint downloads or real-model scoring.
- Dependency installation on a clean machine; ranges are declared, not a frozen original-experiment environment.
- Original Wikipedia evidence, temporal observation provenance, original prompts, or benchmark construction logs.
- Numerical reproduction of paper tables, multi-seed performance, significance, or semantic correctness claims.
- Execution of the copied historical figure scripts and validation of their hard-coded numbers against raw experiments.

GitHub Actions is supplied for Windows/Linux and Python 3.10/3.12, but remote CI has not been run. Local checks cover only the above runtime.
