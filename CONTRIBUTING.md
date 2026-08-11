# Contributing to PhenoCV

Thanks for your interest in improving PhenoCV! This document explains how to
set up a development environment, run tests, and submit changes.

## Code of Conduct

By participating, you agree to uphold our
[Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to
the maintainers via GitHub issues.

## Development setup

PhenoCV keeps heavy GPU dependencies optional so that CPU-only contributors
and CI stay fast.

```bash
# Clone
git clone https://github.com/perseus-wy/PhenoCV.git
cd PhenoCV

# Create an environment (any tool you like)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install in editable mode with dev + optional video deps
pip install -e ".[dev,video]"
```

If you only want to work on the CPU logic (adapters, ROI math, QA metrics),
`pip install -e ".[dev]"` is enough — `torch`/`sam2` are only needed for the
actual GPU video propagation layer, which is lazily imported.

## Running tests

```bash
pytest
```

The test suite is CPU-only and mocks the GPU propagation layer, so it runs
without a GPU or downloaded weights.

## Pull requests

1. Fork the repo and create a topic branch (`feat/...`, `fix/...`).
2. Keep changes focused; add/extend tests for new behavior.
3. Run `pytest` locally before pushing.
4. Open a PR against `main` with a clear description of the change and its
   motivation.

## Coding style

- Format with `black` and lint with `ruff` (config in `pyproject.toml` if
  present).
- Prefer clear, data-source-agnostic APIs. New dataset formats should be
  added as **adapters**, not by patching the core engine.
- Document public functions and keep docstrings bilingual-friendly.

## Reporting issues

Open a GitHub issue with a minimal reproduction (synthetic sample preferred)
and your environment (`phenocv` version, Python, OS, GPU/driver if relevant).
