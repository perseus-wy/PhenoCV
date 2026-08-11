# AGENTS.md — PhenoCV (for OpenAI Codex)

PhenoCV is an open-source **vision toolkit for plant phenotyping**. The first
module is **temporal canopy segmentation**: a few manually labeled keyframes of a
plant/object sequence are propagated to a fully segmented time series with
SAM 2 video propagation.

**Agent skill:** see [`SKILL.md`](./SKILL.md) for the full trigger conditions,
concepts, CLI/API usage, adapter contract, presets, and QA conventions. This file
is a thin entry point — do not duplicate the skill body here.

## Quick commands

```bash
pip install -e ".[dev]"                       # CPU-only core + tests
pytest                                        # 16 CPU tests
python tools/make_demo_sample.py --out samples/demo   # synthetic sample (no real data)

# Real segmentation (needs torch + sam2 + a SAM 2 checkpoint):
phenocv segment --adapter csv --manifest samples/demo/manifest.csv \
  --config configs/default.yaml --preset plant_phenotyping \
  --checkpoint /path/to/sam2.1_hiera_l.pt --model-cfg sam2.1_hiera_l.yaml \
  --output results/demo --device cuda
```

## Contribution / test conventions

- Engine logic (`src/phenocv/engine.py`) must stay **importable without CUDA**;
  `torch`/`sam2` are imported lazily inside `Sam2VideoPropagator`.
- Keep the **core engine data-source agnostic** — never read a dataset layout
  directly in `engine.py`; add an adapter under `src/phenocv/adapters/` instead.
- Every frame must carry a `pred_source` provenance tag (manual / propagated /
  propagated_lowthr / point_rescue / failed_empty).
- Run `pytest` (CPU-only) before opening a PR. See [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Repo map

```
src/phenocv/{engine,cli,config}.py
src/phenocv/adapters/{base,csv_manifest,plant_phenotyping}.py
configs/default.yaml          # propagation params + presets
tests/                        # CPU-only unit + orchestration tests
tools/make_demo_sample.py     # synthetic sample generator (no real data)
docs/{tuning,export_formats,adapter_guide}.md
```
