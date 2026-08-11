# CLAUDE.md — PhenoCV (for Claude Code)

PhenoCV is an open-source **vision toolkit for plant phenotyping**. Two modules:
(1) **temporal canopy segmentation** — a few manually labeled keyframes of a
plant/object sequence are propagated to a fully segmented time series with
SAM 2 video propagation; (2) **phenotype computation** (`src/phenocv/phenotypes/`)
— a pluggable, fail-closed engine that derives 2D shape / RGB vegetation / 3D
canopy height / multispectral traits from a mask (+ optional richer inputs).

**Agent skill:** see [`SKILL.md`](./SKILL.md) for the full trigger conditions,
concepts, CLI/API usage, adapter contract, presets, phenotyping tiers, and QA
conventions. This file is a thin project-instructions entry point — do not
duplicate the skill body here.

## Quick commands

```bash
pip install -e ".[dev]"                       # CPU-only core + tests
pytest                                        # 30 CPU tests (segmentation + phenotypes)
python tools/make_demo_sample.py --out samples/demo   # synthetic sample (no real data)

# Real segmentation (needs torch + sam2 + a SAM 2 checkpoint):
phenocv segment --adapter csv --manifest samples/demo/manifest.csv \
  --config configs/default.yaml --preset plant_phenotyping \
  --checkpoint /path/to/sam2.1_hiera_l.pt --model-cfg sam2.1_hiera_l.yaml \
  --output results/demo --device cuda

# Phenotype computation (CPU-only, numpy+cv2, no torch):
phenocv list-traits -v                        # registry: extractors + input contracts
phenocv phenotype --mask mask.png [--rgb rgb.png] \
  [--depth depth_mm.png --calibration intrinsics.yaml] \
  --out traits.json [--csv traits.csv]
```

## Contribution / test conventions

- Engine logic (`src/phenocv/engine.py`) must stay **importable without CUDA**;
  `torch`/`sam2` are imported lazily inside `Sam2VideoPropagator`.
- Keep the **core engine data-source agnostic** — never read a dataset layout
  directly in `engine.py`; add an adapter under `src/phenocv/adapters/` instead.
- Every frame must carry a `pred_source` provenance tag (manual / propagated /
  propagated_lowthr / point_rescue / failed_empty).
- **Phenotypes stay CPU-only** (numpy + cv2, no torch) and **fail-closed** —
  missing/unobservable → `NaN` + `missing_reason` (or `<name>_error`), never a
  fabricated value. Add a new trait by writing one `TraitExtractor` subclass and
  decorating it with `@register`; the orchestrator picks it up — do not modify it.
- Run `pytest` (CPU-only) before opening a PR. See [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Repo map

```
src/phenocv/{engine,cli,config}.py
src/phenocv/adapters/{base,csv_manifest,plant_phenotyping}.py
src/phenocv/phenotypes/       # pluggable trait engine (CPU-only, no torch)
  base.py                     #   registry + TraitExtractor + @register + INPUT_* + available_for
  shape2d.py rgb_indices.py   #   L1 mask-only / L2 mask+rgb
  geometry3d.py calib.py      #   L3 mask+depth+calibration (plant height, mm)
  multispectral.py            #   L4 mask+multispectral (MS400 4-band indices)
  compute_traits.py           #   orchestrator: available_for → merge rows, per-extractor try/except
configs/default.yaml          # propagation params + presets
tests/                        # CPU-only unit + orchestration tests (incl. test_phenotypes.py)
tools/make_demo_sample.py     # synthetic sample generator (no real data)
docs/{tuning,export_formats,adapter_guide}.md
```

## Phenotype engine — how it fits together

`phenocv.phenotypes.compute_traits(mask=..., rgb=..., depth=...,
calibration=..., multispectral=...)` builds the set of available inputs, calls
`available_for(inputs)` to select every registered extractor whose `requires`
is a subset (sorted by tier), runs each in a try/except, and merges the rows.
Tiers: **1** `shape2d` (mask) · **2** `rgb_vegetation_indices` (mask+rgb) ·
**3** `canopy_3d_geometry` (mask+depth+calibration, plant height in mm) ·
**4** `multispectral_vegetation_indices` (mask+multispectral). Extending =
one `@register`-decorated `TraitExtractor`; never edit the orchestrator. Full
API/CLI/tier reference lives in `SKILL.md` → "Phenotyping / 表型计算".
